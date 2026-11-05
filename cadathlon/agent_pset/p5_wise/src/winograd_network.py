"""Embed WISE's four tile planes into whole CKKS ciphertexts without replication."""
import math
import numpy as np
import scipy.sparse as sp
from schedule import Schedule
from workloads import parameters


def tile_embedding(order, slots):
    """Map WISE logical tile positions to separately zero-padded ciphertext planes."""
    if order.tilesize != (2, 2):
        raise ValueError("native Winograd requires 2x2 tiles")
    plane = order.B * order.C * order.H * order.W // 4
    stride = math.ceil(plane / slots) * slots
    logical = np.arange(4 * plane, dtype=np.int64)
    return (logical // plane) * stride + logical % plane, 4 * stride


def embed_linear(transform, input_map=None, output_map=None, *, input_size=None, output_size=None):
    """Reindex the upstream sparse transform for physical tile-plane padding.

    Only public storage positions change. No new model terms, inputs or answers
    are introduced. WISE still performs diagonal extraction and BSGS planning.
    """
    coo = transform.mat.tocoo()
    rows = coo.row if output_map is None else output_map[coo.row]
    cols = coo.col if input_map is None else input_map[coo.col]
    shape = (output_size or transform.mat.shape[0], input_size or transform.mat.shape[1])
    transform.mat = sp.csr_matrix((coo.data, (rows, cols)), shape=shape)
    if output_map is not None and transform.bias is not None:
        bias = np.zeros(shape[0]); bias[output_map] = transform.bias
        transform.bias = bias
    return transform


def pack_native(wise, source, target, kernel, bias, slots):
    """Pad each public bias plane consistently with native ciphertext outputs."""
    native = wise.pack_conv2d_winograd(source, target, kernel, bn_b=bias)
    output_map, output_size = tile_embedding(target, slots)
    padded_bias = np.zeros(output_size)
    padded_bias[output_map] = native.out_offset
    native.out_offset = padded_bias
    return native.pack(slots)


def build_cnn(case, slots, wise, provenance):
    p = parameters(case)
    batch, side, channels = case["batch"], case["side"], case["channels"]
    # Two trailing zero pixels prevent circular neighbor routing from crossing
    # image borders. Real extents, example count and model weights stay fixed.
    padded = 1 << (side + 1).bit_length()
    if (padded // 2) ** 2 > slots:
        raise ValueError("one image tile plane must fit in a ciphertext for native WISE neighbor routing")
    source = wise.Order(C=1, H=padded, W=padded, H_real=side, W_real=side,
                        batchsize=(batch, 1, 1), tilesize=(2, 2))
    conv = wise.Order(C=channels, H=padded, W=padded, H_real=side, W_real=side,
                      batchsize=(1, batch, 1), tilesize=(2, 2))
    pooled = wise.Order(C=channels, H=padded//2, W=padded//2, H_real=side//2, W_real=side//2,
                        batchsize=(1, batch, 1))
    target = wise.Order(C=case["classes"], H=1, W=1, batchsize=(batch, 1, 1))
    source_map, source_size = tile_embedding(source, slots)
    conv_map, conv_size = tile_embedding(conv, slots)
    inputs = np.array([source_map[source.idx(b, 0, h, w)]
                       for b, _, h, w in np.ndindex((batch, 1, side, side))])
    native = pack_native(wise, source, conv, p["kernel"], p["b1"], slots)
    pool = wise.pack_avgpool2d(conv, pooled, size=2, stride=2, padding=0)
    pool = embed_linear(pool, input_map=conv_map, input_size=conv_size).pack(slots)
    fc = wise.pack_fc(pooled, target, p["w2"], p["b2"]).pack(slots)
    return Schedule(slots, [("winograd", native), ("square", None), ("linear", pool), ("linear", fc)],
                    batch*side*side, batch*case["classes"], provenance, inputs, source_size)


def build_tiled_cnn(case, slots, wise, provenance):
    """WISE Toeplitz convolution with 2x2 planes that make pooling sparse."""
    p = parameters(case)
    batch, side, channels = case["batch"], case["side"], case["channels"]
    source = wise.Order(C=1, H=side, W=side, batchsize=(batch, 1, 1), tilesize=(2, 2))
    # Put distinct images inside each channel plane, matching WISE's batched
    # layouts. Channel expansion then reuses one diagonal across the batch.
    conv = wise.Order(C=channels, H=side, W=side, batchsize=(1, batch, 1), tilesize=(2, 2))
    pooled = wise.Order(C=channels, H=side//2, W=side//2, batchsize=(1, batch, 1))
    target = wise.Order(C=case["classes"], H=1, W=1, batchsize=(batch, 1, 1))
    layers = [("linear", wise.pack_conv2d_toeplitz(source, conv, 1, p["kernel"], bn_b=p["b1"]).pack(slots)),
              ("square", None),
              ("linear", wise.pack_avgpool2d(conv, pooled, size=2, stride=2, padding=0).pack(slots)),
              ("linear", wise.pack_fc(pooled, target, p["w2"], p["b2"]).pack(slots))]
    return Schedule(slots, layers, batch*side*side, batch*case["classes"], provenance,
                    np.asarray(source.perm), batch*side*side)
