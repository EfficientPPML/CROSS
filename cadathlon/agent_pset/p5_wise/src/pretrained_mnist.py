"""Pinned WISE pretrained MNIST CNN, with independent native HerPN reference."""
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import numpy as np

from schedule import Schedule
from wise_source import load_wise
from winograd_network import embed_linear, tile_embedding

ASSETS = {
    "clear/mnist/mnist_cnn_herpn_avgpool_wo_padding_196.pth": "f32f14996c16c6d22c8133e2eba38d0f0bbbaa7b1de954aff69ae3f725a978e0",
    "clear/mnist/HerPN.py": "54d0e3ed16ac2b289a6db6054e7c3f1243ade9ab13e70911aa046fb10d2f94ae",
    "clear/mnist/data/MNIST/raw/t10k-images-idx3-ubyte": "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    "clear/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte": "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
}


def assets(root=None):
    import torch
    root = Path(root or os.environ["WISE_ROOT"]).resolve()
    wise, provenance = load_wise(root)
    for name, expected in ASSETS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned WISE asset changed: {name}")
    checkpoint = torch.load(root / next(iter(ASSETS)), map_location="cpu", weights_only=True)
    image_bytes = (root / "clear/mnist/data/MNIST/raw/t10k-images-idx3-ubyte").read_bytes()
    label_bytes = (root / "clear/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte").read_bytes()
    if struct.unpack(">4I", image_bytes[:16]) != (2051, 10000, 28, 28) or struct.unpack(">2I", label_bytes[:8]) != (2049, 10000):
        raise ValueError("unexpected MNIST test archive")
    images = np.frombuffer(image_bytes, np.uint8, offset=16).reshape(10000, 1, 28, 28)
    labels = np.frombuffer(label_bytes, np.uint8, offset=8)
    provenance = dict(provenance, asset_sha256=ASSETS)
    return wise, provenance, checkpoint, images, labels


def native_model(root, checkpoint):
    """Load the original HerPN module; do not import the training script."""
    import torch
    path = Path(root) / "clear/mnist/HerPN.py"
    spec = importlib.util.spec_from_file_location("p5_wise_original_herpn", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    class CNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
            self.act1 = module.HerPN(32)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.act2 = module.HerPN(64)
            self.fc1 = torch.nn.Linear(3136, 196)
            self.act3 = module.HerPN(196)
            self.fc2 = torch.nn.Linear(196, 10)
        def forward(self, x):
            x = torch.nn.functional.avg_pool2d(self.act1(self.conv1(x)), 2)
            x = torch.nn.functional.avg_pool2d(self.act2(self.conv2(x)), 2).flatten(1)
            return self.fc2(self.act3(self.fc1(x)))
    model = CNN().double()
    model.load_state_dict(checkpoint, strict=True)
    return model.eval()


def coefficients(checkpoint, prefix):
    def get(name): return checkpoint[prefix + "." + name].double().numpy()
    gamma, beta = get("gamma"), get("beta")
    f = [get(f"f_tilde_{i}") for i in range(3)]
    mu = [get(f"running_mean_h{i}") for i in range(3)]
    inv = [1 / np.sqrt(get(f"running_var_h{i}") + 1e-5) for i in range(3)]
    a0 = gamma * (f[0]*(1-mu[0])*inv[0] - f[1]*mu[1]*inv[1] - f[2]*(1/math.sqrt(2)+mu[2])*inv[2]) + beta
    a1 = gamma * f[1] * inv[1]
    a2 = gamma * f[2] * inv[2] / math.sqrt(2)
    if np.any(a2 <= 0):
        raise ValueError("this pinned WISE fusion requires positive quadratic coefficients")
    return a0, a1, a2


def build_pretrained(batch, slots, wise_root=None, *, convolution="toeplitz"):
    """All nine layers of the original checkpoint; no retraining or truncation."""
    if type(batch) is not int or not 1 <= batch <= 64:
        raise ValueError("pretrained batch must be between 1 and 64")
    if convolution not in ("toeplitz", "winograd", "tiled-toeplitz", "hybrid"):
        raise ValueError("unknown convolution schedule")
    wise, provenance, checkpoint, images, labels = assets(wise_root)
    root = Path(wise_root or os.environ["WISE_ROOT"])
    coeffs = [coefficients(checkpoint, name) for name in ("act1", "act2", "act3")]
    weights = {k: v.double().numpy() for k, v in checkpoint.items()}
    native_tiles = convolution in ("winograd", "hybrid")
    def order(c, h, real, c_real=None, tile=1, gap=(1, 1), batch_outer=False):
        return wise.Order(C=c, C_real=c_real, H=h, W=h, H_real=real, W_real=real,
                          batchsize=(batch, 1, 1) if batch_outer else (1, batch, 1),
                          tilesize=(tile, tile), gap=gap)
    # Match the public WISE MNIST packers: a 4x4 first-layer tile becomes
    # the 2x2 Winograd input after pooling; the Toeplitz variant instead
    # uses a 2x2 first tile and channel gaps for its second pooling layer.
    first_tile = {"toeplitz": 1, "winograd": 2, "tiled-toeplitz": 2, "hybrid": 4}[convolution]
    source = order(1, 32, 28, tile=first_tile, batch_outer=True)
    conv1 = order(32, 32, 28, tile=first_tile)
    pool1 = order(32, 16, 14, tile=2 if native_tiles else 1)
    conv2 = order(64, 16, 14, tile=2 if native_tiles else 1)
    pool2 = order(64, 8, 7, gap=(2, 2) if convolution == "tiled-toeplitz" else (1, 1))
    # Preserve the pooled image stride through the first dense layer. These
    # are the published WISE FC layouts: channel gaps for native Winograd,
    # batch-major features for the gap-packed Toeplitz pooling output.
    fc1 = order(256, 1, 1, c_real=196, batch_outer=not native_tiles,
                gap=(64, 1) if native_tiles else (1, 1))
    target = order(10, 1, 1, batch_outer=True)
    def embedding(o):
        if native_tiles and o.tilesize == (2, 2):
            return tile_embedding(o, slots)
        size = o.B*o.C*o.H*o.W
        return np.arange(size), size
    def linear(obj, src, dst):
        im, ni = embedding(src); om, no = embedding(dst)
        return "linear", embed_linear(obj, im, om, input_size=ni, output_size=no).pack(slots)
    def conv(name, src, dst, c):
        kernel, bias = weights[name + ".weight"], weights[name + ".bias"]
        scale = np.sqrt(c[2])
        if not native_tiles or (convolution == "hybrid" and name == "conv1"):
            return linear(wise.pack_conv2d_toeplitz(src, dst, 1, kernel, bn_b=bias, out_scale=scale), src, dst)
        obj = wise.pack_conv2d_winograd(src, dst, kernel, bn_b=bias, out_scale=scale)
        om, size = embedding(dst)
        b = np.zeros(size); b[om] = obj.out_offset; obj.out_offset = b
        return "winograd", obj.pack(slots)
    def herpn(o, c):
        a0, a1, a2 = c
        original = wise.pack_herpn(o, a0, a1 / np.sqrt(a2), a2).pack(slots)
        mapping, size = embedding(o)
        shape = (math.ceil(size/slots), slots)
        out0, out1 = np.zeros(math.prod(shape)), np.zeros(math.prod(shape))
        out0[mapping] = original.a0.ravel()[:len(mapping)]
        out1[mapping] = original.a1.ravel()[:len(mapping)]
        return "herpn", SimpleNamespace(a0=out0.reshape(shape), a1=out1.reshape(shape))
    layers = [conv("conv1", source, conv1, coeffs[0]), herpn(conv1, coeffs[0]),
              linear(wise.pack_avgpool2d(conv1, pool1, size=2, stride=2, padding=0), conv1, pool1),
              conv("conv2", pool1, conv2, coeffs[1]), herpn(conv2, coeffs[1]),
              linear(wise.pack_avgpool2d(conv2, pool2, size=2, stride=2, padding=0), conv2, pool2),
              linear(wise.pack_fc(pool2, fc1, weights["fc1.weight"], weights["fc1.bias"],
                                  out_scale=np.pad(np.sqrt(coeffs[2][2]), (0, 60))), pool2, fc1),
              herpn(fc1, coeffs[2]),
              linear(wise.pack_fc(fc1, target, weights["fc2.weight"], weights["fc2.bias"]), fc1, target)]
    mapping, size = embedding(source)
    input_indices = np.array([mapping[source.idx(b, 0, h, w)] for b, _, h, w in np.ndindex((batch, 1, 28, 28))])
    plan = Schedule(slots, layers, batch*784, batch*10, provenance, input_indices, size)
    # The normalization matches the published training/evaluation transform.
    x = (images[:batch].astype(np.float64)/255 - .1307)/.3081
    return plan, x, labels[:batch], native_model(root, checkpoint)
