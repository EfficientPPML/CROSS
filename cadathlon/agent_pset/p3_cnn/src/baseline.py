"""Unoptimized, real CROSS baseline; no alternative model or small test ring."""


def build_mapping(model, input_shape, *, batch, devices):
    from jaxite_word import Mapping, nn, packing
    program = nn.vectorize(model, input_shape)
    packed = packing.pack(program, packing.PackingPolicy(lazy_constants=True))
    return Mapping(packed, global_batch=batch, devices=devices)
