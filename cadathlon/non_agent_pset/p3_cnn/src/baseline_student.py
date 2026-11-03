"""Opt-in secure CNN baseline with less setup work for student iteration.

The supplied CNN and correctness gates remain unchanged. For original CNNs,
a 44-bit composite scale and eight HYBRID digits derive N=16384; the explicitly
reduced compact-v1 CNNs use a 42-bit scale and N=8192, all at 128-bit security. Arrays are
resident runtime operands instead of multi-gigabyte HLO constants. This
profile needs independent calibration; legacy baseline times do not apply.
"""


def pack_model(model, input_shape):
    """Run the Torch frontend serially before any concurrent HE setup."""
    from jaxite_word import nn, packing
    if getattr(model, '_cadathlon_compact_profile', False):
        policy = packing.PackingPolicy(
            scaling_mod_size=42, first_mod_size=43, register_word_size=24,
            composite_degree=2, dnum=4, lazy_constants=True,
            name="cadathlon_p3_compact_v1",
        )
    else:
        policy = packing.PackingPolicy(
            scaling_mod_size=44, first_mod_size=45, register_word_size=32,
            composite_degree=2, dnum=8, lazy_constants=True,
            name="cadathlon_p3_student",
        )
    return packing.pack(nn.vectorize(model, input_shape), policy)


def build_packed_mapping(packed, *, batch, devices):
    import os
    from jaxite_word import Mapping
    return Mapping(
        packed, global_batch=batch, devices=devices, dynamic_constants=True,
        bsgs_n_jobs=min(8, os.cpu_count() or 1),
    )


def build_mapping(model, input_shape, *, batch, devices):
    return build_packed_mapping(pack_model(model, input_shape), batch=batch, devices=devices)
