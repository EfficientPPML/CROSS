"""Opt-in lower-precision secure P1 reference for local student iteration.

The model and input/output gates are unchanged. A 42-bit composite scale and
four HYBRID digits derive N=8192 at 128-bit security instead of the legacy
60-bit/N=16384 plan. Historical scoring baselines must not be reused for this
profile; it needs its own measurements. The upstream suite adapter selects
this same policy only with P1_BASELINE_PROFILE=student.
"""
from baseline import BaselineService


def packing_policy():
    from jaxite_word import packing
    return packing.PackingPolicy(
        scaling_mod_size=42, first_mod_size=43, register_word_size=24,
        composite_degree=2, dnum=4, lazy_constants=True,
        name="cadathlon_p1_student")


def build_service(model, input_shape, *, batch, devices, keys=None):
    return BaselineService(model, input_shape, batch, devices, keys, policy=packing_policy())
