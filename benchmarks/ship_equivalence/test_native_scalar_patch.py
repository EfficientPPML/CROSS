"""Regression for the isolated degree-1 scalar precision patch.

Set CROSS_EQUIVALENCE_PATCH_TESTS=1 and put the patched CROSS jaxite_word
directory on PYTHONPATH. The production checkout intentionally lacks this
experimental patch, so this test is opt-in.
"""
import json
import os
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('CROSS_EQUIVALENCE_PATCH_TESTS') != '1',
    reason='Requires an isolated CROSS checkout with the scalar precision patch',
)


def test_small_scalar_survives_addition_at_squared_scale():
    import bootstrapping
    import ckks_ctx
    import key_gen

    jax.config.update('jax_enable_x64', True)
    manifest = json.loads(Path(__file__).with_name('common-n32.json').read_text())
    n, q, p = manifest['degree'], manifest['q'], manifest['p']
    scale = manifest['scale']
    ctx = ckks_ctx.CKKSContext(dict(
        degree=n, num_slots=n//2, q_towers=q, p_towers=p,
        scaling_factor=scale, output_scale=scale, p=30, CKKS_M_FACTOR=1,
        max_bits_in_word=31, composite_degree=1, noise_scale_degree=1,
        **key_gen.gen_pke_pair(q, [], n),
    ))
    ctx.program_initialization(total_rotation_indices=[], dnum=2, r=4, c=8, batch=1)
    engine = bootstrapping.Bootstrap(ctx)
    engine.control_gen(level_budget=[1, 1])
    ciphertext = ctx.encrypt(ctx.encode_at_level([.25]*(n//2), 2, scale=scale**2))
    ciphertext._ckks_nsd = 2
    output = engine._add_scalar(ciphertext, 1e-10, 2, noise_scale_deg=2)
    # This constant is resolvable at the actual scale (~2^60). Rounding it at
    # the base scale (~2^30) incorrectly discards it entirely.
    np.testing.assert_allclose(ctx.decrypt_slots(output), .25+1e-10, atol=1e-12, rtol=0)
