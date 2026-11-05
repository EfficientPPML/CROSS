"""Fused HerPN preserves its learned polynomial while consuming one CKKS level."""
import os
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_pset/p5_wise/src"))


@pytest.mark.parametrize("depth", [3, 4, 9])
def test_complete_chain_has_distinct_ntt_primes_and_actual_128_bit_security(depth):
    from cross_backend import profile
    from jaxite_word import he_params
    from sympy import isprime
    spec = profile(depth)
    q, p, degree = spec["q"], spec["p"], spec["degree"]
    assert len(q) == 2*(depth+1)
    assert len(set(q+p)) == len(q+p)
    assert all(isprime(v) and (v-1) % (2*degree) == 0 for v in q+p)
    assert he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, "classical"))


@pytest.mark.skipif(os.environ.get("CROSS_TEST_WISE_HE") != "1", reason="opt-in encrypted WISE integration")
def test_secure_fused_herpn_and_following_linear_composition():
    import jax
    from cross_backend import profile, create_contexts, make_arithmetic
    spec = profile(2)
    client, server = create_contexts(spec, [])
    arithmetic = make_arithmetic(server)
    n = spec["degree"] // 2
    x = np.linspace(-1.25, 1.25, n)
    a1 = np.sin(np.arange(n)) * .75
    a0 = np.cos(np.arange(n)) * .125
    encrypted = client.encrypt_slots(x)
    result = arithmetic.quadratic(encrypted, a1, a0)
    assert arithmetic._level(result) == 1
    assert result._ckks_nsd == 1
    expected = x*x + a1*x + a0
    np.testing.assert_allclose(client.decrypt_slots(result), expected, atol=1e-6, rtol=0)
    following = arithmetic.weighted_sum([result], [np.full(n, .3)], bias=.01)
    jax.block_until_ready(following.polynomial)
    assert arithmetic._level(following) == 0
    np.testing.assert_allclose(client.decrypt_slots(following), expected*.3+.01, atol=1e-6, rtol=0)


@pytest.mark.skipif(os.environ.get("CROSS_TEST_WISE_HE") != "1", reason="opt-in encrypted WISE integration")
def test_pretrained_depth_retains_large_logits_at_final_level():
    from cross_backend import profile, create_contexts, make_arithmetic
    spec = profile(9)
    client, server = create_contexts(spec, [])
    arithmetic = make_arithmetic(server)
    x = np.linspace(-16, 32, spec["degree"]//2)
    result = client.encrypt_slots(x)
    for _ in range(9):
        result = arithmetic.weighted_sum([result], [np.ones(len(x))])
    assert arithmetic._level(result) == 0
    np.testing.assert_allclose(client.decrypt_slots(result), x, atol=1e-5, rtol=0)
