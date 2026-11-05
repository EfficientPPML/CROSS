"""Real encrypted fused WISE sums must equal the separate CROSS primitives."""
import os
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_pset/p5_wise/src"))


@pytest.mark.skipif(os.environ.get("CROSS_TEST_WISE_HE") != "1", reason="opt-in encrypted WISE integration")
def test_fused_products_match_ciphertext_bytes_across_chunks_levels_and_fresh_inputs():
    from cross_backend import profile, create_contexts, make_arithmetic
    spec = profile(3)
    client, server = create_contexts(spec, [])
    ordinary = make_arithmetic(server)
    fused = [make_arithmetic(server, linear_chunk=n) for n in (1, 2, 8)]
    n = spec["degree"]//2
    x = np.linspace(-.8, .9, n)
    weights = [np.cos(np.arange(n)*.013+i)*.1 for i in range(5)]
    bias = np.linspace(-.03, .04, n)
    for offset in (0., .125):
        a, b = client.encrypt_slots(x+offset), client.encrypt_slots(-x*.3+offset)
        lower = ordinary.weighted_sum([a], [np.full(n, .25)], bias=.02)
        # A genuine lower-level ciphertext forces both paths to align levels.
        inputs = [a, lower, b, a, lower]
        expected_slots = sum(v*w for v, w in zip(
            [x+offset, (x+offset)*.25+.02, -x*.3+offset, x+offset, (x+offset)*.25+.02], weights)) + bias
        expected = ordinary.weighted_sum(inputs, weights, bias=bias)
        for arithmetic in fused:
            actual = arithmetic.weighted_sum(inputs, weights, bias=bias)
            np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
            assert actual._ckks_scale == expected._ckks_scale
            assert actual._ckks_nsd == expected._ckks_nsd == 1
            assert tuple(actual.moduli) == tuple(expected.moduli)
            np.testing.assert_allclose(client.decrypt_slots(actual), expected_slots, atol=1e-6, rtol=0)
        # Public controls are reused without embedding previous ciphertexts.
        assert [a.linear_fusion.compiler.compile_count for a in fused] == [1, 2, 1]
    for arithmetic in fused:
        with pytest.raises(ValueError, match="nonempty"):
            arithmetic.weighted_sum([], [])
        with pytest.raises(ValueError, match="round away"):
            arithmetic.weighted_sum([a], [np.full(n, 1e-30)])
        with pytest.raises(ValueError, match="finite"):
            arithmetic.weighted_sum([a], [np.full(n, np.nan)])
