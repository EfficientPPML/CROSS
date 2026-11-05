"""Public rotation caches preserve exact ciphertexts across keys and levels."""
import os
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_pset/p5_wise/src"))


@pytest.mark.skipif(os.environ.get("CROSS_TEST_WISE_HE") != "1", reason="opt-in encrypted WISE integration")
def test_cached_rotations_match_core_across_keys_levels_fresh_inputs_and_eviction():
    from cross_backend import profile, create_contexts, make_arithmetic
    from rotation_cache import CachedRotation
    spec = profile(3)
    slots = spec["degree"]//2
    keys = [1, 17, slots-1]
    client, server = create_contexts(spec, keys)
    ordinary = make_arithmetic(server)
    cached = CachedRotation(server, 128*1024**2)
    tiny = CachedRotation(server, 1)
    x = np.linspace(-.6, .8, slots)
    for offset in (0., .2):
        fresh = client.encrypt_slots(x+offset)
        lower = ordinary.weighted_sum([fresh], [np.ones(slots)])
        # Alternate actual levels; each iteration includes multiple keys.
        for value in (fresh, lower, fresh):
            for index in keys:
                expected = ordinary.rotate(value, index)
                actual = cached.rotate(value, index)
                np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
                assert tuple(actual.moduli) == tuple(expected.moduli)
                assert actual._ckks_scale == expected._ckks_scale
                assert getattr(actual, "_ckks_nsd", 1) == getattr(expected, "_ckks_nsd", 1)
                assert hasattr(actual, "_ckks_nsd") == hasattr(expected, "_ckks_nsd")
                np.testing.assert_allclose(client.decrypt_slots(actual), np.roll(x+offset, -index), atol=1e-6, rtol=0)
        eviction_result = tiny.rotate(lower, 17)
        np.testing.assert_array_equal(eviction_result.polynomial, ordinary.rotate(lower, 17).polynomial)
        assert tiny.cached_bytes == 0
    assert cached.binding_hits > 0 and cached.state_hits > 0
    assert cached.compile_count == 2
    assert 0 < cached.peak_cached_bytes <= cached.cache_bytes
    cached.clear()
    assert not cached.entries and cached.cached_bytes == 0
    np.testing.assert_array_equal(cached.rotate(fresh, 17).polynomial, ordinary.rotate(fresh, 17).polynomial)
    with pytest.raises(ValueError, match="integer"):
        cached.rotate(fresh, True)
