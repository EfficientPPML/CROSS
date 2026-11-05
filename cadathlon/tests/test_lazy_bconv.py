"""Demand-built public BConv controls preserve field, device and owner semantics."""
import copy
import dataclasses
import gc
from pathlib import Path
import pickle
import sys
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'cadathlon/common'))
from jaxite_word import he_params
import jax
import finite_field

jax.config.update('jax_enable_x64', True)
RECORDS = []


def make_cache(cd=2, backend=finite_field.BarrettContext, perf=False):
    # Public-control fixture only, deliberately not an encrypted/security proof.
    cache = he_params.HEParameterCache(
        [12289, 40961, 65537, 114689], [147457, 163841],
        16, 16, 2, cd, finite_field_context=backend, perf_test=perf)
    key = np.zeros((2, 256, 6), dtype=np.uint32)
    cache.initialize(key, key)
    return cache


def assert_public_equal(left, right, seen=None):
    """Compare every stored public control recursively, including backend data."""
    seen = set() if seen is None else seen
    pair = (id(left), id(right))
    if pair in seen:
        return
    seen.add(pair)
    if isinstance(left, (np.ndarray, jax.Array)):
        assert left.dtype == right.dtype and left.shape == right.shape
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_public_equal(left[name], right[name], seen)
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_public_equal(a, b, seen)
    elif hasattr(left, '__dict__') and not isinstance(left, type):
        assert type(left) is type(right)
        assert_public_equal(vars(left), vars(right), seen)
    else:
        assert left == right


@pytest.mark.parametrize('cd', [1, 2])
@pytest.mark.parametrize('backend', [finite_field.BarrettContext, finite_field.MontgomeryContext])
def test_lookup_does_not_build_unused_controls_and_direct_field_is_exact(monkeypatch, cd, backend):
    monkeypatch.delenv('CROSS_EAGER_LEVEL_PARAMS', raising=False)
    calls = []
    original = he_params._build_level_bconv_params
    def track(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(he_params, '_build_level_bconv_params', track)
    cache = make_cache(cd, backend)
    for level in range(cache.max_level + 1):
        for getter in (cache.get_sliced_ntt_q, cache.get_sliced_ff_q, cache.get_psi_q,
                       cache.get_inv_psi_q, cache.get_psi_qp, cache.get_inv_psi_qp):
            getter(level)
    assert calls == []
    for level in range(cache.max_level + 1):
        params = cache._level(level)
        deferred = vars(params)['bconv_params']
        assert isinstance(deferred, he_params._DeferredBConvParameters)
        q = cache.q_moduli_at_level(level)
        expected = original(level, len(q), q, q + cache.p_towers,
                            cache.num_p, cache.dnum, backend, False)
        actual = params.bconv_params
        assert actual is cache.get_bconv_params(level)
        assert calls[-1] == level and calls.count(level) == 1
        assert_public_equal(actual, expected)
        assert isinstance(vars(params)['bconv_params'], he_params.LevelBConvParams)


@pytest.mark.parametrize('eager,perf', [(True, False), (False, True), (True, True)])
def test_eager_and_perf_generation_order_stays_eager(monkeypatch, eager, perf):
    monkeypatch.setenv('CROSS_EAGER_LEVEL_PARAMS', '1' if eager else '0')
    calls = []
    original = he_params._build_level_bconv_params
    def track(*args):
        calls.append((args[0], args[-1]))
        return original(*args)
    monkeypatch.setattr(he_params, '_build_level_bconv_params', track)
    cache = make_cache(perf=perf)
    if eager:
        assert calls == [(level, perf) for level in range(cache.max_level + 1)]
    else:
        assert calls == []
        cache.get_sliced_ntt_q(0)
        assert calls == [(0, True)]
    assert all(isinstance(vars(p)['bconv_params'], he_params.LevelBConvParams)
               for p in cache.level_params.values())


def test_retained_parameters_use_immutable_spec_after_owner_collection(monkeypatch):
    monkeypatch.delenv('CROSS_EAGER_LEVEL_PARAMS', raising=False)
    cache = make_cache()
    params = cache._level(0)
    q = cache.q_moduli_at_level(0)
    expected = cache._build_bconv_params(0, len(q), q, q + cache.p_towers)
    pending = vars(params)['bconv_params']
    assert copy.deepcopy(pending) is pending
    cache_ref = weakref.ref(cache)
    keys = weakref.ref(cache.eval_key_a_full)
    cache.q_towers.reverse()
    cache.p_towers.reverse()
    cache.dnum = 99
    del cache
    jax.clear_caches()
    gc.collect()
    assert cache_ref() is None and keys() is None
    assert_public_equal(params.bconv_params, expected)


def test_retry_manual_replacement_and_dataclass_helpers(monkeypatch):
    monkeypatch.delenv('CROSS_EAGER_LEVEL_PARAMS', raising=False)
    cache = make_cache()
    params = cache._level(0)
    pending = vars(params)['bconv_params']
    original = he_params._build_level_bconv_params
    with monkeypatch.context() as patch:
        def fail(*args):
            raise RuntimeError('injected public-control construction failure')
        patch.setattr(he_params, '_build_level_bconv_params', fail)
        with pytest.raises(RuntimeError, match='injected public-control'):
            params.bconv_params
    assert vars(params)['bconv_params'] is pending
    actual = params.bconv_params
    assert isinstance(actual, he_params.LevelBConvParams)
    assert dataclasses.replace(params).bconv_params is actual
    assert isinstance(dataclasses.asdict(params)['bconv_params'], dict)
    assert '_DeferredBConvParameters' not in repr(params)
    other = cache._level(1)
    other.bconv_params = None
    assert other.bconv_params is None
    replacement = object()
    other.bconv_params = replacement
    assert other.bconv_params is replacement
    assert he_params._build_level_bconv_params is original


def test_eviction_rebuild_keeps_retained_parameters_valid(monkeypatch):
    monkeypatch.delenv('CROSS_EAGER_LEVEL_PARAMS', raising=False)
    monkeypatch.setenv('CROSS_LEVEL_PARAMS_MAX', '1')
    cache = make_cache()
    old = cache._level(0)
    cache.get_sliced_ntt_q(1)
    assert list(cache.level_params) == [1]
    new = cache._level(0)
    assert old is not new
    assert_public_equal(old.bconv_params, new.bconv_params)


@pytest.mark.parametrize('operation', ['pickle4', 'pickle5', 'copy', 'deepcopy'])
def test_unmaterialized_parameters_remain_serializable(monkeypatch, operation):
    monkeypatch.delenv('CROSS_EAGER_LEVEL_PARAMS', raising=False)
    cache = make_cache()
    params = cache._level(0)
    assert isinstance(vars(params)['bconv_params'], he_params._DeferredBConvParameters)
    if operation.startswith('pickle'):
        restored = pickle.loads(pickle.dumps(params, protocol=int(operation[-1])))
    else:
        restored = getattr(copy, operation)(params)
    assert isinstance(vars(params)['bconv_params'], he_params.LevelBConvParams)
    assert isinstance(vars(restored)['bconv_params'], he_params.LevelBConvParams)
    assert_public_equal(restored, params)


def test_delayed_read_keeps_original_construction_device(monkeypatch):
    if len(jax.devices()) < 2:
        pytest.skip('requires two logical CPU devices')
    monkeypatch.delenv('CROSS_EAGER_LEVEL_PARAMS', raising=False)
    first, second = jax.devices()[:2]
    with jax.default_device(first):
        cache = make_cache()
        params = cache._level(0)
    assert vars(params)['bconv_params'].device == first
    with jax.default_device(second):
        controls = params.bconv_params
    assert controls.PInvModq.devices() == {first}
    assert all(a.devices() == {first} for a in controls.select_tower_index)
