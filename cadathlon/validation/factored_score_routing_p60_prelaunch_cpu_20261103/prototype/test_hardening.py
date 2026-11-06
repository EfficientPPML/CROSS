"""Callback failures must stop before another encrypted primitive executes."""
import copy
import numpy as np
import pytest
from test_routing import Arithmetic, Value, backend_free, ShallowScoreLayout, ShallowScoreRouting, SLOTS


@pytest.mark.parametrize('nsd', [True, 1.0, np.int64(1), np.float64(1)])
def test_exact_integer_noise_degree_required(nsd):
    arithmetic = Arithmetic()
    value = Value(np.ones(SLOTS))
    value._ckks_nsd = nsd
    router = ShallowScoreRouting(arithmetic, ShallowScoreLayout(0, 0))
    with pytest.raises(ValueError, match='integer noise-scale'):
        router.forward(value)
    assert not arithmetic.calls


@pytest.mark.parametrize('scale', [True, False, 0, -1, float('nan'), float('inf'),
                                  complex(2**56), np.longdouble('1e-4900'), np.longdouble('1e4900')])
def test_invalid_scale_fails_before_primitive(scale):
    arithmetic = Arithmetic()
    value = Value(np.ones(SLOTS))
    value._ckks_scale = scale
    router = ShallowScoreRouting(arithmetic, ShallowScoreLayout(0, 0))
    with pytest.raises(ValueError):
        router.forward(value)
    assert not arithmetic.calls


@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('callback_index', [1, 2])
@pytest.mark.parametrize('mutation', ['result_bool_nsd', 'result_float_nsd', 'result_payload',
                                     'result_ntt', 'result_level_type', 'result_scale_type',
                                     'input_bool_nsd', 'input_payload', 'context', 'cache',
                                     'q_moduli', 'p_moduli', 'layout', 'layout_bool',
                                     'context_batch_type', 'context_degree_type', 'refresher',
                                     'bound_aware'])
def test_callback_mutations_stop_without_another_primitive(inverse, callback_index, mutation):
    arithmetic = Arithmetic()
    value = Value(np.arange(SLOTS))
    event_count = 0
    calls_at_mutation = None

    def callback(label, result, counts):
        nonlocal event_count, calls_at_mutation
        event_count += 1
        if event_count != callback_index:
            return
        calls_at_mutation = len(arithmetic.calls)
        if mutation == 'result_bool_nsd': result._ckks_nsd = True
        elif mutation == 'result_float_nsd': result._ckks_nsd = 1.0
        elif mutation == 'result_payload': result.polynomial = result.polynomial.copy()
        elif mutation == 'result_ntt': result.ntt_ctx = object()
        elif mutation == 'result_level_type': result.level = float(result.level)
        elif mutation == 'result_scale_type': result._ckks_scale = int(result._ckks_scale)
        elif mutation == 'input_bool_nsd': value._ckks_nsd = True
        elif mutation == 'input_payload': value.polynomial = value.polynomial.copy()
        elif mutation == 'context': arithmetic.ctx = copy.copy(arithmetic.ctx)
        elif mutation == 'cache': arithmetic.ctx._param_cache = copy.copy(arithmetic.ctx._param_cache)
        elif mutation == 'q_moduli': arithmetic.ctx._param_cache.q_towers = [17, 97]
        elif mutation == 'p_moduli': arithmetic.ctx._param_cache.p_towers = [193]
        elif mutation == 'layout': router.layout = ShallowScoreLayout(0, 0)
        elif mutation == 'layout_bool': object.__setattr__(router.layout, 'key_tile', False)
        elif mutation == 'context_batch_type': arithmetic.ctx._param_cache.batch = True
        elif mutation == 'context_degree_type': arithmetic.ctx.degree = float(2*SLOTS)
        elif mutation == 'refresher': arithmetic.refresher = object()
        elif mutation == 'bound_aware': arithmetic.declare_bound = lambda *args: None
        else: raise AssertionError(mutation)

    router = ShallowScoreRouting(arithmetic, ShallowScoreLayout(0, 0), progress=callback)
    with pytest.raises(ValueError):
        (router.inverse if inverse else router.forward)(value)
    assert calls_at_mutation is not None and len(arithmetic.calls) == calls_at_mutation
    assert router._active_input is None


@pytest.mark.parametrize('field,value', [('_ckks_nsd', True), ('_ckks_nsd', 1.0),
                                        ('_ckks_scale', float('inf')), ('level', 4.0),
                                        ('batch', True), ('degree', float(2*SLOTS))])
def test_invalid_primitive_output_rejected_before_callback(field, value):
    arithmetic = Arithmetic()
    original = arithmetic.weighted_sum
    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(result, field, value)
        return result
    arithmetic.weighted_sum = broken
    events = []
    router = ShallowScoreRouting(arithmetic, ShallowScoreLayout(0, 0), progress=lambda *args: events.append(args))
    with pytest.raises((ValueError, RuntimeError)):
        router.forward(Value(np.ones(SLOTS)))
    assert arithmetic.calls.count('weighted_sum') == 1 and not events


def test_reentry_is_rejected_and_callback_exception_releases_owner():
    arithmetic = Arithmetic()
    value = Value(np.ones(SLOTS))
    def recurse(*args): router.forward(value)
    router = ShallowScoreRouting(arithmetic, ShallowScoreLayout(0, 0), progress=recurse)
    with pytest.raises(ValueError, match='reentered'):
        router.forward(value)
    assert router._active_input is None and arithmetic.calls.count('weighted_sum') == 1
    router.progress = None
    output = router.forward(value)
    assert output.level == 3
