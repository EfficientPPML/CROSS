"""Independent gather/scatter checks for every full-slot key mapping."""
import numpy as np
import pytest
from test_routing import Arithmetic, Value, backend_free, ShallowScoreLayout, SLOTS
from encrypted_factored_score_routing import FactoredScoreLayout, FactoredScoreRouting


@pytest.mark.parametrize('tile', range(8))
@pytest.mark.parametrize('offset', range(16))
def test_every_map_and_inverse_with_hostile_padding(tile, offset):
    values = np.random.default_rng(2026110300 + tile*16 + offset).uniform(-100, 100, SLOTS)
    layout = FactoredScoreLayout(tile, offset)
    arithmetic = Arithmetic()
    router = FactoredScoreRouting(arithmetic, layout)
    source = Value(values)
    expected = np.zeros(SLOTS)
    back = np.zeros(SLOTS)
    for row in range(16):
        for head in range(12):
            anchor = row*4096 + head*64
            target = row*4096 + (16*tile + (row+offset) % 16)*16 + head
            expected[target] = values[anchor]/8
            back[anchor] = values[anchor]/8
    out = router.forward(source, gain=.125)
    np.testing.assert_array_equal(out.data, expected)
    recovered = router.inverse(out)
    np.testing.assert_array_equal(recovered.data, back)
    assert (out.level, recovered.level) == (3, 1)
    assert router.counts['rotations'] == (54 if tile else 52)
    assert router.counts['public_multiplications'] == 56
    assert router.counts['actual_composite_rescales'] == 4
    np.testing.assert_array_equal(source.data, values)
    assert source._public_bound == 'deliberately stale'
    assert not hasattr(out, '_public_bound') and not hasattr(recovered, '_public_bound')
    inverse_expected = np.zeros(SLOTS)
    for row in range(16):
        for head in range(12):
            inverse_expected[row*4096+head*64] = values[row*4096+(16*tile+(row+offset)%16)*16+head]
    np.testing.assert_array_equal(router.inverse(Value(values)).data, inverse_expected)


def test_full_key_union_and_every_actual_rotation_are_declared():
    direct = set()
    factored = set()
    for tile in range(8):
        for offset in range(16):
            direct.update(ShallowScoreLayout(tile, offset).rotation_indices())
            layout = FactoredScoreLayout(tile, offset)
            factored.update(layout.rotation_indices())
            arithmetic = Arithmetic()
            shifts = []
            original = arithmetic.rotate
            def recorded(value, shift):
                shifts.append(shift)
                return original(value, shift)
            arithmetic.rotate = recorded
            router = FactoredScoreRouting(arithmetic, layout)
            router.inverse(router.forward(Value(np.ones(SLOTS))))
            assert set(shifts) == set(layout.rotation_indices())
    assert len(direct) == 276 and len(factored) == 66 and factored < direct


@pytest.mark.parametrize('tile', [0, 7])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('mutation', ['input_bool_nsd', 'result_float_nsd', 'input_payload', 'result_payload', 'layout'])
def test_tile_callback_revalidates_original_and_result_before_continuing(tile, inverse, mutation):
    arithmetic = Arithmetic()
    value = Value(np.ones(SLOTS))
    calls_at_mutation = None
    def callback(label, result, counts):
        nonlocal calls_at_mutation
        if label != ('key tile extraction' if inverse else 'key placement'):
            return
        calls_at_mutation = len(arithmetic.calls)
        if mutation == 'input_bool_nsd': value._ckks_nsd = True
        elif mutation == 'result_float_nsd': result._ckks_nsd = 1.0
        elif mutation == 'input_payload': value.polynomial = value.polynomial.copy()
        elif mutation == 'result_payload': result.polynomial = result.polynomial.copy()
        elif mutation == 'layout': object.__setattr__(router.layout, 'key_tile', (tile+1)%8)
    router = FactoredScoreRouting(arithmetic, FactoredScoreLayout(tile, 15), progress=callback)
    with pytest.raises(ValueError):
        (router.inverse if inverse else router.forward)(value)
    assert calls_at_mutation is not None and len(arithmetic.calls) == calls_at_mutation
    assert router._active_input is None


def test_zero_tile_inverse_does_not_strip_caller_tags():
    value = Value(np.ones(SLOTS))
    FactoredScoreRouting(Arithmetic(), FactoredScoreLayout(0, 0)).inverse(value)
    assert value._public_bound == 'deliberately stale'


@pytest.mark.parametrize('tile', [0, 7])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('factor', [.73, 1., 1.17])
def test_real_weighted_stage_aligns_noncanonical_input_without_relabel(tile, inverse, factor):
    arithmetic = Arithmetic()
    rotate = arithmetic.rotate
    weighted_sum = arithmetic.weighted_sum
    observed_scales = []
    def scale_preserving_rotate(value, shift):
        result = rotate(value, shift)
        result._ckks_scale = value._ckks_scale
        return result
    def aligning_weighted_sum(values, weights):
        observed_scales.append([v._ckks_scale for v in values])
        return weighted_sum(values, weights)
    arithmetic.rotate = scale_preserving_rotate
    arithmetic.weighted_sum = aligning_weighted_sum
    value = Value(np.arange(SLOTS))
    value._ckks_scale = arithmetic.scale*factor
    router = FactoredScoreRouting(arithmetic, FactoredScoreLayout(tile, 15))
    output = (router.inverse if inverse else router.forward)(value)
    assert observed_scales[0] and all(x == arithmetic.scale*factor for x in observed_scales[0])
    assert all(x == arithmetic.scale for x in observed_scales[1])
    assert output._ckks_scale == arithmetic.scale and output.level == 3
    assert value._ckks_scale == arithmetic.scale*factor


@pytest.mark.parametrize('scale', [True, 0., -1., float('nan'), float('inf'), np.longdouble('1e4900'), np.longdouble('1e-4900')])
def test_factored_route_rejects_invalid_scale_before_work(scale):
    arithmetic = Arithmetic()
    value = Value(np.ones(SLOTS))
    value._ckks_scale = scale
    router = FactoredScoreRouting(arithmetic, FactoredScoreLayout(7, 15))
    with pytest.raises(ValueError): router.forward(value)
    assert not arithmetic.calls
