"""Public-plan and arithmetic-contract tests; no cryptographic backend."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'checkout/cadathlon/common'), str(HERE)]
from encrypted_newton import scaled_newton_inverse
from public_schedule import PublicSchedule


class ArrayArithmetic:
    def __init__(self, slots):
        self.ctx = SimpleNamespace(num_slots=slots)
        self.operations = []

    def weighted_sum(self, values, weights, *, bias=0.):
        self.operations.append('public_affine')
        return sum(value*weight for value, weight in zip(values, weights)) + bias

    def multiply(self, left, right):
        self.operations.append('multiply')
        return left*right


def test_public_stage_outputs_use_their_declared_update_counts():
    coefficients = json.loads((HERE/'baseline_manifest.json').read_text())['power_coefficients_in_t_2x']
    schedule = PublicSchedule(coefficients)
    contracts = [schedule.stage(128, stage) for stage in (1, 2, 3, 4)]
    assert [contract['iterations'] for contract in contracts] == [9, 12, 13, 13]
    values, initials = [], []
    for contract in contracts:
        a, b = contract['a'], contract['b']
        values.extend([a, np.nextafter(a, b), contract['inactive_dummy'], np.nextafter(b, a), b])
        initials.extend([2/(32*(a+b))]*5)
    values, initials = np.asarray(values), np.asarray(initials)
    original_values, original_initials = values.copy(), initials.copy()
    arithmetic = ArrayArithmetic(len(values))
    result = scaled_newton_inverse(arithmetic, values, normalization_factor=32.,
        initial_estimate=initials, updates=13, capture_updates=range(1, 14))
    assert arithmetic.operations.count('multiply') == 24
    assert arithmetic.operations.count('public_affine') == 13
    np.testing.assert_array_equal(values, original_values)
    np.testing.assert_array_equal(initials, original_initials)
    assert result.final is result.checkpoints[13]
    for stage, contract in enumerate(contracts):
        selected = slice(stage*5, (stage+1)*5)
        # Independent closed-form Newton residual, not a second operation loop.
        n = contract['iterations']
        x, z0 = values[selected], initials[selected]
        expected = (1-(1-32*x*z0)**(2**n))/(32*x)
        np.testing.assert_allclose(result.checkpoints[n][selected], expected, rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(result.checkpoints[n][selected], 1/(32*x), rtol=1e-12, atol=3e-14)
        # Distinct checkpoint objects remain intact as other slots continue.
        if n < 13:
            assert result.checkpoints[n] is not result.final


@pytest.mark.parametrize('updates', [0, -1, True, 1.5])
def test_invalid_update_count_fails_before_ciphertext_operations(updates):
    arithmetic = ArrayArithmetic(2)
    with pytest.raises(ValueError):
        scaled_newton_inverse(arithmetic, object(), normalization_factor=32.,
                              initial_estimate=1., updates=updates)
    assert arithmetic.operations == []


@pytest.mark.parametrize('factor, initial, captures', [
    (True, 1., ()), (0., 1., ()), (float('inf'), 1., ()),
    (32., 0., ()), (32., float('nan'), ()), (32., 1e308, ()),
    (32., [1., 2., 3.], ()), (32., [[1., 2.]], ()),
    (32., 1., (0,)), (32., 1., (2,)), (32., 1., (1, 1)), (32., 1., (True,)),
])
def test_invalid_public_configuration_fails_without_touching_ciphertext(factor, initial, captures):
    arithmetic = ArrayArithmetic(2)
    with pytest.raises(ValueError):
        scaled_newton_inverse(arithmetic, object(), normalization_factor=factor,
            initial_estimate=initial, updates=1, capture_updates=captures)
    assert arithmetic.operations == []


def test_scalar_single_update_has_no_ciphertext_product_or_retained_checkpoints():
    arithmetic = ArrayArithmetic(2)
    x = np.array([.5, 1.])
    result = scaled_newton_inverse(arithmetic, x, normalization_factor=2.,
        initial_estimate=.5, updates=1)
    np.testing.assert_array_equal(result.final, np.array([.75, .5]))
    assert result.checkpoints == {} and result.updates == 1
    assert arithmetic.operations == ['public_affine']
