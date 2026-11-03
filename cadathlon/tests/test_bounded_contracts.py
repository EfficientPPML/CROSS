"""Experimental regressions for conditional ideal bounds; no encrypted proof.

Primitives deliberately clone every metadata field, as Polynomial does.
Only the real bound/relationship wrappers are under test. The separate
secure metadata probe provides actual CKKS evidence; these tests do not.
"""
from fractions import Fraction as F
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
candidates = ([Path(os.environ['CROSS_REPO'])] if 'CROSS_REPO' in os.environ else [])
candidates += [Path.cwd(), *HERE.parents]
ROOT = next(path.resolve() for path in candidates if (path/'jaxite_word').is_dir()
            and (path/'cadathlon/common').is_dir())
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/'cadathlon/common')]
from encrypted_packed_qwen import CompiledArithmetic
from encrypted_packed_transformer import PackedPairLayout
from encrypted_transformer import public_toy_approximations
from encrypted_bounded_arithmetic import (BoundedArithmetic, public_power_coefficients,
    public_polynomial_difference, refresh_public_bound, _fraction_upper)
from encrypted_bounded_transformer import (BoundedPackedTransformer, _matrix_norm_bound,
    _upper_sqrt, public_affine_bounds)


class Value:
    def __init__(self, slots, level=10):
        self.slots = np.asarray(slots, dtype=float)
        self.level = level
        self._ckks_scale = 1.

    def clone(self, slots=None, level=None):
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.slots = self.slots if slots is None else np.asarray(slots)
        result.level = self.level if level is None else level
        return result


class Primitives(CompiledArithmetic):
    """Sibling subclass makes BoundedArithmetic.super() hit this primitive layer."""
    def __init__(self, slots):
        self.ctx = SimpleNamespace(num_slots=slots)
        self.strategy, self.scale, self.refresher = 'power', 1., None
        self.refresh_events, self.declarations = [], []

    def _level(self, value):
        return value.level

    def _lower(self, value, level):
        assert level <= value.level
        return value if level == value.level else value.clone(level=level)

    def checkpoint(self, value, label):
        return value

    def add(self, left, right, *, subtract=False):
        slots = left.slots-right.slots if subtract else left.slots+right.slots
        return left.clone(slots, min(left.level, right.level))

    def multiply(self, left, right):
        left, right = self._ensure_capacity((left, right), 1)
        return left.clone(left.slots*right.slots, min(left.level, right.level)-1)

    def weighted_sum(self, values, weights, *, bias=0.):
        values = self._ensure_capacity(tuple(values), 1)
        slots = sum(value.slots*weight for value, weight in zip(values, weights))+bias
        return values[0].clone(slots, min(value.level for value in values)-1)

    def rotate(self, value, shift):
        return value if shift % self.ctx.num_slots == 0 else value.clone(np.roll(value.slots, -shift))


class Arithmetic(BoundedArithmetic, Primitives):
    def __init__(self, slots=4):
        Primitives.__init__(self, slots)

    def declare_bound(self, value, bound, reason='public circuit contract'):
        self.declarations.append(dict(reason=reason, bound=bound,
                                      observed=float(np.max(np.abs(value.slots)))))
        return BoundedArithmetic.declare_bound(self, value, bound, reason)


def annotated(arithmetic, slots=None, level=10):
    value = Value([.2, -.3, .5, .1] if slots is None else slots, level)
    arithmetic.declare_bound(value, 1., 'source contract')
    value._public_row_l2_bound = 2.
    value._public_head_l2_bounds = (1., 1.5)
    value._public_normalization_affine = ((1., 2., 3., 4.), (0., .1, .2, .3), 2.)
    return value


def relations(value):
    return {name: getattr(value, name) for name in BoundedArithmetic._relations if hasattr(value, name)}


@pytest.mark.parametrize('operation', ['add', 'subtract', 'multiply', 'weighted_sum', 'rotate', 'polynomial'])
def test_message_changing_operations_discard_cloned_correlations(operation):
    arithmetic = Arithmetic()
    left, right = annotated(arithmetic), annotated(arithmetic, [-.4, .2, .1, -.6])
    original_left, original_right = relations(left), relations(right)
    if operation == 'add':
        result = arithmetic.add(left, right)
        expected = left.slots+right.slots
    elif operation == 'subtract':
        result = arithmetic.add(left, right, subtract=True)
        expected = left.slots-right.slots
    elif operation == 'multiply':
        result = arithmetic.multiply(left, right)
        expected = left.slots*right.slots
    elif operation == 'weighted_sum':
        result = arithmetic.weighted_sum([left, right], [2., -3.], bias=.7)
        expected = 2*left.slots-3*right.slots+.7
    elif operation == 'rotate':
        result = arithmetic.rotate(left, -1)
        expected = np.roll(left.slots, 1)
    else:
        approximation = public_toy_approximations('qwen')['silu']
        weights = np.array([1., 0., .5, -1.])
        result = arithmetic.evaluate(left, approximation, output_weights=weights)
        expected = approximation.evaluate_public(left.slots)*weights
    assert result is not left and relations(result) == {}
    assert relations(left) == original_left and relations(right) == original_right
    np.testing.assert_allclose(result.slots, expected, rtol=0, atol=1e-14)
    assert arithmetic.public_bound(result) >= np.max(np.abs(expected))


@pytest.mark.parametrize('shift', [0, 4, -8])
def test_identity_rotation_keeps_message_correlations(shift):
    arithmetic = Arithmetic()
    value = annotated(arithmetic)
    expected = relations(value)
    assert arithmetic.rotate(value, shift) is value
    assert relations(value) == expected


@pytest.mark.parametrize('level', [10, 4])
def test_level_reduction_preserves_message_and_correlations(level):
    arithmetic = Arithmetic()
    value = annotated(arithmetic)
    result = arithmetic._lower(value, level)
    assert result.level == level and value.level == 10
    assert (result is value) == (level == 10)
    assert relations(result) == relations(value)
    assert result._public_bound_provenance == 'source contract'
    np.testing.assert_array_equal(result.slots, value.slots)


@pytest.mark.parametrize('all_relations', [False, True])
def test_hypothetical_fresh_wrapper_refresh_preserves_only_source_correlations(all_relations):
    arithmetic = Arithmetic()
    source = annotated(arithmetic, level=2)
    if not all_relations:
        del source._public_head_l2_bounds, source._public_normalization_affine
    expected, calls = relations(source), []

    def refresh(value, *, public_bound):
        calls.append(public_bound)
        fresh = annotated(arithmetic, value.slots.copy(), level=25)
        fresh._public_row_l2_bound = 999.
        return fresh

    arithmetic.refresher = SimpleNamespace(input_level=2, output_level=25, refresh=refresh)
    left, right = arithmetic._ensure_capacity((source, source), 1)
    assert left is right and left is not source
    assert len(calls) == len(arithmetic.refresh_events) == 1
    assert relations(left) == expected and relations(source) == expected
    assert left._public_absolute_bound == source._public_absolute_bound
    assert left._public_bound_provenance == 'source contract'
    np.testing.assert_array_equal(left.slots, source.slots)


@pytest.mark.parametrize('operation', ['add', 'multiply', 'weighted_sum'])
def test_basic_bounds_round_outward_including_underflow(operation):
    arithmetic = Arithmetic()
    left, right = Value(np.zeros(4)), Value(np.zeros(4))
    a, b = (1., 2.**-54) if operation == 'add' else (2.**-800, 2.**-800)
    arithmetic.declare_bound(left, a)
    arithmetic.declare_bound(right, b)
    if operation == 'add':
        result, exact = arithmetic.add(left, right), F(a)+F(b)
    elif operation == 'multiply':
        result, exact = arithmetic.multiply(left, right), F(a)*F(b)
    else:
        result, exact = arithmetic.weighted_sum([left], [b]), F(a)*F(b)
    assert F(arithmetic.public_bound(result)) >= exact > 0


@pytest.mark.parametrize('exponent', [-1000, -11, -10, -9, 0, 1, 30, 500, 1000])
def test_refresh_rounding_near_binary_powers_covers_exact_slack(exponent):
    boundary = math.ldexp(1., exponent)/1.125
    for value in (math.nextafter(boundary, -math.inf), boundary, math.nextafter(boundary, math.inf)):
        bound = refresh_public_bound(value)
        assert F(bound) >= F(9, 8)*F(value)
        assert bound >= 2.**-10 and math.frexp(bound)[0] == .5


def test_refresh_floor():
    for value in (0., math.ulp(0.), 2.**-30):
        assert refresh_public_bound(value) == 2.**-10


@pytest.mark.parametrize('value', [-1., math.inf, -math.inf, math.nan, sys.float_info.max])
def test_refresh_invalid_or_overflow_range_is_rejected(value):
    with pytest.raises(ValueError):
        refresh_public_bound(value)


def test_row_norm_is_not_operator_norm_and_root_rounds_outward():
    matrix = [[F(1)]*16 for _ in range(4)]
    assert math.sqrt(sum(x*x for x in matrix[0])) == 4.
    assert _matrix_norm_bound(matrix) == 8.
    assert F(math.sqrt(3))**2 < 3
    assert F(_upper_sqrt(F(3)))**2 >= 3
    assert _matrix_norm_bound([[F(1), F(1), F(1)]]) == _upper_sqrt(F(3))


@pytest.mark.parametrize('power', [-3000, -1500, 0, 1000, 2046])
def test_root_scaling_handles_underflow_and_large_rationals(power):
    exact = F(2)**power
    result = _upper_sqrt(exact)
    assert math.isfinite(result) and F(result)**2 >= exact
    previous = math.nextafter(result, -math.inf)
    assert previous < 0 or F(previous)**2 < exact


@pytest.mark.parametrize('value', [F(-1), F(2)**2048])
def test_root_rejects_negative_or_overflow(value):
    with pytest.raises(ValueError):
        _upper_sqrt(value)


def test_affine_bound_includes_gamma_beta_and_bias():
    weight = np.array([[1., -2., 3.], [-4., 5., -6.]])
    gamma, beta, bias = (-2., .5, 3.), (.1, -.2, .3), (.01, -.04)
    energy = 2.5
    coordinate, row_l2 = public_affine_bounds(weight, bias, gamma, beta, energy)
    for vector in ([2.5, 0., 0.], [0., -2.5, 0.], [0., 0., 2.5], [1.5, -2., 0.]):
        outputs = [sum(F(float(w))*(F(g)*F(x)+F(b)) for w, g, x, b in zip(row, gamma, vector, beta))+F(offset)
                   for row, offset in zip(weight, bias)]
        assert max(map(abs, outputs)) <= F(coordinate)
        assert sum(value*value for value in outputs) <= F(row_l2)**2


def test_segmented_windows_need_absolute_group_bounds_before_masking():
    graph = object.__new__(BoundedPackedTransformer)
    graph.layout = PackedPairLayout(1, 1, 32, 64)
    graph.features = 32
    graph.arithmetic = Arithmetic(64)
    graph.ctx = graph.arithmetic.ctx
    squares = np.zeros(64)
    squares[15] = squares[32] = 23.9
    value = Value(squares)
    graph.arithmetic.declare_bound(value, 23.9)
    result = graph.segmented_sum(value, 32, 1, factor=1/16, bias=1e-5, group_l1_bound=24.)
    raw = next(row for row in graph.arithmetic.declarations if row['reason'].startswith('sliding window'))
    assert raw['observed'] == 47.8 and raw['bound'] == 48.
    assert raw['observed'] > refresh_public_bound(24.)
    np.testing.assert_allclose(result.slots, 23.9/16+1e-5, rtol=0, atol=1e-14)
    signed = np.array([100., -100., -100., 100.])
    windows = signed+np.roll(signed, -1)
    np.testing.assert_array_equal(windows[::2], [0., 0.])
    assert max(abs(windows)) == 200.  # Signed aligned sums cannot replace group L1.


def exact_chebyshev(coefficients, t):
    if len(coefficients) == 1:
        return coefficients[0]
    previous, current = F(1), t
    result = coefficients[0]+coefficients[1]*t
    for coefficient in coefficients[2:]:
        previous, current = current, 2*t*current-previous
        result += coefficient*current
    return result


def exact_actual_power_circuit(approximation, output_weight, t):
    """Evaluate the runtime's rounded public constants with exact arithmetic."""
    lo, hi = approximation.domain
    x = (F(lo)+F(hi))/2 + (F(hi)-F(lo))*t/2
    coefficients = np.polynomial.chebyshev.cheb2poly(approximation.coefficients)
    if approximation.input_power_basis:
        coefficients = coefficients/hi**np.arange(len(coefficients))
        argument = x
    else:
        argument = F(2/(hi-lo))*x+F(-(hi+lo)/(hi-lo))
    weighted = [F(float(np.float64(coefficient)*np.float64(output_weight))) for coefficient in coefficients]
    # Horner is independent of the helper's affine expansion/Chebyshev conversion.
    result = F(0)
    for coefficient in reversed(weighted):
        result = result*argument+coefficient
    return result


@pytest.mark.parametrize('name,weight', [('inverse_sqrt', 1.), ('reciprocal', 1/3),
                                       ('reciprocal', 1/7), ('silu', 1.)])
def test_exact_power_pullback_and_difference_bound(name, weight):
    approximation = public_toy_approximations('qwen')[name]
    actual = public_power_coefficients(approximation, weight)
    nominal = tuple(F(weight)*F(value) for value in approximation.coefficients)
    exact_difference = sum(abs(a-b) for a, b in zip(actual, nominal))
    difference_bound = public_polynomial_difference(approximation, weight)
    assert F(difference_bound) >= exact_difference
    # More distinct exact points than the degree prove polynomial identity.
    for index in range(len(actual)+1):
        t = F(-1)+F(2*index, len(actual))
        expected = exact_actual_power_circuit(approximation, weight, t)
        assert exact_chebyshev(actual, t) == expected
        assert abs(expected-exact_chebyshev(nominal, t)) <= F(difference_bound)


def test_fraction_bound_preserves_a_positive_subnormal_result():
    exact = F(math.ulp(0.))/2
    assert _fraction_upper(exact) == math.ulp(0.)
