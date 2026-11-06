"""Experimental public-bound scheduling for genuine homomorphic refresh.

Bounds describe IDEAL plaintext circuit contracts, excluding CKKS/refresh error, and are never inferred from
ciphertext slots. Declaring a contract does not range-check or clip data.
The 12.5% range slack is not a stochastic noise budget or domain-membership proof.
This module is used only by explicit experimental bounded adapters.
Complete encrypted deep-case gates remain required before validation claims.
"""
from functools import lru_cache
from fractions import Fraction
import math
import numpy as np

from encrypted_packed_qwen import CompiledArithmetic


def _up(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('public magnitude propagation overflowed or became invalid')
    return 0. if value == 0 else math.nextafter(value, math.inf)


def _fraction_upper(value):
    result = float(value)
    if not math.isfinite(result) or value < 0:
        raise ValueError('public bound exceeds floating-point capacity')
    while Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


@lru_cache(maxsize=256)
def public_power_coefficients(approximation, output_weight=1.):
    """Exact Chebyshev coefficients of the actual public power circuit.

    The coefficients returned are Fractions in canonical t on [-1,1]. This
    includes binary64 affine normalization, cheb2poly conversion, optional
    direct-input SiLU division, and every final output-weight multiplication.
    It excludes CKKS encoding/operation error and inspects no ciphertext.
    """
    lo, hi = approximation.domain
    coefficients = np.polynomial.chebyshev.cheb2poly(approximation.coefficients)
    if approximation.input_power_basis:
        coefficients = coefficients / hi ** np.arange(len(coefficients))
        alpha, beta = Fraction(hi), Fraction(0)
    else:
        slope = Fraction(2/(hi-lo))
        alpha = slope*(Fraction(hi)-Fraction(lo))/2
        beta = slope*(Fraction(hi)+Fraction(lo))/2 + Fraction(-(hi+lo)/(hi-lo))
    weighted = [Fraction(float(x*output_weight)) for x in coefficients]
    powers = [Fraction(0) for _ in weighted]
    for n, coefficient in enumerate(weighted):
        for k in range(n+1):
            powers[k] += coefficient*math.comb(n,k)*alpha**k*beta**(n-k)
    chebyshev = [Fraction(0) for _ in powers]
    basis = [Fraction(1)]
    for coefficient in powers:
        for k, term in enumerate(basis):
            chebyshev[k] += coefficient*term
        next_basis = [Fraction(0) for _ in range(len(basis)+1)]
        for k, term in enumerate(basis):
            if k == 0:
                next_basis[1] += term
            else:
                next_basis[k-1] += term/2
                next_basis[k+1] += term/2
        basis = next_basis
    return tuple(chebyshev)


@lru_cache(maxsize=256)
def public_polynomial_difference(approximation, output_weight=1.):
    """Certified public coefficient-L1 difference from weight*p_chebyshev."""
    actual = public_power_coefficients(approximation, output_weight)
    return _fraction_upper(sum(abs(a-Fraction(output_weight)*Fraction(c))
                               for a,c in zip(actual, approximation.coefficients)))


def refresh_public_bound(bound):
    """Public power-of-two range with slack for approximate arithmetic."""
    if not math.isfinite(bound) or bound < 0:
        raise ValueError('finite nonnegative public magnitude bound required')
    widened = _up(1.125*bound)
    # log2 can round a value immediately above a power of two back to the
    # integer exponent. frexp preserves that strict range distinction.
    fraction, exponent = math.frexp(max(2.**-10, widened))
    try:
        rounded = math.ldexp(1., exponent-1 if fraction == .5 else exponent)
    except OverflowError as error:
        raise ValueError('public refresh range exceeds floating-point capacity') from error
    if not math.isfinite(rounded):
        raise ValueError('public refresh range exceeds floating-point capacity')
    return rounded


@lru_cache(None)
def normalized_inverse_bound(approximation):
    """Upper envelope for |sqrt(x)*p(x)| on the declared variance interval.

    The public grid is widened by a global derivative bound, not used as a
    sample-only maximum. |T_n|<=1 and |T_n'|<=n^2 give the derivative envelope;
    conservative floating-point slack is included. No activations are read.
    """
    lo, hi = approximation.domain
    if lo <= 0:
        raise ValueError('positive variance domain required')
    c = np.asarray(approximation.coefficients, dtype=float)
    magnitude = float(np.abs(c).sum())
    derivative = float(np.sum(np.arange(len(c))**2 * np.abs(c)))
    lipschitz = magnitude/(2*math.sqrt(lo)) + math.sqrt(hi)*2*derivative/(hi-lo)
    points = 65537
    x = np.linspace(lo, hi, points)
    t = (2*x-lo-hi)/(hi-lo)
    sampled = float(np.max(np.abs(np.sqrt(x)*np.polynomial.chebyshev.chebval(t, c))))
    result = sampled + lipschitz*(hi-lo)/(2*(points-1)) + 1e-11*max(1., magnitude*math.sqrt(hi))
    result = _up(result + _up(math.sqrt(hi)*public_polynomial_difference(approximation)))
    if not math.isfinite(result):
        raise ValueError('normalization bound exceeds floating-point capacity')
    return result


@lru_cache(None)
def polynomial_envelope(approximation):
    """Public polynomial grid widened by its global Chebyshev derivative."""
    c = np.asarray(approximation.coefficients, dtype=float)
    t = np.linspace(-1., 1., 65537)
    values = np.polynomial.chebyshev.chebval(t, c)
    derivative = float(np.sum(np.arange(len(c))**2*np.abs(c)))
    margin = derivative/65536 + 1e-11*max(1., float(np.sum(np.abs(c))))
    margin = _up(margin + public_polynomial_difference(approximation))
    return float(values.min())-margin, float(values.max())+margin


@lru_cache(None)
def reciprocal_ratio_bound(approximation, output_weight=1.):
    """Envelope for |x*p(x)|, including a global derivative grid margin."""
    lo, hi = approximation.domain
    if lo <= 0:
        raise ValueError('positive denominator domain required')
    if not math.isfinite(output_weight) or output_weight <= 0:
        raise ValueError('positive finite public reciprocal output weight required')
    c = np.asarray(approximation.coefficients, dtype=float)
    x = np.linspace(lo, hi, 65537)
    t = (2*x-lo-hi)/(hi-lo)
    magnitude = float(np.sum(np.abs(c)))
    derivative = magnitude + hi*2/(hi-lo)*float(np.sum(np.arange(len(c))**2*np.abs(c)))
    nominal = (float(np.max(np.abs(x*np.polynomial.chebyshev.chebval(t,c))))
               + derivative*(hi-lo)/(2*65536) + 1e-11*max(1.,hi*magnitude))
    difference = _fraction_upper(Fraction(hi)*Fraction(public_polynomial_difference(approximation, output_weight))/Fraction(output_weight))
    return _up(nominal+difference)


class BoundedArithmetic(CompiledArithmetic):
    """Tracked public magnitudes with a real keyless bounded bootstrap."""
    _relations = ('_public_row_l2_bound', '_public_head_l2_bounds',
                  '_public_normalization_affine')

    def __init__(self, context, refresher=None, *, low_memory=True):
        from encrypted_refresh import BoundedSecureRefresh
        if refresher is not None and (not isinstance(refresher, BoundedSecureRefresh) or refresher.ctx is not context):
            raise TypeError('bounded arithmetic requires BoundedSecureRefresh on the same context')
        super().__init__(context, low_memory=low_memory)
        self.refresher = refresher
        self.refresh_events = []

    @classmethod
    def forget_relations(cls, value):
        """Generic arithmetic does not preserve correlated norm contracts.

        Polynomial clones copy their complete __dict__; retaining a source's
        affine or energy contract on a different message can underbound it.
        Only semantic operations may explicitly restore these relationships.
        """
        for name in cls._relations:
            value.__dict__.pop(name, None)
        return value

    @classmethod
    def preserve_relations(cls, source, result):
        if source is result:
            return result
        cls.forget_relations(result)
        for name in cls._relations:
            if hasattr(source, name):
                setattr(result, name, getattr(source, name))
        return result

    @staticmethod
    def public_bound(value):
        bound = getattr(value, '_public_absolute_bound', None)
        if bound is None or not math.isfinite(bound) or bound < 0:
            raise ValueError('ciphertext requires an explicit finite public magnitude contract')
        return bound

    def declare_bound(self, value, bound, reason='public circuit contract'):
        bound = float(bound)
        if not math.isfinite(bound) or bound < 0:
            raise ValueError('finite nonnegative public magnitude contract required')
        self._level(value)
        value._public_absolute_bound = bound
        value._public_bound_provenance = reason
        return value

    def _ensure_capacity(self, values, required):
        values = tuple(values)
        if self.refresher is None:
            return values
        reserve = self.refresher.input_level
        minimum = required + reserve
        output, refreshed = [], {}
        for value in values:
            level = self._level(value)
            bound = self.public_bound(value)
            if level < minimum:
                if level < reserve:
                    raise ValueError(f'bounded refresh needs {reserve} input levels; only {level} remain')
                if self.refresher.output_level < minimum:
                    raise ValueError(f'bounded refresh supplies {self.refresher.output_level} levels; operation needs {minimum}')
                original = value
                if id(original) not in refreshed:
                    scale_bound = refresh_public_bound(bound)
                    self.checkpoint(value, 'bounded homomorphic refresh starting')
                    value = self.refresher.refresh(value, public_bound=scale_bound)
                    if self._level(value) < minimum:
                        raise ValueError('bounded homomorphic refresh returned insufficient capacity')
                    value = self.preserve_relations(original, value)
                    value = self.declare_bound(value, bound, getattr(original, '_public_bound_provenance', 'public arithmetic bound'))
                    self.checkpoint(value, 'bounded homomorphic refresh complete')
                    refreshed[id(original)] = value
                    self.refresh_events.append(dict(input_level=level, output_level=self._level(value),
                        required=required, public_bound=scale_bound, circuit_bound=bound, provenance=getattr(original, '_public_bound_provenance', 'public arithmetic bound')))
                else:
                    value = refreshed[id(original)]
            output.append(value)
        return tuple(output)

    def _lower(self, value, level):
        bound = self.public_bound(value)
        result = self.preserve_relations(value, super()._lower(value, level))
        return self.declare_bound(result, bound, getattr(value, '_public_bound_provenance', 'public arithmetic bound'))

    def add(self, left, right, *, subtract=False):
        bound = _fraction_upper(Fraction(self.public_bound(left)) + Fraction(self.public_bound(right)))
        return self.declare_bound(self.forget_relations(super().add(left, right, subtract=subtract)), bound, 'triangle bound for addition/subtraction')

    def rotate(self, value, shift):
        bound = self.public_bound(value)
        result = super().rotate(value, shift)
        if result is not value:
            self.forget_relations(result)
        return self.declare_bound(result, bound, 'public permutation')

    def multiply(self, left, right):
        bound = _fraction_upper(Fraction(self.public_bound(left)) * Fraction(self.public_bound(right)))
        return self.declare_bound(self.forget_relations(super().multiply(left, right)), bound, 'product of public operand bounds')

    def weighted_sum(self, values, weights, *, bias=0.):
        values, weights = tuple(values), tuple(weights)
        if not values or len(values) != len(weights):
            raise ValueError('nonempty equally sized ciphertexts and public weights required')
        bound = Fraction(0)
        for value, weight in zip(values, weights):
            bound += Fraction(self.public_bound(value))*Fraction(float(np.max(np.abs(weight))))
        bound = _fraction_upper(bound + Fraction(float(np.max(np.abs(bias)))))
        return self.declare_bound(self.forget_relations(super().weighted_sum(values, weights, bias=bias)), bound, 'public weighted-sum triangle bound')

    def evaluate(self, value, approximation, *, output_weights=1.):
        value = self.declare_bound(value, max(abs(x) for x in approximation.domain),
                                   f'{approximation.name}: declared input domain {approximation.domain}')
        result = super().evaluate(value, approximation, output_weights=output_weights)
        # Pull the actual rounded power circuit back into exact Chebyshev
        # coefficients before applying |T_n|<=1. Output masks/count factors
        # are public and often contain only one or two distinct values.
        bound = _fraction_upper(max(sum(map(abs, public_power_coefficients(approximation, float(weight))))
                                    for weight in np.unique(output_weights)))
        return self.declare_bound(self.forget_relations(result), bound, 'actual public polynomial coefficient envelope')
