"""Pinned layer0 BERT GELU64, using real static-capacity CKKS primitives.

The public input domain is a caller obligation. No refresh, codec, range
inspection, monomial conversion or implicit bound propagation occurs here.
All physical padding is evaluated as P(0); the next dense map must mask it.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
import numpy as np

from encrypted_factored_score_routing import FactoredScoreLayout, FactoredScoreRouting
from encrypted_scale_planning import positive_scale, drop_divisor, weighted_sum_scale, multiply_scale

MANIFEST_SHA256 = '4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
COEFFICIENT_SHA256 = 'ed5e604dcfb1b9cf1d783160c758a822144ff51c666e9980508abdf4174b7d75'


@dataclass(frozen=True)
class Gelu64Contract:
    manifest_bytes: bytes = field(repr=False)
    domain: tuple = field(init=False)
    coefficients: tuple = field(init=False)
    degree: int = field(init=False)

    def __post_init__(self):
        if type(self.manifest_bytes) is not bytes or hashlib.sha256(self.manifest_bytes).hexdigest() != MANIFEST_SHA256:
            raise ValueError('exact frozen v2 BERT manifest required')
        row = json.loads(self.manifest_bytes)['gelu']['0']
        polynomial = row['polynomial']
        coefficients = np.asarray(polynomial['coefficients'], dtype='<f8')
        if (row['strategy'] != 'balanced Chebyshev' or row['degree'] != 64
                or polynomial['name'] != 'gelu' or tuple(polynomial['domain']) != (-21., 14.)
                or coefficients.shape != (65,) or not np.isfinite(coefficients).all()
                or hashlib.sha256(coefficients.tobytes()).hexdigest() != COEFFICIENT_SHA256):
            raise ValueError('frozen layer0 GELU64 contract differs')
        object.__setattr__(self, 'domain', (-21., 14.))
        object.__setattr__(self, 'coefficients', tuple(map(float, coefficients)))
        object.__setattr__(self, 'degree', 64)


def _typed(value):
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_typed(x) for x in value)
    return type(value), value


def _contract_state(contract):
    return tuple(_typed(getattr(contract, name, None)) for name in
                 ('manifest_bytes', 'domain', 'coefficients', 'degree'))


def verify_contract(contract):
    if not isinstance(contract, Gelu64Contract):
        raise TypeError('pinned GELU64 contract required')
    if _contract_state(contract) != _contract_state(Gelu64Contract(contract.manifest_bytes)):
        raise ValueError('typed GELU64 contract changed')
    return _contract_state(contract)


def gelu64_plan(q_moduli, *, input_level, canonical_scale, input_scale):
    q = tuple(q_moduli)
    if (type(input_level) is not int or input_level < 16 or len(q) != 2*(input_level+1)
            or len(set(q)) != len(q) or any(type(x) is not int or x <= 1 for x in q)):
        raise ValueError('GELU64 needs fourteen levels and two continuation levels')
    canonical, source = positive_scale(canonical_scale), positive_scale(input_scale)
    stages = []
    def weighted(name, inputs):
        level = min(x['level'] for x in inputs)
        plan = weighted_sum_scale([x['scale'] for x in inputs], canonical, drop_divisor(q, level))
        output = dict(name=name, level=level-1, scale=plan['output_scale'])
        stages.append(dict(kind='weighted', input_level=level, **output))
        return output
    t = weighted('domain map', [dict(level=input_level, scale=source)])
    terms = {1: t}
    def term(n):
        if n not in terms:
            left, right = term(n//2), term((n+1)//2)
            level = min(left['level'], right['level'])
            product = dict(name=f'product{n}', level=level-1,
                scale=multiply_scale(left['scale'], right['scale'], drop_divisor(q, level)))
            stages.append(dict(kind='ciphertext_product', input_level=level, **product))
            terms[n] = weighted(f'T{n}', [product] if n%2 == 0 else [product, t])
        return terms[n]
    output = weighted('output', [term(n) for n in range(1, 65)])
    assert output['level'] == input_level-14
    return dict(input_level=input_level, input_scale=source, canonical_scale=canonical,
                output_level=output['level'], output_scale=output['scale'], stages=stages,
                consumed_levels=14, reserved_output_levels=2, ciphertext_products=63,
                weighted_operations=65, actual_composite_rescales=128,
                physical_padding='P(0), not zero; following dense source mask required')


class BertV2Gelu:
    def __init__(self, arithmetic, contract, *, progress=None):
        state = verify_contract(contract)
        if progress is not None and not callable(progress):
            raise TypeError('progress must be callable or None')
        self.arithmetic, self.contract, self.progress = arithmetic, contract, progress
        self._metadata = FactoredScoreRouting(arithmetic, FactoredScoreLayout(0, 0))
        self._bound = arithmetic, contract, progress, self._metadata
        self._contract_state = state
        self._active = False
        self.last_plan = None
        self._guard()

    def _guard(self):
        if any(a is not b for a, b in zip((self.arithmetic, self.contract, self.progress, self._metadata), self._bound)):
            raise ValueError('construction-bound GELU owner changed')
        if _contract_state(self.contract) != self._contract_state:
            raise ValueError('construction-bound GELU contract changed')
        self._metadata._guard_context()

    def evaluate_tile(self, value):
        self._guard()
        if self._active: raise ValueError('GELU evaluation cannot be reentered')
        self._metadata._validate_ciphertext(value, minimum_levels=16, canonical=False)
        arithmetic = self.arithmetic
        plan = gelu64_plan(arithmetic.ctx._param_cache.q_moduli_at_level(arithmetic._level(value)),
                          input_level=arithmetic._level(value), canonical_scale=arithmetic.scale,
                          input_scale=value._ckks_scale)
        expected = {row['name']: row for row in plan['stages']}
        source, source_state = value, self._metadata._value_fingerprint(value)
        retained = {}
        terms = {}
        self.last_plan = None
        self._active = True
        counts = dict(ciphertext_products=0, weighted_operations=0)
        def guard():
            self._guard()
            if (self._metadata._value_fingerprint(source) != source_state
                    or any(self._metadata._value_fingerprint(v) != state for v, state in retained.values())):
                raise ValueError('GELU source or retained Chebyshev branch changed')
        def verify(name, result):
            guard()
            self._metadata._validate_ciphertext(result, minimum_levels=0, canonical=False)
            row = expected[name]
            if arithmetic._level(result) != row['level'] or _typed(result._ckks_scale) != _typed(row['scale']):
                raise RuntimeError('GELU primitive metadata differs from actual public plan')
            for key in tuple(vars(result)):
                if key.startswith('_public_'): delattr(result, key)
            return result
        def note(name, result):
            result = verify(name, result)
            state = self._metadata._value_fingerprint(result)
            if self.progress is not None: self.progress(name, result, dict(counts))
            guard()
            if self._metadata._value_fingerprint(result) != state:
                raise ValueError('GELU callback changed result metadata or payload')
            return result
        try:
            lo, hi = self.contract.domain
            t = arithmetic.weighted_sum([value], [2/(hi-lo)], bias=-(hi+lo)/(hi-lo))
            counts['weighted_operations'] += 1
            terms[1] = note('domain map', t)
            retained[1] = terms[1], self._metadata._value_fingerprint(terms[1])
            def term(n):
                if n not in terms:
                    left, right = term(n//2), term((n+1)//2)
                    guard()
                    product = arithmetic.multiply(left, right)
                    counts['ciphertext_products'] += 1
                    product = verify(f'product{n}', product)
                    if n%2 == 0:
                        result = arithmetic.weighted_sum([product], [2.], bias=-1.)
                    else:
                        result = arithmetic.weighted_sum([product, terms[1]], [2., -1.])
                    counts['weighted_operations'] += 1
                    terms[n] = note(f'T{n}', result)
                    retained[n] = terms[n], self._metadata._value_fingerprint(terms[n])
                return terms[n]
            values = [term(n) for n in range(1, 65)]
            guard()
            result = arithmetic.weighted_sum(values, list(self.contract.coefficients[1:]),
                                             bias=self.contract.coefficients[0])
            counts['weighted_operations'] += 1
            result = note('output', result)
            if counts != dict(ciphertext_products=63, weighted_operations=65):
                raise RuntimeError('fixed GELU operation inventory differs')
            self.last_plan = plan
            return result
        finally:
            retained.clear()
            terms.clear()
            self._active = False
