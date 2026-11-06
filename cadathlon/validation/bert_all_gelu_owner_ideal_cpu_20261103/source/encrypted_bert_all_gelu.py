"""Pinned twelve-layer BERT GELUs, using real static-capacity CKKS primitives.

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



@dataclass(frozen=True)
class BertGeluContract:
    manifest_bytes: bytes = field(repr=False)
    layer: int
    domain: tuple = field(init=False)
    coefficients: tuple = field(init=False)
    degree: int = field(init=False)

    def __post_init__(self):
        if type(self.manifest_bytes) is not bytes or hashlib.sha256(self.manifest_bytes).hexdigest() != MANIFEST_SHA256:
            raise ValueError('exact frozen v2 BERT manifest required')
        if type(self.layer) is not int or not 0 <= self.layer < 12:
            raise ValueError('exact public layer integer in0..11 required')
        row = json.loads(self.manifest_bytes)['gelu'][str(self.layer)]
        polynomial = row['polynomial']
        coefficients = np.asarray(polynomial['coefficients'], dtype='<f8')
        if (row['strategy'] != 'balanced Chebyshev' or type(row['degree']) is not int
                or not 2 <= row['degree'] <= 192 or polynomial['name'] != 'gelu'
                or polynomial['domain'] != row['domain']
                or coefficients.shape != (row['degree']+1,) or not np.isfinite(coefficients).all()
                or hashlib.sha256(coefficients.tobytes()).hexdigest() != polynomial['coefficient_f64le_sha256']):
            raise ValueError('frozen layer GELU contract differs')
        object.__setattr__(self, 'domain', tuple(map(float, polynomial['domain'])))
        object.__setattr__(self, 'coefficients', tuple(map(float, coefficients)))
        object.__setattr__(self, 'degree', row['degree'])


def _typed(value):
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_typed(x) for x in value)
    return type(value), value


def _contract_state(contract):
    return tuple(_typed(getattr(contract, name, None)) for name in
                 ('manifest_bytes', 'layer', 'domain', 'coefficients', 'degree'))


def verify_contract(contract):
    if not isinstance(contract, BertGeluContract):
        raise TypeError('pinned GELU contract required')
    if _contract_state(contract) != _contract_state(BertGeluContract(contract.manifest_bytes, contract.layer)):
        raise ValueError('typed GELU contract changed')
    return _contract_state(contract)


def gelu_plan(q_moduli, *, degree, input_level, canonical_scale, input_scale):
    if type(degree) is not int or not 2 <= degree <= 192:
        raise ValueError('supported public balanced degree integer in2..192 required')
    depth = 2*math.ceil(math.log2(degree))+2
    q = tuple(q_moduli)
    if (type(input_level) is not int or input_level < depth+2 or len(q) != 2*(input_level+1)
            or len(set(q)) != len(q) or any(type(x) is not int or x <= 1 for x in q)):
        raise ValueError('GELU needs its fixed depth plus two continuation levels')
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
    output = weighted('output', [term(n) for n in range(1, degree+1)])
    assert output['level'] == input_level-depth
    return dict(input_level=input_level, input_scale=source, canonical_scale=canonical,
                output_level=output['level'], output_scale=output['scale'], stages=stages,
                consumed_levels=depth, reserved_output_levels=2, ciphertext_products=degree-1,
                weighted_operations=degree+1, actual_composite_rescales=2*degree,
                physical_padding='P(0), not zero; following dense source mask required')


class BertGelu:
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

    def _validate(self, value, minimum_levels):
        self._metadata._validate_ciphertext(value, minimum_levels=minimum_levels, canonical=False)
        if _typed(getattr(value, 'degree_layout', None)) != _typed(self.arithmetic.ctx.degree_layout):
            raise ValueError('GELU ciphertext layout differs from borrowed context')

    def evaluate_tile(self, value):
        self._guard()
        if self._active: raise ValueError('GELU evaluation cannot be reentered')
        self._validate(value, 2*math.ceil(math.log2(self.contract.degree))+4)
        arithmetic = self.arithmetic
        plan = gelu_plan(arithmetic.ctx._param_cache.q_moduli_at_level(arithmetic._level(value)),
                          degree=self.contract.degree, input_level=arithmetic._level(value), canonical_scale=arithmetic.scale,
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
            self._validate(result, 0)
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
            # The callback cannot supply a proved semantic range for this
            # owner. Remove annotations it adds to this or any live branch.
            for current in (result, *(v for v, _ in retained.values())):
                for key in tuple(vars(current)):
                    if key.startswith('_public_'): delattr(current, key)
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
            values = [term(n) for n in range(1, self.contract.degree+1)]
            guard()
            result = arithmetic.weighted_sum(values, list(self.contract.coefficients[1:]),
                                             bias=self.contract.coefficients[0])
            counts['weighted_operations'] += 1
            result = note('output', result)
            if counts != dict(ciphertext_products=self.contract.degree-1, weighted_operations=self.contract.degree+1):
                raise RuntimeError('fixed GELU operation inventory differs')
            self.last_plan = plan
            return result
        finally:
            retained.clear()
            terms.clear()
            self._active = False
