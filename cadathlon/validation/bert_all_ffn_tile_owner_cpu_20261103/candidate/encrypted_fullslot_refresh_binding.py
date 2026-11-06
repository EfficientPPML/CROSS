"""Borrow and validate an actual full-slot bounded refresh engine.

This validates public ownership/layout/capacity, not encrypted range or error.
It uses the same engine checks as the separately frozen softmax assembly.
"""
from collections.abc import Mapping
import math
from encrypted_factored_score_routing import FactoredScoreLayout, FactoredScoreRouting


def typed(value):
    if isinstance(value, Mapping):
        return type(value), tuple(sorted((k, typed(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return type(value), tuple(typed(v) for v in value)
    return type(value), value


class FullSlotRefreshBinding:
    def __init__(self, arithmetic, refresher):
        from encrypted_refresh import BoundedSecureRefresh, refresh_plan
        if not isinstance(refresher, BoundedSecureRefresh):
            raise TypeError('actual bounded refresh adapter required')
        self.arithmetic, self.refresher = arithmetic, refresher
        self._metadata = FactoredScoreRouting(arithmetic, FactoredScoreLayout(0, 0))
        # Derive the complete inventory independently of the supplied adapter.
        # Otherwise an empty, self-consistent plan could bypass every key check.
        self._expected_plan = typed(refresh_plan(131072, 65536, (4, 4),
                                                 max_level=arithmetic.ctx.max_level))
        self._helpers = tuple(refresher.compiled_primitives or ())
        self._keys = arithmetic.ctx._param_cache.raw_rotation_keys
        self._maps = arithmetic.ctx._param_cache.coef_maps
        self._objects = self._object_state()
        self._public = self._public_state()
        self._active = False
        self.check()

    def _object_state(self):
        r = self.refresher
        return (self.arithmetic, r, self._metadata, r.ctx, r.evaluator, r.engine,
                r.compiled_primitives, r.plan, getattr(r.engine, 'ctx', None))

    def _public_state(self):
        r = self.refresher
        return (typed(r.plan), tuple(typed(getattr(r, name)) for name in
                ('input_level', 'output_level', 'meta_precision', 'public_bound')),
                typed(r.evaluator.scale), typed(getattr(r.evaluator, 'strategy', None)),
                id(getattr(r.evaluator, 'progress', None)), tuple(typed(getattr(r.engine, name)) for name in
                ('_degree', '_m', '_num_slots', '_gap', '_sparse', '_composite_degree',
                 '_level_budget', '_dc_subtract')))

    def check(self):
        if any(a is not b for a, b in zip(self._object_state(), self._objects)):
            raise ValueError('borrowed full-slot refresh objects changed')
        if self._public_state() != self._public:
            raise ValueError('borrowed full-slot refresh public contract changed')
        self._metadata._guard_context()
        r = self.refresher; ctx = self.arithmetic.ctx; cache = ctx._param_cache
        engine = r.engine; view = engine.ctx
        if (r.ctx is not ctx or r.evaluator.ctx is not ctx or engine._cache is not cache
                or view._param_cache is not cache or r.compiled_primitives is None
                or view.secret_key is not None or view.parameters.get('secret_key') is not None
                or getattr(r.evaluator, 'refresher', None) is not None
                or callable(getattr(r.evaluator, 'declare_bound', None))
                or typed(r.evaluator.scale) != typed(self.arithmetic.scale)
                or typed(ctx.max_level) != (int, 49)
                or typed(r.plan) != self._expected_plan
                or typed(r.input_level) != (int, 2) or typed(r.output_level) != (int, 25)
                or typed(r.meta_precision) != (int, 10)
                or typed(engine._degree) != (int, 131072) or typed(engine._m) != (int, 262144)
                or typed(engine._num_slots) != (int, 65536) or typed(engine._gap) != (int, 1)
                or typed(engine._sparse) != (bool, False)
                or typed(engine._composite_degree) != (int, 2)
                or typed(engine._level_budget) != (list, ((int, 4), (int, 4)))
                or engine._dc_subtract is not False
                or typed(r.plan.get('period')) != (int, 65536)
                or typed(r.plan.get('bootstrap_slots')) != (int, 65536)
                or tuple(r.plan.get('level_budget', ())) != (4, 4)):
            raise ValueError('full-slot keyless meta10/4x4 refresh with25 output levels required')
        helper_methods = ('rotate', 'relinearize', 'rescale', 'multiply_encoded', 'multiply')
        if (type(r.compiled_primitives) is not tuple or len(r.compiled_primitives) != 5
                or any(helper is not original or getattr(helper, 'ctx', None) is not ctx
                       or not callable(getattr(helper, method, None))
                       for helper, original, method in zip(r.compiled_primitives, self._helpers, helper_methods))):
            raise ValueError('five supplied compiled refresh helpers must share the borrowed context')
        if any(typed(getattr(view, n, None)) != typed(getattr(ctx, n, None)) for n in
               ('degree', 'num_slots', 'composite_degree', 'dnum', 'degree_layout')):
            raise ValueError('bootstrap shallow context differs from borrowed ring')
        for name in ('q_towers', 'p_towers'):
            if tuple(getattr(view, name, getattr(cache, name))) != tuple(getattr(cache, name)):
                raise ValueError('bootstrap ordered Q/P differs from actual ring')
        keys, maps = getattr(cache, 'raw_rotation_keys', None), getattr(cache, 'coef_maps', None)
        if (getattr(cache, 'per_level_rotation_keys', None) is not False
                or keys is not self._keys or maps is not self._maps
                or not isinstance(keys, Mapping) or not isinstance(maps, Mapping)):
            raise ValueError('supplied reusable refresh keys/maps required')
        for key in r.plan['rotations']:
            row = keys.get(key)
            if (not isinstance(row, Mapping) or row.get('a') is None
                    or row.get('b') is None or maps.get(key) is None):
                raise ValueError('missing supplied refresh key/map')

    def _validate(self, value, minimum_levels):
        self._metadata._validate_ciphertext(value, minimum_levels=minimum_levels, canonical=False)
        if typed(getattr(value, 'degree_layout', None)) != typed(self.arithmetic.ctx.degree_layout):
            raise ValueError('refresh ciphertext layout differs from borrowed context')

    def refresh(self, value, *, public_bound):
        self.check()
        if self._active: raise ValueError('borrowed refresh binding cannot be reentered')
        if type(public_bound) is not float or public_bound not in (2., 16., 32., 64., 128.):
            raise ValueError('only the fixed B2/B16/B32/B64/B128 public contracts are supported')
        self._validate(value, 2)
        before = self._metadata._value_fingerprint(value)
        self._active = True
        try:
            result = self.refresher.refresh(value, public_bound=public_bound)
            self.check()
            if self._metadata._value_fingerprint(value) != before:
                raise ValueError('refresh changed its borrowed source ciphertext')
            self._validate(result, 25)
            if (self.arithmetic._level(result) != 25
                    or not math.isclose(result._ckks_scale, self.arithmetic.scale, rel_tol=1e-12)):
                raise ValueError('actual refresh output level/scale differs from the static contract')
            for name in tuple(vars(result)):
                if name.startswith('_public_'): delattr(result, name)
            return result
        finally:
            self._active = False
