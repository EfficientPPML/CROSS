"""Experimental four-baby BERT score routing on the common p60 profile.

Borrowed serial arithmetic performs every rotation, addition and rescale.
This module changes operation order; public algebra is not encrypted accuracy
or performance evidence. It owns neither the context nor evaluation keys.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
import numpy as np
from encrypted_factored_score_routing import FactoredScoreLayout, SLOTS
from encrypted_shallow_score_routing import ShallowScoreLayout
from encrypted_scale_planning import drop_divisor, weighted_sum_scale

Q_SHA = 'fc2b841869ef2b2b556209323ed033c5a177344d1de395fc94e8a5865df39a49'
P_SHA = 'c03a649ab504238a4f24d57d7fe8e49f2a445c5d842579494cd243a7043d1d49'
DELTA = 1.1467305003699405e18


def typed(value):
    if isinstance(value, (tuple, list)):
        return type(value), tuple(typed(x) for x in value)
    return type(value), value


def accessor(owner, name):
    method = getattr(owner, name)
    return getattr(method, '__self__', None), getattr(method, '__func__', method)


def same_accessors(current, expected):
    return len(current) == len(expected) and all(
        len(a) == len(b) and all(x is y for x, y in zip(a, b))
        for a, b in zip(current, expected))


def required_rotation_indices():
    indices = {sign*stride*k % SLOTS for sign in (-1, 1)
               for stride, ks in ((63, (1, 2, 3, 4, 8)),
                                  (16, (1, 2, 3, 4, 8, 12))) for k in ks}
    indices.update(sign*256*k % SLOTS for sign in (-1, 1) for k in range(1, 8))
    return tuple(sorted(indices))


class BSGSScoreRouting:
    """Same sparse map and two-level depth, with seven weighted groups."""
    def __init__(self, arithmetic, layout, *, progress=None):
        from encrypted_packed_qwen import CompiledArithmetic
        from support import require_secure_execution
        if not isinstance(arithmetic, CompiledArithmetic):
            raise TypeError('CompiledArithmetic required')
        if type(layout) is not FactoredScoreLayout:
            raise TypeError('exact FactoredScoreLayout required')
        layout.__post_init__()
        if progress is not None and not callable(progress):
            raise TypeError('progress must be callable or None')
        self.arithmetic, self.layout, self.progress = arithmetic, layout, progress
        self.ctx = arithmetic.ctx
        self._cache = self.ctx._param_cache
        self._borrowed = (arithmetic, self.ctx, self._cache, layout, progress)
        self._helpers = tuple(getattr(arithmetic, n) for n in
                              ('rotation', '_plaintext_multiply', '_rescale'))
        self._keys, self._maps = self._cache.raw_rotation_keys, self._cache.coef_maps
        self._key_objects = self._key_state()
        self._secure = require_secure_execution
        self._layout_state = typed((layout.key_tile, layout.offset))
        self._layout_dependencies = tuple(getattr(owner, name) for owner, name in (
            (FactoredScoreLayout, 'tile_shift'), (ShallowScoreLayout, 'row_keys'),
            (ShallowScoreLayout, 'head_diagonals'), (ShallowScoreLayout, 'key_diagonals')))
        self._context_state = self._context_fingerprint()
        self._methods = self._accessors()
        self._active = False
        self._retained = []
        self._failed_arrays = []
        self.counts = Counter()
        self.last_run = None
        self._guard()

    def _key_state(self):
        k = self._cache
        if (getattr(k, 'per_level_rotation_keys', None) is not False or
                not isinstance(k.raw_rotation_keys, Mapping) or not isinstance(k.coef_maps, Mapping)):
            raise ValueError('supplied reusable public key/map inventory required')
        states = []
        for shift in required_rotation_indices():
            pair, permutation = k.raw_rotation_keys.get(shift), k.coef_maps.get(shift)
            if not isinstance(pair, Mapping) or permutation is None:
                raise ValueError('missing required public rotation pair/map')
            arrays = []
            for name in ('a', 'b'):
                array = pair.get(name)
                if (array is None or tuple(getattr(array, 'shape', ())) != (8, 114, 131072) or
                        str(getattr(array, 'dtype', None)) != 'uint32' or
                        isinstance(array, np.ndarray) and array.flags.writeable):
                    raise ValueError('read-only max-level p60 public key array required')
                arrays.append(id(array))
            if (tuple(getattr(permutation, 'shape', ())) != (131072,) or
                    str(getattr(permutation, 'dtype', None)) != 'int32' or
                    isinstance(permutation, np.ndarray) and permutation.flags.writeable):
                raise ValueError('immutable full-ring public rotation map required')
            states.append((shift, id(pair), *arrays, id(permutation)))
        # Identity/shape checks preserve the borrowed inventory. Authentication
        # and exclusion of external mmap writes remain the loader's obligation.
        return tuple(states)

    def _accessors(self):
        a = self.arithmetic
        names = ('rotate', 'weighted_sum', 'add', '_level', '_finish', '_prepare',
                 '_lower', '_ensure_capacity', '_encode_public', '_public_slots',
                 '_clear_working_caches')
        pairs = [accessor(a, n) for n in names]
        for helper, methods in ((a.rotation, ('rotate', 'clear')),
                                (a._plaintext_multiply, ('multiply', 'multiply_encoded', 'clear')),
                                (a._rescale, ('rescale', 'clear'))):
            pairs.extend(accessor(helper, n) for n in methods)
        pairs.extend(accessor(self.layout, n) for n in
                     ('__post_init__', 'head_diagonals', 'key_diagonals'))
        return tuple(pairs)

    def _context_fingerprint(self):
        c, k = self.ctx, self.ctx._param_cache
        return tuple(typed(getattr(owner, name, None)) for owner, names in (
            (c, ('degree', 'num_slots', 'composite_degree', 'dnum', 'max_level',
                 'scaling_factor', 'degree_layout', 'q_towers', 'p_towers')),
            (k, ('degree', 'batch', 'composite_degree', 'dnum', 'degree_layout',
                 'q_towers', 'p_towers')))
            for name in names)

    def _guard(self):
        a, c = self.arithmetic, self.ctx
        current = (a, c, c._param_cache, self.layout, self.progress)
        if any(x is not y for x, y in zip(current, self._borrowed)):
            raise ValueError('borrowed owner/context/layout/callback changed')
        if a.ctx is not c or self._cache is not c._param_cache:
            raise ValueError('borrowed arithmetic context changed')
        if self._failed_arrays:
            raise ValueError('previous synchronization failure remains unresolved')
        if self._context_fingerprint() != self._context_state:
            raise ValueError('context metadata changed')
        if typed((self.layout.key_tile, self.layout.offset)) != self._layout_state:
            raise ValueError('layout metadata changed')
        dependencies = tuple(getattr(owner, name) for owner, name in (
            (FactoredScoreLayout, 'tile_shift'), (ShallowScoreLayout, 'row_keys'),
            (ShallowScoreLayout, 'head_diagonals'), (ShallowScoreLayout, 'key_diagonals')))
        if (type(self.layout) is not FactoredScoreLayout or
                any(x is not y for x, y in zip(dependencies, self._layout_dependencies))):
            raise ValueError('layout inherited accessor changed')
        self.layout.__post_init__()
        k = c._param_cache
        from finite_field import BarrettContext
        if k.ff_context_cls is not BarrettContext:
            raise ValueError('exact standard Barrett field family required')
        if (k.raw_rotation_keys is not self._keys or k.coef_maps is not self._maps or
                self._key_state() != self._key_objects):
            raise ValueError('construction-bound public key/map inventory changed')
        for owner, name, expected in (
            (c, 'degree', 131072), (c, 'num_slots', 65536),
            (c, 'composite_degree', 2), (c, 'dnum', 8), (c, 'max_level', 49),
            (c, 'scaling_factor', DELTA), (c, 'degree_layout', (256, 512)),
            (k, 'degree', 131072), (k, 'batch', 1), (k, 'composite_degree', 2),
            (k, 'dnum', 8), (k, 'degree_layout', (256, 512)),
            (a, 'scale', DELTA), (a, 'low_memory', True)):
            if typed(getattr(owner, name, None)) != typed(expected):
                raise ValueError('fixed common p60 serial profile required: '+name)
        for name, digest in (('q_towers', Q_SHA), ('p_towers', P_SHA)):
            towers = getattr(k, name)
            if (not isinstance(towers, (tuple, list)) or
                    any(type(x) is not int for x in towers) or
                    hashlib.sha256(json.dumps(list(towers)).encode()).hexdigest() != digest or
                    typed(getattr(c, name)) != typed(towers)):
                raise ValueError('exact ordered common-profile moduli required')
        if (c.secret_key is not None or k.secret_key is not None or
                c.parameters.get('secret_key') is not None):
            raise ValueError('keyless borrowed context required')
        if (getattr(a, 'refresher', None) is not None or
                getattr(a, 'progress', None) is not None or
                callable(getattr(a, 'declare_bound', None))):
            raise ValueError('ordinary arithmetic without implicit callbacks/refresh required')
        if any(getattr(a, n) is not helper for n, helper in zip(
                ('rotation', '_plaintext_multiply', '_rescale'), self._helpers)):
            raise ValueError('compiled helper changed')
        if any(helper.ctx is not c for helper in self._helpers):
            raise ValueError('compiled helper context changed')
        if not same_accessors(self._accessors(), self._methods):
            raise ValueError('borrowed arithmetic/layout accessor changed')
        from support import require_secure_execution
        if require_secure_execution is not self._secure:
            raise ValueError('security accessor changed')

    def _fingerprint(self, value):
        ntt = value.ntt_ctx
        field = ntt.ff_ctx
        return (tuple(typed(getattr(value, n, None)) for n in
                      ('batch', 'num_elements', 'degree', 'r', 'c', 'degree_layout', 'precision',
                       'num_moduli', 'moduli', '_ckks_nsd', '_ckks_scale')),
                id(value.polynomial), tuple(value.polynomial.shape), str(value.polynomial.dtype),
                frozenset(value.polynomial.devices()), id(ntt), id(field), type(field),
                typed(tuple(ntt.moduli)), typed(tuple(field.moduli)),
                typed(field.computation_format_is_standard))

    def _retain(self, value):
        # Save the array separately: a mutable wrapper is not a storage owner.
        self._retained.append((value, value.polynomial, self._fingerprint(value)))
        return value

    def _check_retained(self):
        self._guard()
        for value, payload, state in self._retained:
            if value.polynomial is not payload or self._fingerprint(value) != state:
                raise ValueError('retained ciphertext metadata or payload changed')

    def _validate(self, value, *, level=None, scale=None, minimum=0):
        self._guard()
        from jaxite_word import Polynomial
        if not isinstance(value, Polynomial):
            raise TypeError('actual ciphertext required')
        self._secure(self.ctx, value)
        for name, expected in (('batch', 1), ('num_elements', 2), ('degree', 131072),
                               ('r', 256), ('c', 512),
                               ('degree_layout', (256, 512)), ('precision', 32),
                               ('_ckks_nsd', 1)):
            if typed(getattr(value, name, None)) != typed(expected):
                raise ValueError('ciphertext typed metadata differs: '+name)
        actual = self.arithmetic._level(value)
        if type(actual) is not int or not minimum <= actual <= 49:
            raise ValueError('insufficient or invalid ciphertext level')
        if level is not None and actual != level:
            raise ValueError('primitive changed expected level')
        if (type(value.num_moduli) is not int or value.num_moduli != 2*(actual+1) or
                tuple(value.moduli) != tuple(self._cache.q_towers[:2*(actual+1)]) or
                any(type(x) is not int for x in value.moduli)):
            raise ValueError('ciphertext ordered Q prefix differs')
        ntt = value.ntt_ctx
        field = ntt.ff_ctx
        if (type(field) is not self._cache.ff_context_cls or
                typed(field.computation_format_is_standard) != typed(True) or
                tuple(ntt.moduli) != tuple(value.moduli) or
                tuple(field.moduli) != tuple(value.moduli)):
            raise ValueError('ciphertext NTT/field metadata differs')
        s = value._ckks_scale
        if type(s) is not float or not math.isfinite(s) or s <= 0:
            raise ValueError('finite positive binary64 ciphertext scale required')
        if scale is not None and typed(s) != typed(scale):
            raise ValueError('primitive changed expected binary64 scale')
        if (tuple(value.polynomial.shape) != (1, 2, 256, 512, 2*(actual+1)) or
                str(value.polynomial.dtype) != 'uint32' or len(value.polynomial.devices()) != 1):
            raise ValueError('single-device full-slot uint32 payload required')

    def _forget(self, value):
        if value is self._retained[0][0]:
            return
        for name in tuple(vars(value)):
            if name.startswith('_public_'):
                delattr(value, name)

    def _note(self, label, value):
        self._check_retained()
        self._forget(value)
        if self.progress is not None:
            self.progress(label, value, dict(self.counts))
        self._check_retained()
        for retained, _, _ in self._retained:
            self._forget(retained)

    def _rotate(self, value, shift):
        self._check_retained()
        self._validate(value)
        level, scale = self.arithmetic._level(value), value._ckks_scale
        shift %= SLOTS
        if shift == 0:
            return value
        out = self._retain(self.arithmetic.rotate(value, shift))
        self._validate(out, level=level, scale=scale)
        self._check_retained()
        self.counts['rotations'] += 1
        self._forget(out)
        return out

    def _add(self, left, right):
        self._check_retained()
        level, scale = self.arithmetic._level(left), left._ckks_scale
        self._validate(left, level=level, scale=scale)
        self._validate(right, level=level, scale=scale)
        out = self._retain(self.arithmetic.add(left, right))
        self._validate(out, level=level, scale=scale)
        self._check_retained()
        self.counts['ciphertext_additions'] += 1
        self._forget(out)
        return out

    def _stage(self, value, diagonals, *, stride, terms, label):
        self._check_retained()
        self._validate(value, minimum=1)
        level = self.arithmetic._level(value)
        scale = weighted_sum_scale([value._ckks_scale]*4, DELTA,
                                   drop_divisor(self._cache.q_towers, level))['output_scale']
        masks = {}
        for shift, mask in diagonals:
            if type(shift) is not int or shift % SLOTS in masks:
                raise ValueError('unique exact integer diagonals required')
            if type(mask) is not np.ndarray or mask.dtype != np.dtype('float64') or mask.shape != (SLOTS,) or not np.isfinite(mask).all():
                raise ValueError('finite public full-slot masks required')
            masks[shift % SLOTS] = np.frombuffer(mask.tobytes(), dtype='float64')
        if set(masks) != {stride*k % SLOTS for k in range(terms)}:
            raise ValueError('exact arithmetic progression required')
        # Only original input, stage input, four baby arrays and current group/
        # accumulator survive a group. All primitive results synchronize before
        # retirement under the required low_memory arithmetic contract.
        base = list(self._retained)
        babies = [value] + [self._rotate(value, stride*b) for b in range(1, 4)]
        baby_refs = list(self._retained)
        result = None
        for first in range(0, terms, 4):
            giant = stride*first
            weights = [np.frombuffer(np.roll(masks[(giant+stride*b) % SLOTS], giant).tobytes(), dtype='float64') for b in range(4)]
            self._check_retained()
            partial = self._retain(self.arithmetic.weighted_sum(babies, weights))
            self._validate(partial, level=level-1, scale=scale)
            self._check_retained()
            self.counts['public_multiplications'] += 4
            self.counts['ciphertext_additions'] += 3
            self.counts['actual_composite_rescales'] += 1
            self._forget(partial)
            moved = self._rotate(partial, giant)
            result = moved if result is None else self._add(result, moved)
            result_ref = (result, result.polynomial, self._fingerprint(result))
            self._retained = baby_refs + [result_ref]
        self.counts['public_multiplication_levels'] += 1
        self._note(label, result)
        self._retained = base + [(result, result.polynomial, self._fingerprint(result))]
        return result

    def _evaluate(self, value, *, inverse, gain=1.):
        self._guard()
        if self._active:
            raise ValueError('serial routing cannot be reentered')
        self._validate(value, minimum=2)
        self._active = True
        self.last_run = None
        self._retained = []
        self._retain(value)
        start = self.counts.copy()
        try:
            if inverse:
                local = self._rotate(value, self.layout.tile_shift)
                self._note('key tile extraction', local)
                key = self._stage(local, self.layout.key_diagonals(inverse=True),
                                  stride=16, terms=16, label='key extraction')
                out = self._stage(key, self.layout.head_diagonals(inverse=True),
                                  stride=-63, terms=12, label='head expansion')
            else:
                head = self._stage(value, self.layout.head_diagonals(gain=gain),
                                   stride=63, terms=12, label='head compaction')
                local = self._stage(head, self.layout.key_diagonals(),
                                    stride=-16, terms=16, label='local key placement')
                out = self._rotate(local, -self.layout.tile_shift)
                self._note('key placement', out)
            self._check_retained()
            self._validate(out, level=self.arithmetic._level(value)-2)
            self.last_run = dict(input_level=self.arithmetic._level(value),
                output_level=self.arithmetic._level(out), input_scale=value._ckks_scale,
                output_scale=out._ckks_scale, counts=dict(self.counts-start),
                inverse=inverse, gain=gain)
            return out
        except BaseException as primary:
            # Inspect original arrays, never just the possibly replaced wrappers.
            arrays = tuple({id(payload): payload for _, payload, _ in self._retained}.values())
            try:
                import jax
                jax.block_until_ready(arrays)
            except BaseException as cleanup:
                self._failed_arrays.append(arrays)
                primary.add_note('routing original-array synchronization failed: '+type(cleanup).__name__)
            raise
        finally:
            self._retained = []
            self._active = False

    def forward(self, value, *, gain=1.):
        if type(gain) is not float or gain not in (1., .125):
            raise ValueError('exact supported public float gain required')
        return self._evaluate(value, inverse=False, gain=gain)

    def inverse(self, value):
        return self._evaluate(value, inverse=True)
