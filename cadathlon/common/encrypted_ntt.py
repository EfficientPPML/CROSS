"""Isolated public-only NTT subset provider; no global dispatch/cache mutation.

Owner and derived contexts are weakly referenced. This is a candidate injection
seam, not a live runtime feature. Only canonical standard Barrett, uint32
polynomial layouts, and one physical device are supported. Unsupported inputs
return None so the ordinary constructor retains its existing behavior.
"""
from collections import Counter, OrderedDict
import importlib
from numbers import Integral
import weakref

import jax
import jax.numpy as jnp
import numpy as np

FIELDS = ('ntt_bat_tf_step1', 'ntt_tf_step2', 'ntt_bat_tf_step3',
          'intt_bat_tf_step1', 'intt_tf_step2', 'intt_bat_tf_step3')


def _ints(values):
    if not isinstance(values, (list, tuple)) or not values:
        return None
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        return None
    return tuple(int(x) for x in values)


def _devices(array):
    if not isinstance(array, jax.Array):
        return None
    devices = array.devices()
    return next(iter(devices)) if len(devices) == 1 else None


def _field_signature(field, moduli, device):
    # These are small public arrays; checking them prevents an altered field
    # object from borrowing tables under a stale type/moduli-only identity.
    import math
    s = tuple(2 * math.ceil(math.log2(q)) for q in moduli)
    w = tuple(min(x, 32) for x in s)
    sw = tuple(a - b for a, b in zip(s, w))
    m = tuple(math.floor(2 ** bits / q) for bits, q in zip(s, moduli))
    specs = [('barrett_s', s), ('barrett_w', w), ('barrett_s_w', sw), ('barrett_m', m)]
    if any(tuple(getattr(field, name, ())) != expected for name, expected in specs):
        return None
    for name, expected, dtype in [('m', m, 'uint64'), ('moduli_reduction', moduli, 'uint64'),
                                  ('w', w, 'uint16'), ('s_w', sw, 'uint16')]:
        array = getattr(field, name, None)
        if _devices(array) != device or str(array.dtype) != dtype or array.shape != (len(moduli),):
            return None
        if not np.array_equal(np.asarray(array), np.asarray(expected, dtype=dtype)):
            return None
    return (type(field), moduli, s, w, sw, m, device)


class ScopedNTTSubsets:
    """One owner, bounded weak bindings, and no secret/key references."""
    def __init__(self, owner, *, max_entries=16):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        self._owner = weakref.ref(owner)
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self.stats = Counter()
        self._closed = False
        q, p = owner.ntt_q_max, owner.ntt_p
        self._source_refs = (weakref.ref(q), weakref.ref(p))
        self._source_type = type(q)
        module = importlib.import_module(type(q).__module__)
        self._supported_type = module.NTTCiphertextBarrettContext
        self._field_type = module.ff_context.BarrettContext
        self._layout = (owner.r, owner.c)
        self._degree = owner.degree
        self._moduli = (_ints(owner.q_towers), _ints(owner.p_towers))
        self._source_identity = tuple(self._identity(x) for x in (q, p))
        self._device = _devices(q.ntt_tf_step2)

    @staticmethod
    def _identity(source):
        return (id(source), tuple(source.moduli), source.r, source.c,
                source.transform_length, tuple(id(getattr(source, name)) for name in FIELDS))

    def _fallback(self, reason):
        self.stats['fallback'] += 1
        self.stats['fallback_' + reason] += 1
        return None

    def close(self):
        """Drop provider bookkeeping. Existing caller-owned contexts stay valid."""
        owner = self._owner()
        for name in ('rotation_polynomial_factory', 'multiplication_polynomial_factory',
                     'rescale_polynomial_factory'):
            factory = getattr(owner, name, None)
            if getattr(factory, '__self__', None) is self:
                setattr(owner, name, None)
        self._entries.clear()
        self._source_refs = ()
        self._owner = lambda: None
        self._closed = True

    def clear(self):
        """Release weak lookup entries; neither masters nor caller arrays owned."""
        self._entries.clear()

    @property
    def live_entries(self):
        return sum(ref() is not None for ref, _ in self._entries.values())

    def _sources(self, device):
        owner = self._owner()
        if owner is None or self._closed or bool(getattr(owner, 'perf_test', False)):
            return None
        if ((owner.r, owner.c) != self._layout or owner.degree != self._degree
                or (_ints(owner.q_towers), _ints(owner.p_towers)) != self._moduli
                or owner.ff_context_cls is not self._field_type):
            return None
        sources = tuple(ref() for ref in self._source_refs)
        if len(sources) != 2 or any(source is None for source in sources):
            return None
        if sources != (owner.ntt_q_max, owner.ntt_p):
            return None
        r, c = self._layout
        shapes = ((r, 4, r, 4), (r, c), (c, 4, c, 4),
                  (c, 4, c, 4), (r, c), (r, 4, r, 4))
        for source, moduli, identity in zip(sources, self._moduli, self._source_identity):
            if (type(source) is not self._supported_type or self._identity(source) != identity
                    or _ints(source.moduli) != moduli or len(set(moduli)) != len(moduli)
                    or type(source.ff_ctx) is not self._field_type
                    or not source.ff_ctx.computation_format_is_standard
                    or _ints(source.ff_ctx.moduli) != moduli
                    or _field_signature(source.ff_ctx, moduli, device) is None):
                return None
            for name, leading in zip(FIELDS, shapes):
                array = getattr(source, name)
                dtype = 'uint64' if 'tf_step2' in name else 'uint8'
                if (array.shape != (*leading, len(moduli)) or str(array.dtype) != dtype
                        or (isinstance(array, jax.Array) and _devices(array) != device)
                        or not isinstance(array, (jax.Array, np.ndarray))):
                    return None
        return sources

    def get(self, moduli, parameters, *, precision=32, device=None):
        """Return a borrowed-table context, or None to use ordinary dispatch."""
        if self._closed:
            return self._fallback('closed')
        requested = _ints(moduli)
        if requested is None or len(set(requested)) != len(requested):
            return self._fallback('moduli')
        if (type(precision) is not int or precision != 32 or not isinstance(parameters, dict)
                or set(parameters) != {'r', 'c', 'finite_field_context'}
                or any(type(parameters[k]) is not int for k in ('r', 'c'))
                or (parameters['r'], parameters['c']) != self._layout
                or self._layout[0] * self._layout[1] != self._degree):
            return self._fallback('layout_precision_parameters')
        if device is None or device != self._device:
            return self._fallback('device')
        field = parameters['finite_field_context']
        if (type(field) is not self._field_type or not field.computation_format_is_standard
                or _ints(field.moduli) != requested
                or _field_signature(field, requested, device) is None):
            return self._fallback('field')
        sources = self._sources(device)
        if sources is None:
            return self._fallback('owner_or_sources')
        lookup = {q: (source, i) for source in sources for i, q in enumerate(source.moduli)}
        if len(lookup) != sum(len(source.moduli) for source in sources) or any(q not in lookup for q in requested):
            return self._fallback('unknown_modulus')
        key = (requested, self._layout, self._degree, precision, self._field_type, device)
        entry = self._entries.get(key)
        if entry is not None:
            previous, identity = entry[0](), entry[1]
            if previous is not None and self._identity(previous) == identity:
                # A context retains its own field; verify that a caller did not
                # mutate it between weak-cache hits.
                if (type(previous.ff_ctx) is self._field_type
                        and previous.ff_ctx.computation_format_is_standard
                        and _ints(previous.ff_ctx.moduli) == requested
                        and _field_signature(previous.ff_ctx, requested, device) is not None):
                    self._entries.move_to_end(key); self.stats['hits'] += 1
                    return previous
            self._entries.pop(key, None)
        spans = []
        for q in requested:
            source, index = lookup[q]
            if spans and spans[-1][0] is source and spans[-1][2] == index:
                spans[-1][2] += 1
            else:
                spans.append([source, index, index + 1])
        result = object.__new__(self._supported_type)
        result.ff_ctx = field
        result.r, result.c = self._layout
        result.transform_length = self._degree
        result.moduli = list(requested)
        result.num_bytes = 4
        result.parameters = dict(parameters)
        result.psi_list = [lookup[q][0].psi_list[lookup[q][1]] for q in requested]
        result.omega_list = [lookup[q][0].omega_list[lookup[q][1]] for q in requested]
        with jax.default_device(device):
            for name in FIELDS:
                arrays = [getattr(source, name)[..., lo:hi] for source, lo, hi in spans]
                join = np.concatenate if all(isinstance(array, np.ndarray) for array in arrays) else jnp.concatenate
                setattr(result, name, arrays[0] if len(arrays) == 1 else join(arrays, axis=-1))
        # A barrier gives a clear lifetime boundary before an owning kernel is
        # permitted to release prior controls. The array payloads can still be
        # copied/materialized; this provider makes no zero-copy/HBM claim.
        jax.block_until_ready([getattr(result, name) for name in FIELDS])
        self._entries[key] = (weakref.ref(result), self._identity(result))
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        self.stats['builds'] += 1
        return result

    def polynomial_factory(self, shapes, parameters=None, *, payload=None):
        """Explicit isolated-kernel injection; fallback preserves constructor."""
        module = importlib.import_module('polynomial')
        original = module.Polynomial
        if parameters is None or 'ntt_ctx' in parameters or 'moduli' not in parameters:
            return original(shapes, parameters, payload=payload)
        factory = parameters.get('finite_field_context', self._field_type)
        if factory is not self._field_type:
            return original(shapes, parameters, payload=payload)
        # Device choice matches ordinary uncommitted construction. If caller
        # enters another default-device scope, get() refuses master reuse.
        field = factory(moduli=parameters['moduli'])
        device = _devices(field.m)
        layout = tuple(shapes.get('degree_layout', ()))
        if len(layout) != 2 or shapes.get('degree') != self._degree:
            return original(shapes, parameters, payload=payload)
        selected = self.get(parameters['moduli'],
            {'r': layout[0], 'c': layout[1], 'finite_field_context': field},
            precision=shapes.get('precision'), device=device)
        if selected is None:
            return original(shapes, parameters, payload=payload)
        return original(shapes, dict(parameters, ntt_ctx=selected), payload=payload)


def enable_scoped_ntt_subsets(context, *, max_entries=16):
    """Opt in one initialized keyless evaluator; returns its closeable owner.

    This affects future HERot, HEMul and rescale-facade controls for this context.
    HEMul inherits its factory for nested rescale controls; the tensor-only
    helper keeps its own path. Disabled split-rescale branches stay disabled.
    Existing controls remain valid, and unrelated contexts retain ordinary
    construction. Disable before context teardown by calling close().
    """
    from encrypted_polynomial import EncryptedPolynomialEvaluator
    EncryptedPolynomialEvaluator(context)
    # No ciphertext exists at opt-in time. Match the execution gate's public
    # noise/distribution assumptions here; each actual operation still checks
    # its real ciphertext with require_secure_execution.
    import math
    from jaxite_word import he_params
    noise = float(context.parameters.get('sigma', he_params.DEFAULT_SIGMA))
    if not math.isfinite(noise) or noise < he_params.DEFAULT_SIGMA:
        raise ValueError('context noise sigma is below the security table assumption')
    if context.parameters.get('secret_distribution', he_params.SECRET_DISTRIBUTION) != he_params.SECRET_DISTRIBUTION:
        raise ValueError('security table requires uniform_ternary secrets')
    cache = context._param_cache
    for name in ('rotation_polynomial_factory', 'multiplication_polynomial_factory',
                     'rescale_polynomial_factory'):
        if getattr(cache, name) is not None:
            raise ValueError(f'this context already owns a {name}')
    provider = ScopedNTTSubsets(cache, max_entries=max_entries)
    cache.rotation_polynomial_factory = provider.polynomial_factory
    cache.multiplication_polynomial_factory = provider.polynomial_factory
    cache.rescale_polynomial_factory = provider.polynomial_factory
    return provider
