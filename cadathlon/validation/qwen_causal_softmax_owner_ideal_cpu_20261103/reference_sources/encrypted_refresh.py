"""Experimental, evaluation-key-only CKKS homomorphic refresh.

The sparse bootstrap operates on a *publicly specified periodic packing* in a
full-slot ciphertext. Every encrypted value and every public mask in a circuit
must repeat with ``period`` slots. The server cannot inspect that invariant;
the circuit's packing rules must establish it. No client callback, encryption,
decryption, or secret-dependent key generation is used here.

This adapter uses the repository's existing Bootstrap engine. It is not a new
bootstrap construction or an assertion that complete transformer references
have passed their accuracy/performance gates.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import tempfile


class ClientRotationKeyStore:
    """Client setup owner for disk-backed *public* evaluation keys.

    ``generate`` never stores its secret-key argument on the returned owner or
    on disk. ``keys`` can be passed directly as pregenerated_rotation_keys.
    Release evaluator contexts before closing this owner. A caller may select
    a larger local filesystem with ``directory``; insufficient space fails
    before expensive key generation begins.
    """

    @classmethod
    def generate(cls, secret_key, q, p, *, degree, dnum, rotations, directory=None):
        import numpy as np
        from jaxite_word import he_params, key_gen

        rotations = tuple(sorted(set(rotations)))
        if (type(degree) is not int or degree < 1 or type(dnum) is not int or dnum < 1
                or not q or not p or any(type(index) is not int for index in rotations)):
            raise ValueError('valid ring parameters and integer rotations required')
        target = he_params.SecurityTarget(128, he_params.CostModel.CLASSICAL)
        if not he_params.qp_is_secure(q, p, degree, target):
            raise ValueError('public key store requires a 128-bit-secure Q/P ring')
        he_params.assert_p_coverage(q, p, dnum)
        partition = (len(q) + dnum - 1) // dnum
        parts = (len(q) + partition - 1) // partition
        expected_shape = (parts, len(q) + len(p), degree)
        required = len(rotations) * (2 * math.prod(expected_shape) * 4 + 8192)
        parent = Path(directory) if directory is not None else Path(tempfile.gettempdir())
        if shutil.disk_usage(parent).free < required + 64 * 1024**2:
            raise ValueError(f'public rotation keys require {required} bytes plus 64 MiB free in {parent}')
        owner = cls()
        owner._directory = tempfile.TemporaryDirectory(prefix='cross-public-rotation-keys-', dir=parent)
        owner.keys = {}
        owner.required_bytes = required
        owner.closed = False
        try:
            for index in rotations:
                generated = key_gen.gen_rotation_key(secret_key, q, p, index, dnum=dnum,
                    noise_std=he_params.DEFAULT_SIGMA, noise_scale=1)[index]
                stored = {}
                for name in ('a', 'b'):
                    array = np.asarray(generated[name], dtype=np.uint32)
                    if array.shape != expected_shape:
                        raise ValueError('generated rotation-key shape differs from the ring contract')
                    path = Path(owner._directory.name) / f'{index}-{name}.npy'
                    mapped = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint32, shape=expected_shape)
                    mapped[:] = array
                    mapped.flush()
                    del mapped
                    stored[name] = np.load(path, mmap_mode='r', allow_pickle=False)
                owner.keys[index] = stored
                del generated, stored, array
            return owner
        except BaseException:
            owner.close()
            raise

    def close(self):
        if self.closed:
            return
        self.keys.clear()
        self._directory.cleanup()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class RefreshProfile:
    degree: int = 131072
    depth: int = 52
    scale_bits: int = 56
    first_bits: int = 61
    dnum: int = 8

    def parameters(self):
        """Generate and verify the actual complete evaluation-key Q/P ring."""
        from jaxite_word import he_params, util
        from jaxite_word.composite_prime_gen import composite_prime_gen

        for name in ('degree', 'depth', 'scale_bits', 'first_bits', 'dnum'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.degree & (self.degree - 1):
            raise ValueError('degree must be a power of two')
        q = composite_prime_gen(2, 2 * (self.depth + 1), self.first_bits,
                                self.scale_bits, 2 * self.degree, 31)
        p = util.generate_p_towers(q, self.dnum, self.degree, aux_bits=30)
        target = he_params.SecurityTarget(128, he_params.CostModel.CLASSICAL)
        if not he_params.qp_is_secure(q, p, self.degree, target):
            raise ValueError('actual bootstrap Q/P ring exceeds the 128-bit security ceiling')
        he_params.assert_p_coverage(q, p, self.dnum)
        scale = float(q[-2]) * q[-1]
        return dict(degree=self.degree, num_slots=self.degree // 2,
                    scaling_factor=scale, output_scale=scale,
                    q_towers=q, p_towers=p, composite_degree=2,
                    p=self.scale_bits, CKKS_M_FACTOR=1, max_bits_in_word=61,
                    noise_scale_degree=1)


def refresh_plan(degree, period, level_budget=(2, 2), *, max_level):
    """Return the public rotation contract and conservative normalized capacity."""
    from jaxite_word.bootstrapping import Bootstrap

    if (type(degree) is not int or degree < 8 or degree & (degree - 1)
            or type(period) is not int or period < 1 or period & (period - 1)
            or period > degree // 2 or (degree // 2) % period):
        raise ValueError('period must be a power-of-two divisor of N/2')
    # The sparse engine needs two transform levels; repeating a one/two-slot
    # period in a four-slot transform preserves the same packing invariant.
    slots = max(4, period)
    if len(level_budget) != 2 or any(type(n) is not int or n < 2 or n > int(math.log2(slots))
                                     for n in level_budget):
        raise ValueError('sparse level_budget requires two valid budgets >= 2')
    if type(max_level) is not int or max_level < 0:
        raise ValueError('max_level must be a nonnegative integer')
    depth = Bootstrap.compute_bootstrap_nq(2, level_budget, secret_key_dist='uniform_ternary')['boot_depth']
    # The engine produces nsd=2. One genuine rescale restores nsd=1;
    # one plaintext multiplication/rescale restores the evaluator's scale.
    capacity = max_level - depth - 2
    if capacity < 1:
        raise ValueError('chain has no usable levels after normalized bootstrap')
    return dict(period=period, bootstrap_slots=slots, level_budget=tuple(level_budget),
                boot_depth=depth, output_level=capacity,
                rotations=Bootstrap.required_rotation_indices(degree, slots, level_budget))


class SecureRefresh:
    """Refresh periodic ciphertexts with only retained public evaluation keys."""

    input_level = 1

    def __init__(self, context, *, period, level_budget=(2, 2), low_memory=True, compiled=False,
                 preserve_dispatch_cache=False):
        from encrypted_polynomial import EncryptedPolynomialEvaluator
        from jaxite_word.bootstrapping import Bootstrap

        # Reuse the strict real-context, Q/P-security and secret-key checks.
        self.evaluator = EncryptedPolynomialEvaluator(
            context, strategy='power', low_memory=low_memory,
            compiled_multiply=compiled, compiled_linear=compiled,
            preserve_dispatch_cache=preserve_dispatch_cache)
        self.ctx = context
        cache = context._param_cache
        if context.num_slots != context.degree // 2:
            raise ValueError('refresh adapter requires full-slot periodic ciphertexts')
        if (tuple(cache.q_towers) != tuple(context.q_towers)
                or tuple(cache.p_towers) != tuple(context.p_towers)
                or cache.degree != context.degree or cache.composite_degree != 2):
            raise ValueError('live primitive cache differs from the verified execution ring')
        if cache.per_level_rotation_keys:
            raise ValueError('refresh requires retained pregenerated rotation keys')
        self.plan = refresh_plan(context.degree, period, level_budget, max_level=context.max_level)
        missing = set(self.plan['rotations']) - set(cache.raw_rotation_keys)
        if missing:
            raise ValueError(f'missing pregenerated bootstrap rotation keys: {sorted(missing)}')
        # Share the exact initialized ring and its primitive accessors. Only
        # the transform's public periodicity changes; the normal full-slot
        # codec/context remains untouched.
        view = copy.copy(context)
        view.num_slots = self.plan['bootstrap_slots']
        self.engine = Bootstrap(view)
        self.engine._low_mem = bool(low_memory)
        self.engine._dc_subtract = False  # never construct an encrypted template
        self.engine.control_gen(level_budget=list(level_budget), secret_key_dist='uniform_ternary')
        self.engine.setup_key()  # registers maps; never generates secret keys
        self.refresh_count = 0
        self.compiled_primitives = _install_compiled_primitives(self.engine, context) if compiled else None
        if low_memory:
            checkpoint = self.engine._stage_checkpoint

            def release_working_caches(tag, ct=None):
                # Preserve the engine's own barriers and hoisted-plaintext
                # cleanup first. Sharing only completed NTT metadata avoids
                # retaining duplicate tables through the polynomial schedule.
                checkpoint(tag, ct)
                if ct is not None:
                    self.evaluator._finish(ct)
                if self.compiled_primitives is not None:
                    for helper in self.compiled_primitives:
                        helper.clear()
                self.evaluator._clear_working_caches()
                self.evaluator._active_level = None

            self.engine._stage_checkpoint = release_working_caches

    @property
    def output_level(self):
        return self.plan['output_level']

    def refresh(self, value):
        """Return a refreshed, nsd=1 ciphertext at the canonical evaluator scale."""
        return self._refresh_canonical(value)

    def _bootstrap(self, value):
        return self.engine.bootstrap(value)

    def _refresh_canonical(self, value):
        from support import require_secure_execution

        level = self.evaluator._level(value)
        if level < 1:
            raise ValueError('refresh must be scheduled before the final level is exhausted')
        # Truncation preserves both the represented message and its scale.
        # The engine's AdjustCiphertext then consumes the remaining level.
        depleted = self.evaluator._lower(value, 1)
        result = self._bootstrap(depleted)
        require_secure_execution(self, result)
        level = self.ctx._param_cache.level_for_num_q(result.num_moduli)
        nsd = getattr(result, '_ckks_nsd', None)
        if nsd == 2:
            if level < 1:
                raise ValueError('bootstrap output lacks a normalization level')
            if self.compiled_primitives is None:
                result = self.ctx.he_rescale[level, level - 1].rescale(result)
            else:
                result = self.compiled_primitives[2].rescale(result, level, level - 1)
            # A real composite rescale has just removed one scale factor.
            result._ckks_nsd = 1
        elif nsd != 1:
            raise ValueError(f'unexpected bootstrap output noise-scale degree: {nsd}')
        result = self.evaluator.weighted_sum([result], [1.])
        if self.evaluator._level(result) < self.plan['output_level']:
            raise ValueError('bootstrap returned fewer levels than its advertised contract')
        require_secure_execution(self, result)
        self.refresh_count += 1
        return result


class _Operation:
    """Delegate only unmodified attributes to the original eager facade."""
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        if name in self.overrides:
            return self.overrides[name]
        return getattr(self.original(), name)


class _Accessor:
    def __init__(self, original, bind):
        self.original, self.bind = original, bind

    def __getitem__(self, key):
        return self.bind(key, lambda: self.original[key])

    def __getattr__(self, name):
        return getattr(self.original, name)


def _install_compiled_primitives(engine, context):
    """Bind compiled operations only on this engine's private context view."""
    from encrypted_primitives import (CompiledRelinearization, CompiledRescale,
                                     CompiledPlaintextMultiply, CompiledTensorOnly)
    from encrypted_rotation import CompiledRotation
    rotations = CompiledRotation(context)
    relinearization = CompiledRelinearization(context)
    rescale = CompiledRescale(context)
    plaintext = CompiledPlaintextMultiply(context)
    tensor = CompiledTensorOnly(context)

    def bind_rotation(key, original):
        level, index = key
        def rotate(value):
            if context._param_cache.level_for_num_q(value.num_moduli) != level:
                raise ValueError('rotation level differs from bound accessor')
            return rotations.rotate(value, index)
        return _Operation(original, rotate=rotate)

    def bind_multiply(level, original):
        cache = context._param_cache
        if type(level) is not int or not 0 <= level < cache.max_level:
            raise ValueError('compiled multiply output level must be below the maximum level')
        def relinearize(value):
            if cache.level_for_num_q(value.num_moduli) != level + 1:
                raise ValueError('relinearization level differs from bound accessor')
            return relinearization.relinearize(value)
        def multiply(left, right):
            if cache.level_for_num_q(left.num_moduli) != level + 1:
                raise ValueError('tensor-product level differs from bound accessor')
            return tensor.multiply(left, right)
        return _Operation(original, relinearize=relinearize, hemul_no_relin=multiply,
                          input_num_moduli=cache.num_q_at_level(level + 1))

    engine.ctx._he_rot = _Accessor(context.he_rot, bind_rotation)
    engine.ctx._he_mul = _Accessor(context.he_mul, bind_multiply)
    engine.ctx._he_rescale = _Accessor(context.he_rescale,
        lambda key, original: _Operation(original, rescale=lambda value: rescale.rescale(value, *key)))
    engine.ctx._ptct_mul = _Accessor(context.ptct_mul,
        lambda level, original: _Operation(original, _mul_encoded=
            lambda value, encoded: plaintext.multiply_encoded(value, encoded, level)))
    return rotations, relinearization, rescale, plaintext, tensor


class BoundedSecureRefresh(SecureRefresh):
    """Experimental compiled refresh with an explicit public magnitude bound.

    Reserve two input levels for real range preconditioning. The final real
    plaintext multiplication combines scale normalization, bound restoration
    and averaging over the known periodic layout, preserving normalized output
    capacity. Bounds may vary by public circuit operation. Values beyond the
    supplied bound are outside this experimental contract. Large-ring numerical
    validation remains required before use as a transformer reference.
    """
    input_level = 2

    def __init__(self, context, *, public_bound=1., meta_precision=None, compiled=True,
                 preserve_dispatch_cache=False, **kwargs):
        self.public_bound = self._bound(public_bound)
        if meta_precision is not None and (type(meta_precision) is not int or not 1 <= meta_precision <= 62):
            raise ValueError('meta_precision must be an integer in [1, 62] or None')
        self.meta_precision = meta_precision
        super().__init__(context, compiled=compiled,
            preserve_dispatch_cache=preserve_dispatch_cache, **kwargs)

    @staticmethod
    def _bound(value):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError('public magnitude bound must be numeric')
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('public magnitude bound must be positive and finite')
        return value

    @property
    def output_level(self):
        return self.plan['output_level']

    def _bootstrap(self, value):
        if self.meta_precision is None:
            return self.engine.bootstrap(value)
        return self.engine.meta_bootstrap(value, precision=self.meta_precision)

    def refresh(self, value, *, public_bound=None):
        from support import require_secure_execution

        bound = self.public_bound if public_bound is None else self._bound(public_bound)
        if self.evaluator._level(value) < self.input_level:
            raise ValueError('public preconditioning requires two remaining input levels')
        if bound != 1.:
            value = self.evaluator.weighted_sum([value], [1. / bound])
        result = self._bootstrap(self.evaluator._lower(value, 1))
        require_secure_execution(self, result)
        level = self.ctx._param_cache.level_for_num_q(result.num_moduli)
        nsd = getattr(result, '_ckks_nsd', None)
        if nsd == 2:
            if level < 1:
                raise ValueError('bootstrap output lacks a normalization level')
            if self.compiled_primitives is None:
                result = self.ctx.he_rescale[level, level - 1].rescale(result)
            else:
                result = self.compiled_primitives[2].rescale(result, level, level - 1)
            result._ckks_nsd = 1
        elif nsd != 1:
            raise ValueError(f'unexpected bootstrap output noise-scale degree: {nsd}')

        # The input layout publicly repeats every period. Project back into
        # that same subspace with real rotations and additions; no slots are
        # inspected. Folding the average/bound into the scale alignment saves
        # the two separate plaintext multiplication/rescale operations.
        period = self.plan['period']
        shift = period
        while shift < self.ctx.num_slots:
            rotated = (self.evaluator.rotate(result, shift) if self.compiled_primitives is None
                       else self.compiled_primitives[0].rotate(result, shift))
            result = self.evaluator.add(result, rotated)
            shift *= 2
        aliases = self.ctx.num_slots // period
        result = self.evaluator.weighted_sum([result], [bound / aliases])
        if self.evaluator._level(result) < self.output_level:
            raise ValueError('combined normalization returned insufficient output levels')
        require_secure_execution(self, result)
        self.refresh_count += 1
        return result
