#!/usr/bin/env python3
"""Opt-in real secure bootstrap validation; client decryption only at the end."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cadathlon/common'))
from encrypted_refresh import (RefreshProfile, SecureRefresh, BoundedSecureRefresh,
                               ClientRotationKeyStore, refresh_plan)
# Load experimental dependencies before capturing the source snapshot.
import encrypted_polynomial, encrypted_compilation, encrypted_rotation, encrypted_primitives


def source_hashes():
    """Include lazy core/helper dependencies, not just the top-level adapter."""
    paths = {Path(__file__).resolve()}
    paths.update((ROOT / 'jaxite_word').glob('*.py'))
    paths.update((ROOT / 'cadathlon/common').glob('*.py'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def measurement_status(*, output_finite, error, refresh_error, before, after):
    numerical = (output_finite is True and all(math.isfinite(value) and 0 <= value < 1e-3
                                             for value in (error, refresh_error)))
    unchanged = bool(before) and before == after
    return dict(passed=numerical and unchanged, numerical_passed=numerical,
                source_hashes_unchanged=unchanged)


@contextmanager
def cancellation_handlers():
    signals = (signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    def cancel(signum, frame):
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(f'refresh validation cancelled by signal {signum}')
    try:
        for sig in signals:
            signal.signal(sig, cancel)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _cleanup(contexts, key_store, active_error):
    """Attempt every release, retaining an evaluation error as the primary one."""
    import jax
    from jaxite_word import ckks_ctx

    errors = []
    while contexts:
        context = contexts.pop()
        if context is not None:
            try:
                ckks_ctx.evict_context_caches(context)
            except BaseException as error:
                errors.append(error)
        context = None
    for release in (jax.clear_caches, gc.collect):
        try:
            release()
        except BaseException as error:
            errors.append(error)
    if key_store is not None:
        try:
            key_store.close()
        except BaseException as error:
            errors.append(error)
    if errors:
        message = 'refresh cleanup failed: ' + '; '.join(str(error) for error in errors)
        if active_error is not None:
            active_error.add_note(message)
        else:
            for error in errors:
                if not isinstance(error, Exception):
                    error.add_note(message)
                    raise error
            raise RuntimeError(message) from errors[0]


def run(*, degree=65536, depth=22, period=4, dnum=8, scale_bits=56,
        level_budget=(2, 2), low_memory=True, continuation=True, compiled=False,
        amplitude=.75, public_bound=None, meta_precision=None, disk_keys=False):
    import jax
    from jaxite_word import CKKSContext, he_params, key_gen, ckks_ctx
    from support import require_secure_execution

    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('amplitude must be finite and positive')
    bounded = public_bound is not None or meta_precision is not None
    if bounded:
        public_bound = BoundedSecureRefresh._bound(1. if public_bound is None else public_bound)
    if meta_precision is not None and (type(meta_precision) is not int or not 1 <= meta_precision <= 62):
        raise ValueError('meta_precision must be an integer in [1, 62] or None')
    if bounded and amplitude > public_bound:
        raise ValueError('validation amplitude exceeds the declared public bound')
    jax.config.update('jax_enable_x64', True)
    started = time.perf_counter()
    source_hashes_start = source_hashes()
    params = RefreshProfile(degree, depth, scale_bits, 61, dnum).parameters()
    q, p = params['q_towers'], params['p_towers']
    plan = refresh_plan(degree, period, level_budget, max_level=depth)
    r = 256
    params['degree_layout'] = (r, degree // r)

    def progress(message):
        print(f'[{time.perf_counter()-started:.1f}s] {message}', file=sys.stderr, flush=True)

    progress(f'secure ring N={degree} Q={len(q)} P={len(p)} log2QP={math.log2(math.prod(q+p)):.3f}; {len(plan["rotations"])} rotation keys')
    client = server = refresh = key_store = checkpoint = report_checkpoint = None
    try:
        # This is the client/offline phase. Convert retained public keys to
        # native uint32 arrays immediately to bound Python object overhead.
        keys = key_gen.gen_pke_pair(q, [], degree)
        evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=dnum,
                                                noise_std=he_params.DEFAULT_SIGMA, noise_scale=1)
        evaluation = [jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1)
                      for name in ('a', 'b')]
        if disk_keys:
            progress('generating client-owned disk-backed public rotation keys')
            key_store = ClientRotationKeyStore.generate(keys['secret_key'], q, p,
                degree=degree, dnum=dnum, rotations=plan['rotations'])
            rotations = key_store.keys
            progress(f'public rotation keys prepared: {key_store.required_bytes} bytes on disk')
        else:
            rotations = {}
            for index in plan['rotations']:
                key = key_gen.gen_rotation_key(keys['secret_key'], q, p, index, dnum=dnum,
                    noise_std=he_params.DEFAULT_SIGMA, noise_scale=1)[index]
                rotations[index] = {name: np.asarray(key[name], dtype=np.uint32) for name in ('a', 'b')}
                del key
                progress(f'rotation keys {len(rotations)}/{len(plan["rotations"])}')
        client = CKKSContext(dict(params, **keys))
        server = CKKSContext(dict(params, evaluation_key=evaluation))
        server.program_initialization(list(plan['rotations']), dnum=dnum, r=r, c=degree // r,
                                      batch=1, pregenerated_rotation_keys=rotations)
        del keys, evaluation, rotations
        gc.collect()
        progress('separate keyless server initialized')

        def forbidden(*args, **kwargs):
            raise AssertionError('server encryption, decryption, or key generation is forbidden')

        for name in ('encrypt', 'encrypt_slots', 'decrypt', 'decrypt_slots', 'decode'):
            setattr(server, name, forbidden)
        assert server.secret_key is None
        assert server._param_cache.secret_key is None
        assert server.parameters.get('secret_key') is None
        if bounded:
            refresh = BoundedSecureRefresh(server, period=period, level_budget=level_budget,
                low_memory=low_memory, compiled=compiled,
                public_bound=public_bound, meta_precision=meta_precision)
        else:
            refresh = SecureRefresh(server, period=period, level_budget=level_budget,
                low_memory=low_memory, compiled=compiled)
        checkpoint = refresh.engine._stage_checkpoint

        def report_checkpoint(tag, ct=None):
            checkpoint(tag, ct)
            progress(f'bootstrap stage: {tag}')

        refresh.engine._stage_checkpoint = report_checkpoint
        setup_seconds = time.perf_counter() - started
        progress('bootstrap constants prepared')
        pattern = np.linspace(-amplitude, amplitude, period)
        values = np.tile(pattern, degree // 2 // period)
        encrypted = client.encrypt_slots(values)
        encrypted = server.he_level_reduce[depth, refresh.input_level].level_reduce(encrypted)
        # The only remaining access to client.secret_key is final decryption.
        generation_names = ('gen_pke_pair', 'gen_evaluation_key', 'gen_rotation_key')
        # Historical flat imports and package imports can identify distinct
        # key_gen modules. Guard both runtime spellings used by primitives.
        modules = {module for name, module in sys.modules.items()
                   if name in ('key_gen', 'jaxite_word.key_gen')}
        originals = {(module, name): getattr(module, name)
                     for module in modules for name in generation_names}
        for module, name in originals:
            setattr(module, name, forbidden)
        try:
            refreshed_at = time.perf_counter()
            result = refresh.refresh(encrypted)
            jax.block_until_ready(result.polynomial)
            refresh_seconds = time.perf_counter() - refreshed_at
            progress('homomorphic refresh complete')
            actual_level = server._param_cache.level_for_num_q(result.num_moduli)
            manifest = require_secure_execution(refresh, result)[0]
            refreshed = result
            # A subsequent real ciphertext multiplication proves that the
            # refreshed output is usable by the same polynomial evaluator.
            if continuation:
                result = refresh.evaluator.multiply(result, result)
                jax.block_until_ready(result.polynomial)
        finally:
            for (module, name), original in originals.items():
                setattr(module, name, original)
        actual = np.asarray(client.decrypt_slots(result))
        refreshed_actual = np.asarray(client.decrypt_slots(refreshed))
        expected = values ** 2 if continuation else values
        error = float(np.max(np.abs(actual - expected)))
        refresh_error = float(np.max(np.abs(refreshed_actual - values)))
        relative_error = float(np.max(np.abs(actual-expected) / np.abs(expected)))
        refresh_relative_error = float(np.max(np.abs(refreshed_actual-values) / np.abs(values)))
        source_hashes_end = source_hashes()
        finite = bool(np.isfinite(actual).all() and np.isfinite(refreshed_actual).all())
        status = measurement_status(output_finite=finite, error=error, refresh_error=refresh_error,
                                    before=source_hashes_start, after=source_hashes_end)
        report = dict(**status, backend=jax.default_backend(),
                      device=str(jax.devices()[0]), profile=dict(degree=degree, depth=depth,
                      period=period, dnum=dnum, scale_bits=scale_bits), plan=plan,
                      actual_output_level=actual_level, continuation=continuation,
                      max_abs_error=error, refresh_max_abs_error=refresh_error, output_finite=finite,
                      max_relative_error=relative_error, refresh_max_relative_error=refresh_relative_error,
                      amplitude=amplitude, public_bound=public_bound, meta_precision=meta_precision,
                      compiled_primitives=compiled, absolute_error_gate=1e-3,
                      disk_backed_public_keys=bool(disk_keys),
                      public_key_store_bytes=key_store.required_bytes if key_store is not None else None,
                      source_hashes_at_end=source_hashes_end,
                      setup_seconds=setup_seconds, refresh_seconds=refresh_seconds,
                      total_seconds=time.perf_counter()-started, actual_ring=manifest,
                      timing_scope='component timing through client validation; excludes final cleanup and process startup',
                      server_has_secret_key=False, runtime_key_generation_forbidden=True,
                      source_hashes_at_start=source_hashes_start,
                      refreshed_sample=refreshed_actual[:period].tolist(), final_sample=actual[:period].tolist())
        progress(f'max absolute error {error:.6g}')
        return report
    finally:
        contexts = [client, server]
        refresh = client = server = checkpoint = report_checkpoint = None
        _cleanup(contexts, key_store, sys.exc_info()[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degree', type=int, default=65536)
    parser.add_argument('--depth', type=int, default=22)
    parser.add_argument('--period', type=int, default=4)
    parser.add_argument('--dnum', type=int, default=8)
    parser.add_argument('--scale-bits', type=int, default=56)
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--amplitude', type=float, default=.75)
    parser.add_argument('--public-bound', type=float)
    parser.add_argument('--meta-precision', type=int)
    parser.add_argument('--disk-keys', action='store_true', help='store client-generated public rotation keys in temporary read-only arrays')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'passed': False, 'error': 'refresh validation has not completed'}) + '\n')
    try:
        with cancellation_handlers():
            report = run(degree=args.degree, depth=args.depth, period=args.period,
                         dnum=args.dnum, scale_bits=args.scale_bits, level_budget=(args.budget, args.budget),
                         compiled=args.compiled, amplitude=args.amplitude,
                         public_bound=args.public_bound, meta_precision=args.meta_precision,
                         disk_keys=args.disk_keys)
    except BaseException as error:
        report = dict(passed=False, error=f'{type(error).__name__}: {error}',
                      notes=list(getattr(error, '__notes__', ())))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        raise
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('actual_ring', 'plan')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
