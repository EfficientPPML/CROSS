"""Run the compact-v1 two-CNN competition schedule within a wall-time budget.

The parent enforces the budget over a fresh worker process, including setup,
requests, validation and cleanup. Setup and requests are serial. The two
published compact CNNs are distinct from the original full-model/SOTA track;
this schedule does not claim trained accuracy or a SOTA comparison.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

SRC = Path(__file__).resolve().parent
CASES = ('student1.in', 'student2.in')


def prepare_all(cases, build):
    """Collect every constructor outcome before returning any live mapping."""
    mappings, failures = {}, []
    for case in cases:
        try:
            mappings[case] = build(case)
        except Exception as error:
            failures.append(f'{case}: {type(error).__name__}: {error}')
        except BaseException as error:
            release_all(mappings, primary_error=error)
            raise
    return mappings, failures


def release_all(prepared, *, primary_error=None):
    failures = []
    interruptions = []
    released = set()
    for name, (mapping, _) in prepared.items():
        if id(mapping) in released:
            continue
        released.add(id(mapping))
        try:
            mapping.release()
        except BaseException as error:
            failures.append(f'{name}: {type(error).__name__}: {error}')
            if not isinstance(error, Exception):
                interruptions.append(error)
    if failures:
        message = 'cleanup failed: ' + '; '.join(failures)
        if primary_error is not None:
            primary_error.add_note(message)
        elif interruptions:
            interruptions[0].add_note(message)
            raise interruptions[0]
        else:
            raise RuntimeError(message)


def worker(report_path, *, baseline=False):
    import runner
    from support import cross_root, read_json, load_module
    from reference import make_case
    import baseline_student

    started = time.monotonic()
    report = {'passed': False, 'cases': {}, 'schedule': list(CASES),
              'profile': 'compact-v1', 'gate_version': 'compact-v1',
              'mode': 'baseline' if baseline else 'candidate', 'setup_workers': 1,
              'trained_accuracy_evaluated': False}
    prepared = {}
    try:
        root = cross_root()
        import jax
        jax.config.update('jax_enable_x64', True)
        from jaxite_word.mapping import defer_cache_clearing
        devices = tuple(jax.devices('tpu')[:1])
        if not devices:
            raise RuntimeError('This evaluation requires a TPU')
        report['devices'] = [str(device) for device in devices]
        report['device_kinds'] = [getattr(device, 'device_kind', 'unknown') for device in devices]
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('jax', 'jaxlib', 'libtpu', 'numpy', 'torch')}
        report['versions']['python'] = sys.version.split()[0]
        paths = [SRC / 'baseline_student.py', SRC / 'runner.py', SRC / 'evaluate.py',
                 SRC / 'compact_models.py', SRC / 'reference.py']
        if not baseline:
            paths.append(SRC / 'cnn_opt.py')
        paths += [root / 'jaxite_word' / name for name in
                  ('mapping.py', 'packing.py', 'ckks_ctx.py', 'bsgs.py',
                   'composite_prime_gen.py')]
        paths += [SRC.parent / 'test' / name for case in CASES
                  for name in (case, case.replace('.in', '.out'))]
        report['source_sha256'] = {(str(path.relative_to(root)) if path.is_relative_to(root)
                                   else 'problem/' + path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in paths}
        report['fixture_sha256'] = {
            name: {suffix: hashlib.sha256((SRC.parent / 'test' / name).with_suffix(suffix).read_bytes()).hexdigest()
                   for suffix in ('.in', '.out')}
            for name in CASES
        }
        report['model_source_sha256'] = hashlib.sha256((SRC / 'compact_models.py').read_bytes()).hexdigest()
        implementation = baseline_student if baseline else load_module(SRC / 'cnn_opt.py')
        models = {}
        for name in CASES:
            case = read_json(SRC.parent / 'test' / name)
            model, shape, _, _ = make_case(case)
            # Torch FX tracing installs process-wide patches and must not
            # overlap another trace, even for independent model objects.
            packed = baseline_student.pack_model(model, shape) if baseline else None
            models[name] = (model, shape, packed, case['batch'])

        def build(name):
            model, shape, packed, batch = models[name]
            began = time.monotonic()
            if baseline:
                mapping = implementation.build_packed_mapping(packed, batch=batch, devices=devices)
            else:
                mapping = implementation.build_mapping(model, shape, batch=batch, devices=devices)
            return mapping, time.monotonic() - began

        # Keep already-prepared AOT mappings alive while collecting any
        # construction failures; release every owned mapping before clearing
        # process-wide tracing caches, including failure paths.
        with defer_cache_clearing():
            primary_error = None
            try:
                prepared, failures = prepare_all(CASES, build)
                if failures:
                    raise RuntimeError('; '.join(failures))
                for name in CASES:
                    args = SimpleNamespace(case=SRC.parent / 'test' / name,
                                           reference=False, baseline=False,
                                           student_baseline=baseline, warmups=1, repeats=3)
                    report['cases'][name] = runner.run(args, prepared=prepared[name])
                    report['cases'][name]['model'] = read_json(args.case)['model']
                report['passed'] = True
            except BaseException as error:
                primary_error = error
                raise
            finally:
                release_all(prepared, primary_error=primary_error)
        report['device_memory'] = [device.memory_stats() for device in devices]
    except BaseException as error:
        report['passed'] = False
        report['error'] = f'{type(error).__name__}: {error}'
        report['error_notes'] = list(getattr(error, '__notes__', ()))
        if not isinstance(error, Exception):
            raise
    finally:
        report['worker_elapsed_s'] = time.monotonic() - started
        report['peak_host_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['passed'] else 1


def stop_process_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    process.wait()


def process_group_alive(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


@contextmanager
def termination_signals():
    signals = (signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    def terminate(signum, frame):
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        raise SystemExit(128 + signum)
    try:
        for sig in signals:
            signal.signal(sig, terminate)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def evaluate(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'budget_s': args.budget_s, 'profile': 'compact-v1',
              'gate_version': 'compact-v1', 'schedule': list(CASES),
              'mode': 'baseline' if getattr(args, 'baseline', False) else 'candidate',
              'trained_accuracy_evaluated': False}
    with (args.output / 'evaluation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker_path = args.output / 'worker.json'
        worker_path.unlink(missing_ok=True)
        path = args.output / 'report.json'
        path.unlink(missing_ok=True)
        pending_error = None
        process = None
        with (args.output / 'worker.log').open('w') as log:
            try:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), '--worker-report', str(worker_path)]
                    + (['--baseline'] if getattr(args, 'baseline', False) else []),
                    cwd=SRC, env=dict(os.environ, JAX_PLATFORMS='tpu', PYTHONUNBUFFERED='1'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                code = process.wait(timeout=max(0, args.budget_s - (time.monotonic() - started)))
                report['worker_returncode'] = code
                if worker_path.exists():
                    report.update(json.loads(worker_path.read_text()))
                else:
                    report['error'] = 'worker produced no report; see worker.log'
                if code != 0:
                    report['passed'] = False
                    report.setdefault('error', f'worker exited {code}; see worker.log')
            except subprocess.TimeoutExpired:
                report['error'] = 'evaluation exceeded wall-time budget'
                stop_process_group(process)
            except Exception as error:
                report['passed'] = False
                report['error'] = f'{type(error).__name__}: {error}'
            except BaseException as error:
                report['passed'] = False
                report['error'] = f'{type(error).__name__}: {error}'
                pending_error = error
            finally:
                if process is not None and (process.poll() is None or process_group_alive(process.pid)):
                    report['passed'] = False
                    report.setdefault('error', 'worker left descendant processes; owned group stopped')
                    stop_process_group(process)
        report['elapsed_s'] = time.monotonic() - started
        report['within_budget'] = report['elapsed_s'] <= args.budget_s
        report['passed'] = report['passed'] and report['within_budget']
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        if pending_error is not None:
            raise pending_error
    print(json.dumps({'passed': report['passed'], 'elapsed_s': report['elapsed_s'],
                      'report': str(path), 'error': report.get('error')}, allow_nan=False))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-s', type=float, default=300.)
    parser.add_argument('--baseline', action='store_true', help='Measure the compact encrypted reference')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker-report', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_report:
        return worker(args.worker_report, baseline=args.baseline)
    if not math.isfinite(args.budget_s) or not 0 < args.budget_s <= 300:
        parser.error('--budget-s must be positive, finite, and at most 300')
    if args.output is None:
        args.output = SRC / 'suite' / 'run' / ('compact-baseline' if args.baseline else 'compact-candidate')
    args.output = args.output.resolve()
    try:
        with termination_signals():
            return evaluate(args)
    except Exception as error:
        print(json.dumps({'passed': False, 'error': f'{type(error).__name__}: {error}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
