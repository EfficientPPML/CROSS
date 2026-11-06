"""Run an entire named P1 evaluation, archive every case, and enforce wall time.

Assets must be staged before invoking this command. The pinned external
harness remains unchanged. The student schedule uses C3/C4 and a separately
calibrated secure ring; legacy retains C1-C6. Setup, all three repeats,
archiving, validation, and daemon cleanup count toward --budget-s.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

from daemon_control import stop_daemon
import install_suite
import score
from p1_evidence import archive_hashes, model_identity, sha256

SUITE = Path(__file__).resolve().parent


def source_hashes(cross):
    sources = {('problem/' + str(path.relative_to(SUITE.parent))): path
               for path in SUITE.parent.rglob('*.py')
               if not {'run', 'results', 'calibration'}.intersection(path.relative_to(SUITE.parent).parts)}
    sources.update({('CROSS/' + str(path.relative_to(cross))): path
                    for path in (cross / 'jaxite_word').rglob('*.py')})
    support = SUITE.parents[3] / 'common' / 'support.py'
    if support.is_file():
        sources['common/support.py'] = support
    if (SUITE / 'suite_config.json').is_file():
        sources['problem/suite/suite_config.json'] = SUITE / 'suite_config.json'
    return {name: sha256(path) for name, path in sorted(sources.items())}


@contextlib.contextmanager
def cancellation_handlers():
    def cancelled(signum, frame):
        raise InterruptedError(f'evaluation cancelled by signal {signum}')
    previous = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def cancellation_shield():
    """A repeated cancellation must not interrupt bounded release/rollback."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def evaluation_lock():
    """Only one evaluator may own this checkout's daemon and archive paths."""
    path = SUITE / 'run' / 'evaluation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'an evaluation already owns {path}; use a separate checkout') from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()) + '\n')
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def terminate_process_group(process):
    """Reap descendants too when the parent exits before its children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # wait() observes only the parent. A child may ignore SIGTERM and
        # retain device handles after that parent has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_command(command, *, cwd, env, log, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('evaluation wall-time budget exhausted')
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            code = process.wait(timeout=remaining)
        except BaseException:
            # The harness launches subprocesses; stop the owned process group,
            # not just its parent. Resident daemons are cleaned up separately.
            with cancellation_shield():
                terminate_process_group(process)
            raise
        if code:
            terminate_process_group(process)
            raise RuntimeError(f'command exited {code}: {command}')
        # A successful leader can leave a child behind. Its return code does
        # not prove that the process group released accelerator resources.
        terminate_process_group(process)


def cleanup():
    failures = []
    for pidfile in (SUITE / 'run').glob('*/service.pid'):
        try:
            pid = int(pidfile.read_text().strip())
            stop_daemon(pid, timeout=5)
            pidfile.unlink(missing_ok=True)
        except FileNotFoundError:
            continue
        except Exception as error:
            failures.append(f'{pidfile}: {error}')
    if failures:
        raise RuntimeError('daemon cleanup failed: ' + '; '.join(failures))


def publish_calibration(staged, destination, deadline):
    """Publish only after the whole evaluation passes, retaining a good prior baseline."""
    if time.monotonic() > deadline:
        raise TimeoutError('evaluation budget exhausted before calibration publication')
    previous = destination.read_bytes() if destination.exists() else None
    staged.replace(destination)
    completed = time.monotonic()
    if completed > deadline:
        if previous is None:
            destination.unlink()
        else:
            rollback = staged.with_name('previous-calibration.json')
            rollback.write_bytes(previous)
            rollback.replace(destination)
        raise TimeoutError('evaluation budget exhausted during calibration publication')
    return completed


def finalize_reports(result, report_paths, *, started, deadline,
                     staged=None, destination=None, snapshot=None):
    """Keep calibration publication reversible through final I/O and budget checks."""
    backup, had_previous, attempted = None, False, False

    def refresh_elapsed():
        result['elapsed_s'] = time.monotonic()-started
        result['within_budget'] = time.monotonic() <= deadline

    def write(path):
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    try:
        refresh_elapsed()
        if result['passed'] and not result['within_budget']:
            raise TimeoutError('evaluation budget exhausted before final reporting')
        if staged is not None and result['passed']:
            # Prepare rollback bytes before committing; rollback itself then
            # requires only rename/unlink even if subsequent writes fail.
            backup = staged.with_name('previous-published-calibration.json')
            had_previous = destination.exists()
            if had_previous:
                shutil.copyfile(destination, backup)
            if snapshot is not None:
                write(snapshot)
            attempted = True
            publish_calibration(staged, destination, deadline)
        paths = [*report_paths, *([snapshot] if snapshot is not None and result['passed'] else [])]
        refresh_elapsed()
        for path in paths:
            write(path)
        refresh_elapsed()
        if result['passed'] and not result['within_budget']:
            raise TimeoutError('evaluation budget exhausted during final reporting')
        # Record elapsed time through the first complete reporting pass and
        # check the deadline again after the final required writes.
        for path in paths:
            write(path)
        if result['passed'] and time.monotonic() > deadline:
            raise TimeoutError('evaluation budget exhausted during final reporting')
    except Exception as error:
        result['passed'] = False
        result.setdefault('error', f'final publication/reporting failed: {type(error).__name__}: {error}')
        if attempted:
            try:
                if had_previous:
                    backup.replace(destination)
                else:
                    destination.unlink(missing_ok=True)
            except OSError as rollback_error:
                result['calibration_rollback_error'] = str(rollback_error)
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)
        refresh_elapsed()
        for path in report_paths:
            try:
                write(path)
            except OSError as report_error:
                # Never leave a previously written passed:true report after
                # a failed final write. Missing evidence is safely unscoreable.
                path.unlink(missing_ok=True)
                result.setdefault('report_errors', []).append(str(report_error))
    # Retain any unused aggregate rollback file in run/ scratch. Deleting it
    # after commit would introduce another fallible post-publication action;
    # make clean removes this scratch without touching published calibration.


def run(args):
    started = time.monotonic()
    deadline = started + args.budget_s
    label = ('student-' if args.profile == 'student' else '') + ('baseline' if args.baseline else 'candidate')
    folder = SUITE / 'run' / ('evaluation-' + label)
    folder.mkdir(parents=True, exist_ok=True)
    canonical_report = folder / 'report.json'
    report_path = args.report or canonical_report
    report_paths = [canonical_report] + ([report_path] if report_path != canonical_report else [])
    manifest = SUITE / ('baseline_manifest_student.json' if args.profile == 'student' else 'baseline_manifest.json')
    staged_manifest = folder / 'baseline-calibration.json'
    evaluation_id = uuid.uuid4().hex
    env = dict(os.environ, P3_SUITE_DIR=str(SUITE),
               CROSS_ROOT=os.environ.get('CROSS_ROOT', str(SUITE.parents[4])),
               P1_BASELINE_PROFILE=args.profile,
               P1_EVALUATION_ID=evaluation_id,
               P1_REQUIRE_TPU='1')
    env.pop('P3_BASELINE', None)
    if args.baseline:
        env['P3_BASELINE'] = '1'
    result = {'passed': False, 'profile': args.profile, 'label': label,
              'mode': 'baseline' if args.baseline else 'candidate',
              'gate_version': score.STUDENT_GATE_VERSION if args.profile == 'student' else 'legacy',
              'evaluation_id': evaluation_id, 'cleanup_passed': False, 'require_tpu': True,
              'source_files_unchanged': False,
              'archive_sha256': {},
              'budget_s': args.budget_s, 'cases': {}, 'logs': str(folder),
              'versions': {}}
    # Invalidate the prior success before any new work, even with --report.
    # A failed attempt must not leave a scoreable old default report behind.
    for path in report_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        path.write_text(json.dumps({**result, 'error': 'evaluation has not completed'}, indent=2) + '\n')
    folders = {}
    try:
        ranked = score.CONFIG['evaluation_profiles'][args.profile]['case_ids']
        cases = [c for c in score.CONFIG['cases'] if c['id'] in ranked]
        if not ranked or len(ranked) != len(set(ranked)) or {c['id'] for c in cases} != set(ranked):
            raise ValueError('evaluation profile must name a nonempty set of known cases')
        result.update(seed=score.CONFIG['determinism']['seed'],
            runs_per_case=score.CONFIG['determinism']['num_runs'], suite_commit=score.CONFIG['suite']['commit'])
        root = args.suite_root.resolve() if args.suite_root else install_suite.suite_root()
        env['MLINF_ROOT'] = str(root)
        result['versions'] = {name: importlib.metadata.version(name)
                             for name in ('jax', 'jaxlib', 'libtpu', 'numpy', 'torch')}
        result['versions']['python'] = sys.version.split()[0]
        if args.profile == 'student' and not 0 < args.budget_s <= 300:
            raise ValueError('student evaluation budget must be at most 300 seconds')
        result['model_identity'] = model_identity(SUITE.parent.parent)
        result['source_sha256'] = source_hashes(Path(env['CROSS_ROOT']))
        ok, detail = install_suite.harness_is_pristine(root)
        if not ok:
            raise RuntimeError(detail)
        cleanup()
        install_suite.install(root)
        for case in cases:
            env['P3_DEVICES'] = str(case['devices'])
            case_started = time.monotonic()
            with (folder / (case['id'] + '.log')).open('w') as log:
                run_command([sys.executable, 'harness/run_submission.py', str(case['instance']),
                             '--seed', str(score.CONFIG['determinism']['seed']), '--num_runs',
                             str(score.CONFIG['determinism']['num_runs'])],
                            cwd=root, env=env, log=log, deadline=deadline)
                run_command([sys.executable, str(SUITE / 'archive.py'), '--case', case['id'],
                             '--label', label, '--suite-root', str(root)],
                            cwd=root, env=env, log=log, deadline=deadline)
            records, meta, archive_folder = score.read_case(root, case['instance'],
                                               score.CONFIG['determinism']['num_runs'], case['id'], label)
            summary = score.summarize(records, case['batch'], meta)
            if summary['baseline_profile'] != args.profile:
                raise ValueError('archived baseline profile differs from this evaluation')
            passed, reasons = score.gate(summary, score.CONFIG['gates'])
            result['cases'][case['id']] = {'passed': bool(passed), 'reasons': reasons,
                                         'elapsed_s': time.monotonic() - case_started, **summary}
            folders[case['id']] = archive_folder
            result['archive_sha256'][case['id']] = archive_hashes(archive_folder, score.CONFIG['determinism']['num_runs'])
            cleanup()
            if not passed:
                raise RuntimeError(f"{case['id']} failed: {reasons}")
        result['passed'] = True
    except (Exception, SystemExit) as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        with cancellation_shield():
            try:
                cleanup()
                result['cleanup_passed'] = True
            except Exception as error:
                result['passed'] = False
                result['cleanup_error'] = str(error)
            try:
                result['source_sha256_after'] = source_hashes(Path(env['CROSS_ROOT']))
                result['source_files_unchanged'] = result.get('source_sha256') == result['source_sha256_after']
                if not result['source_files_unchanged']:
                    result['passed'] = False
                    result.setdefault('error', 'source files changed during evaluation')
            except Exception as error:
                result['passed'] = False
                result.setdefault('error', f'source verification failed: {error}')
            result['elapsed_s'] = time.monotonic() - started
            result['within_budget'] = result['elapsed_s'] <= args.budget_s
            result['passed'] = result['passed'] and result['within_budget']
            if result['passed'] and args.profile == 'student':
                try:
                    score.validate_student_evaluation(result, label=label, folders=folders)
                except Exception as error:
                    result['passed'] = False
                    result['error'] = f'evaluation evidence failed: {error}'
            snapshot = None
            if args.baseline and result['passed']:
                try:
                    payload = score.calibration_payload(result['cases'], args.profile, result)
                    staged_manifest.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
                    if args.profile == 'student':
                        snapshot = score.calibration_report_path(label, evaluation_id)
                        snapshot.parent.mkdir(parents=True, exist_ok=True)
                        archived = {}
                        for cid, source in folders.items():
                            target = snapshot.parent / 'archives' / cid
                            target.mkdir(parents=True, exist_ok=True)
                            for name in result['archive_sha256'][cid]:
                                shutil.copy2(source / name, target / name)
                            archived[cid] = target
                        score.validate_student_evaluation(result, label=label, folders=archived, mode='baseline')
                except Exception as error:
                    result['passed'] = False
                    result['error'] = f'calibration preparation failed: {error}'
            finalize_reports(result, report_paths, started=started, deadline=deadline,
                staged=staged_manifest if args.baseline and result['passed'] else None,
                destination=manifest, snapshot=snapshot)
    print(json.dumps({'passed': result['passed'], 'elapsed_s': result['elapsed_s'],
                      'report': str(report_path), 'error': result.get('error')}, allow_nan=False))
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('legacy', 'student'), default='student')
    parser.add_argument('--baseline', action='store_true', help='run reference and emit a verified calibration')
    parser.add_argument('--budget-s', type=float, default=300.)
    parser.add_argument('--suite-root', type=Path, default=None)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.budget_s) or args.budget_s <= 0:
        parser.error('--budget-s must be positive and finite')
    if args.profile == 'student' and args.budget_s > 300:
        parser.error('student --budget-s must be at most 300')
    try:
        with cancellation_handlers(), evaluation_lock():
            return run(args)
    except Exception as error:
        print(json.dumps({'passed': False, 'error': f'{type(error).__name__}: {error}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
