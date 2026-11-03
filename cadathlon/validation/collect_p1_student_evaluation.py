"""Collect one fresh, complete P1 student baseline and its exact source snapshot.

Validation-only driver. Running without --prepare-only requires an exclusive
TPU lease. Source copying and the offline self-score are outside the separately
reported evaluator process wall time. No private keys or per-image evidence
are copied. The evaluator remains responsible for its 300-second gate.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_evaluator(repo):
    suite = repo / 'cadathlon/non_agent_pset/p1_benchmark/src/suite'
    sys.path.insert(0, str(suite))
    spec = importlib.util.spec_from_file_location('p1_evidence_evaluator', suite / 'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_path(name, evaluator, repo):
    if name.startswith('problem/'):
        return evaluator.SUITE.parent / name.removeprefix('problem/')
    if name.startswith('CROSS/'):
        return repo / name.removeprefix('CROSS/')
    if name == 'common/support.py':
        return repo / 'cadathlon/common/support.py'
    raise ValueError(f'unknown measured source location: {name}')


def snapshot_sources(output, evaluator, repo):
    hashes = evaluator.source_hashes(repo)
    inventory = {}
    for name, expected in hashes.items():
        source = source_path(name, evaluator, repo)
        # .txt preserves exact bytes while preventing discovery as active tests.
        destination = output / 'sources' / (name + '.txt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination) != expected:
            raise RuntimeError(f'source changed while archiving: {name}')
        inventory[name] = {'sha256': expected, 'snapshot': str(destination.relative_to(output)),
                           'bytes': destination.stat().st_size}
    if evaluator.source_hashes(repo) != hashes:
        raise RuntimeError('measured sources changed during the startup snapshot')
    write_json(output / 'source_inventory.json', inventory)
    return hashes


@contextlib.contextmanager
def forward_signals(process, signals):
    previous = {}

    def forward(signum, frame):
        # The evaluator handles the first cancellation and shields its cleanup.
        # Repeated signals must not interrupt that bounded cleanup.
        if not signals:
            signals.append(signum)
            if process.poll() is None:
                process.send_signal(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def preserve_outputs(output, evaluator, report):
    suite = evaluator.SUITE
    canonical = suite / 'run/evaluation-student-baseline'
    for name in ('report.json', 'C3.log', 'C4.log'):
        source = canonical / name
        if source.is_file():
            shutil.copyfile(source, output / name)
    manifest = suite / 'baseline_manifest_student.json'
    if manifest.is_file():
        shutil.copyfile(manifest, output / 'baseline_manifest_student.json')
    identity = report.get('evaluation_id', '')
    if not isinstance(identity, str) or re.fullmatch('[0-9a-f]{32}', identity) is None:
        return
    calibration = suite / 'calibration/student-baseline' / identity
    if (calibration / 'report.json').is_file():
        destination = output / 'calibration'
        destination.mkdir()
        shutil.copyfile(calibration / 'report.json', destination / 'report.json')
    # Copy only aggregate case/run summaries, never evidence/, keys or arrays.
    for case in ('C3', 'C4'):
        source = calibration / 'archives' / case
        if not source.is_dir():
            source = suite / 'results' / f'{case}_student-baseline'
        target = output / 'archives' / case
        target.mkdir(parents=True)
        for name in ('case.json', 'results-1.json', 'results-2.json', 'results-3.json'):
            if (source / name).is_file():
                shutil.copyfile(source / name, target / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new evidence directory; must not exist')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--suite-root', type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='copy/verify sources without launching any evaluation')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'passed': False, 'scope': 'complete P1 student-v2 encrypted baseline; offline self-score recorded separately',
              'prepared_only': args.prepare_only, 'repo': str(repo), 'wrapper_pid': os.getpid(),
              'wrapper_sha256': digest(Path(__file__)), 'source_files_unchanged': False,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'signals': []}
    try:
        evaluator = load_evaluator(repo)
        shutil.copyfile(Path(__file__), output / 'collector.py.txt')
        hashes = snapshot_sources(output, evaluator, repo)
        result['source_sha256_before'] = hashes
        command = [args.python, str(evaluator.SUITE / 'evaluate.py'),
                   '--profile', 'student', '--baseline', '--budget-s', '300']
        if args.suite_root is not None:
            command += ['--suite-root', str(args.suite_root.resolve())]
        env = dict(os.environ, CROSS_ROOT=str(repo))
        env.pop('JAX_PLATFORMS', None)
        if args.suite_root is not None:
            env['MLINF_ROOT'] = str(args.suite_root.resolve())
        result['command'] = command
        result['environment_overrides'] = {'CROSS_ROOT': str(repo), 'JAX_PLATFORMS': '<unset>'}
        result['inherited_thread_settings'] = {name: env.get(name) for name in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS', 'TPU_VISIBLE_CHIPS')}
        canonical = evaluator.SUITE / 'run/evaluation-student-baseline/report.json'
        previous = {}
        if canonical.is_file():
            shutil.copyfile(canonical, output / 'previous_report.json')
            try:
                previous = json.loads(canonical.read_text())
            except ValueError as error:
                result['previous_report_error'] = str(error)
        if not isinstance(previous, dict):
            previous = {}
        result['previous_evaluation_id'] = previous.get('evaluation_id')
        manifest = evaluator.SUITE / 'baseline_manifest_student.json'
        if manifest.is_file():
            shutil.copyfile(manifest, output / 'previous_baseline_manifest_student.json')
        write_json(output / 'collection.json', result)
        if args.prepare_only:
            result['scope'] = 'source snapshot/preparation only; no TPU evaluation or self-score launched'
            result['source_sha256_after'] = evaluator.source_hashes(repo)
            result['source_files_unchanged'] = result['source_sha256_after'] == hashes
            return 0
        with (output / 'evaluator.log').open('w') as log:
            evaluation_started = time.monotonic()
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            result['evaluator_pid'] = process.pid
            print(json.dumps({'evaluator_pid': process.pid, 'output': str(output)}), flush=True)
            with forward_signals(process, result['signals']):
                try:
                    result['returncode'] = process.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    result['outer_timeout'] = True
                    process.send_signal(signal.SIGTERM)
                    try:
                        result['returncode'] = process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        result['forced_kill'] = True
                        result['returncode'] = process.wait(timeout=10)
            result['external_evaluation_elapsed_s'] = time.monotonic() - evaluation_started
        result['source_sha256_after'] = evaluator.source_hashes(repo)
        result['source_files_unchanged'] = result['source_sha256_after'] == hashes
        result['evaluation_peak_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if canonical.is_file():
            # Keep raw evidence even if a failed attempt left malformed JSON.
            shutil.copyfile(canonical, output / 'report.json')
        report = json.loads(canonical.read_text()) if canonical.is_file() else {}
        result['fresh_evaluation_id'] = bool(report.get('evaluation_id')) and report['evaluation_id'] != previous.get('evaluation_id')
        result['report_sources_match_snapshot'] = report.get('source_sha256') == hashes == report.get('source_sha256_after')
        preserve_outputs(output, evaluator, report)
        score_command = [args.python, str(evaluator.SUITE / 'score.py'),
                         '--evaluation-profile', 'student', '--label', 'student-baseline']
        score_started = time.monotonic()
        scored = subprocess.run(score_command, cwd=repo, env=env, capture_output=True, text=True, timeout=60)
        (output / 'self_score.stdout').write_text(scored.stdout)
        (output / 'self_score.stderr').write_text(scored.stderr)
        result['self_score_command'] = score_command
        result['self_score_returncode'] = scored.returncode
        result['offline_self_score_elapsed_s'] = time.monotonic() - score_started
        self_score = json.loads(scored.stdout)
        write_json(output / 'self_score.json', self_score)
        result['passed'] = (result['returncode'] == 0 and not result['signals']
            and result['external_evaluation_elapsed_s'] <= 300 and report.get('passed') is True
            and result['fresh_evaluation_id'] and result['source_files_unchanged']
            and result['report_sources_match_snapshot'] and scored.returncode == 0
            and self_score.get('passed') is True)
        return 0 if result['passed'] else 1
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        return 1
    finally:
        result['collection_elapsed_s'] = time.monotonic() - started
        write_json(output / 'collection.json', result)
        print(json.dumps({'passed': result['passed'], 'prepared_only': args.prepare_only,
                          'output': str(output), 'error': result.get('error')}), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
