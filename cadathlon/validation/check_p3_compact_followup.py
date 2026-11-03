"""Preserve a fresh compact-v1 TPU evaluation and its exact runtime sources.

Run only while this process owns the TPU. The evaluator enforces its 300-second
budget; this wrapper separately times the whole evaluator process and records
its source archive, raw output, report and baseline self-score.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'cadathlon/non_agent_pset/p3_cnn/src'


def manifest():
    paths = {Path(__file__).resolve(), ROOT / 'cadathlon/common/support.py'}
    for folder in (ROOT / 'jaxite_word', ROOT / 'demos', SRC):
        paths.update(folder.rglob('*.py'))
    paths.update(SRC.parent / 'test' / name for name in
                 ('student1.in', 'student1.out', 'student2.in', 'student2.out'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wait_for_evaluator(command, *, env, output):
    process = None
    cancelled = []
    def cancel(signum, frame):
        if not cancelled:
            cancelled.append(signum)
            if process is not None and process.poll() is None:
                process.terminate()
    previous = {sig: signal.signal(sig, cancel) for sig in
                (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    record = {'outer_timeout': False, 'forced_kill': False}
    started = time.monotonic()
    try:
        with (output / 'stdout.json').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            record['pid'] = process.pid
            if cancelled:
                process.terminate()
            try:
                process.wait(timeout=360)
            except subprocess.TimeoutExpired:
                record['outer_timeout'] = True
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    record['forced_kill'] = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record['returncode'] = process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                record['forced_kill'] = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record['cancelled_signals'] = cancelled
        record['evaluator_process_elapsed_s'] = time.monotonic() - started
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write_json(output / 'evaluator-process.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new evidence directory')
    parser.add_argument('--prepare-only', action='store_true', help='archive sources without starting the TPU')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start_hashes = manifest()
    write_json(output / 'source-start.json', start_hashes)
    archive_path = output / 'sources.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name in start_hashes:
            archive.add(ROOT / name, arcname=name)
    # Reject a source change during snapshot construction before starting TPU work.
    if manifest() != start_hashes:
        raise RuntimeError('sources changed while archiving the runtime')
    env = dict(os.environ, CROSS_ROOT=str(ROOT), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env.pop('JAX_PLATFORMS', None)
    command = [sys.executable, str(SRC / 'evaluate.py'), '--baseline',
               '--budget-s', '300', '--output', str(output / 'evaluation')]
    evidence = {'scope': 'fresh compact-v1 full TPU baseline evaluation; no trained accuracy claim',
                'passed': False, 'command': command, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'environment': {key: env.get(key) for key in
                                ('CROSS_ROOT', 'JAX_PLATFORMS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                'source_sha256_start': start_hashes,
                'source_archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    write_json(output / 'process.json', evidence)
    if args.prepare_only:
        evidence['prepared_only'] = True
        write_json(output / 'process.json', evidence)
        print(json.dumps({'prepared_only': True, 'encrypted_evaluation_passed': False,
                          'source_files': len(start_hashes)}))
        return 0
    evidence.update(wait_for_evaluator(command, env=env, output=output))
    try:
        evidence['result'] = json.loads((output / 'evaluation/report.json').read_text())
        report_path = str(output / 'evaluation/report.json')
        if evidence['returncode'] == 0 and not evidence['cancelled_signals']:
            with (output / 'self-score.json').open('w') as stdout, (output / 'self-score.log').open('w') as stderr:
                scored = subprocess.run([sys.executable, str(SRC / 'score_compact.py'),
                                         '--result', report_path, '--baseline', report_path],
                                        cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            evidence['self_score_returncode'] = scored.returncode
            evidence['self_score'] = json.loads((output / 'self-score.json').read_text())
    except (OSError, ValueError) as error:
        evidence['error'] = str(error)
    end_hashes = manifest()
    write_json(output / 'source-end.json', end_hashes)
    evidence['source_sha256_end'] = end_hashes
    evidence['source_files_unchanged'] = start_hashes == end_hashes
    evidence['whole_process_within_300s'] = evidence['evaluator_process_elapsed_s'] <= 300
    evidence['passed'] = (evidence['returncode'] == 0 and evidence.get('result', {}).get('passed') is True
                          and not evidence['cancelled_signals'] and not evidence['outer_timeout']
                          and evidence['whole_process_within_300s']
                          and evidence.get('self_score_returncode') == 0 and evidence['source_files_unchanged'])
    write_json(output / 'process.json', evidence)
    print(json.dumps({key: evidence.get(key) for key in
                      ('passed', 'returncode', 'evaluator_process_elapsed_s', 'source_files_unchanged', 'error')}))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
