"""Check P2 staging from the actual release archive without initializing JAX."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
STAGES = {'client_key_generation', 'server_preprocess_model', 'client_encode_encrypt_input',
          'server_encrypted_compute', 'client_decrypt_decode'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Use a new report path')
    archive = ROOT / 'cadathlon/contestant-release.tar.gz'
    report = {'scope': 'Extracted release dependency/staging/preflight check; no encrypted model execution.',
              'passed': False, 'release_sha256': digest(archive),
              'checker_sha256': digest(Path(__file__)), 'commands': []}
    env = dict(os.environ, CROSS_ROOT=str(ROOT), JAX_PLATFORMS='preflight_must_not_initialize_jax',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    report['environment'] = {key: env[key] for key in
                             ('CROSS_ROOT', 'JAX_PLATFORMS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='cross-released-bert-stage-') as temporary:
            work = Path(temporary)
            released, staged = work / 'release', work / 'staged'
            released.mkdir()
            with tarfile.open(archive) as stream:
                stream.extractall(released, filter='data')
            src = released / 'agent_pset/p2_bert/src'
            command = [sys.executable, str(src / 'stage_benchmark.py'), '--upstream',
                       str(args.upstream.resolve()), '--destination', str(staged)]
            run = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True, check=False)
            report['commands'].append(dict(command=command, returncode=run.returncode,
                                           stdout=run.stdout, stderr=run.stderr))
            if run.returncode:
                raise RuntimeError('Released BERT staging failed')
            manifest = json.loads((staged / 'cross_staging_manifest.json').read_text())
            for name, expected in manifest['common_sha256'].items():
                assert digest(staged / 'submissions/_cross_common' / name) == expected
                assert digest(released / 'common' / name) == expected
            for name, expected in manifest['harness_sha256'].items():
                assert digest(staged / name) == expected
                assert digest(args.upstream / name) == expected
            report['staging_manifest'] = manifest
            for adapter in ('bert_reference.py', 'bert_packed_reference.py', 'bert_bounded_reference.py'):
                command = [sys.executable, str(staged / 'submissions/runner.py'), '--preflight',
                           '--implementation', str(staged / 'submissions' / adapter)]
                run = subprocess.run(command, cwd=staged, env=env, text=True, capture_output=True, check=False)
                report['commands'].append(dict(command=command, returncode=run.returncode,
                                               stdout=run.stdout, stderr=run.stderr))
                result = json.loads(run.stdout)
                assert run.returncode == 1, 'Toy adapter must still report missing full-model stages'
                assert result['implementation_status'] == 'experimental-unvalidated'
                assert set(result['missing_interfaces']) == STAGES
                assert result['ready_for_execution'] is False and result['encryption_validated'] is False
            report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['elapsed_s'] = time.perf_counter() - started
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report.get(key) for key in ('passed', 'elapsed_s', 'error')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
