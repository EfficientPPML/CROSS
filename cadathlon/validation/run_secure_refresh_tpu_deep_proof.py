"""Coordinated TPU validation wrapper; run only after the root grants its lease."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/home/jianming_gatech/CROSS')
PREFIX = Path('/tmp/secure-refresh-tpu-deep-v1')
REPORT = PREFIX.with_suffix('.json')
LOG = PREFIX.with_suffix('.log')
EVIDENCE = PREFIX.with_suffix('.evidence.json')
PYTHON = '/home/jianming_gatech/miniconda3/envs/jaxite/bin/python'
SOURCES = sorted([*ROOT.joinpath('jaxite_word').glob('*.py'), *(
    ROOT/'cadathlon/common'/name for name in (
        'encrypted_refresh.py', 'encrypted_polynomial.py', 'encrypted_compilation.py',
        'encrypted_primitives.py', 'encrypted_rotation.py', 'support.py')),
    ROOT/'cadathlon/validation/check_secure_refresh.py'])


def hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in SOURCES}


def main():
    if any(path.exists() for path in (REPORT, LOG, EVIDENCE)):
        raise RuntimeError('Proof paths already exist; choose a new run identifier.')
    command = ['taskset', '-c', '172-175', PYTHON,
        'cadathlon/validation/check_secure_refresh.py', '--degree', '131072',
        '--depth', '49', '--dnum', '4', '--period', '128', '--budget', '4',
        '--scale-bits', '56', '--compiled', '--amplitude', '14.2',
        '--public-bound', '16', '--meta-precision', '10', '--disk-keys', '--report', str(REPORT)]
    environment = dict(os.environ, JAX_PLATFORMS='tpu', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4')
    startup = hashes()
    evidence = dict(command=command, source_hashes_at_start=startup,
                    profile_scope='Planned packed Qwen2 deep refresh ring and period128; not complete-model validation.',
                    started_unix_seconds=time.time(), python=sys.version,
                    wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    EVIDENCE.write_text(json.dumps(evidence, indent=2)+'\n')
    start = time.perf_counter()
    maximum_rss_kib = 0
    with LOG.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT)
        evidence['pid'] = process.pid
        EVIDENCE.write_text(json.dumps(evidence, indent=2)+'\n')
        print(f'PID {process.pid}; report {REPORT}', flush=True)
        while process.poll() is None:
            try:
                for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                    if line.startswith(('VmRSS:', 'VmHWM:')):
                        maximum_rss_kib = max(maximum_rss_kib, int(line.split()[1]))
            except FileNotFoundError:
                pass
            time.sleep(1)
        return_code = process.wait()
    ending = hashes()
    evidence.update(exit_code=return_code, process_reaped=True,
        wall_seconds=time.perf_counter()-start, max_observed_process_rss_kib=maximum_rss_kib,
        memory_scope='Sampled Linux process VmRSS/VmHWM, not TPU HBM peak.',
        source_hashes_at_end=ending, source_hashes_unchanged=ending == startup,
        report_present=REPORT.exists())
    numerical_ok = False
    if REPORT.exists():
        report = json.loads(REPORT.read_text())
        evidence['numerical_report_passed'] = report.get('passed', False)
        numerical_ok = (report.get('passed') is True and report.get('backend') == 'tpu'
                        and report.get('actual_output_level', -1) >= 25
                        and report.get('server_has_secret_key') is False
                        and report.get('runtime_key_generation_forbidden') is True)
    evidence['passed'] = return_code == 0 and ending == startup and numerical_ok
    EVIDENCE.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({key: value for key, value in evidence.items()
                      if not key.startswith('source_hashes')}), flush=True)
    return return_code if return_code else (0 if evidence['passed'] else 2)


if __name__ == '__main__':
    raise SystemExit(main())
