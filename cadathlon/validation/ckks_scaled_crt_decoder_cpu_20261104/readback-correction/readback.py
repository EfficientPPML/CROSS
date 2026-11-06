"""Public source capture/readback around the externally owned CPU test handle."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104')
EXTRA = ['before/ckks_ctx.py', 'source_origin.json', 'public_geometry.json',
         'candidate.patch', 'test_real_decode.py', 'run_cpu.py', 'DESIGN.md', 'proof.py']

def digest(data):
    return hashlib.sha256(data).hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')

def sources():
    original = json.loads((ROOT / 'source_origin.json').read_bytes())['source_hashes']
    return sorted(['checkout/' + name for name in original] + EXTRA)

def readback(root, expected):
    got = {name: digest((root / name).read_bytes()) for name in expected}
    if got != expected:
        raise RuntimeError('source readback mismatch')
    return got

def prepare(output):
    output.mkdir(parents=False, exist_ok=False)
    expected = {name: digest((ROOT / name).read_bytes()) for name in sources()}
    dump(output / 'sources-before.json', expected)
    execution = output / 'execution'
    execution.mkdir()
    for name in expected:
        target = execution / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    readback(execution, expected)
    with tarfile.open(output / 'sources.tar.gz', 'w:gz') as archive:
        for name in expected:
            archive.add(execution / name, arcname=name)
    dump(output / 'prepared.json', {'source_count': len(expected),
         'source_manifest_sha256': digest((output / 'sources-before.json').read_bytes()),
         'source_archive_sha256': digest((output / 'sources.tar.gz').read_bytes()),
         'execution_root': str(execution), 'scope': 'clear CPU arithmetic only'})
    print(json.dumps(json.loads((output / 'prepared.json').read_bytes()), sort_keys=True))

def absent(identifier, *, group=False):
    try:
        (os.killpg if group else os.kill)(identifier, 0)
    except ProcessLookupError:
        return True
    return False

def finalize(output, session, exit_code, reap_tool):
    assert type(session) is int and session > 0 and exit_code == 0 and reap_tool
    expected = json.loads((output / 'sources-before.json').read_bytes())
    after = readback(ROOT, expected)
    readback(output / 'execution', expected)
    with tarfile.open(output / 'sources.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(expected) and all(m.isfile() for m in members)
        observed = {m.name: digest(archive.extractfile(m).read()) for m in members}
        assert observed == expected
    worker = json.loads((output / 'worker/report.json').read_bytes())
    assert worker['passed'] is True and worker['pytest_exit_code'] == 0
    assert worker['jax_backend'] == 'cpu' and worker['device_platforms'] == ['cpu']
    assert worker['device_access_denials'] == []
    assert all(x == 'libtpu' for x in worker['import_guard_denials'])
    assert worker['real_crt'] is True and worker['real_fft'] is True
    for name in ('ciphertext_decryption', 'model_forward', 'key_generation', 'encrypted_failure_recovered'):
        assert worker[name] is False
    modules = worker['loaded_local_modules']
    assert modules and all(row['sha256'] == expected[row['path']] for row in modules.values())
    assert modules['ckks_ctx']['path'] == 'checkout/jaxite_word/ckks_ctx.py'
    cases = ET.parse(output / 'worker/tests.xml').getroot().findall('.//testcase')
    assert len(cases) == 35
    assert all(case.find('failure') is None and case.find('error') is None and case.find('skipped') is None for case in cases)
    assert absent(worker['pid']) and absent(worker['pgid'], group=True)
    original = json.loads((ROOT / 'source_origin.json').read_bytes())
    original_root = Path('/tmp/cross-optimized-twelve-20261103/checkout')
    readback(original_root, original['source_hashes'])
    dump(output / 'sources-after.json', after)
    result = {'passed': True, 'source_count': len(expected), 'tests': len(cases),
              'source_manifest_sha256': digest((output / 'sources-before.json').read_bytes()),
              'source_archive_sha256': digest((output / 'sources.tar.gz').read_bytes()),
              'worker_report_sha256': digest((output / 'worker/report.json').read_bytes()),
              'session': session, 'owner_reported_exit_code': exit_code,
              'owner_reported_reaped': True, 'reap_tool_chunk': reap_tool,
              'observed_pid': worker['pid'], 'observed_pgid': worker['pgid'],
              'pid_absent': True, 'group_absent': True,
              'frozen199_current_unchanged': True, 'frozen199_root': str(original_root), 'current_execution_archive_equal': True,
              'scope': 'clear synthetic CPU CRT/FFT arithmetic; no HE/model result',
              'actual_toy5_throw_site_proved': False, 'actual_toy5_recovered': False}
    dump(output / 'report.json', result)
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'finalize'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--session', type=int)
    parser.add_argument('--exit-code', type=int)
    parser.add_argument('--reap-tool')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
    else:
        finalize(args.output, args.session, args.exit_code, args.reap_tool)
