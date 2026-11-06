"""Offline source integrity and recorded test verification for routing repair."""
from pathlib import Path
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    for name, expected in json.loads((HERE / 'artifacts.json').read_text()).items():
        assert sha((HERE / name).read_bytes()) == expected, name
    report = json.loads((HERE / 'report.json').read_text())
    assert report['passed'] and report['tests_exit_code'] == report['baseline_reproduction_exit_code'] == 0
    assert report['source_files_unchanged'] and report['source_archive_verified']
    before = report['source_hashes_before']
    assert before == report['source_hashes_after']
    assert sha((HERE / 'sources.tar.gz').read_bytes()) == report['source_archive_sha256']
    with tarfile.open(HERE / 'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(before) == len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name: sha(archive.extractfile(m).read()) for m in members} == before
    reproduction = json.loads((HERE / 'baseline_reproduction.json').read_text())
    assert reproduction['baseline_sha256'] == '0ce87c1c580e27bf793fc1d23eca63b64f6f14de63e36e1feea77d0481d83e2c'
    assert reproduction['baseline_accepts_boolean_input_nsd']
    assert reproduction['baseline_continues_after_input_callback_mutation']
    assert reproduction['baseline_returns_boolean_output_nsd']
    assert '252 passed' in (HERE / 'tests.log').read_text()
    terminal = json.loads((HERE / 'terminal.json').read_text())
    assert terminal['exit_code'] == 0 and terminal['session_reaped']
    return dict(passed=True, source_files=len(before), tests=252, encrypted_rerun=False, main_changed=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
