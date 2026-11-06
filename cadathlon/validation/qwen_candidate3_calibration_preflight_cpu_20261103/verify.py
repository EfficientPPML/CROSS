"""Verify the saved public preflight without importing model or HE code."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    expected = json.loads((ROOT / 'artifact_manifest.json').read_bytes())
    for name, value in expected.items():
        assert digest((ROOT / name).read_bytes()) == value, name
    report = json.loads((ROOT / 'preflight-v1/report.json').read_bytes())
    before = report['source_hashes_before']
    assert report['passed'] is True and report['test_exit_code'] == 0
    assert report['checkpoint_model_evaluated'] is False
    assert report['validation_outputs_read'] is False
    assert report['source_files_unchanged'] is True
    assert len(before) == 55 and before == report['source_hashes_after']
    archive = ROOT / 'preflight-v1/sources.tar.gz'
    assert digest(archive.read_bytes()) == report['source_archive_sha256']
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == len(before)
        assert len({m.name for m in members}) == len(members)
        assert all(m.isfile() for m in members)
        actual = {m.name: digest(tar.extractfile(m).read()) for m in members}
    assert actual == before
    assert before['candidate-v3/manifest.json'] == (
        '7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
    )
    for name in ('candidate_gate.py', 'check_approximate.py', 'cleanup.py',
                 'collector_requirement.json', 'preflight_collector.py'):
        assert digest((ROOT / name).read_bytes()) == before[name]
    assert '68 passed, 19 subtests passed' in (ROOT / 'preflight-v1/tests.log').read_text()
    print('PASS: 55 exact sources; 68 tests plus 19 subtests; no checkpoint-model claim.')


if __name__ == '__main__':
    main()
