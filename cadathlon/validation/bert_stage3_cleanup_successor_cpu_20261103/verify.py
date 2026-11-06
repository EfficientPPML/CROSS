"""Offline exact-byte verification of positive and preserved negative runs."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()

for name, expected in json.loads((ROOT / 'artifact_manifest.json').read_bytes()).items():
    assert sha((ROOT / name).read_bytes()) == expected, name
for name, count, code in (('negative-v1', 3, 2), ('negative-v2', 3, 1), ('source-bound-v1', 5, 0)):
    report = json.loads((ROOT / name / 'report.json').read_bytes())
    before = report['source_hashes_before']
    assert report['source_files_unchanged'] is True
    assert len(before) == count and before == report['source_hashes_after']
    archive = ROOT / name / 'sources.tar.gz'
    assert sha(archive.read_bytes()) == report['source_archive_sha256']
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == count == len({m.name for m in members})
        assert all(m.isfile() for m in members)
        actual = {m.name: sha(tar.extractfile(m).read()) for m in members}
    assert actual == before
    assert report.get('exit_code', report.get('test_exit_code')) == code
    if name == 'source-bound-v1':
        assert report['passed'] is True and report['source_archive_verified'] is True
        assert report['tests'] == dict(tests=30, failures=0, errors=0, skipped=0)
        assert sha((ROOT / 'cleanup.py').read_bytes()) == before['candidate/cleanup.py']
        assert sha((ROOT / 'test_cleanup.py').read_bytes()) == before['test_cleanup.py']
        assert sha((ROOT / 'signal_probe.py').read_bytes()) == before['signal_probe.py']
assert sha((ROOT / 'cleanup.py').read_bytes()) == '1798455cdd88d4bbac5e1769e6b3226676ccb9901e7be6faf383e7ad063c1ef6'
assert '5 failed' in (ROOT / 'negative-v2/tests.log').read_text()
print('PASS: 30 scoped tests; five final sources; both historical negative runs retained.')
