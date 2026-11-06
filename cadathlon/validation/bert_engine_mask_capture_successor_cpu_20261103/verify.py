"""Offline source/negative/positive verification without model imports."""
from pathlib import Path
import hashlib
import json
import tarfile

P = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
for name, h in json.loads((P / 'artifact_manifest.json').read_bytes()).items():
    assert sha((P / name).read_bytes()) == h, name
for name, code, tests in (('negative-v1', 1, dict(tests=8, failures=8, errors=0, skipped=0)),
                          ('source-bound-v1', 0, dict(tests=37, failures=0, errors=0, skipped=0))):
    r = json.loads((P / name / 'report.json').read_bytes())
    assert r['test_exit_code'] == code and r['tests'] == tests
    assert r['child_reaped'] is True and r['sources_unchanged'] is True
    assert r['original_frozen_sources_unchanged'] is True and r['archive_verified'] is True
    before = r['source_hashes_before']
    assert len(before) == 9 and before == r['source_hashes_after']
    archive = P / name / 'sources.tar.gz'
    assert sha(archive.read_bytes()) == r['source_archive_sha256']
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == 9 == len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name: sha(tar.extractfile(m).read()) for m in members} == before
    assert before['before/original_full_model_engine.py'] == '841fb5a5f635a10a5e2f3f84b04361816deda11ab1869f0472e1acc00507f009'
    if code == 0:
        assert r['passed'] is True
        assert sha((P / 'full_model_engine.py').read_bytes()) == before['candidate/full_model_engine.py']
        assert sha((P / 'test_mask_capture.py').read_bytes()) == before['candidate/test_mask_capture.py']
    else:
        assert r['passed'] is False and r['negative_reproduced'] is True
assert sha((P / 'full_model_engine.py').read_bytes()) == 'a5384484269139f67db18ea90b3a99f2952265321d84bc04964433d9dede3f53'
print('PASS: 37 scoped tests, 8 preserved negatives, 9 exact sources per run.')
