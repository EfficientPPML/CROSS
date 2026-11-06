"""Offline preflight verification; never read external fresh inputs."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
for name,h in json.loads((ROOT/'artifact_manifest.json').read_bytes()).items():
    assert sha((ROOT/name).read_bytes())==h,name
for name,count in (('development-v1',None),('development-v2',132),('preflight-v3',135)):
    folder=ROOT/name
    r=json.loads((folder/('failure.json' if count is None else 'report.json')).read_bytes())
    before=r['archived_source_hashes'] if count is None else r['source_hashes_before']
    assert len(before)==74
    archive=folder/'sources.tar.gz'
    assert sha(archive.read_bytes())==r['archive_sha256' if count is None else 'source_archive_sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert len(members)==74==len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name:sha(tar.extractfile(m).read()) for m in members}==before
    assert not any(n.startswith('validation_inputs/') for n in before)
    assert 'fresh_freeze/sources.tar.gz' not in before
    assert r['fresh_input_bodies_read'] is False and r['checkpoint_model_evaluated'] is False
    if count is None:
        assert r['observed_exit_code']==1 and r['session_reaped'] is True and r['tests_started'] is False
    else:
        assert r['passed'] is True and r['test_exit_code']==0
        assert r['source_files_unchanged'] is True and r['source_archive_verified'] is True
        assert r['source_hashes_after']==before
        assert r['tests']==dict(tests=count,errors=0,failures=0,skipped=0)
        assert r['read_guard']['forbidden_reads_blocked']==[]
        assert r['calibration_prerequisite']['calibration_contexts']==416
    if name=='preflight-v3':
        for f in ('validate_heldout.py','calibration_gate.py','validation_gate.py','fresh_gate.py',
                  'validation_cleanup.py','validation_terminal.py','validation_protocol.json','validation_requirement.json'):
            assert sha((ROOT/f).read_bytes())==before[f]
        req=json.loads((ROOT/'validation_requirement.json').read_bytes())
        assert len(req['expected_runtime_source_hashes'])==55
        assert all(before[n]==h for n,h in req['expected_runtime_source_hashes'].items())
        assert req['candidate_manifest_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
        assert req['validation_input_sha256']=='b20f28fc66260d8bd0ec704d6138da9a60ba513a53111d789cfd10f431dd80af'
print('PASS: final116 tests+19subtests,74 exact sources; fresh bodies remain external/unread.')
