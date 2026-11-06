"""Offline source/proof verification, without importing model or runtime code."""
from pathlib import Path
import hashlib,json,tarfile


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    artifacts=json.loads((root/'artifact_hashes.json').read_text())
    actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
    assert actual==artifacts
    report=json.loads((root/'source-bound-v2/report.json').read_text())
    assert all(report[k] is True for k in ('passed','sources_unchanged','archive_verified','original_frozen_sources_unchanged','child_reaped'))
    assert report['tests']==dict(tests=25,failures=0,errors=0,skipped=0) and report['test_exit_code']==0
    before=report['source_hashes_before'];assert before==report['source_hashes_after'] and len(before)==13
    archive=root/'source-bound-v2/sources.tar.gz'
    assert digest(archive)==report['source_archive_sha256']=='00759e282efcc3f6518071454638c47243a79b85769f82c9c9a6bd9d96ae1df0'
    with tarfile.open(archive) as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==13
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==before
    for n,h in before.items():assert digest(root/n)==h
    applicability=json.loads((root/'applicability.json').read_text())
    assert applicability['passed'] and applicability['three_result_files_equal'] and not applicability['original_files_modified']
    assert digest(root/'candidate-v2.patch')==applicability['patch_sha256']=='26e3a9e0283387405243e8e2942ebc07d153a269b80399ae538f0526107fd1b1'
    assert digest(root/'candidate-v2/full_model_engine.py')==applicability['final_runtime_sha256']=='2c68a0d7125bbdc314845553162cc0f9b07ea7fec283481d4b143f2130ddf129'
    negatives=json.loads((root/'negative_source_binding.json').read_text())
    for name,path in (('original_negative_engine_sha256','before/full_model_engine.py'),('original_negative_test_sha256','before/test_cleanup_regressions.py'),('signal_negative_engine_sha256','signal-install-before.py'),('signal_negative_test_sha256','candidate-v2/test_signal_installation.py')):
        assert digest(root/path)==negatives[name]
    terminal=json.loads((root/'terminal.json').read_text())
    assert terminal['final_command_exit_code']==0 and terminal['final_command_reaped']
    assert terminal['original_negative']==dict(exit_code=1,reaped=True,failed=5,passed=4)
    assert terminal['signal_install_negative']==dict(exit_code=1,reaped=True,failed=2,passed=2)
    assert '5 failed, 4 passed' in (root/'negative-v1.log').read_text()
    assert '2 failed, 2 passed' in (root/'signal-install-negative-v1.log').read_text()
    return dict(passed=True,sources=13,tests=25,scope='Offline mocked cleanup evidence only; no HE/model or live adoption.')


if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),sort_keys=True))
