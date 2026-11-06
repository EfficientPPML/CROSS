"""Offline assembly verification; no runtime/backend/model imports."""
from pathlib import Path
import hashlib,json,tarfile


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def members(path):
    with tarfile.open(path) as t:
        rows=t.getmembers();assert len(rows)==len({m.name for m in rows})
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in rows)
        return {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in rows}


def verify(root):
    expected=json.loads((root/'artifact_hashes.json').read_text())
    actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
    assert actual==expected
    r=json.loads((root/'source-bound-v1/report.json').read_text())
    assert r['passed'] and r['session_reaped'] and r['authoritative_exit_code']==0 and r['exec_session']==67332
    assert r['tests']==dict(tests=10,failures=0,errors=0,skipped=0)
    archive=root/'source-bound-v1/sources.tar.gz'
    assert digest(archive)==r['source_archive_sha256']=='d240ac039e830710a51ddc809c3c60b31198e3ee5ebee696ffd392888e35c04d'
    content=members(archive);assert len(content)==198 and content==r['source_hashes_before']==r['source_hashes_after']
    assembly=json.loads((root/'assembly_report.json').read_text());assert assembly['passed']
    assert assembly['source_file_count']==196 and assembly['shared_core_file_count']==30 and assembly['joined_ideal_proof_complete']
    assert not assembly['runtime_modules_imported'] and not assembly['model_assets_keys_ciphertexts_read'] and not assembly['actual_server_compute_started']
    assert {n:h for n,h in content.items() if n.startswith(('runtime/','joined/'))}==assembly['source_hashes']
    stage=members(root/'origins/stage3_sources.tar.gz');joined=members(root/'origins/joined_sources.tar.gz')
    assert len(stage)==76 and len(joined)==137
    for n,h in stage.items():
        if n.startswith('runtime/'):
            assert content[n]==(assembly['sole_runtime_overlay']['after'] if n=='runtime/stage/full_model_engine.py' else h)
    assert {n[len('joined/'):]:h for n,h in content.items() if n.startswith('joined/')}==joined
    assert digest(root/'origins/stage3_sources.tar.gz')==assembly['stage3_source_archive_sha256']
    assert digest(root/'origins/joined_sources.tar.gz')==assembly['joined_source_archive_sha256']
    assert digest(root/'origins/stage3_preflight_report.json')==assembly['stage3_report_sha256']
    assert digest(root/'origins/joined_ideal_report.json')==assembly['joined_report_sha256']
    assert digest(root/'origins/engine_before.py')==assembly['sole_runtime_overlay']['before']
    resolution=json.loads((root/'import_resolution.json').read_text());assert resolution['passed']
    assert set(resolution['equal_basename_collisions'])=={'server_public_model','support'}
    for name,row in resolution['module_resolution'].items():assert content[row['path']]==row['sha256']
    for name,path in resolution['critical_expected'].items():assert resolution['module_resolution'][name]['path']==path
    return dict(passed=True,assembly_files=196,source_archive_files=198,tests=10,scope='Source/static resolution only; no actual server computation.')


if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),sort_keys=True))
