from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_sources(path):
    with tarfile.open(path) as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms}) and all(m.isfile() for m in ms)
        return {m.name:t.extractfile(m).read() for m in ms}
def main():
    inventory=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    assert {p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__'not in p.parts}==inventory
    r=json.loads((ROOT/'report.json').read_bytes());p=json.loads((ROOT/'installed-probe-v1.json').read_bytes())
    assert r['passed'] is True and r['terminal_scope_completed'] is True and r['returncode']==0
    assert r['test_counts']==dict(tests=169,failures=0,errors=0,skipped=0)
    assert r['source_count']==89 and r['source_files_unchanged'] is True and r['source_archive_verified'] is True
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='e4b87b2bf1aef11712ebeeaf880264036c85aee504d2be15c311b5348ba80567'
    source=read_sources(ROOT/'sources.tar.gz');hashes={n:hashlib.sha256(v).hexdigest() for n,v in source.items()}
    assert len(source)==89 and hashes==r['source_hashes_before']==r['source_hashes_after']==r['snapshot_hashes']
    assert p['passed'] is True and p['denial_count']==p['restored_count']==21 and len(set(p['denied']))==21
    assert p['source_hashes_before']==p['source_hashes_after']==hashes and p['actual_client_decode'] is False
    assert p['native_model_modules_loaded'] is False and len(p['loaded_local_modules'])==50
    assert sha(ROOT/'parent_sources.tar.gz')=='3943ad416e8e8f6ad4e4161a605c613d2be2d79d9a11dea5e59960f54504e9d2'
    parent=read_sources(ROOT/'parent_sources.tar.gz');assert len(parent)==88
    assert {n for n in parent if parent[n]!=source[n]}=={'source_origin.json','dependencies.json'}
    assert set(source)-set(parent)=={'installed_guard_probe.py'}
    assert sum(n.startswith('runtime/') for n in source)==59
    deps=json.loads(source['dependencies.json']);assert deps['ready'] is True
    for stage in (3,4):assert hashlib.sha256(source[f'stage{stage}_source_manifest.json']).hexdigest()==deps[f'stage{stage}_source_inventory_sha256']
    terminal=json.loads((ROOT/'terminal.json').read_bytes())
    assert terminal['preflight_exit_code']==terminal['installed_probe_exit_code']==0 and terminal['preflight_reaped'] is True
    print(json.dumps(dict(passed=True,source_count=89,mock_tests=169,installed_denials=21,actual_client_decode=False)))
if __name__=='__main__':main()
