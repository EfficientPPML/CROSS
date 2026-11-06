"""Offline hash/source/mock-scope verifier; never opens retained private paths."""
from pathlib import Path
import hashlib,json,tarfile

ROOT=Path(__file__).resolve().parent
PINS={
    'report.json':'5499d1f5d23a7b6ac1ca6f37f7370e43eef12f4ea0703c92213fa97895b57d3e',
    'sources.tar.gz':'3943ad416e8e8f6ad4e4161a605c613d2be2d79d9a11dea5e59960f54504e9d2',
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(directory):
    report=json.loads((directory/'report.json').read_bytes())
    assert report['passed'] is True and report['terminal_scope_completed'] is True
    assert report['returncode']==0 and report['process_reaped'] is True
    assert report['test_counts']==dict(tests=169,failures=0,errors=0,skipped=0)
    assert report['actual_client_decode'] is False and report['private_payloads_read'] is False
    expected=report['source_hashes_before']
    assert len(expected)==report['source_count']==88
    assert expected==report['source_hashes_after']==report['snapshot_hashes']
    assert report['source_files_unchanged'] is True and report['source_archive_verified'] is True
    assert sha(directory/'sources.tar.gz')==report['source_archive_sha256']
    with tarfile.open(directory/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len(members)==88==len({m.name for m in members})
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..'not in Path(m.name).parts for m in members)
        source={m.name:archive.extractfile(m).read() for m in members}
    assert {n:hashlib.sha256(v).hexdigest() for n,v in source.items()}==expected
    origin=json.loads(source['source_origin.json'])
    assert all(expected[n]==v for n,v in origin['copied_source_sha256'].items())
    assert sum(n.startswith('runtime/') for n in expected)==59
    deps=json.loads(source['dependencies.json']);assert deps['ready'] is False
    for stage in (3,4):
        assert hashlib.sha256(source[f'stage{stage}_source_manifest.json']).hexdigest()==deps[f'stage{stage}_source_inventory_sha256']
    assert len(json.loads(source['stage3_source_manifest.json']))==80
    assert len(json.loads(source['stage4_source_manifest.json']))==217
    return source

def main():
    inventory=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__'not in p.parts}
    assert actual==inventory
    assert all(sha(ROOT/n)==v for n,v in PINS.items())
    final=verify(ROOT)
    old=verify(ROOT/'history/source-bound-preflight-v1')
    assert {n for n in final if final[n]!=old[n]}=={'preflight.py','source_origin.json'}
    terminal=json.loads((ROOT/'terminal.json').read_bytes())
    assert terminal['exit_code']==0 and terminal['reaped'] is True
    print(json.dumps(dict(passed=True,source_count=88,test_count=169,actual_client_decode=False,historical_source_preserved=True)))

if __name__=='__main__':main()
