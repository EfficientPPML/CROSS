"""Offline public preflight check; never opens real assets or private fixtures."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    inventory=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    assert {p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}==inventory
    assert sha(ROOT/'report.json')=='a759c4b0ed85d124813bb6e38a97c27f08f5714dd0d0fbb7f5e5fa8cf86d6347'
    r=json.loads((ROOT/'report.json').read_bytes());b=r['source_hashes_before']
    assert all(r[n] is True for n in ('passed','terminal_scope_completed','cleanup_completed','process_reaped','source_files_unchanged','source_archive_verified'))
    assert r['returncode']==0 and r['actual_preparation'] is False
    assert len(b)==r['source_count']==27 and b==r['source_hashes_after']==r['snapshot_hashes']
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='1518e0d0da361611db7800b50661ed205b0cfa6edc6234526250415952b8e7f4'
    with tarfile.open(ROOT/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len(members)==len({m.name for m in members})==27
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==b
    suites=list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
    counts={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','failures','errors','skipped')}
    assert counts==r['test_counts']==dict(tests=32,failures=0,errors=0,skipped=0)
    assert r['read_guard']==json.loads((ROOT/'read_guard.json').read_bytes())==dict(attempted_real_asset_reads=[])
    terminal=json.loads((ROOT/'terminal.json').read_bytes())
    assert terminal['authoritative_session_id']==32066 and terminal['observed_exit_code']==0 and terminal['session_reaped'] is True
    print(json.dumps(dict(passed=True,source_count=27,mock_tests=32,real_asset_read_attempts=0,actual_preparation=False)))
if __name__=='__main__':main()
