from pathlib import Path
import json,hashlib,tarfile
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 inv=json.loads((P/'artifact_hashes.json').read_bytes());assert {n:sha(P/n)for n in inv}==inv
 r=json.loads((P/'report.json').read_bytes());expected=r['source_hashes_before']
 assert r['passed']is False and r['complete']is True and len(expected)==98
 assert expected==r['source_hashes_after']==r['execution_source_hashes_before']==r['execution_source_hashes_after']
 assert all(r[n]is True for n in ('cleanup_completed','source_files_unchanged','execution_files_unchanged','terminal_scope_completed','termination_scope_completed'))
 assert sha(P/'sources.tar.gz')==r['source_archive_sha256']
 with tarfile.open(P/'sources.tar.gz')as t:
  ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==98 and all(m.isfile()for m in ms)
  assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in ms}==expected
 for name in ('S1','S8'):
  w=json.loads((P/name/'report.json').read_bytes());assert w['passed']is False and w['complete']is False and w['cleanup_completed']is True
  assert w['source_hashes_before']==w['source_hashes_after']==expected
  assert w['failure_detail']['message']=='completed original calibration prerequisite failed'
  assert 'asset_sha256_before'not in w and 'embedding_sha256'not in w and 'owner_observations'not in w
 assert all(c['direct_returncode']==1 and all(c[n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))for c in r['children'])
 print('PASS preserved98-source prerequisite failure; no numerical gates')
if __name__=='__main__':verify()
