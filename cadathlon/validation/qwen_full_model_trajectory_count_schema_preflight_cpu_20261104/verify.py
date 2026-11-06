from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 inv=json.loads((P/'artifact_hashes.json').read_bytes())
 assert {n:sha(P/n) for n in inv}==inv
 r=json.loads((P/'preflight-v2/report.json').read_bytes());before=r['source_hashes_before']
 assert len(before)==101 and r['passed']is True and r['complete']is True
 for n in ('source_files_unchanged','execution_files_unchanged','cleanup_completed','termination_scope_completed','terminal_scope_completed','zero_audited_body_reads'):assert r[n]is True,n
 assert before==r['source_hashes_after']==r['execution_source_hashes_before']==r['execution_source_hashes_after']
 assert sha(P/'preflight-v2/sources.tar.gz')==r['source_archive_sha256']
 with tarfile.open(P/'preflight-v2/sources.tar.gz')as t:
  ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==101 and all(m.isfile()for m in ms)
  assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in ms}==before
 old=json.loads((P/'runtime_predecessor.json').read_bytes())['source_hashes'];assert len(old)==87 and all(before[n]==h for n,h in old.items())
 suites=ET.parse(P/'preflight-v2/tests.xml').getroot().findall('testsuite')
 assert {n:sum(int(s.get(n,0))for s in suites)for n in ('tests','errors','failures','skipped')}==dict(tests=86,errors=0,failures=0,skipped=0)
 audit=json.loads((P/'preflight-v2/audit.json').read_bytes());assert audit['passed'] and not audit['denied_attempts'] and not audit['backend_modules']
 assert len(r['children'])==1 and all(all(c.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))for c in r['children'])
 print('PASS101 sources /86 prelaunch tests; no learned trajectory or HE claim')
if __name__=='__main__':verify()
