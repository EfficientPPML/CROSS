from pathlib import Path
import ast,hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(n):return json.loads((P/n).read_bytes())
def archive(n,pins):
 with tarfile.open(P/n)as z:
  m=z.getmembers();assert len(m)==len({x.name for x in m})==len(pins)and all(x.isfile()for x in m)
  raw={x.name:z.extractfile(x).read()for x in m}
 assert {n:hashlib.sha256(b).hexdigest()for n,b in raw.items()}==pins
 return raw
r=load('preflight-v1/report.json');s=load('startup.json');t=load('terminal_readback.json');pins=s['source_hashes_before']
assert len(pins)==260 and pins==r['source_hashes_start']==r['source_hashes_end']==r['execution_hashes_end']
assert all(r[k]is True for k in ('passed','complete','cleanup_completed','source_verified','terminal_scope_completed','termination_scope_completed','all_tests_executed_without_skip'))
assert r['execute']is False and r['test_cases']==r['expected_test_cases']==371
assert r['preflight_audit']['backend_or_private_attempts']==[]and r['preflight_audit']['zero_forbidden_attempts']is True
assert r['fresh_source_guard']['passed']is True and r['fresh_source_guard']['fresh']is True and r['fresh_source_guard']['source_count']==260 and r['fresh_source_guard']['manifest_sha256']==s['external_manifest_sha256']
assert t['session']==51618 and t['exit_code']==0 and t['reaped']is True and t['actual_execution_authorized']is False
assert t['report_sha256']==sha(P/'preflight-v1/report.json')and t['source_archive_sha256']==sha(P/'preflight-v1/sources.tar.gz')==r['source_archive_sha256']
assert t['all_current_startup_execution_before_after_archive_match']is True
c=r['test_child'];assert c['direct_returncode']==0 and all(c[k]is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
x=ET.parse(P/'preflight-v1/tests.xml');assert len(x.findall('.//testcase'))==371 and not any(x.findall('.//'+k)for k in ('failure','error','skipped'))
raw=archive('preflight-v1/sources.tar.gz',pins);assert json.loads(raw['execution_authorization.json'])=={'ready':False}
old=load('preserved253/report.json');assert old['passed']is True and old['test_cases']==276
before=archive('preserved253/sources.tar.gz',old['source_hashes_start']);runtime=[n for n in before if n.startswith('runtime/')];assert len(runtime)==199 and all(raw[n]==before[n]for n in runtime)
a,b=ast.parse(raw['component_arithmetic.py']),ast.parse(before['component_arithmetic.py']);d=lambda n:ast.dump(n,include_attributes=False)
for n in b.body:
 if isinstance(n,(ast.ClassDef,ast.FunctionDef)):
  q=next(v for v in a.body if type(v)is type(n)and v.name==n.name)
  if n.name!='exercise':assert d(n)==d(q)
  else:
   t1=next(v for v in n.body if isinstance(v,ast.Try));t2=next(v for v in q.body if isinstance(v,ast.Try));assert [d(v)for v in t1.body]==[d(v)for v in t2.body]
v3=load('preserved258-development/report.json');assert v3['source_count']==258 and v3['development_full_tests']==338 and v3['exit_code']==0 and v3['reaped']is True
archive('preserved258-development/sources.tar.gz',v3['source_hashes'])
for folder,failures,passes in [('negative-nested-cleanup',3,1),('negative-sticky-primary',2,1)]:
 x=ET.parse(P/folder/'tests.xml');assert len(x.findall('.//failure'))==failures and len(x.findall('.//testcase'))==failures+passes
x=ET.parse(P/'independent-spec-successor/tests.xml');assert len(x.findall('.//testcase'))==28 and not any(x.findall('.//'+k)for k in ('failure','error','skipped'))
artifact=load('artifact_hashes.json');actual={str(p.relative_to(P))for p in P.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts};assert set(artifact)==actual
for name,pin in artifact.items():assert sha(P/name)==pin,name
print('VERIFIED: disabled Toy4 V4,260 sources/371 checks, original199 and numerical AST unchanged, retained253/258 and negative evidence; no actual diagnostic launch.')
