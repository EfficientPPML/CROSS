from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return json.loads((P/name).read_bytes())
r=load('preflight-v1/report.json');s=load('startup.json');t=load('terminal_readback.json');pins=s['source_hashes_before']
assert len(pins)==253 and pins==r['source_hashes_start']==r['source_hashes_end']==r['execution_hashes_end']
assert all(r[k] is True for k in ('passed','complete','cleanup_completed','source_verified','terminal_scope_completed','termination_scope_completed','all_tests_executed_without_skip'))
assert r['execute'] is False and r['test_cases']==r['expected_test_cases']==276
assert r['preflight_audit']['backend_or_private_attempts']==[] and r['preflight_audit']['zero_forbidden_attempts'] is True
fresh=r['fresh_source_guard'];assert fresh['passed'] is True and fresh['fresh'] is True and fresh['source_count']==253 and fresh['manifest_sha256']==s['external_manifest_sha256']
assert t['session']==78016 and t['exit_code']==0 and t['reaped'] is True and t['actual_execution_authorized'] is False
assert t['report_sha256']==sha(P/'preflight-v1/report.json')=='12d76acc517b861c68c1e2f5a298fad8709331c66a9d7657de75e8ec2561873c'
assert sha(P/'preflight-v1/sources.tar.gz')==r['source_archive_sha256']==t['source_archive_sha256']
child=r['test_child'];assert child['direct_returncode']==0 and all(child[k] is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
assert t['all_current_startup_execution_before_after_archive_match'] is True
xml=ET.parse(P/'preflight-v1/tests.xml');assert len(xml.findall('.//testcase'))==276 and not any(xml.findall('.//'+n)for n in ('failure','error','skipped'))
with tarfile.open(P/'preflight-v1/sources.tar.gz')as z:
 members=z.getmembers();assert len(members)==len({m.name for m in members})==253 and all(m.isfile()for m in members)
 raw={m.name:z.extractfile(m).read()for m in members};assert {n:hashlib.sha256(b).hexdigest()for n,b in raw.items()}==pins
 assert json.loads(raw['execution_authorization.json'])=={'ready':False}
 origin=json.loads(raw['successor_origin.json']);old=origin['predecessor_sources'];assert len(old)==246
 assert all(pins[n]==h for n,h in old.items()if n.startswith('runtime/')or n=='component_arithmetic.py')
 assert len([n for n in old if n.startswith('runtime/')])==199
for version,count,passed in (('v1',198,False),('v2',200,True)):
 folder='predecessor-development-source-'+version;p=load(folder+'/report.json');assert p['passed'] is passed
 xml=ET.parse(P/folder/'tests.xml');assert len(xml.findall('.//testcase'))==count and not any(xml.findall('.//'+n)for n in ('failure','error','skipped'))
 with tarfile.open(P/folder/'sources.tar.gz')as z:
  m=z.getmembers();assert len(m)==len({x.name for x in m})==246
  assert {x.name:hashlib.sha256(z.extractfile(x).read()).hexdigest()for x in m}==p['source_hashes_start']
assert load('negative-response-ownership/report.json')['returncode']==1
assert load('negative-response-ownership/output-ownership-report.json')['returncode']==1
assert load('predecessor-review/report.json')['current_unchanged'] is True
inventory=load('artifact_hashes.json')
assert set(inventory)=={str(p.relative_to(P))for p in P.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts}
for name,digest in inventory.items():assert sha(P/name)==digest,name
print('VERIFIED: disabled Toy4 successor253 sources/276 tests, exact original199+arithmetic, preserved246/negatives, source/process/outer gates; no HE/backend launch.')
