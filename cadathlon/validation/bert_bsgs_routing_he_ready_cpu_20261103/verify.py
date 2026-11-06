from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as E
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
pins=json.loads((p/'artifact_sha256.json').read_bytes())
actual={x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file() and '__pycache__'not in x.parts}
assert actual==set(pins)|{'artifact_sha256.json'}
for n,h in pins.items():assert sha(p/n)==h,n
r=json.loads((p/'report.json').read_bytes());assert r['passed']is True and r['execute']is False and r['terminal_scope_completed']is True
assert r['source_hashes_start']==r['source_hashes_end']==json.loads((p/'source_manifest.json').read_bytes()) and len(r['source_hashes_start'])==147
assert sha(p/'sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(p/'sources.tar.gz') as z:
 m=z.getmembers();assert len(m)==len({x.name for x in m})==147 and all(x.isfile() for x in m)
 bodies={x.name:z.extractfile(x).read() for x in m}
 assert {n:hashlib.sha256(b).hexdigest() for n,b in bodies.items()}==r['source_hashes_start']
assert json.loads(bodies['execution_authorization.json'])=={'ready':True}
old=json.loads(bodies['predecessor/report.json']);assert old['passed']is True and old['execute']is False and len(old['source_hashes_start'])==142
assert hashlib.sha256(bodies['predecessor/report.json']).hexdigest()=='d070b71ba2f1e471258468d1143a59e1309f479ab0f564274482898580192993'
assert hashlib.sha256(bodies['predecessor/sources.tar.gz']).hexdigest()==old['source_archive_sha256']
overlay=json.loads(bodies['successor_overlays.json']);assert len(overlay)==10
for n,h in old['source_hashes_start'].items():assert hashlib.sha256(bodies[n]).hexdigest()==overlay.get(n,h)
for name,count in (('mock_tests',402),('installed_denials',9)):
 cases=E.parse(p/(name+'.xml')).findall('.//testcase');assert len(cases)==count
 assert all(c.find(k)is None for c in cases for k in ('failure','error','skipped'))
 assert r[name]==dict(tests=count,errors=0,failures=0,skipped=0,exit_code=0)
assert all(r[k]is True for k in ('all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed'))
assert all(c['direct_child_reaped']is True and c['direct_returncode']==0 and c['group_empty_verified']is True and c['cleanup_completed']is True and not c['cleanup_errors'] for c in r['children'])
t=json.loads((p/'terminal_observation.json').read_bytes());assert t['observed_exit_code']==0 and t['reaped']is True and t['report_sha256']==sha(p/'report.json')
print('PASS:147 exact sources,402 synthetic +9 installed checks; no encrypted result in this preparation')
