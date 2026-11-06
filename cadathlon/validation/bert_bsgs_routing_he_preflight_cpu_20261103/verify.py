from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as E
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
pins=json.loads((p/'artifact_sha256.json').read_bytes())
actual={x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file() and '__pycache__'not in x.parts}
assert actual==set(pins)|{'artifact_sha256.json'}
for n,h in pins.items():assert sha(p/n)==h,n
r=json.loads((p/'report.json').read_bytes());assert r['passed']is True and r['execute']is False and r['terminal_scope_completed']is True
assert r['source_hashes_start']==r['source_hashes_end']==json.loads((p/'source_manifest.json').read_bytes()) and len(r['source_hashes_start'])==142
assert sha(p/'sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(p/'sources.tar.gz') as z:
 m=z.getmembers();assert len(m)==len({x.name for x in m})==142 and all(x.isfile() for x in m)
 assert {x.name:hashlib.sha256(z.extractfile(x).read()).hexdigest() for x in m}==r['source_hashes_start']
for name,count in (('mock_tests',361),('installed_denials',9)):
 cases=E.parse(p/(name+'.xml')).findall('.//testcase');assert len(cases)==count
 assert all(c.find(k)is None for c in cases for k in ('failure','error','skipped'))
 assert r[name]==dict(tests=count,errors=0,failures=0,skipped=0,exit_code=0)
assert all(c['reaped']is True and c['exit_code']==0 for c in r['children'])
t=json.loads((p/'terminal_observation.json').read_bytes());assert t['observed_exit_code']==0 and t['reaped']is True and t['report_sha256']==sha(p/'report.json')
print('PASS:142 exact sources,361 synthetic +9 installed checks; disabled preparation only, known successor repairs required')
