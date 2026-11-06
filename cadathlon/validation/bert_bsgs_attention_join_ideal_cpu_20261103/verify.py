from pathlib import Path
import hashlib,json,tarfile,math,xml.etree.ElementTree as ET
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
load=lambda q:json.loads(q.read_bytes())
pins=load(p/'artifact_sha256.json')
assert set(pins)=={x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file() and '__pycache__'not in x.parts and x.name!='artifact_sha256.json'}
for n,h in pins.items():assert sha(p/n)==h,n
r=load(p/'report.json');expected=r['source_hashes_before']
assert len(expected)==17 and expected==r['source_hashes_after']==r['execution_source_hashes_after']
for n,h in expected.items():assert sha(p/'candidate'/n)==h,n
assert sha(p/'sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(p/'sources.tar.gz')as z:
 ms=z.getmembers();assert len(ms)==len({m.name for m in ms})==17 and all(m.isfile() for m in ms)
 assert {m.name:hashlib.sha256(z.extractfile(m).read()).hexdigest() for m in ms}==expected
assert all(r[k]is True for k in ('passed','termination_scope_completed','terminal_scope_completed','all_processes_reaped_and_owned_group_empty','sources_unchanged','source_archive_verified'))
assert r['encrypted_execution']is False and r['model_forward']is False and r['cleanup_errors']==[]
proc=r['process'];assert proc['direct_returncode']==0 and all(proc[k]is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed')) and proc['cleanup_errors']==[]
cases=ET.parse(p/'tests.xml').findall('.//testcase');assert len(cases)==22 and not any(x.find('failure')is not None or x.find('error')is not None or x.find('skipped')is not None for x in cases)
assert r['tests']==dict(tests=22,errors=0,failures=0,skipped=0)
assert sha(p/'ideal_results.jsonl')==r['numeric_sha256']
rows=[json.loads(x) for x in (p/'ideal_results.jsonl').read_text().splitlines()]
delta=1.1467305003699405e18
assert len(rows)==6 and {(x['operation'],x['input_level'],x['input_scale']) for x in rows}=={(op,L,s) for op in ('scores','values') for L,s in ((49,delta),(25,math.nextafter(delta,math.inf)),(25,delta*1.21))}
for x in rows:
 assert x['physical_slots']==65536 and x['gate']==1e-8 and x['output_level']==x['input_level']-3 and math.isfinite(x['max_abs_error']) and 0<=x['max_abs_error']<=1e-8
 c=x['counts'];assert c['route_rotations']==1520 and c['route_actual_composite_rescales']==896 and c['ciphertext_products']==128 and c['head_rotations']==768 and c['row_alignment_rotations']==120
summary=dict(fixtures=6,physical_slots=sum(x['physical_slots'] for x in rows),absolute_gate=1e-8,maximum_absolute_error=max(x['max_abs_error'] for x in rows))
assert summary==r['numeric_summary']
t=load(p/'terminal_review.json');assert t['session']==9052 and t['exit_code']==0 and t['reaped']is True and t['both_pids_absent']is True and t['owned_group_absent']is True
assert t['report_sha256']==sha(p/'report.json') and t['source_archive_sha256']==sha(p/'sources.tar.gz') and t['numeric_summary']==summary
print('PASS:17 exact sources,22 ideal attention checks,393216 slots; no HE/full-model claim')
