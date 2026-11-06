from pathlib import Path
import hashlib,json,tarfile,math
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
load=lambda q:json.loads(q.read_bytes())
pins=load(p/'artifact_sha256.json')
assert {x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file() and '__pycache__'not in x.parts}==set(pins)|{'artifact_sha256.json'}
for n,h in pins.items():assert sha(p/n)==h,n
reports={};workers={};expected=None
for stage in ('fixture','server','validate'):
 d=p/stage;r=load(d/'report.json');w=load(d/'worker/worker_report.json');reports[stage]=r;workers[stage]=w
 assert r['stage']==w['stage']==stage and r['worker_exit_code']==0
 assert all(r[k]is True for k in ('passed','execute','worker_reaped','worker_group_empty_verified','all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed','terminal_scope_completed','source_files_unchanged','source_archive_verified'))
 assert r['cleanup_errors']==[] and all(w[k]is True for k in ('passed','phase_complete','loader_cleanup_completed','source_files_unchanged','terminal_scope_completed')) and w['forbidden_attempts']==[]
 assert sha(d/'worker/worker_report.json')==r['worker_report_sha256'] and sha(d/'sources.tar.gz')==r['source_archive_sha256']
 assert r['source_hashes_start']==r['source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']==load(d/'source_manifest.json') and len(r['source_hashes_start'])==147
 if expected is None:expected=r['source_hashes_start']
 assert r['source_hashes_start']==expected
 with tarfile.open(d/'sources.tar.gz') as z:
  m=z.getmembers();assert len(m)==len({x.name for x in m})==147 and all(x.isfile() for x in m)
  assert {x.name:hashlib.sha256(z.extractfile(x).read()).hexdigest() for x in m}==expected
 assert r['readback']==w['readback'] and r['request_contract']==w['request_contract']
 assert r['request_contract']==reports['fixture']['request_contract']
 for child in r['children']:assert child['direct_child_reaped']is True and child['direct_returncode']==0 and child['group_empty_verified']is True and child['cleanup_errors']==[]
for role,stage,count in [('inputs','fixture',2),('outputs','server',18)]:
 m=load(p/role/'manifest.json');r=reports[stage]
 assert m==r['readback']['manifest'] and sha(p/role/'manifest.json')==r['readback']['manifest_sha256'] and r['readback']['tile_count']==count
 for i in range(count):
  q=p/role/f'tile{i}_transport.json';t=load(q)
  assert sha(q)==m['tiles'][i]['manifest_sha256'] and t['tile_index']==i and t['tile_count']==count and t['request_id']==r['request_contract']['nonce']
assert reports['server']['readback']==reports['validate']['readback']
assert workers['server']['completed_ciphertexts']==18 and workers['server']['all_inputs_authenticated_before_server']is True
assert workers['server']['arithmetic']['route_totals']==dict(ciphertext_additions=156,composite_rescales=42,critical_public_levels=12,public_products=168,rotations=72)
w=workers['validate'];rows=w['numeric_checkpoints'];assert len(rows)==18 and w['all18_authenticated_before_private_read']is True
for i,r in enumerate(rows):
 assert r['index']==i and r['slots']==65536 and r['absolute_gate']==1e-6 and r['passed']is True and r['numeric_metrics_finite']is True
 assert math.isfinite(r['max_abs_error']) and 0<=r['max_abs_error']<=1e-6 and math.isfinite(r['nrmse'])
summary=dict(absolute_gate=1e-6,checkpoints=18,maximum_absolute_error=max(x['max_abs_error'] for x in rows),passed=True,physical_slots=sum(x['slots'] for x in rows))
assert w['numeric_summary']==reports['validate']['numeric_summary']==summary
review=load(p/'root_final_review.json');assert review['passed']is True and review['numeric_summary']==summary
for stage in reports:
 assert review['report_pins'][stage]==sha(p/stage/'report.json')
 assert review['observed_sessions'][stage]['exit_code']==0 and review['observed_sessions'][stage]['reaped']is True
assert load(p/'root_server_review.json')['all_payloads_authenticated']is True
print('PASS:147 frozen sources; genuine18-checkpoint encrypted BSGS routing,1179648 slots,maxabs',summary['maximum_absolute_error'])
