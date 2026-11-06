from pathlib import Path
import hashlib,json,tarfile,math
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'run-v1/report.json').read_bytes());receipt=json.loads((P/'terminal_readback.json').read_bytes());s=json.loads((P/'summary.json').read_bytes())
assert r['passed'] is True and r['complete'] is True and r['preflight_only'] is False
assert r['fixed_absolute_error_gate']==1e-6 and r['errors']==r['cleanup_errors']==r['collector_cleanup_errors']==[]
assert all(r[k] is True for k in ('cleanup_completed','source_archive_verified','source_files_unchanged','input_payload_and_metadata_unchanged','source_closure_verified_before_decode','server_complete_before_any_decode','keyless_server_initialized'))
assert all(r[k] is False for k in ('full_attention_evaluated','full_block_evaluated','full_model_evaluated','refresh_performed','calibrated_timing'))
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==128
assert len(r['codec_guards'])==7 and len(r['key_generation_guards'])==2
assert r['ring']['actual_qp_is_secure_checked'] is True and r['ring']['P_coverage_passed'] is True
assert r['ring']['rotation_indices']==[] and r['ring']['degree']==131072 and r['ring']['dnum']==8
assert list(r['checks'])==['prestage','domain map','T2','T4','T8','T16','T32','T64','output']
assert [v['level']for v in r['checks'].values()]==[24,23,21,19,17,15,13,11,10]
for n,v in r['checks'].items():
 assert v['passed'] is True and v['shape_and_finite'] is True and v['metadata_ok'] is True
 assert v['slots_checked']==65536 and v['gate']==1e-6 and math.isfinite(v['max_abs_error']) and 0<=v['max_abs_error']<=1e-6
 assert v['level']==v['expected_metadata']['level'] and v['scale']==v['expected_metadata']['scale'] and v['nsd']==1
 assert s['checkpoints'][n]=={k:v[k]for k in ('passed','max_abs_error','padding_max_abs_error','slots_checked','level','scale')}
assert max(v['max_abs_error']for v in r['checks'].values())==s['worst_checkpoint_error']==4.930477581766013e-9
assert r['checks']['output']['max_abs_error']==1.964828300060617e-11
assert r['checks']['prestage']['domain_diagnostic']['decoded_all_slots_inside'] is False
assert s['total_checkpoint_slots']==589824 and s['native_gelu_diagnostic']==r['native_gelu_diagnostic']
assert sha(P/'run-v1/report.json')==receipt['report_sha256']
assert sha(P/'run-v1/sources.tar.gz')==receipt['source_archive_sha256']==r['source_archive_sha256']
with tarfile.open(P/'run-v1/sources.tar.gz')as t:
 m=t.getmembers();assert len(m)==len({x.name for x in m})==128 and all(x.isfile()for x in m)
 assert {x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest()for x in m}==r['source_hashes_before']
assert receipt['session']==35253 and receipt['exit_code']==0 and receipt['reaped'] is True
assert all(receipt[k] is True for k in ('worker_absent','owned_group_absent','all128_current_sources_equal_start_end_archive'))
assert 'Exit status: 0'in(P/'run-v1.time.txt').read_text()
inventory=json.loads((P/'artifact_hashes.json').read_bytes())
assert set(inventory)=={str(p.relative_to(P))for p in P.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts}
for name,digest in inventory.items():assert sha(P/name)==digest,name
print('VERIFIED: genuine GELU64 all9 fixed gates/589824 slots PASS;128 exact sources; exit0/reaped and cleanup complete. Component scope only.')
