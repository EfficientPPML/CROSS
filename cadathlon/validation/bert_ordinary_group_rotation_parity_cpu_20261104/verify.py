"""Read-only public terminal verification; never imports archived runtime code."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, tarfile, io, math
parser=argparse.ArgumentParser()
parser.add_argument('--current',type=Path,help='Optional original source tree; reads only the259 manifest entries.')
args=parser.parse_args()
out=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def safe(name):
 p=PurePosixPath(name)
 return bool(name) and not p.is_absolute() and '..' not in p.parts and str(p)==name
index=json.loads((out/'files.sha256.json').read_bytes())
assert type(index)is dict and all(safe(name) for name in index)
assert {str(x.relative_to(out)) for x in out.rglob('*') if x.is_file()}==set(index)|{'files.sha256.json'}
for name,digest in index.items():
 path=out/name
 assert not path.is_symlink() and path.is_file() and sha(path.read_bytes())==digest,name
parent_raw=(out/'report.json').read_bytes();worker_raw=(out/'worker_report.json').read_bytes();p=json.loads(parent_raw);w=json.loads(worker_raw)
assert sha(parent_raw)=='dae30871ca548367b873f4fcf2c65e29867413c0286f83d124797f433befaea3'
assert sha(worker_raw)=='f94a9be90a458a6b398c0245005f5504f3d42c82af26a771a2afffdad3961273'
expected=p['source_hashes_start'];assert len(expected)==259
assert expected==p['source_hashes_end']==p['execution_source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']==json.loads((out/'source_manifest.json').read_bytes())
if args.current is not None:
 assert all(sha((args.current/n).read_bytes())==h for n,h in expected.items())
archive=(out/'sources.tar.gz').read_bytes();assert sha(archive)==p['source_archive_sha256']=='56afba21d5e6f4c4805b202565b8f1bb6360b976de07369beb6dbf77bc371d59'
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as z:
 members=z.getmembers();assert len(members)==259 and len({m.name for m in members})==259 and all(m.isfile() and not m.issym() and not m.islnk() and safe(m.name) for m in members)
 assert {m.name:sha(z.extractfile(m).read())for m in members}==expected
for name in ['passed','execute','actual_rotation_parity','actual_execution_attempted','all_processes_reaped','external_workspaces_never_removed','operation_complete','terminal_scope_completed','source_files_unchanged','source_archive_verified','io_workspace_retained','cleanup_completed']:assert p[name]is True,name
for name in ['passed','actual_rotations_started','all_rotations_completed','request_readback_complete','upload_payloads_unchanged','source_files_unchanged','compiled_owner_cleanup_completed','loader_cleanup_completed','keygen_codecs_native_private_model_forbidden','terminal_scope_completed']:assert w[name]is True,name
assert not p.get('error')and not w.get('error')and not p.get('cleanup_errors')and not w.get('cleanup_errors')and not w.get('forbidden_attempts')
assert p['public_plan']==w['public_plan']and p['prerequisite']==w['prerequisite']and p['reviewed_sources']==w['reviewed_sources']
assert p['request_readback']==w['request_readback']and p['arithmetic']==w['arithmetic']
for field,value in w.items():
 if field.startswith('loaded_local_modules'):
  for row in value.values():assert expected[row['path']]==row['sha256']
record=p['worker_process'];assert record==json.loads((out/'process.json').read_bytes())
assert record['pid']==w['pid']==1878109 and type(record['returncode'])is int and record['returncode']==0 and record['timeout_s']is None
assert not record.get('error')
for name in ['process_reaped','public_key_store_removed','owned_directory_handle_closed']:assert record[name]is True,name
assert record['result']['passed']is True and record['result']['worker_report_sha256']==sha(worker_raw)
assert record['result']==json.loads((out/'stdout.json').read_bytes())
# The generic run_case initializer is intentionally preserved; the selected
# collector admits its typed exit/cleanup/result and authenticated worker proof.
assert record['passed']is False
plan=p['public_plan'];params=plan['parameters'];arith=w['arithmetic'];rows=arith['comparisons']
assert rows==w['completed_comparisons']and len(rows)==66 and arith['passed']is True and arith['input_unchanged']is True
assert params['degree']==131072 and params['num_slots']==65536 and params['dnum']==8 and params['composite_degree']==2 and params['degree_layout']==[256,512]
q=params['q_towers'];aux=params['p_towers'];assert len(q)==100 and len(aux)==14
assert all(type(x)is int and x>1 for x in q+aux)
qp_bits=(math.prod(q)*math.prod(aux)).bit_length();assert qp_bits==3403 and qp_bits<=3523
assert all(math.prod(aux)>=math.prod(q[start:start+13])for start in range(0,100,13))
groups=(('head',tuple(63*i for i in range(12))),('local_key',tuple(-16*i for i in range(16))),('edges',(0,63,63,-16,0)))
wanted=[dict(level=level,group=name,index=index,normalized_index=index%65536,sequence=sequence)for level in (49,25)for name,indices in groups for sequence,index in enumerate(indices)]
for row,req in zip(rows,wanted):
 for name,value in req.items():assert type(row[name])is type(value)and row[name]==value
 assert row['passed']is True and row['metadata_equal']is True and type(row['mismatched_words'])is int and row['mismatched_words']==0
 towers=(row['level']+1)*2;assert type(row['compared_words'])is int and row['compared_words']==2*131072*towers
 assert row['zero_alias']is(req['normalized_index']==0)and row['left']==row['right']
 value=row['left'];assert value['shape']==[1,2,256,512,towers]and value['moduli']==q[:towers]and value['dtype']=='uint32'and value['degree_layout']==[256,512]
 assert type(value['precision'])is int and value['precision']==32 and type(value['nsd'])is int and value['nsd']==1
 assert type(value['scale'])is float and value['scale']==params['scaling_factor']and value['scale_hex']==value['scale'].hex()
 assert type(value['payload_sha256'])is str and len(value['payload_sha256'])==64
for name,value in [('actual_rotation_calls',116),('nonzero_pair_comparisons',58),('zero_alias_comparisons',8),('shared_decompositions',6)]:assert type(arith[name])is int and arith[name]==plan[name]==value
assert len(arith['groups'])==len(plan['rows'])==6
for observed,planned in zip(arith['groups'],plan['rows']):
 assert (observed['level'],observed['group'])==(planned['level'],planned['group'])
 assert observed['stats']==dict(mode='digits',requests=planned['requests'],decompositions=1,digit_bytes=planned['digit_bytes'],result_bytes=planned['result_bytes'])
 assert type(observed['elapsed_group_s'])is float and math.isfinite(observed['elapsed_group_s'])and observed['elapsed_group_s']>0
reduction=arith['level_reduction'];assert all(reduction[name]is True for name in ['passed','payload_prefix_equal','ordered_q_prefix'])
assert reduction['input']['shape']==[1,2,256,512,100]and reduction['output']['shape']==[1,2,256,512,52]
assert reduction['input']['moduli']==q and reduction['output']['moduli']==q[:52]
assert reduction['input']['scale']==reduction['output']['scale']==params['scaling_factor']
owner=json.loads((out/'owner-terminal.json').read_bytes())
assert owner['schema']=='ordinary-group-parity-owner-terminal-v1'
assert owner['passed']is True and owner['owner_reaped']is True and type(owner['owner_exit_code'])is int and owner['owner_exit_code']==0
assert owner['owner_session']==42804 and owner['owner_tool_chunk']=='711c01'
assert owner['report_sha256']==sha(parent_raw) and owner['worker_report_sha256']==sha(worker_raw)
assert owner['source_archive_sha256']==sha(archive) and owner['source_manifest_sha256']==sha((out/'source_manifest.json').read_bytes())
assert owner['review_script_sha256']==sha((out/'owner-readback.py.txt').read_bytes())
assert owner['source_count']==259 and owner['current_execution_start_end_archive_equal']is True
assert owner['observed_absent_pids']==[1878080,1878082,1878109] and owner['observed_absent_groups']==[1878080,1878109]
assert owner['owned_temporary_process_directory_absent']is True and owner['borrowed_external_stores_retained_reported']is True
assert owner['nested_process_passed_initializer']is False
assert owner['total_compared_words']==sum(row['compared_words'] for row in rows)==1314914304
for field in ['all_word_and_metadata_equal','level49_to25_prefix_equal','public_p_coverage','loader_cleanup_completed','compiled_owner_cleanup_completed']:
 assert owner[field]is True
assert owner['parent_elapsed_s']==p['elapsed_s'] and owner['worker_elapsed_s']==w['elapsed_s']
assert owner['cpu_affinity']==p['cpu_affinity']==w['cpu_affinity']==[168,169,170,171]
references=json.loads((out/'public_references.json').read_bytes())
assert references['prerequisite']==p['prerequisite'] and references['reviewed_sources']==p['reviewed_sources']
progress=[json.loads(line) for line in (out/'stderr.log').read_text().splitlines()]
assert len(progress)==66
for emitted,observed in zip(progress,rows):
 assert all(type(emitted[key])is type(observed[key]) and emitted[key]==observed[key] for key in emitted)
root_raw=(out/'root-review.json').read_bytes()
assert sha(root_raw)=='b77006bbb9e19d9c62a8f7f5c5c92ec0b8f57efb84d3dd8be642129a07546ad9'
review=json.loads(root_raw)
assert review['schema']=='ordinary-group-parity-root-review-v1' and review['passed']is True
assert review['source_count']==259 and review['current_execution_start_end_archive_exact']is True and review['pure_selected_collector_gate_passed']is True
assert review['pairs']==66 and review['nonzero_pairs']==58 and review['zero_alias_pairs']==8
assert review['rotation_calls']==116 and review['shared_decompositions']==6
assert review['compared_uint32_words']==1314914304 and review['mismatched_words']==0
assert review['generic_worker_process_passed']is False
assert review['loaded_origins']==owner['source_origins']
assert review['review_script_sha256']==sha((out/'root-review-driver.py.txt').read_bytes())
root_pins={
 '/tmp/bert-ordinary-group-parity-ready-20261103/actual-v1/report.json':'report.json',
 '/tmp/bert-ordinary-group-parity-ready-20261103/actual-v1/source_manifest.json':'source_manifest.json',
 '/tmp/bert-ordinary-group-parity-ready-20261103/actual-v1/sources.tar.gz':'sources.tar.gz',
 '/tmp/bert-ordinary-group-parity-ready-20261103/actual-v1/worker/worker_report.json':'worker_report.json',
 '/tmp/bert-ordinary-group-parity-terminal-20261104/owner-terminal.json':'owner-terminal.json',
 '/tmp/check_ordinary_group_terminal_20261104.py':'owner-readback.py.txt',
}
assert set(review['pins'])==set(root_pins)
for original,local in root_pins.items():assert review['pins'][original]==sha((out/local).read_bytes())
assert review['borrowed_directory_identities_retained']==dict(io=[66305,2383254],keys=[24,1563],model=[66305,2383041])
print(json.dumps(dict(passed=True,public_files=len(index)+1,sources=259,compared_pairs=66,total_compared_words=1314914304,current_sources_checked=args.current is not None,scope='Public source/report and recorded owner-terminal checks only; no payload reads, runtime imports or experiments.'),sort_keys=True))
