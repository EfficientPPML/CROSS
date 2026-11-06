from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--current',type=Path);args=p.parse_args()
manifest=json.loads((P/'artifact_hashes.json').read_bytes())
assert {x.relative_to(P).as_posix():sha(x.read_bytes()) for x in P.rglob('*') if x.is_file() and x.name!='artifact_hashes.json'}==manifest
r=json.loads((P/'measured/report.json').read_bytes());w=json.loads((P/'measured/worker/worker_report.json').read_bytes());proc=json.loads((P/'measured/worker/process.json').read_bytes());expected=json.loads((P/'measured/source_manifest.json').read_bytes())
assert len(expected)==231 and r['source_hashes_start']==r['source_hashes_end']==r['execution_source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']==expected
assert all(r[n] is True for n in ('passed','execute','actual_rotation_parity','actual_execution_attempted','operation_complete','all_processes_reaped','source_files_unchanged','source_archive_verified','cleanup_completed','terminal_scope_completed','external_workspaces_never_removed','io_workspace_retained'))
assert all(w[n] is True for n in ('passed','all_rotations_completed','source_files_unchanged','compiled_owner_cleanup_completed','loader_cleanup_completed','upload_payloads_unchanged','keygen_codecs_native_private_model_forbidden','terminal_scope_completed')) and not w['forbidden_attempts']
assert proc['returncode']==0 and all(proc[n] is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
assert proc['result']['passed'] is True and proc['result']['worker_report_sha256']==sha((P/'measured/worker/worker_report.json').read_bytes())
assert sha((P/'measured/sources.tar.gz').read_bytes())==r['source_archive_sha256']
with tarfile.open(P/'measured/sources.tar.gz') as t:
 rows=t.getmembers();assert len(rows)==231==len({m.name for m in rows})
 assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in rows)
 assert {m.name:sha(t.extractfile(m).read()) for m in rows}==expected
if args.current:
 assert {n:sha((args.current/n).read_bytes()) for n in expected}==expected
 assert {n:sha((args.current/'actual-v1/execution_source'/n).read_bytes()) for n in expected}==expected
assert r['arithmetic']==w['arithmetic'];a=w['arithmetic'];schedule=[(L,k,i) for L in (49,25) for i,k in enumerate((1,1,64,64,1,64))]
assert [(x['level'],x['index'],x['sequence']) for x in a['comparisons']]==schedule
assert a['passed'] is True and a['actual_rotation_calls']==24 and a['input_unchanged'] is True
for x in a['comparisons']:
 assert x['passed'] is True and x['metadata_equal'] is True and x['mismatched_words']==0 and x['compared_words']==2*131072*2*(x['level']+1) and x['left']==x['right']
for x,bytes_ in zip(a['levels'],(1913651200,554696704)):
 assert x['retained_entries']==2 and x['retained_bytes']==bytes_ and x['stats_delta']['hits']==4 and x['stats_delta']['published']==2
red=a['level_reduction'];assert all(red[n] is True for n in ('passed','payload_prefix_equal','ordered_q_prefix'))
assert red['input']['scale']==red['output']['scale'] and red['input']['nsd']==red['output']['nsd']==1
assert red['output']['moduli']==red['input']['moduli'][:52]
assert r['cpu_affinity']==w['cpu_affinity']==[140,141,142,143]
terminal=json.loads((P/'session_terminal.json').read_bytes());assert terminal['session_id']==41655 and terminal['observed_exit_code']==0 and terminal['tool_session_reaped'] is True and terminal['all_three_absent'] is True
print(json.dumps({'passed':True,'source_count':231,'exact_pairs':12,'actual_rotations':24,'compared_uint32_words':sum(x['compared_words'] for x in a['comparisons']),'current_checked':args.current is not None,'scope':'representation parity only'}))
