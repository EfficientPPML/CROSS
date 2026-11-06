"""Public recorded-result verification only; no measured runtime imports."""
from pathlib import Path
import argparse,hashlib,json,math,os,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(current=False):
 r=json.loads((P/'run-v1/report.json').read_bytes());m=json.loads((P/'sources.json').read_bytes());o=json.loads((P/'owner-terminal.json').read_bytes())
 assert sha(P/'run-v1/report.json')=='f4c8e201cec1b643781e42e3f8ad542cb3e48f39cdd43418e1c9ce88b56f8f3e'
 assert sha(P/'run-v1/sources.tar.gz')==r['source_archive_sha256']=='edbe516b8af583027268b54fee4b504c38cf7ba075c5a213820f90c2a58d161a'
 assert len(m)==113 and m==r['source_hashes_before']==r['source_hashes_after']
 with tarfile.open(P/'run-v1/sources.tar.gz')as t:
  rows=t.getmembers();assert len(rows)==len({x.name for x in rows})==113 and all(x.isfile()for x in rows)
  assert {x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest()for x in rows}==m
 assert all(r[n]is True for n in ('passed','complete','source_archive_verified','source_files_unchanged','input_payload_and_metadata_unchanged','server_complete_before_any_decode','cleanup_completed','owned_key_directory_removed'))
 assert all(r[n]is False for n in ('preflight_only','refresh_performed','full_attention_evaluated','full_block_evaluated','full_model_evaluated','calibrated_timing'))
 assert r['errors']==r['cleanup_errors']==r['collector_cleanup_errors']==[]
 levels={'score_mean':36,'normalized_t':35,'base_exp':31,'mean_exp':29,'inverse':18,'base_times_inverse':17,'probabilities':16}
 assert set(r['checks'])=={f+':'+n for f in ('all128_distinct16','partial9_per_row_edge_keys')for n in levels}
 assert r['fixed_absolute_error_gate']==1e-6
 for name,c in r['checks'].items():
  assert all(c[n]is True for n in ('passed','shape_and_finite','error_arithmetic_finite','metadata_ok'))
  assert c['slots_checked']==65536 and c['nsd']==1 and c['level']==levels[name.split(':')[1]]
  assert math.isfinite(c['max_abs_error'])and 0<=c['max_abs_error']<=1e-6 and c['scale']==c['expected_scale']
  assert len(c['execution_rings'])==1
  ring=c['execution_rings'][0];assert ring['degree']==131072 and ring['num_slots']==65536 and ring['num_q']==100 and ring['num_p']==14 and ring['dnum']==8 and ring['max_level']==49
  assert ring['q_towers']==r['ring']['q_towers']and ring['p_towers']==r['ring']['p_towers']
 assert r['ring']['actual_qp_is_secure_checked']is True and r['ring']['P_coverage_passed']is True and r['ring']['actual_QP_bit_length']==3403
 assert set(r['codec_guards'])=={'encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode'}
 labels=[p['label']for p in r['phases']];all_done=labels.index('all14 ciphertext checkpoints complete; final client validation begins');client_done=labels.index('client all-slot validation complete');assert all_done<client_done
 assert all(labels.index(f+' seven ciphertext checkpoints complete')<all_done for f in ('all128_distinct16','partial9_per_row_edge_keys'))
 assert o['session_id']==44272 and o['exit_code']==0 and o['reaped']is True and o['terminal_tool_chunk']=='e5927b'
 assert all(o[n]is True for n in ('worker_pid_absent','parent_pid_absent','group_absent','owned_key_directory_removed','current_sources_equal_before_after_archive','all_fixed_gates_pass'))
 assert o['report_sha256']==sha(P/'run-v1/report.json')and o['source_archive_sha256']==sha(P/'run-v1/sources.tar.gz')
 if current:
  root=Path('/tmp/bert-initial-softmax-corrected-component-20261103');assert all(sha(root/n)==h for n,h in m.items())
  assert not Path(r['owned_public_key_directory']).exists()and not Path('/proc/1222982').exists()and not Path('/proc/1222978').exists()
  try:os.killpg(1222978,0)
  except ProcessLookupError:pass
  else:raise AssertionError('recorded group still exists')
 index=json.loads((P/'artifact_hashes.json').read_bytes());assert index=={str(p.relative_to(P)):sha(p)for p in P.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'}
 print(json.dumps(dict(passed=True,sources=113,checkpoints=14,checked_slots=917504,current_checked=current,max_error=max(v['max_abs_error']for v in r['checks'].values()))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');a=p.parse_args();verify(a.current)
