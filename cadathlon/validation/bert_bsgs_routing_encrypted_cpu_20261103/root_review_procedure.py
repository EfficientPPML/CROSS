from pathlib import Path
import hashlib,json,os,sys,tarfile,math,datetime

root=Path('/tmp/bert-bsgs-routing-he-ready-20261103')
sys.path.insert(0,str(root))
import collect_component as c
p=c.protocol
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
expected=p.hashes(p.sources());assert len(expected)==147
profile,plan=p.check_sources(expected);p.require_ready()
p.authenticate_keys(Path('/tmp/bert-full-key-generation-component-20261103/actual-v1'),Path('/dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d'),profile=profile,plan=plan)
pins={'fixture':'7c2e7f91cddefe24fdabece73af5914d941bdd5a9f98c92df7b7863581539eef','server':'da26ae7398f5bb4af58407c9425cdf642b0a5aad5c584286b88c1689f5a8899c','validate':'3e3510ff1f519fe77e9c3e64faeb9e853b858404fd80f0f2c910a63d30d0a991'}
reports={};workers={}
for stage,pin in pins.items():
 run=root/f'actual-{stage}-v1';r=json.loads((run/'report.json').read_bytes())
 assert sha(run/'report.json')==pin
 for n,h in expected.items():assert sha(root/n)==sha(run/'execution_source'/n)==h,n
 r=p.terminal_gate(run,pin,stage=stage,expected_sources=expected,workspace=Path(r['workspace']),workspace_identity=r['workspace_identity'],request=r['request_contract'])
 w=json.loads((run/'worker/worker_report.json').read_bytes());reports[stage]=r;workers[stage]=w
 assert not Path(f'/proc/{w["pid"]}').exists()
 for child in r['children']:
  assert child['direct_child_reaped'] is True and child['direct_returncode']==0 and child['group_empty_verified'] is True and child['cleanup_errors']==[]
  try:os.killpg(child['process_group'],0)
  except ProcessLookupError:pass
  else:raise AssertionError('owned group survives')
assert reports['server']['readback']==reports['validate']['readback']
w=workers['validate'];r=reports['validate'];rows=w['numeric_checkpoints'];assert len(rows)==18
proof=json.loads((root/'proof_plan.json').read_bytes())
for i,row in enumerate(rows):
 assert row['index']==i and row['slots']==65536 and row['absolute_gate']==1e-6
 assert row['passed'] is True and row['numeric_metrics_finite'] is True
 assert math.isfinite(row['max_abs_error']) and 0<=row['max_abs_error']<=1e-6 and math.isfinite(row['nrmse'])
assert r['numeric_summary']==w['numeric_summary']==dict(absolute_gate=1e-6,checkpoints=18,maximum_absolute_error=max(x['max_abs_error'] for x in rows),passed=True,physical_slots=sum(x['slots'] for x in rows))
assert w['all18_authenticated_before_private_read'] is True
review=dict(passed=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_count=147,current_execution_archive_verified=True,three_terminal_gates_authenticated=True,all_phase_workers_and_groups_absent=True,server_readback_equals_validator_readback=True,all18_authenticated_before_private_read=True,numeric_summary=r['numeric_summary'],report_pins=pins,observed_sessions={'fixture':{'session':5310,'exit_code':0,'reaped':True},'server':{'session':89442,'exit_code':0,'reaped':True},'validate':{'session':6747,'exit_code':0,'reaped':True}},scope='Root independent public terminal/metric/source verification. Full ciphertext readback already independently recorded in server.json. No new decode or model run.')
out=root/'root-review/final.json';assert not out.exists();out.write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review))
