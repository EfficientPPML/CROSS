from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
root=Path('/tmp/qwen-proc-self-root-review-20261104')
cases=(('client24','/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v2',68374,'934453','170b97',24),('keygen32','/tmp/qwen-keygen-proc-self-focused-proof-20261104/preparation-v2',67367,'562eca','74d265',32))
def inventory(p):
 assert not any(x.is_symlink()for x in p.rglob('*'))
 return {x.relative_to(p).as_posix():sha(x.read_bytes())for x in p.rglob('*')if x.is_file()}
for label,p,session,launch_chunk,terminal_chunk,count in cases:
 proof=Path(p);pre_raw=(root/(label+'-preexecution-review.json')).read_bytes();pre=json.loads(pre_raw)
 grant_raw=(root/(label+'-run-grant.json')).read_bytes();grant=json.loads(grant_raw)
 assert grant['authorized']is True and grant['root_review_sha256']==sha(pre_raw)
 for name,h in pre['harness_pins'].items():assert sha((proof/name).read_bytes())==h
 cfg=json.loads((proof/'run_config.json').read_bytes());before=(proof/'sources-before.json').read_bytes();expected=json.loads(before)
 assert sha(before)==pre['source_manifest_sha256']and len(expected)==pre['source_count']
 assert inventory(Path(cfg['source_root']))==inventory(proof/'execution')==expected
 assert (proof/'sources-after.json').read_bytes()==(proof/'execution-after.json').read_bytes()==before
 assert sha((proof/'sources.tar.gz').read_bytes())==pre['source_archive_sha256']
 with tarfile.open(proof/'sources.tar.gz')as tf:
  members=tf.getmembers();assert len(members)==len({m.name for m in members})==len(expected)
  assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
  assert {m.name:sha(tf.extractfile(m).read())for m in members}==expected
 xml=ET.parse(proof/'tests.xml').getroot();counts={name:len(list(xml.iter(tag)))for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
 assert counts==dict(tests=count,failures=0,errors=0,skipped=0)
 ordered=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')];assert ordered==cfg['selected_xml_cases']
 report=json.loads((proof/'report.json').read_bytes());audit=json.loads((proof/'worker-audit.json').read_bytes());live=json.loads((proof/'live-observation.json').read_bytes())
 assert report['passed']is True and report['tests']==counts and report['sources_unchanged']is True
 assert report['selected_nodes']==audit['collected_nodes']==cfg['selected_nodes']and report['selected_xml_cases']==ordered
 assert report['child_returncode']==0 and report['child_reaped']is True and report['child_group_absent']is True
 assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False and audit['pytest_exit_code']==0
 assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
 assert audit['forbidden_roots']==cfg['forbidden_roots']and audit['affinity']==cfg['cpu_affinity']
 assert audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']
 assert report['actual_execution_authorized']is False and report['model_execution']is False and report['encrypted_execution']is False
 assert report['actual_key_generation']is False and report['actual_client_role_execution']is False
 for pid in (live['outer_pid'],live['worker_pid']):
  try:os.kill(pid,0)
  except ProcessLookupError:pass
  else:raise AssertionError('terminal PID remains')
 for pgid in set((live['outer_pgid'],live['worker_pgid'])):
  try:os.killpg(pgid,0)
  except ProcessLookupError:pass
  else:raise AssertionError('terminal group remains')
 owner=dict(schema='qwen-proc-self-root-direct-owner-v1',label=label,session_id=session,launch_tool_chunk=launch_chunk,terminal_tool_chunk=terminal_chunk,exit_code=0,direct_session_reaped=True,root_attempted_live_observation='e8ad54; both had already exited; no independent live observation claimed',healthy_execution_deadline=False,automatic_retry=False)
 owner_path=root/(label+'-direct-owner.json');owner_path.write_bytes(pretty(owner))
 names=('report.json','worker-audit.json','tests.xml','tests.log','live-observation.json','sources-before.json','sources-after.json','execution-after.json','sources.tar.gz','executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json','handoff.json')
 evidence={n:sha((proof/n).read_bytes())for n in names}
 result=dict(schema='qwen-proc-self-root-terminal-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),label=label,passed=True,counts=counts,ordered_selection_exact=True,source_current_execution_before_after_archive_equal=True,source_count=len(expected),source_manifest_sha256=pre['source_manifest_sha256'],root_preexecution_review_sha256=sha(pre_raw),root_run_grant_sha256=sha(grant_raw),direct_owner_receipt_sha256=sha(owner_path.read_bytes()),evidence=evidence,elapsed_s=report['elapsed_s'],outer_and_worker_pid_and_group_absent=True,independent_live_observation=False,backend_imported=False,actual_role_execution=False,capacity_evaluated=False,scope=cfg['scope'],child_artifact_retention='separate bounded per-case review follows; no additional scope claimed here',review_script_sha256=sha(Path(__file__).read_bytes()))
 path=root/(label+'-terminal-review.json');path.write_bytes(pretty(result))
 print(json.dumps(dict(label=label,path=str(path),sha256=sha(path.read_bytes()),report_sha256=evidence['report.json'],passed=True,tests=count)))
