from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
proof=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v1')
root=Path('/tmp/qwen-toy5-checkpoint-root-review-20261104')
pre_raw=(root/'preexecution-review.json').read_bytes();pre=json.loads(pre_raw)
grant_raw=(root/'run-grant.json').read_bytes();grant=json.loads(grant_raw)
assert sha(pre_raw)=='7648cfeab362fb30de97c81e4d0feeec703d6cd33076705940b0b6bb43e53db4'
assert sha(grant_raw)=='e515902df92b28486ef33bf8e890d60753c813a0814dd1e3b7f5c0aa1ab5cedb' and grant['authorized']is True
for n,h in pre['harness_pins'].items():assert sha((proof/n).read_bytes())==h
cfg=json.loads((proof/'run_config.json').read_bytes());before=(proof/'sources-before.json').read_bytes();expected=json.loads(before)
def inventory(p):
 assert not any(q.is_symlink()for q in p.rglob('*'))
 return {q.relative_to(p).as_posix():sha(q.read_bytes())for q in p.rglob('*')if q.is_file()}
assert len(expected)==261 and inventory(Path(cfg['source_root']))==inventory(proof/'execution')==expected
assert (proof/'sources-after.json').read_bytes()==(proof/'execution-after.json').read_bytes()==before
assert sha(before)==pre['source_manifest_sha256'] and sha((proof/'sources.tar.gz').read_bytes())==pre['source_archive_sha256']
with tarfile.open(proof/'sources.tar.gz')as tf:
 members=tf.getmembers();assert len(members)==len({m.name for m in members})==261
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==expected
xml=ET.parse(proof/'tests.xml').getroot();cases=list(xml.iter('testcase'))
counts={name:len(list(xml.iter(tag)))for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=48,failures=6,errors=0,skipped=0)
ordered=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in cases];assert ordered==cfg['selected_xml_cases']
failures=[dict(classname=c.attrib['classname'],name=c.attrib['name'],message=c.find('failure').attrib.get('message'))for c in cases if c.find('failure')is not None]
report=json.loads((proof/'report.json').read_bytes());audit=json.loads((proof/'worker-audit.json').read_bytes());live=json.loads((proof/'live-observation.json').read_bytes())
assert report['passed']is False and report['tests']==counts and report['sources_unchanged']is True
assert report['child_returncode']==1 and report['child_reaped']is True and report['child_group_absent']is True
assert report['selected_nodes']==audit['collected_nodes']==cfg['selected_nodes'] and report['selected_xml_cases']==ordered
assert audit['pytest_exit_code']==1 and audit['denied']==audit['loaded_backend']==[] and audit['numpy_imported']is True
assert audit['forbidden_roots']==cfg['forbidden_roots'] and audit['affinity']==cfg['cpu_affinity']
assert audit['runtime_sources_before']==audit['runtime_sources_after']==cfg['runtime_source_expectations']
assert audit['runtime_module_origins']=={n:dict(file=p,spec=p)for n,p in cfg['runtime_module_origins'].items()}
for n,h in cfg['runtime_source_expectations'].items():assert sha(Path(n).read_bytes())==h
assert audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']
for key in ('actual_execution_authorized','model_execution','encrypted_execution','actual_key_generation','actual_client_role_execution','first_encrypted_divergence_claim','capacity_measurement'):assert report[key]is False
for pid in(live['outer_pid'],live['worker_pid']):
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise AssertionError('terminal PID remains')
for pgid in set((live['outer_pgid'],live['worker_pgid'])):
 try:os.killpg(pgid,0)
 except ProcessLookupError:pass
 else:raise AssertionError('terminal process group remains')
assert json.loads((proof/'owned-pytest-metadata.json').read_bytes())=={}
owner=dict(schema='q5-focused-root-direct-owner-v1',session_id=35774,launch_tool_chunk='7d216d',terminal_tool_chunk='07788f',exit_code=1,direct_session_reaped=True,root_observation_tool='700a12; both processes already exited; no independent live observation',healthy_execution_deadline=False,automatic_retry=False)
owner_path=root/'direct-owner.json';owner_path.write_bytes(pretty(owner))
names=('report.json','worker-audit.json','tests.xml','tests.log','live-observation.json','sources-before.json','sources-after.json','execution-after.json','sources.tar.gz','executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json','handoff.json','owned-pytest-metadata.json')
evidence={n:sha((proof/n).read_bytes())for n in names}
result=dict(schema='q5-focused-root-terminal-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evidence_verified=True,test_outcome='FAIL',passed=False,counts=counts,passed_testcases=42,failures=failures,ordered_selection_exact=True,source_current_execution_before_after_archive_equal=True,source_count=261,source_manifest_sha256=pre['source_manifest_sha256'],root_preexecution_review_sha256=sha(pre_raw),root_run_grant_sha256=sha(grant_raw),direct_owner_receipt_sha256=sha(owner_path.read_bytes()),evidence=evidence,elapsed_s=report['elapsed_s'],outer_and_worker_pid_and_group_absent=True,independent_live_observation=False,backend_imported=False,genuine_numpy_imported=True,actual_role_execution=False,capacity_evaluated=False,scope=cfg['scope'],diagnosis='First assertion shows85 fixture captures versus47 intended; six failures preserved. Source-only fixture semantic-selection diagnosis follows; no candidate/runtime finding inferred yet.',script_sha256=sha(Path(__file__).read_bytes()))
path=root/'terminal-review.json';path.write_bytes(pretty(result))
print(json.dumps(dict(path=str(path),sha256=sha(path.read_bytes()),report_sha256=evidence['report.json'],test_outcome='FAIL',counts=counts,evidence_verified=True)))
