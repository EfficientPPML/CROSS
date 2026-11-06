from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
root=Path('/tmp/qwen-server-admission-snapshot-root-review-v2-20261104')
proof=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v2')
pre_raw=(root/'preexecution-review.json').read_bytes();grant_raw=(root/'run-grant.json').read_bytes()
assert sha(pre_raw)=='774a73cbd3f9a69e686662253fd4d0acec73e2c8888eba7040eee2bf79c52787'
assert sha(grant_raw)=='64d002a861256e6e8a82a0ad365acb95f8c63f24ec5579ef8299e39948d55bd6'
pre=json.loads(pre_raw);grant=json.loads(grant_raw);assert grant['authorized']is True
for n,h in pre['harness_pins'].items():assert sha((proof/n).read_bytes())==h
cfg=json.loads((proof/'run_config.json').read_bytes());before=(proof/'sources-before.json').read_bytes();expected=json.loads(before)
def inventory(p):
 assert not any(q.is_symlink()for q in p.rglob('*'))
 return{q.relative_to(p).as_posix():sha(q.read_bytes())for q in p.rglob('*')if q.is_file()}
assert len(expected)==38 and inventory(Path(cfg['source_root']))==inventory(proof/'execution')==expected
assert (proof/'sources-after.json').read_bytes()==(proof/'execution-after.json').read_bytes()==before
assert sha(before)==pre['source_manifest_sha256'] and sha((proof/'sources.tar.gz').read_bytes())==pre['source_archive_sha256']
with tarfile.open(proof/'sources.tar.gz')as tf:
 members=tf.getmembers();assert len(members)==len({m.name for m in members})==38
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==expected
xml=ET.parse(proof/'tests.xml').getroot();counts={name:len(list(xml.iter(tag)))for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=18,failures=0,errors=0,skipped=0)
ordered=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')];assert ordered==cfg['selected_xml_cases']
report=json.loads((proof/'report.json').read_bytes());audit=json.loads((proof/'worker-audit.json').read_bytes());live=json.loads((proof/'live-observation.json').read_bytes())
assert sha((proof/'report.json').read_bytes())=='fb736e9786a60c142fc523c77044e1d0f514bf55281128922bd6416630e37fa4'
assert report['passed']is True and report['tests']==counts and report['sources_unchanged']is True
assert report['child_returncode']==audit['pytest_exit_code']==0 and report['child_reaped']is True and report['child_group_absent']is True
assert report['selected_nodes']==audit['collected_nodes']==cfg['selected_nodes']and report['selected_xml_cases']==ordered
assert audit['denied']==audit['loaded_backend']==[] and audit['numpy_imported']is False and audit['affinity']==[112,113,114,115]
assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
assert audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']
for n in('actual_execution_authorized','model_execution','encrypted_execution','actual_key_generation','actual_client_role_execution','actual_proofs_present','full_scientific_admission_tested'):assert report[n]is False
def absent(pid):
 assert type(pid)is int and pid>0
 for fn in(os.kill,os.killpg):
  try:fn(pid,0)
  except ProcessLookupError:pass
  else:raise AssertionError('owned PID/group remains')
assert live['outer_pid']==live['outer_pgid'];absent(live['outer_pid']);absent(live['worker_pid'])
metadata=json.loads((proof/'owned-pytest-metadata.json').read_bytes());assert len(metadata)==report['retained_owned_files']==89
regular={};aliases={}
for n,row in metadata.items():
 assert not Path(n).is_absolute()and '..'not in Path(n).parts and n.startswith('pytest-owned/')
 p=proof/n
 if row['kind']=='regular':
  assert p.is_file()and not p.is_symlink();raw=p.read_bytes();assert len(raw)==row['bytes']and sha(raw)==row['sha256'];regular[n]=row
 else:
  assert row['kind']=='owned_pytest_alias'and p.is_symlink()and p.parent==proof/'pytest-owned'and p.name.endswith('current')
  assert p.lstat().st_uid==os.geteuid()and os.readlink(p)==row['target']
  target=(proof/row['resolved_target']);assert target.resolve()==p.resolve()==target and target.parent==proof/'pytest-owned'and target.is_dir()and not target.is_symlink()
  aliases[n]=row
assert len(regular)==72 and len(aliases)==17 and not any((proof/'tmp').iterdir())
records=json.loads((proof/'test-child-terminals.json').read_bytes());assert records==report['child_records']and len(records)==report['early_parent_records_validated']==18
expected_cases=[n.split('[',1)[1][:-1]for n in cfg['selected_nodes']];pids=[]
for case,row in zip(expected_cases,records,strict=True):
 parent=row['parent_terminal'];terminal=row['terminal'];result=row['result'];pid=result['pid'];pids.append(pid)
 assert type(pid)is int and pid>0 and result['pgid']==parent['pid']==parent['owned_session_pgid']==terminal['pid']==terminal['child_reported_pgid']==pid
 assert parent['scenario']==terminal['scenario']==result['scenario']==case
 assert parent['exit_code']==terminal['exit_code']==0 and parent['reaped']is terminal['reaped']is True and parent['group_absent']is terminal['group_absent']is True
 assert parent['observed_pgid']==terminal['observed_pgid']and parent['observed_pgid']in(None,pid)
 assert parent['child_result_read']is False and result['assertions_completed']is True and result['original_validator_identities_restored']is True
 assert result['backend_denials']==[]and result['backend_imported']is False and result['scientific_admission_tested']is False and result['actual_proof_instances']is False
 assert result['recording_full_validation_seam']is True and result['graph_type_count']==13
 for field,value in(('parent_terminal_path',parent),('terminal_path',terminal),('result_path',result)):
  n=row[field];assert n in regular and json.loads((proof/n).read_bytes())==value
 assert row['log_path']in regular and sha((proof/row['log_path']).read_bytes())==row['log_sha256']
 assert sha((proof/row['parent_terminal_path']).read_bytes())==row['parent_terminal_sha256'];absent(pid)
assert len(set(pids))==18
owner=dict(schema='snapshot-root-direct-owner-v2',session_id=91616,launch_tool_chunk='e197dd',terminal_tool_chunk='70406f',exit_code=0,direct_session_reaped=True,independent_live_observation=False,late_observation_tool='7c0404',healthy_execution_deadline=False,automatic_retry=False)
owner_path=root/'direct-owner.json';owner_path.write_bytes(pretty(owner))
names=('report.json','worker-audit.json','tests.xml','tests.log','live-observation.json','sources-before.json','sources-after.json','execution-after.json','sources.tar.gz','executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json','handoff.json','owned-pytest-metadata.json','test-child-terminals.json')
evidence={n:sha((proof/n).read_bytes())for n in names}
record=dict(schema='snapshot-root-terminal-review-v2',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,evidence_verified=True,counts=counts,source_count=38,runtime_files=28,source_current_execution_before_after_archive_equal=True,root_preexecution_review_sha256=sha(pre_raw),root_run_grant_sha256=sha(grant_raw),direct_owner_receipt_sha256=sha(owner_path.read_bytes()),evidence=evidence,elapsed_s=report['elapsed_s'],retained_regular_files=72,retained_alias_records=17,early_parent_records=18,positive_child_records=18,all_child_pids_and_groups_absent=True,outer_worker_pids_and_groups_absent=True,independent_live_observation=False,scope=cfg['scope'],full_scientific_admission_tested=False,backend_imported=False,guard_integrated_into_actual_server=False,script_sha256=sha(Path(__file__).read_bytes()))
out=root/'terminal-review.json';out.write_bytes(pretty(record));print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),report_sha256=evidence['report.json'],passed=True,tests=18,regular_files=72,aliases=17)))
