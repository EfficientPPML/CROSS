from pathlib import Path
import datetime,hashlib,json,os,stat,tarfile,xml.etree.ElementTree as ET
p=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v2');o=Path('/tmp/qwen-server-snapshot-integration-root-test-review-20261104');sha=lambda b:hashlib.sha256(b).hexdigest();canon=lambda v:(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
c=json.loads((p/'run_config.json').read_bytes());h=json.loads((p/'handoff.json').read_bytes());r=json.loads((p/'report.json').read_bytes());m=json.loads((p/'sources-before.json').read_bytes())
assert sha((p/'report.json').read_bytes())=='98c2642ab0cdc5dd5e7871c3d55bd6b8ff7b1da3e7b76e6e3de29db06f56afbb'
for n,v in h['files'].items():assert sha((p/n).read_bytes())==v
for root in (Path(c['source_root']),p/'execution'):
 assert not any(f.is_symlink()for f in root.rglob('*'))
 assert {f.relative_to(root).as_posix():sha(f.read_bytes())for f in root.rglob('*')if f.is_file()}==m and len(m)==405
for n in ('sources-after.json','execution-after.json'):assert json.loads((p/n).read_bytes())==m
with tarfile.open(p/'sources.tar.gz')as t:
 members=t.getmembers();assert len(members)==405==len({x.name for x in members})and all(x.isfile()for x in members)
 assert {x.name:sha(t.extractfile(x).read())for x in members}==m
assert sha((p/'sources.tar.gz').read_bytes())==r['source_archive_sha256']==c['source_archive_sha256']
assert r['passed']is True and r['tests']=={'tests':36,'failures':0,'errors':0,'skipped':0}
assert r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True and r['sources_unchanged']is True
for n in ('actual_client_role_execution','actual_device_initialized','actual_execution_authorized','actual_key_generation','actual_proofs_present','backend_context_construction','encrypted_execution','full_scientific_admission_tested','genuine_capacity_measured','genuine_cli_bootstrap','model_execution','numpy_imported','observed_child_module_origin_table'):assert r[n]is False
x=ET.parse(p/'tests.xml').getroot();nodes=list(x.iter('testcase'));assert len(nodes)==36 and all(len(n)==0 for n in nodes)
assert [{'classname':n.attrib['classname'],'name':n.attrib['name']}for n in nodes]==c['selected_xml_cases']==r['selected_xml_cases']
a=json.loads((p/'worker-audit.json').read_bytes());assert a['collected_nodes']==c['selected_nodes']==r['selected_nodes']
assert a['denied']==a['loaded_backend']==[]and a['numpy_imported']is False and a['pytest_exit_code']==0 and a['affinity']==[112,113,114,115]
assert a['runtime_sources_before']==a['runtime_sources_after']==c['runtime_source_expectations']and len(a['runtime_sources_before'])==391
metadata=json.loads((p/'owned-pytest-metadata.json').read_bytes());regular={n:v for n,v in metadata.items()if v['kind']=='regular'};aliases={n:v for n,v in metadata.items()if v['kind']=='owned_pytest_alias'}
assert len(regular)==150 and len(aliases)==15 and len(metadata)==r['retained_owned_files']==165
for n,v in regular.items():
 f=p/n;assert f.is_file()and not f.is_symlink()and f.stat().st_size==v['bytes']and sha(f.read_bytes())==v['sha256']
for n,v in aliases.items():
 f=p/n;assert f.parent==p/'pytest-owned'and f.is_symlink()and f.lstat().st_uid==os.geteuid()
 raw=os.readlink(f);assert raw==v['target']and '..'not in Path(raw).parts
 target=f.parent/raw;assert target==target.resolve(strict=True)and target==p/v['resolved_target']and target.parent==p/'pytest-owned'and target.is_dir()and not target.is_symlink()
actual=set()
for leaf in (p/'tmp',p/'pytest-owned'):
 for base,dirs,files in os.walk(leaf,followlinks=False):
  for name in list(dirs):
   f=Path(base)/name
   if f.is_symlink():actual.add(f.relative_to(p).as_posix());dirs.remove(name)
  for name in files:actual.add((Path(base)/name).relative_to(p).as_posix())
assert actual==set(metadata)
records=json.loads((p/'test-child-terminals.json').read_bytes());assert records==r['child_records']and len(records)==36
pids=[];scenarios=[]
for row in records:
 result=json.loads((p/row['result_path']).read_bytes());terminal=json.loads((p/row['terminal_path']).read_bytes());parent=json.loads((p/row['parent_terminal_path']).read_bytes())
 assert row['result']==result and row['terminal']==terminal and row['parent_terminal']==parent
 pid=result['pid'];assert type(pid)is int and pid==result['pgid']==terminal['pid']==terminal['child_reported_pgid']==parent['pid']==parent['owned_session_pgid']
 assert result['scenario']==terminal['scenario']==parent['scenario'];scenarios.append(result['scenario']);pids.append(pid)
 for rr in (parent,terminal):assert rr['exit_code']==0 and rr['reaped']is True and rr['group_absent']is True and rr['observed_pgid']in(None,pid)
 assert parent['child_result_read']is False
 assert sha((p/row['parent_terminal_path']).read_bytes())==row['parent_terminal_sha256']and sha((p/row['log_path']).read_bytes())==row['log_sha256']
 for n in ('assertions_completed','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','runtime_sources_unchanged','definition_only_entry'):assert result[n]is True
 for n in ('genuine_cli_bootstrap','genuine_backend_imported','scientific_admission','capacity_measurement','model_key_ciphertext_body_reads','actual_server_execution'):assert result[n]is False
 assert result['backend_denials']==[]and result['source_count']==391
assert len(set(pids))==36 and scenarios==[n.split('[',1)[1][:-1]for n in c['selected_nodes']]
fixtures=[n for n in regular if n.endswith('/tiny-public-proof.json')];assert len(fixtures)==6 and all(regular[n]['bytes']<=1024 for n in fixtures)
for row in records:
 f=(Path(row['result_path']).parent/'tiny-public-proof.json').as_posix();assert(f in fixtures)==row['result']['scenario'].startswith('process_')
live=json.loads((p/'live-observation.json').read_bytes());assert live['outer_pid']==664985 and live['worker_pid']==a['pid']==665002
allp=[664985,665002]+pids
for pid in allp:
 for group in (False,True):
  try:(os.killpg if group else os.kill)(pid,0)
  except ProcessLookupError:pass
  else:raise AssertionError((pid,group,'present'))
owner={'session':46330,'launch_chunk':'5005f4','terminal_chunk':'98ea2c','exit_code':0,'root_owned_direct_completion_and_reap':True,'outer_pid':664985,'worker_pid':665002,'independent_live_chunk':'fbfe6b','independent_live_observation':json.loads((o/'live-v2-observation.json').read_bytes())};(o/'direct-owner-v2.json').write_bytes(canon(owner))
review={'schema':'qwen-snapshot-integration-terminal-root-review-v2','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'report_sha256':sha((p/'report.json').read_bytes()),'source_count':405,'runtime_count':391,'current_execution_before_after_archive_exact':True,'XML_order_collection_exact36_PASS':True,'elapsed_s':r['elapsed_s'],'early_parent_records':36,'positive_child_records':36,'regular_child_artifacts':144,'owned_tiny_public_fixtures':6,'observed_owned_aliases':15,'all38_PIDs_and_groups_absent':allp,'direct_owner_sha256':sha((o/'direct-owner-v2.json').read_bytes()),'grant_sha256':sha((o/'run-v2-grant.json').read_bytes()),'failed_v1_receipt_sha256':sha((o/'failed-terminal-v1.json').read_bytes()),'source_manifest_sha256':c['source_manifest_sha256'],'source_archive_sha256':c['source_archive_sha256'],'scope':'Synthetic retained-runtime integration only; explicit recording seams and definition-only entry. No genuine native, capacity, scientific, model, HE, CLI-bootstrap or timing improvement claim.','rerun':False,'prior25FAIL11PASS_preserved':True}
b=canon(review);(o/'terminal-v2-review.json').write_bytes(b);print(json.dumps({'receipt':str(o/'terminal-v2-review.json'),'sha256':sha(b),'passed':True,'cases':36,'absent_pid_group_pairs':len(allp)}))
