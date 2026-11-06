from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
p=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v1');o=Path('/tmp/qwen-server-snapshot-integration-root-test-review-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest();canon=lambda v:(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
c=json.loads((p/'run_config.json').read_bytes());m=json.loads((p/'sources-before.json').read_bytes())
for root in (Path(c['source_root']),p/'execution'):
 assert not any(f.is_symlink()for f in root.rglob('*'))
 assert {f.relative_to(root).as_posix():sha(f.read_bytes())for f in root.rglob('*')if f.is_file()}==m
for n in ('sources-after.json','execution-after.json'):assert json.loads((p/n).read_bytes())==m
with tarfile.open(p/'sources.tar.gz')as t:
 members=t.getmembers();assert len(members)==404==len({x.name for x in members}) and all(x.isfile()for x in members)
 assert {x.name:sha(t.extractfile(x).read())for x in members}==m
assert sha((p/'sources.tar.gz').read_bytes())==c['source_archive_sha256']
x=ET.parse(p/'tests.xml').getroot();rows=list(x.iter('testcase'))
assert [{'classname':n.attrib['classname'],'name':n.attrib['name']}for n in rows]==c['selected_xml_cases']
failed=[n for n in rows if n.find('failure')is not None];passed=[n for n in rows if len(n)==0]
assert len(rows)==36 and len(failed)==25 and len(passed)==11 and not list(x.iter('error'))and not list(x.iter('skipped'))
audit=json.loads((p/'worker-audit.json').read_bytes());live=json.loads((p/'live-observation.json').read_bytes())
assert audit['pytest_exit_code']==1 and audit['denied']==audit['loaded_backend']==[] and audit['numpy_imported']is False
assert audit['collected_nodes']==c['selected_nodes'] and audit['runtime_sources_before']==audit['runtime_sources_after']==c['runtime_source_expectations']
assert audit['pid']==live['worker_pid']==646144 and live['outer_pid']==646142
base=p/'pytest-owned';directories=[f for f in base.iterdir()if f.is_dir()and not f.is_symlink()];assert len(directories)==36
cases={};fail_names={n.attrib['name'].split('[',1)[1][:-1]for n in failed}
for d in directories:
 early=json.loads((d/'parent-terminal.json').read_bytes());name=early['scenario'];assert name not in cases
 assert type(early['pid'])is int and early['pid']>0 and early['owned_session_pgid']==early['pid']
 assert early['reaped']is True and early['group_absent']is True and early['child_result_read']is False
 assert early['observed_pgid']in(None,early['pid'])
 raw=(d/'child.log').read_bytes()
 if name in fail_names:
  assert early['exit_code']==1 and not(d/'result.json').exists()and not(d/'child-terminal.json').exists()
  assert b'capture_runtime' in raw and b'self.full_boundary,self.capacity' in raw and b'TypeError: original source-fixed Python callback required' in raw
 else:
  assert early['exit_code']==0
  result=json.loads((d/'result.json').read_bytes());terminal=json.loads((d/'child-terminal.json').read_bytes())
  assert result['scenario']==terminal['scenario']==name and result['pid']==result['pgid']==terminal['pid']==early['pid']
  assert terminal['reaped']is True and terminal['group_absent']is True and terminal['exit_code']==0
  for n in ('assertions_completed','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','runtime_sources_unchanged'):assert result[n]is True
  for n in ('genuine_cli_bootstrap','genuine_backend_imported','scientific_admission','capacity_measurement','model_key_ciphertext_body_reads','actual_server_execution'):assert result[n]is False
 cases[name]={'pid':early['pid'],'exit_code':early['exit_code'],'early_record_sha256':sha((d/'parent-terminal.json').read_bytes()),'log_sha256':sha(raw),'path':str(d)}
assert set(cases)=={n.split('[',1)[1][:-1]for n in c['selected_nodes']}
absence=[]
for pid in [646142,646144]+[v['pid']for v in cases.values()]:
 for group in (False,True):
  try:(os.killpg if group else os.kill)(pid,0)
  except ProcessLookupError:pass
  else:raise AssertionError((pid,group,'still present'))
 absence.append(pid)
assert not(p/'report.json').exists()
owner={'session':31389,'launch_chunk':'4c43f9','terminal_chunk':'915588','exit_code':1,'root_owned_direct_completion_and_reap':True,'outer_pid':646142,'worker_pid':646144,'independent_live_observation_chunk':'300315','independent_live_observation':json.loads((o/'live-observation.json').read_bytes())}
(o/'direct-owner-failed-v1.json').write_bytes(canon(owner))
r={'schema':'qwen-snapshot-integration-failed-terminal-root-review-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outcome':'Failed synthetic integration run; preserve intact, no complete PASS','source_count':404,'runtime_count':391,'current_execution_before_after_archive_exact':True,'static_XML_and_collected36_exact':True,'tests':{'passed':11,'failed':25,'errors':0,'skipped':0},'pytest_elapsed_s':34.09,'failure_scope':'All25 direct cases reject bound fixture callbacks before runtime issuer completion/scenario assertions. Original391 runtime FunctionType contract is retained; six recorded process and five terminal-function cases passed.','harness_failure':'Success-only144-artifact inspector rejects incomplete child positive/results; no report.json produced','early_parent_records':36,'positive_child_records':11,'all38_PIDs_and_groups_absent':absence,'case_records':cases,'worker_audit_sha256':sha((p/'worker-audit.json').read_bytes()),'tests_xml_sha256':sha((p/'tests.xml').read_bytes()),'tests_log_sha256':sha((p/'tests.log').read_bytes()),'run_grant_sha256':sha((o/'run-grant.json').read_bytes()),'direct_owner_sha256':sha((o/'direct-owner-failed-v1.json').read_bytes()),'source_manifest_sha256':c['source_manifest_sha256'],'source_archive_sha256':c['source_archive_sha256'],'actual_or_scientific_or_capacity_or_HE_authority':False,'source_or_evidence_edits':False,'rerun':False}
b=canon(r);(o/'failed-terminal-v1.json').write_bytes(b)
print(json.dumps({'receipt':str(o/'failed-terminal-v1.json'),'sha256':sha(b),'passed':11,'failed':25,'early_records':36,'absent_pids_and_groups':len(absence)}))
