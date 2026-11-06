from pathlib import Path
import ast,datetime,hashlib,json,tarfile
p=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v1')
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
h=json.loads((p/'handoff.json').read_bytes())
assert sha((p/'handoff.json').read_bytes())=='ad7b69f004da28a838094861bd384bc8759556951991252bff0f6691a86dedfe'
for n,v in h['files'].items():assert sha((p/n).read_bytes())==v,n
assert h['driver_sha256']=='77862d27858c55aee305223507e7feadf3d20376511d194f0d41b72dadb23057'
assert h['worker_sha256']=='9ee25a101d4611e17b376d9b4729dae44bcf9f804d34a72ae340096337d65361'
assert h['config_sha256']=='d804adc4a8b4f2d94bc21ec95976b5a3f43c2a003b0b03c7734e7125ad701090'
c=json.loads((p/'run_config.json').read_bytes());mraw=(p/'sources-before.json').read_bytes();m=json.loads(mraw)
assert mraw==compact(m) and sha(mraw)==c['source_manifest_sha256']=='9fa73f2f59e6bef71a2de475a09e0aaccb25d56375f7a06452cac7b75e4afc1c'
assert len(m)==404 and c['source_count']==404 and c['runtime_count']==391
assert c['cpu_affinity']==[112,113,114,115] and c['runtime_module_origins']=={}
assert c['synthetic_only']is True and c['actual_execution_authorized']is False
for root in (Path(c['source_root']),p/'execution'):
 observed={}
 for file in root.rglob('*'):
  assert not file.is_symlink()
  if file.is_file():observed[file.relative_to(root).as_posix()]=sha(file.read_bytes())
 assert observed==m
assert sha((p/'sources.tar.gz').read_bytes())==c['source_archive_sha256']=='bbc4c0bd3ce4bc6a1cd0efa10a7afc799e280887132869b846849fee10b2e713'
with tarfile.open(p/'sources.tar.gz')as t:
 members=t.getmembers();assert len(members)==404==len({x.name for x in members})
 assert all(x.isfile() and not Path(x.name).is_absolute() and '..'not in Path(x.name).parts for x in members)
 assert {x.name:sha(t.extractfile(x).read())for x in members}==m
runtime=json.loads((p/'execution/runtime_sources.json').read_bytes());assert runtime==c['runtime_source_expectations'] and len(runtime)==391
assert all(m['runtime/'+n]==v for n,v in runtime.items())
selection=json.loads((p/'execution/selected_tests.json').read_bytes())
assert c['selected_nodes']==selection['static_nodeids'] and c['selected_files']==['test_snapshot_integration.py'] and len(c['selected_nodes'])==36
assert sha(canon(c['selected_nodes']))==c['selected_nodes_sha256']
assert sha(canon(c['selected_xml_cases']))==c['selected_xml_cases_sha256']
assert c['selected_xml_cases']==[{'classname':n.split('.py::')[0],'name':n.split('::')[1]}for n in c['selected_nodes']]
assert c['selection_sha256']==m['selected_tests.json'] and c['seams_sha256']==m['SEAMS.json'] and c['entry_fixture_provenance_sha256']==m['entry_fixture_provenance.json']
assert c['pytest_ini_sha256']==m['pytest.ini']
for n in ('executed_driver.py','worker_driver.py'):ast.parse((p/n).read_bytes(),filename=str(p/n))
driver=ast.parse((p/'executed_driver.py').read_bytes());values={t.id:ast.literal_eval(n.value) for n in driver.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name) and t.id in ('CONFIG_SHA256','WORKER_SHA256')}
assert values=={'CONFIG_SHA256':h['config_sha256'],'WORKER_SHA256':h['worker_sha256']}
def ownership(tree):
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
 withnode=next(n for n in fn.body if isinstance(n,ast.With))
 return next(n for n in withnode.body if isinstance(n,ast.Try))
old=ast.parse((p/'before_executed_driver.py').read_bytes())
assert ast.dump(ownership(old),include_attributes=False)==ast.dump(ownership(driver),include_attributes=False)
proposal=json.loads((p/'launch-proposal.json').read_bytes());assert proposal['argv']==h['argv']
assert proposal['launch_authorized']is False
for n in ('report.json','tests.log','tests.xml','tmp','pytest-owned','worker-audit.json','live-observation.json'):
 assert not(p/n).exists() and not(p/n).is_symlink(),n
out=Path('/tmp/qwen-server-snapshot-integration-root-test-review-20261104')
r={'schema':'qwen-snapshot-integration-root-harness-source-review-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'handoff_sha256':sha((p/'handoff.json').read_bytes()),'driver_sha256':h['driver_sha256'],'worker_sha256':h['worker_sha256'],'config_sha256':h['config_sha256'],'proposal_sha256':sha((p/'launch-proposal.json').read_bytes()),'exact404_current_execution_archive':True,'runtime391_exact':True,'static36_selection_and_XML':True,'source_map_compact_canonical':True,'all_handoff_files_exact':True,'original_device50_immediate_Popen_ownership_AST_exact':True,'no_result_or_temp_artifacts':True,'expected_artifacts':{'child_evidence':144,'owned_tiny_public_fixtures':6,'aliases':'observed, no fixed assumed count'},'cpu_affinity':[112,113,114,115],'root_reads':['5ea526','ae437f','dee719','714d35','d76b0b'],'scope':'36 synthetic integration cases, disclosed definition-only entry and source-owned numeric names; no actual native/scientific/capacity/HE/CLI bootstrap claims','findings':[],'status':'Source review passed; independent peer reviews and separate root launch grant remain','candidate_imports':False,'collection':False,'tests':False,'launch_authorized':False}
b=canon(r);(out/'preexecution-review.json').write_bytes(b)
print(json.dumps({'receipt':str(out/'preexecution-review.json'),'sha256':sha(b)}))
