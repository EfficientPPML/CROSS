from pathlib import Path
import ast, datetime, hashlib, json, tarfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()
compact=lambda value:(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
pretty=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
ROOT=Path('/tmp/qwen-proc-self-root-review-20261104')
CASES={
 'client24':dict(proof='/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v2',count=472,tests=24,cores=[108,109,110,111],nested_count=464,nested_map='runtime-sources.json',runtime='/tmp/qwen-actual-client-role-runtime-v2-20261104',prior='/tmp/qwen-actual-client-role-runtime-20261104',changed='qwen_client_dependencies.py',nested_sha='072d3900906ad0ad113cc21f7c52b660fcec394ddfec2ccc29787b02fa36590c',map='208049d311bc26bb8db8ce71c610c6c484dfbec064e1b26bbbe5f51483a1ec40',archive='9ed75e3c56eaf743a34f591d6978af69394de93a8ea806a26bb5fb9b98d4a617',config='c2bb4d4d172f6130992ee149c00c9b328d0a89901155a4bc59c2ab2f8ddd61eb',driver='0abf93eed934872faf79aa22590de829897baccd050ba1b523aaf9c2aefa55bb',worker='14b20e9830cafacf472774e4c4fbf10814e07a8ad93ec05224020b847e32e246',handoff='0b21aa92f7679de930b0f52ebb68a5b740624a037ca353759aa1f1264f64f0bb',proposal='673fa9208f72d8dd41e3ba8dac59ffbe90797ba2bbe9a62e00b8feb674d8a1b9'),
 'keygen32':dict(proof='/tmp/qwen-keygen-proc-self-focused-proof-20261104/preparation-v2',count=92,tests=32,cores=[112,113,114,115],nested_count=85,nested_map='runtime_sources.json',runtime='/tmp/qwen-keygen-actual-runtime-v3-20261104',prior='/tmp/qwen-keygen-actual-runtime-v2-20261104',changed='keygen_dependencies.py',nested_sha='01e5f8681ff5731d8d19d8f488b692bcf38025b83a0dbb3036b536a3d9f1726e',map='bb9bee8a343f45e65fdb344240bdf3b5f7fc66b9ef5167c67909b11e9d81d53d',archive='f6368d99e879e5e9e3629b05a12fc09c4748709ff3c393f7811e606303b9bc56',config='ea21fcc6fd3c4d6e72141fd8417218746014f77702d1306fd63d8abd3f9c2634',driver='4a36879a675ea674fdfe9781dbd8a81fbfba797d31d183b9360b7ae8a9478312',worker='e151f7706e792e01c4cddc10029ac03a10dd95e4417042edbd0b23e425d0a8f3',handoff='0dfbb394a1d86ed663992549e4e18ce5dabbcd995a1e7a2b3a4c028400476d08',proposal='9ac08d40c98af15266447cb99e70442de44a8258011d91624a119bbdf15082ad')}
def inventory(root):
 assert root.is_absolute() and root.resolve()==root
 assert not any(p.is_symlink()for p in root.rglob('*'))
 return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}
def selection(source,files):
 nodes=[]
 for name in files:
  tree=ast.parse((source/name).read_bytes());constants={}
  for obj in tree.body:
   if isinstance(obj,ast.Assign)and len(obj.targets)==1 and isinstance(obj.targets[0],ast.Name):
    try:constants[obj.targets[0].id]=ast.literal_eval(obj.value)
    except (ValueError,TypeError):pass
   if not isinstance(obj,ast.FunctionDef)or not obj.name.startswith('test_'):continue
   params=[d for d in obj.decorator_list if isinstance(d,ast.Call)and isinstance(d.func,ast.Attribute)and d.func.attr=='parametrize']
   assert len(params)<=1
   suffixes=['']
   if params:
    p=params[0];assert not p.keywords and ast.literal_eval(p.args[0])=='case'
    values=constants[p.args[1].id]if isinstance(p.args[1],ast.Name)else ast.literal_eval(p.args[1])
    assert all(type(x)is str for x in values)
    suffixes=['['+x+']'for x in values]
   nodes.extend(name+'::'+obj.name+x for x in suffixes)
 return nodes
def main():
 ROOT.mkdir(exist_ok=False)
 for label,v in CASES.items():
  proof=Path(v['proof']);raw=(proof/'run_config.json').read_bytes();assert sha(raw)==v['config'];cfg=json.loads(raw)
  source=Path(cfg['source_root']);mapping_raw=(proof/'sources-before.json').read_bytes();expected=json.loads(mapping_raw)
  assert sha(mapping_raw)==v['map']and compact(expected)==mapping_raw and len(expected)==v['count']
  assert inventory(source)==inventory(proof/'execution')==expected
  archive=proof/'sources.tar.gz';assert sha(archive.read_bytes())==v['archive']
  with tarfile.open(archive)as tf:
   members=tf.getmembers();assert len(members)==len({m.name for m in members})==v['count']
   assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
   assert {m.name:sha(tf.extractfile(m).read())for m in members}==expected
  parent_raw=(source/v['nested_map']).read_bytes();parent=json.loads(parent_raw)
  assert sha(parent_raw)==v['nested_sha']and compact(parent)==parent_raw and len(parent)==v['nested_count']
  assert {n[8:]:h for n,h in expected.items()if n.startswith('runtime/')}==parent==inventory(Path(v['runtime']))
  before=inventory(Path(v['prior']));assert set(before)==set(parent)
  changed=[n for n in parent if parent[n]!=before[n]];assert changed==[v['changed']]
  nodes=selection(source,cfg['selected_files']);assert len(nodes)==v['tests']and nodes==cfg['selected_nodes']
  xml=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in nodes]
  assert xml==cfg['selected_xml_cases']and sha(pretty(nodes))==cfg['selected_nodes_sha256']and sha(pretty(xml))==cfg['selected_xml_cases_sha256']
  assert cfg['cpu_affinity']==v['cores']and cfg['runtime_source_expectations']==cfg['runtime_module_origins']=={}
  assert cfg['actual_execution_authorized']is False and cfg['synthetic_only']is True and'numpy'in cfg['forbidden_roots']
  pins={'executed_driver.py':v['driver'],'worker_driver.py':v['worker'],'handoff.json':v['handoff'],'launch-proposal.json':v['proposal'],'run_config.json':v['config']}
  for n,h in pins.items():assert sha((proof/n).read_bytes())==h
  for n in ('executed_driver.py','worker_driver.py'):ast.parse((proof/n).read_bytes())
  for n in expected:
   if n.endswith('.py'):ast.parse((source/n).read_bytes())
  absent=('report.json','tests.log','tests.xml','live-observation.json','worker-audit.json','sources-after.json','execution-after.json','pytest-owned')
  assert not any((proof/n).exists()for n in absent)
  proposal=json.loads((proof/'launch-proposal.json').read_bytes());assert proposal['execution_authorized']is False and proposal['automatic_retry']is False and proposal['healthy_job_deadline']is False
  record=dict(schema='qwen-proc-self-root-preexecution-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,label=label,source_count=v['count'],nested_count=v['nested_count'],nested_runtime_exact=True,runtime_changed_files=changed,source_manifest_sha256=v['map'],source_archive_sha256=v['archive'],runtime_manifest_sha256=v['nested_sha'],harness_pins=pins,static_ordered_nodes=nodes,source_current_execution_archive_equal=True,prior_outputs_absent=list(absent),source_only=True,scope=cfg['scope'],execution_authorized=False,review_script_sha256=sha(Path(__file__).read_bytes()))
  path=ROOT/(label+'-preexecution-review.json');path.write_bytes(pretty(record))
  print(json.dumps(dict(label=label,path=str(path),sha256=sha(path.read_bytes()),passed=True,static_count=len(nodes))))
if __name__=='__main__':main()
