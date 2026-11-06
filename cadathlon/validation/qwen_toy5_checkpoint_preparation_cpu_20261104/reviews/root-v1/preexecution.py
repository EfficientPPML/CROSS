from pathlib import Path
import ast, datetime, hashlib, json, tarfile
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
proof=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v1')
root=Path('/tmp/qwen-toy5-checkpoint-root-review-20261104')
pins={'handoff.json':'f43116c2b628131578c2a1bf38d0770d67b4e8acaaaf943bc22853dfeb74fc4e','executed_driver.py':'d32615321f455240e3bbd599a29b64c1ac5395f1c02e67f45d6358269b662241','worker_driver.py':'d2c04361908ba05b84b4b69369dd1cef94091def36454d6863dfe077b67bd8e1','run_config.json':'c6e5dcc8fe8118f7a2b7751af9ef3cbf1b4b98e4e4b5397f9a25720fe584a14b','launch-proposal.json':'4298e2f95ada6936a9960bdc6d12e278ed86c67f0e77d2969ce86c973bd285a1'}
for n,h in pins.items():assert sha((proof/n).read_bytes())==h,n
handoff=json.loads((proof/'handoff.json').read_bytes())
for n,h in handoff['files'].items():assert sha((proof/n).read_bytes())==h,n
cfg=json.loads((proof/'run_config.json').read_bytes());source=Path(cfg['source_root'])
def inventory(p):
 assert p.is_absolute() and p.resolve()==p and not any(q.is_symlink() for q in p.rglob('*'))
 return {q.relative_to(p).as_posix():sha(q.read_bytes()) for q in p.rglob('*') if q.is_file()}
raw=(proof/'sources-before.json').read_bytes();mapping=json.loads(raw)
assert sha(raw)==cfg['source_manifest_sha256']=='91a4bc8c1715f8900edb49e97cc9c2be696b30a5037b5efd4a136ac088664339'
assert raw==compact(mapping) and len(mapping)==261 and inventory(source)==inventory(proof/'execution')==mapping
archive=proof/'sources.tar.gz';assert sha(archive.read_bytes())==cfg['source_archive_sha256']=='6d890c097757327e1f4de6a624a1496dd40e466703210e617975bbf64e26e21a'
with tarfile.open(archive)as tf:
 members=tf.getmembers();assert len(members)==len({m.name for m in members})==261
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==mapping
parent_raw=(source/'runtime_sources.json').read_bytes();parent=json.loads(parent_raw)
assert sha(parent_raw)=='71c960c8d82f7d617ccb8167ac5bfbc526d5cdf54c2458b954ebfdfbb84f1ed3' and len(parent)==250
assert {n[8:]:h for n,h in mapping.items()if n.startswith('runtime/')}==parent==inventory(Path('/tmp/qwen-toy5-checkpoint-component-20261104'))
nodes=[];counts={}
for name in cfg['selected_files']:
 before=len(nodes);tree=ast.parse((source/name).read_bytes())
 for obj in tree.body:
  if not isinstance(obj,ast.FunctionDef) or not obj.name.startswith('test_'):continue
  params=[d for d in obj.decorator_list if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='parametrize']
  assert len(params)<=1;suffixes=['']
  if params:
   p=params[0];values=ast.literal_eval(p.args[1]);kw={k.arg:ast.literal_eval(k.value)for k in p.keywords}
   assert set(kw)=={'ids'} and len(kw['ids'])==len(values) and all(type(v)is str for v in kw['ids'])
   suffixes=['['+v+']'for v in kw['ids']]
  nodes.extend(name+'::'+obj.name+s for s in suffixes)
 counts[name]=len(nodes)-before
selection=json.loads((source/'selected_tests.json').read_bytes())
assert len(nodes)==48 and nodes==cfg['selected_nodes']==selection['ordered_nodes'] and counts==selection['counts']
xml=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in nodes]
assert xml==cfg['selected_xml_cases']==selection['ordered_xml_cases']
assert sha(pretty(nodes))==cfg['selected_nodes_sha256'] and sha(pretty(xml))==cfg['selected_xml_cases_sha256']
npins=json.loads((source/'numpy_pins.json').read_bytes())
assert sha((source/'numpy_pins.json').read_bytes())==cfg['numpy_pins_sha256']=='a696bcc87f4c3be479c56ade4046e8b50e49f5c2100d0e7ca5f1c1929b35972d'
assert npins['sources']==cfg['runtime_source_expectations'] and npins['required_origins']==cfg['runtime_module_origins']
assert len(npins['sources'])==4 and set(npins['required_origins'])=={'numpy'}
for n,h in npins['sources'].items():assert sha(Path(n).read_bytes())==h
assert cfg['cpu_affinity']==[108,109,110,111] and cfg['actual_execution_authorized']is False and cfg['synthetic_only']is True
for n in mapping:
 if n.endswith('.py'):ast.parse((source/n).read_bytes())
for n in ('executed_driver.py','worker_driver.py'):ast.parse((proof/n).read_bytes())
def ownerblock(path):
 tree=ast.parse(path.read_bytes())
 return ast.dump(next(n for n in ast.walk(tree)if isinstance(n,ast.Try)and n.body and isinstance(n.body[0],ast.Assign)and any(isinstance(x,ast.Name)and x.id=='child_pgid'for x in n.body[0].targets)),include_attributes=False)
assert ownerblock(proof/'executed_driver.py')==ownerblock(Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v2/executed_driver.py'))
absent=['tests.log','tests.xml','report.json','worker-audit.json','live-observation.json','sources-after.json','execution-after.json','pytest-owned']
assert not any((proof/n).exists()for n in absent)
proposal=json.loads((proof/'launch-proposal.json').read_bytes())
assert proposal['authorized']is False and proposal['automatic_retry']is False and proposal['healthy_deadline']is False
peer=Path('/tmp/qwen-toy5-checkpoint-security-source-review-20261104/review.json')
assert sha(peer.read_bytes())=='ae43cafaa0ef5475c3bfc13402dbadd3ef1217e3297f8b46a35a6ba455f2ef93'
record=dict(schema='q5-checkpoint-root-preexecution-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,source_count=261,parent_count=250,parent_exact=True,static_cases=48,counts=counts,source_current_execution_archive_equal=True,source_manifest_sha256=cfg['source_manifest_sha256'],source_archive_sha256=cfg['source_archive_sha256'],harness_pins=pins,static_nodes=nodes,prior_outputs_absent=absent,source_only=True,candidate_imports=False,execution_authorized=False,owned_Popen_block_exact=True,numpy_four_sources_exact=True,security_peer_sha256=sha(peer.read_bytes()),scope=cfg['scope'],script_sha256=sha(Path(__file__).read_bytes()))
root.mkdir(exist_ok=False);path=root/'preexecution-review.json';path.write_bytes(pretty(record))
print(json.dumps(dict(path=str(path),sha256=sha(path.read_bytes()),passed=True,source_count=261,static_cases=48)))
