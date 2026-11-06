from pathlib import Path
import ast,hashlib,json,tarfile,datetime
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda x:(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
pretty=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
source=Path('/tmp/qwen-keygen-actual-runtime-tests-20261104')
proof=Path('/tmp/qwen-keygen-actual-focused-proof-20261104/preparation-v1')
review=Path('/tmp/qwen-key-generation-actual-root-review-20261104')
configraw=(proof/'run_config.json').read_bytes();config=json.loads(configraw)
assert sha(configraw)=='fc96c6011ef2a04cc3101fcfde685e08a503078e12a6f876021e8ce6eb409b99'
raw=(proof/'sources-before.json').read_bytes();expected=json.loads(raw)
assert sha(raw)=='59d8d58cd42d6beb8265b632b9c08277de2d913f5f0190364ca2d40689e3d8d3'
assert compact(expected)==raw and len(expected)==97
for root in (source,proof/'execution'):
 assert not any(p.is_symlink() for p in root.rglob('*'))
 assert {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}==expected
archive=proof/'sources.tar.gz'
assert sha(archive.read_bytes())=='11a7c027a38452239fd164520e0789423054147f01aae38953cbfc54c576999f'
with tarfile.open(archive)as tar:
 members=tar.getmembers();assert len(members)==len({m.name for m in members})==97
 assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
parentraw=(source/'parent85_sources.json').read_bytes();parent=json.loads(parentraw)
assert sha(parentraw)=='28782d1927a4334c778dc8429d92637776bfe6a97a6a42e0d25f99cc82f35699'
assert len(parent)==85 and {n[8:]:v for n,v in expected.items()if n.startswith('runtime/')}==parent
original=Path('/tmp/qwen-keygen-actual-runtime-v2-20261104')
assert {p.relative_to(original).as_posix():sha(p.read_bytes())for p in original.rglob('*')if p.is_file()}==parent
nodes=[];counts={}
for name in config['selected_files']:
 tree=ast.parse((source/name).read_bytes());before=len(nodes)
 for fn in tree.body:
  if not isinstance(fn,ast.FunctionDef)or not fn.name.startswith('test_'):continue
  params=[d for d in fn.decorator_list if isinstance(d,ast.Call)and isinstance(d.func,ast.Attribute)and d.func.attr=='parametrize']
  assert len(params)<=1
  suffixes=['']
  if params:
   values=ast.literal_eval(params[0].args[1]);assert all(type(x)is str for x in values)
   assert not params[0].keywords
   suffixes=['['+x+']'for x in values]
  nodes.extend(name+'::'+fn.name+s for s in suffixes)
 counts[name]=len(nodes)-before
assert len(nodes)==61 and nodes==config['selected_nodes']
xml=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in nodes]
assert xml==config['selected_xml_cases']
assert sha(pretty(nodes))==config['selected_nodes_sha256']and sha(pretty(xml))==config['selected_xml_cases_sha256']
assert config['cpu_affinity']==[112,113,114,115]and config['runtime_source_expectations']==config['runtime_module_origins']=={}
assert 'numpy'in config['forbidden_roots']and config['genuine_backend_imports_authorized']is False
pins={'executed_driver.py':'27f3b76b22afbd956092d4d84f443da301cc237e2fe50172a69374660539e4eb','worker_driver.py':'9d2d36620f356dbeb9dca7ee44137a88051acd79e539cba31120a2e7586efc9e','handoff.json':'d212a3e7221f78d86f37ca1bdd0a7130ef334b50a2f6a84eae1fb1d767e2b1f3','launch-proposal.json':'4655d145a2a5a9dbced85dc7099d08b02996d5290d7a8663c78e8b504e898338'}
for n,h in pins.items():assert sha((proof/n).read_bytes())==h
for n in ('executed_driver.py','worker_driver.py'):ast.parse((proof/n).read_bytes())
peer=Path('/tmp/qwen-keygen-runtime-focused-peer-review-20261104/review.json');assert sha(peer.read_bytes())=='2497b7eb9c5582ff53b2ec5920133fae4928e3e9b1c80597aaf2e99b831d916d'
pv=json.loads(peer.read_bytes());assert pv['passed']is True and pv['source_hashes']==expected and pv['static_selected_nodes']==nodes
result=dict(schema='qwen-keygen97-root-preexecution-source-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,source_count=97,nested85_exact=True,static_count=61,per_file=counts,source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive.read_bytes()),config_sha256=sha(configraw),harness_pins=pins,source_peer_review_sha256=sha(peer.read_bytes()),source_only=True,execution_authorized=False,temporary_module_name_doubles='Three explicitly reviewed run-flow recording cases, removed before guard completion.',scope='Source/current/execution/archive and ordered AST61 verified. Root read full harness and entry cases; full focused test review delegated to pinned peer. No imports, collection, test execution, real backend, key/model/CT bodies or capacity.',review_script_sha256=sha(Path(__file__).read_bytes()))
path=review/'keygen97-preexecution-review.json'
with path.open('xb')as f:f.write(pretty(result))
print(json.dumps(dict(path=str(path),sha256=sha(path.read_bytes()),passed=True,static_count=61)))
