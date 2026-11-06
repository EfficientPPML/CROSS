"""Stdlib source inventory/AST capture only; never imports candidate code."""
from pathlib import Path
import ast,hashlib,json,tarfile,difflib
root=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-20261104')
out=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-handoff-20261104');out.mkdir(exist_ok=False)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(name,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();(out/name).write_bytes(raw);return sha(raw)
parent=json.loads((root/'parent250_sources.json').read_bytes());assert len(parent)==250
for name,pin in parent.items():
 assert sha((root/'parent'/name).read_bytes())==pin
 assert sha((Path('/tmp/qwen-toy5-checkpoint-component-20261104')/name).read_bytes())==pin
copies=json.loads((root/'copy_origins.json').read_bytes());assert len(copies)==339
for name,row in copies.items():assert sha((root/name).read_bytes())==row['sha256']==sha(Path(row['path']).read_bytes())
for name in ('q5_checkpoint_arithmetic.py','q5_checkpoint_contract.py','q5_actual_metadata_replay.py','q5_checkpoint_oracle.py','q5_checkpoint_entry.py'):
 assert (root/name).read_bytes()==(root/'parent'/name).read_bytes()
core=json.loads((root/'core_source_manifest.json').read_bytes());assert len(core)==34
assert sha((root/'core_source_manifest.json').read_bytes())=='11353173bbf8e2796813d0feb6f69fac308775c7b13cc164900170f32bc62088'
for name,pin in core.items():assert sha((root/name).read_bytes())==pin
sources={p.relative_to(root).as_posix():p for p in sorted(root.rglob('*'))if p.is_file()}
assert not any(p.is_symlink()for p in root.rglob('*'))
python_count=0
for name,p in sources.items():
 if p.suffix=='.py':ast.parse(p.read_bytes(),filename=name);python_count+=1
# Exact six original definition ASTs and byte sources.
old_selection=json.loads((root/'parent/source_selection.json').read_bytes());definitions={}
for name,row in old_selection['original_dependencies'].items():
 path=root/'common'/Path(row['path']).name;raw=path.read_bytes();assert sha(raw)==row['source_sha256']
 node=next(n for n in ast.parse(raw).body if getattr(n,'name',None)==name)
 pin=sha(ast.dump(node,include_attributes=False).encode());assert pin==row['definition_ast_sha256'];definitions[name]={'path':str(path),'source_sha256':sha(raw),'definition_ast_sha256':pin}
# Verify the original loader body after only declared admission/name substitution.
a=next(n for n in ast.parse((root/'q5_key_loader.py').read_bytes()).body if getattr(n,'name',None)=='_load_bound')
b=next(n for n in ast.parse(Path('/tmp/qwen-key-context-adapter-v2-20261104/qwen_key_loader.py').read_bytes()).body if getattr(n,'name',None)=='_load_for_recording')
a.name=b.name
assert isinstance(a.body[0],ast.Expr)and isinstance(a.body[0].value,ast.Call)
a.body[0]=b.body[0]
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
oldtoy=next(n for n in ast.parse((root/'reference/frozen/cadathlon/agent_pset/p4_qwen/src/reference.py').read_bytes()).body if getattr(n,'name',None)=='toy_weights')
newtoy=next(n for n in ast.parse((root/'q5_toy_weights.py').read_bytes()).body if getattr(n,'name',None)=='toy_weights')
assert ast.dump(oldtoy,include_attributes=False)==ast.dump(newtoy,include_attributes=False)
# All explicitly selected local from-import names exist syntactically.
selected=json.loads((root/'source_selection.json').read_bytes());mods={k:v for table in selected['role_modules'].values()for k,v in table.items()}
exports={}
for module,relative in mods.items():
 names=set()
 for n in ast.parse((root/relative).read_bytes()).body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):names.add(n.name)
  elif isinstance(n,(ast.Assign,ast.AnnAssign)):
   targets=n.targets if isinstance(n,ast.Assign)else[n.target]
   for target in targets:
    if isinstance(target,ast.Name):names.add(target.id)
  elif isinstance(n,(ast.Import,ast.ImportFrom)):
   names.update(a.asname or a.name.split('.')[0]for a in n.names)
 exports[module]=names
unresolved=[]
for p in root.glob('q5_*.py'):
 for n in ast.walk(ast.parse(p.read_bytes())):
  if isinstance(n,ast.ImportFrom)and n.module in exports:
   for alias in n.names:
    if alias.name!='*'and alias.name not in exports[n.module]:unresolved.append([p.name,n.module,alias.name])
assert not unresolved,unresolved
# Exact layer18 is independently preserved.
held=Path('/tmp/qwen-toy5-actual-key-io-layer-20261104');oldmap=json.loads(Path('/tmp/qwen-toy5-actual-key-io-layer-handoff-20261104/sources.json').read_bytes())
for name,pin in oldmap.items():assert sha((held/name).read_bytes())==pin
# Fresh finalized patch for all18 compared Python bytes.
layer_delta={}
for p in held.glob('*.py'):
 raw=(root/p.name).read_bytes()
 if raw!=p.read_bytes():layer_delta[p.name]={'old_sha256':sha(p.read_bytes()),'new_sha256':sha(raw)}
report=dict(schema='qwen-toy5-source-only-inventory-v1',source_count=len(sources),python_ast_count=python_count,parent250_exact=True,copied339_exact=True,core34_exact=True,original6_definition_asts=definitions,original_loader_body_ast_exact_after_declared_admission_rename=True,original_toy_weights_ast_exact=True,old_layer18_preserved=True,layer18_delta=layer_delta,local_from_import_names_resolve=True,candidate_imported=False,test_collection=False,tests_executed=False,numeric_backend_imported=False,body_reads=False,actual_role_issuer=False)
report_pin=write('source-review.json',report)
rows={name:sha(p.read_bytes())for name,p in sources.items()};raw=(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n').encode();(out/'sources.json').write_bytes(raw)
with tarfile.open(out/'sources.tar.gz','w:gz')as tar:
 for name,p in sources.items():tar.add(p,arcname=name,recursive=False)
with tarfile.open(out/'sources.tar.gz','r:gz')as tar:
 members=tar.getmembers();assert len(members)==len(rows)and all(m.isfile()for m in members)
 for m in members:assert sha(tar.extractfile(m).read())==rows[m.name]
new_files=sorted(set(rows)-set(copies));write('explicit-additions.json',dict(exact_copies=len(copies),added_or_adapted_names=new_files,count=len(new_files)))
handoff=dict(schema='qwen-toy5-disabled-role-successor-source-handoff-v1',source_root=str(root),source_count=len(rows),source_manifest=str(out/'sources.json'),source_manifest_encoding='compact sorted JSON with final newline',source_manifest_sha256=sha(raw),source_archive=str(out/'sources.tar.gz'),source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),source_review_sha256=report_pin,source_only=True,source_held=True,candidate_imports=False,collection=False,tests=False,actual_entry_enabled=False,actual_runtime_issuer=False,resource_or_role_grants=None,focused_proposal='FOCUSED_TEST_PROPOSAL.md',entry='q5_actual_entry.py',actual_scope_limit='source-selected real callgraph and ownership adapters; complete runtime/audit/native/device/resource issuer remains explicitly unavailable',keypins={n:rows[n]for n in ('q5_key_contract.py','q5_key_loader.py','q5_key_store.py','q5_key_producer.py','q5_checkpoint_transport.py','q5_ciphertext_owner.py','q5_fixture_assets.py','q5_role_authority.py','q5_actual_dependencies.py','q5_actual_process.py','q5_actual_bootstrap.py','q5_actual_entry.py')})
handoff_pin=write('handoff.json',handoff)
Path(out/'capture_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'source_count':len(rows),'map':sha(raw),'tar':handoff['source_archive_sha256'],'handoff':handoff_pin,'source_review':report_pin,'keypins':handoff['keypins']},indent=2))
