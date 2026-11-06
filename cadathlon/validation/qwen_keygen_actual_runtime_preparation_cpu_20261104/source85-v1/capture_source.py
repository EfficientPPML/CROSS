"""Capture the held source draft. AST/data reads only, no candidate imports."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('/tmp/qwen-keygen-actual-runtime-20261104')
out=Path(__file__).parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canonical=lambda obj:(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
pretty=lambda obj:(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(path):return json.loads(path.read_bytes())
def inventory():
    paths=list(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in sorted(paths)if p.is_file()}
def assignment(path,name):
    t=ast.parse(path.read_text())
    return next(ast.literal_eval(n.value)for n in t.body if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id==name for v in n.targets))

sources=inventory();assert len(sources)==85
assert read(root/'keygen_source_inventory.json')==dict(schema='qwen-keygen-source-inventory-v1',files=sorted(sources))
copies=read(root/'keygen_selected_copies.json');assert len(copies)==74
for name,row in copies.items():
    assert sources[name]==row['sha256']==sha(Path(row['path']).read_bytes())
parent28=read(Path('/tmp/qwen-keygen-authority-boundary-v2-handoff-20261104/sources.json'))
assert len(parent28)==28 and all(sources[n]==h for n,h in parent28.items())
minimum=assignment(root/'keygen_boundary_pins.py','REQUIRED_ACTUAL_SOURCES')
assert len(minimum)==58 and all(sources[n]==h for n,h in minimum.items())
core=read(root/'core_source_manifest.json')
assert len(core)==34
for n,h in core.items():
    path=n if n.startswith('runtime/')else 'runtime/jaxite_word/'+n
    assert sources[path]==h
origin=read(root/'keygen_implementation_origin.json')
for row in origin['references'].values():assert sha(Path(row['path']).read_bytes())==row['sha256']
parsed=[]
for name in sources:
    if name.endswith('.py'):ast.parse((root/name).read_text(),filename=name);parsed.append(name)
assert not any(n.startswith('test_')or n.endswith('/conftest.py')for n in sources)
assert read(root/'keygen_entry_status.json')==dict(schema='qwen-keygen-entry-status-v1',actual_entry_ready=False)
dep_fields=assignment(root/'keygen_dependencies.py','DEPENDENCY_FIELDS')
assert sorted(dep_fields)==read(root/'keygen_policy_contract.json')['dependency']['fields']
for target,source in [('keygen_entry.py','model_entry.py'),('keygen_output.py','model_output.py')]:
    before=Path('/tmp/qwen-model-store-readonly-v2-20261104')/source
    patch=''.join(difflib.unified_diff(before.read_text().splitlines(True),(root/target).read_text().splitlines(True),fromfile='readonly42/'+source,tofile='keygen85/'+target))
    (out/(target+'.patch')).write_text(patch)
new=sorted(set(sources)-set(copies));assert len(new)==11
(out/'sources.json').write_bytes(canonical(sources))
(out/'sources-pretty.json').write_bytes(pretty(sources))
with tarfile.open(out/'sources.tar.gz','w:gz')as archive:
    for n in sources:archive.add(root/n,arcname=n,recursive=False)
with tarfile.open(out/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==85 and all(m.isfile()for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==sources
assert inventory()==sources
result=dict(schema='qwen-keygen-actual-source-handoff-v1',source_root=str(root),source_count=85,
    source_manifest=str(out/'sources.json'),source_manifest_sha256=sha((out/'sources.json').read_bytes()),
    source_manifest_encoding='canonical sorted compact JSON plus newline; this is the actual gate digest',
    pretty_inventory_sha256=sha((out/'sources-pretty.json').read_bytes()),
    source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),exact_selected_copies=74,exact_parent28=True,
    exact_required_prerequisites58=True,exact_core34=True,new_files=new,overlays_to_selected_copies=[],
    new_runtime_pins={n:sources[n]for n in origin['new_runtime']},static_parsed_python_files=len(parsed),
    shared_helper_sha256=sources['qwen_backend_identity.py'],writer_sha256=sources['qwen_key_store.py'],
    original_recording_producer_selected_as_provenance_only=True,
    selected_original_client39_pin=sources['client_role_source_pins.py'],response_root_successor_selected=False,
    dependency_policy_instance_supplied=False,actual_entry_ready=False,scope=dict(source_authoring=True,ast_and_source_checks=True,
        candidate_imports=False,tests_or_collection=False,installed_backend_imports=False,body_reads=False,
        current_capacity_or_store_metadata=False,rng=False,actual_grant=False),
    limitations=origin['limitations'],independent_source_review_required=True)
(out/'handoff.json').write_bytes(pretty(result))
print(json.dumps(dict(source_count=85,map=result['source_manifest_sha256'],archive=result['source_archive_sha256'],handoff=sha((out/'handoff.json').read_bytes()),new_runtime_pins=result['new_runtime_pins']),sort_keys=True))
