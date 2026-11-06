"""Stdlib source-only capture; never loads the guard or its candidate imports."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/tmp/qwen-server-admission-snapshot-source-20261104')
OUT=Path(__file__).parent
PURE=Path('/tmp/qwen-server-public-packet-20261104')
SERVER=Path('/tmp/qwen-server-actual-runtime-20261104')
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
P=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():H(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}

pure_expected=json.loads(Path('/tmp/qwen-server-public-packet-sources-v1-20261104.json').read_bytes())
assert inventory(PURE)==pure_expected
raw=(ROOT/'qwen_admission_snapshot.py').read_bytes();tree=ast.parse(raw)
imports={n.asname:n.name for node in tree.body if isinstance(node,ast.Import)for n in node.names if n.asname}
fields=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_FIELDS'for t in n.targets))
assert isinstance(fields,ast.Tuple)and len(fields.elts)==13
covered={};refs={}
for pair in fields.elts:
    target,names=pair.elts
    assert isinstance(target,ast.Attribute)and isinstance(target.value,ast.Name)
    module=imports[target.value.id];path=PURE/(module+'.py');data=path.read_bytes()
    assert H(data)==pure_expected[path.name]
    cls=next(n for n in ast.parse(data).body if isinstance(n,ast.ClassDef)and n.name==target.attr)
    annotations=tuple(n.target.id for n in cls.body if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name))
    assert ast.literal_eval(names)==annotations
    covered[module+'.'+target.attr]=list(annotations)
module_names=json.loads((PURE/'packet_import_closure.json').read_bytes())['local_modules']
for module in module_names:
    p=PURE/(module+'.py');refs[str(p)]={'sha256':H(p.read_bytes()),'kind':'exact held packet27 pure helper'}
for name in('qwen_server_backend.py','qwen_key_loader.py','qwen_stage_composition.py','qwen_backend_composition.py'):
    p=SERVER/name;refs[str(p)]={'sha256':H(p.read_bytes()),'kind':'root draft source call-hierarchy observation only'}
source_refs={'schema':'qwen-admission-snapshot-source-refs-v1','packet27_manifest_sha256':'4961e6dbe93b4905d99130a873e90bcfdddfc711ed51aaef032bb3544308edff',
    'pure_helper_modules':module_names,'references':refs,'complete_closed_field_coverage':covered,
    'source_only':True,'candidate_imported':False,'tests_or_proofs_present':False,
    'runtime_admission_process':'Absent at initial trace; created during authoring, not read or reviewed here'}
(ROOT/'source_refs.json').write_bytes(P(source_refs))
assert 'json'not in {n.name for node in tree.body if isinstance(node,ast.Import)for n in node.names}
checks=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='AdmissionSnapshot')
hot=next(n for n in checks.body if isinstance(n,ast.FunctionDef)and n.name=='check')
called_names={n.func.id for n in ast.walk(hot)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)}
assert not {'authenticate_packet','authenticate_scientific','_FULL_PACKET_CHECK','sha','canonical'}&called_names
mapping=inventory(ROOT);assert len(mapping)==4
(OUT/'sources.json').write_bytes(C(mapping))
with tarfile.open(OUT/'sources.tar.gz','w:gz')as t:
    for name in mapping:t.add(ROOT/name,arcname=name,recursive=False)
with tarfile.open(OUT/'sources.tar.gz')as t:
    members=t.getmembers();assert len(members)==4 and all(m.isfile()for m in members)
    assert {m.name:H(t.extractfile(m).read())for m in members}==mapping
assert inventory(PURE)==pure_expected and inventory(ROOT)==mapping
handoff={'schema':'qwen-admission-snapshot-source-handoff-v1','source_root':str(ROOT),'source_count':4,
    'source_manifest_sha256':H(C(mapping)),'source_archive_sha256':H((OUT/'sources.tar.gz').read_bytes()),
    'guard_sha256':mapping['qwen_admission_snapshot.py'],'closed_admission_types':13,
    'all_declared_instance_fields_statically_matched':True,'held_pure27_unchanged':True,
    'pure_helper_count':len(module_names),'tests_collected_or_run':False,'candidate_imported':False,
    'actual_proof_instances_present':False,'runtime_or_capacity_measurements':False,
    'full_process_integration_performed':False,'actual_execution_authorized':False}
(OUT/'handoff.json').write_bytes(P(handoff))
print(json.dumps(handoff,sort_keys=True))
