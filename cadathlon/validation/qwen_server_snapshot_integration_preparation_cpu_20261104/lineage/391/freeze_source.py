"""Freeze the narrow callback-binding successor using stdlib source/AST only."""
import ast
import difflib
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile

H=Path(__file__).resolve().parent
N=Path('/tmp/qwen-server-snapshot-runtime-v2-20261104')
P=Path('/tmp/qwen-server-snapshot-runtime-20261104')
PH=Path('/tmp/qwen-server-snapshot-runtime-handoff-20261104')
BASE=Path('/tmp/qwen-server-actual-runtime-20261104')
PEER=Path('/tmp/qwen-server-snapshot-integration-security-review-20261104/source390/review.json')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def canonical(value):return(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def read(path):
    a=path.lstat();assert stat.S_ISREG(a.st_mode)and not path.is_symlink(),path
    raw=path.read_bytes();z=path.lstat()
    assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),path
    return raw
def inventory(root):
    out={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),p
        if p.is_file():out[p.relative_to(root).as_posix()]=read(p)
        else:assert p.is_dir(),p
    return out
def mapping(raw):return{n:sha(b)for n,b in raw.items()}
def archive_mapping(path):
    out={}
    with tarfile.open(path,'r:gz')as t:
        for m in t.getmembers():
            assert m.isfile()and m.name not in out and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts,m.name
            with t.extractfile(m)as f:out[m.name]=sha(f.read())
    return out
def top(raw):return{n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
dump=lambda node:ast.dump(node,include_attributes=False)
parent=inventory(P);old=json.loads(read(PH/'sources.json'))
assert len(parent)==390 and mapping(parent)==old==archive_mapping(PH/'sources.tar.gz')
assert sha(read(PH/'sources.json'))=='4a9fd8e86386aa0411ada600877117161549cd0b79cbe0b3a21a026508729c40'
assert sha(read(PH/'sources.tar.gz'))=='9bd175e441360b4dcdff986c435ac8bfc2f728b824502df0c12de80d959f4a1b'
base=inventory(BASE);base_map=json.loads(read(PH/'parent387-sources.json'))
assert len(base)==387 and mapping(base)==base_map==archive_mapping(PH/'parent387-sources.tar.gz')
assert sha(read(PEER))=='1dbe5b6e7c9af727de387256488c7efe4e7a304ea642a86d2801081ad0286f0c'
raw=inventory(N)
codes=('qwen_server_process.py','qwen_server_backend.py')
assert {n for n in old if sha(raw[n])!=old[n]}==set(codes)|{'SNAPSHOT_INTEGRATION.md'}
assert set(raw)==set(old)
process=ast.parse(raw[codes[0]]);backend=ast.parse(raw[codes[1]])
old_process=ast.parse(parent[codes[0]])
def literal(tree,name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id==name))
closure=literal(process,'SNAPSHOT_BINDING_SOURCES')
assert closure==literal(old_process,'SNAPSHOT_BINDING_SOURCES')+('cleanup_origin.py',)and len(closure)==21
assert sha(raw['cleanup_origin.py'])=='4543b26bf1168e29bc21375c2f80a9aa99648205a9fd9aea6d5a8059c6d18175'
assert literal(process,'EXPECTED_ID_FIELDS')==literal(old_process,'EXPECTED_ID_FIELDS')
current_functions=top(raw[codes[0]]);old_functions=top(parent[codes[0]])
changes={n:[k for k,v in top(parent[n]).items()if dump(v)!=dump(top(raw[n])[k])]for n in codes}
assert set(changes[codes[0]])=={'require_runtime','RuntimeAdmission','_runtime','run'}
assert set(changes[codes[1]])=={'_runtime','ActualServerDependencies','ServerBackend'}
assert dump(current_functions['_runtime_action'])==dump(old_functions['_runtime_action'])
assert dump(current_functions['_capture_snapshot'])==dump(old_functions['_capture_snapshot'])
assert dump(current_functions['_expected_ids_bytes'])==dump(old_functions['_expected_ids_bytes'])
backend_import=next(i for i,n in enumerate(process.body)if isinstance(n,ast.Import)and any(a.name=='qwen_server_backend'for a in n.names))
for name in('RuntimeAdmission','require_runtime','_expected_ids_bytes'):
    assert next(i for i,n in enumerate(process.body)if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)<backend_import
imports=[n for n in backend.body if isinstance(n,ast.ImportFrom)and n.module=='qwen_server_process']
assert len(imports)==1 and [a.name for a in imports[0].names]==['RuntimeAdmission','require_runtime','_expected_ids_bytes']
assert not any(isinstance(n,ast.ImportFrom)and n.module=='qwen_server_process'for f in top(raw[codes[1]]).values()for n in ast.walk(f))
def retained(fn,wanted):
    actual={a.arg:ast.unparse(v)for a,v in zip(fn.args.kwonlyargs,fn.args.kw_defaults)if v is not None}
    assert actual==wanted,(fn.name,actual,wanted)
retained(current_functions['require_runtime'],{'_dispatch':'_runtime_action'})
methods={n.name:n for n in current_functions['RuntimeAdmission'].body if isinstance(n,ast.FunctionDef)}
for name in('check','expected_ids_bytes','input_digest'):retained(methods[name],{'_require':'require_runtime'})
for name in('allow_public_key_reads','admit_server_bodies','require_public_key_read_scope','require_server_body_scope'):retained(methods[name],{'_dispatch':'_runtime_action'})
for name in('_runtime','run'):retained(current_functions[name],{'_require_runtime':'require_runtime'})
bf=top(raw[codes[1]])
retained(bf['_runtime'],{'_require':'require_runtime'})
bm={n.name:n for n in bf['ServerBackend'].body if isinstance(n,ast.FunctionDef)}
for name in('__init__','check'):retained(bm[name],{'_check_runtime':'_runtime'})
private_keywords={'_dispatch','_require','_require_runtime','_check_runtime'}
for tree in(process,backend):
    assert not any(isinstance(n,ast.Call)and any(k.arg in private_keywords for k in n.keywords)for n in ast.walk(tree))
# Normalize only the declared defaults/call-name retention and moved imports;
# existing runtime and backend control bodies must otherwise remain identical.
class Normalize(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        node=self.generic_visit(node)
        pairs=[(a,d)for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults)if a.arg not in private_keywords]
        node.args.kwonlyargs=[a for a,d in pairs];node.args.kw_defaults=[d for a,d in pairs]
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name):
            names={'_require':'require_runtime','_require_runtime':'require_runtime','_check_runtime':'_runtime','_dispatch':'_runtime_action'}
            if node.func.id in names:node.func.id=names[node.func.id]
        return node
normalize=Normalize()
for name in('require_runtime','RuntimeAdmission','_runtime','run'):
    a=ast.parse(parent[codes[0]]);b=ast.parse(raw[codes[0]])
    aa=next(n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
    bb=next(n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
    assert dump(aa)==dump(normalize.visit(bb)),name
for name in('_runtime','ActualServerDependencies','ServerBackend'):
    a=top(parent[codes[1]])[name];b=top(raw[codes[1]])[name]
    class RemoveProcessImport(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module=='qwen_server_process'else node
    assert dump(RemoveProcessImport().visit(a))==dump(normalize.visit(b)),name
for name in('qwen_admission_snapshot.py','qwen_server_resource.py','qwen_server_dependencies.py','qwen_server_entry.py','qwen_server_entry_status.json','execution_authorization.json','qwen_finalizer.py','cleanup_origin.py'):
    assert raw[name]==parent[name]
assert sha(raw['qwen_admission_snapshot.py'])=='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
assert json.loads(raw['qwen_server_entry_status.json'])['actual_entry_ready']is False
assert json.loads(raw['execution_authorization.json'])['actual_execution_enabled']is False
history=dict(schema='qwen-server-snapshot-retained-dispatch-source-v2',source_only=True,
    predecessor390=dict(root=str(P),source_map_sha256=sha(read(PH/'sources.json')),source_archive_path=str(PH/'sources.tar.gz'),source_archive_sha256=sha(read(PH/'sources.tar.gz'))),
    predecessor387=dict(root=str(BASE),source_map_sha256=sha(read(PH/'parent387-sources.json')),source_archive_sha256=sha(read(PH/'parent387-sources.tar.gz'))),
    findings_receipt=dict(path=str(PEER),sha256=sha(read(PEER))),
    corrections=['Add exact cleanup_origin source/module/code/export binding on read->finish->protect_body/select->is_cleanup/mark path.','Retain original process/backend dispatcher references before callbacks; no dynamic dispatcher import at backend entry.'],
    observed_execution_failure=False,imports=False,tests=False,actual_authorization=False,registry_scope_unchanged=True,
    retention_keywords='Internal source defaults; all selected source calls keep fixed arguments and never override them. No arbitrary private registry/global/default/closure mutation defense.',
    literal_guard_unchanged=True,capacity_denial_and_false_readiness_unchanged=True,
    code_overlays={n:dict(before=old[n],after=sha(raw[n]))for n in codes})
(N/'snapshot_dispatch_successor.json').write_bytes(canonical(history))
provenance=json.loads(parent['snapshot_integration.json'])
provenance.update(schema='qwen-server-snapshot-integration-source-v2',status='held_unexecuted_source_successor',source_count=391,
    predecessor390=history['predecessor390'],correction_history='snapshot_dispatch_successor.json',
    callback_source_closure={n:sha(raw[n])for n in closure},
    additions=['qwen_admission_snapshot.py','SNAPSHOT_INTEGRATION.md','snapshot_integration.json','snapshot_dispatch_successor.json'],
    code_overlays={n:dict(before=base_map[n],after=sha(raw[n]))for n in('qwen_server_process.py','qwen_server_backend.py','qwen_server_entry.py')},
    backend_module_capture='Pure declarations loaded after original process dispatcher/class definitions and before snapshot capture; no backend/context construction.',
    original_dispatch='Retained source-defined function defaults enter original runtime control before checking public dispatcher replacements; fixed ordinary source call arguments.')
(N/'snapshot_integration.json').write_bytes(canonical(provenance))
raw=inventory(N);m=mapping(raw)
assert len(m)==391 and set(m)-set(old)=={'snapshot_dispatch_successor.json'}
overlays=sorted(n for n in old if m[n]!=old[n]);assert overlays==['SNAPSHOT_INTEGRATION.md','qwen_server_backend.py','qwen_server_process.py','snapshot_integration.json']
assert sum(m[n]==old[n]for n in old)==386
assert sum(m[n]==base_map[n]for n in base_map)==384
(H/'sources.json').write_bytes(canonical(m))
closure_map={n:m[n]for n in closure};(H/'callback-source-closure.json').write_bytes(canonical(closure_map))
patch=''.join(''.join(difflib.unified_diff(parent[n].decode().splitlines(True),raw[n].decode().splitlines(True),fromfile='held390/'+n,tofile='successor391/'+n))for n in overlays)
(H/'source.patch').write_text(patch)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT)as t:
    for name,data in raw.items():
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';t.addfile(info,io.BytesIO(data))
(H/'sources.tar.gz').write_bytes(gzip.compress(buf.getvalue(),mtime=0))
assert archive_mapping(H/'sources.tar.gz')==m==mapping(inventory(N))
assert mapping(inventory(P))==old and mapping(inventory(BASE))==base_map
review=dict(schema='qwen-server-snapshot-dispatch-static-review-v2',passed=True,source_files=391,parent390_preserved=True,parent387_preserved=True,
    unchanged_from390=386,unchanged_from387=384,changed_top_definitions=changes,callback_sources=21,literal_cleanup_origin=True,
    normalized_runtime_and_backend_control_AST_unchanged=True,original_registry_dispatch_AST_unchanged=True,
    fixed_retention_call_shapes=True,pure_import_order_source_checked=True,guard_resource_entry_and_false_status_exact=True,
    imports=False,collection=False,tests=False,probes=False,actual_authorization=False,script_sha256=sha(read(Path(__file__))))
(H/'static-review.json').write_bytes(canonical(review))
handoff=dict(schema='qwen-server-snapshot-integration-held-handoff-v2',root=str(N),source_count=391,source_map_sha256=sha(read(H/'sources.json')),
    source_archive_sha256=sha(read(H/'sources.tar.gz')),source_archive_bytes=(H/'sources.tar.gz').stat().st_size,
    parent390_map_sha256=sha(read(PH/'sources.json')),parent390_archive_sha256=sha(read(PH/'sources.tar.gz')),
    parent387_map_sha256=sha(read(PH/'parent387-sources.json')),source_patch_sha256=sha(read(H/'source.patch')),
    callback_closure_sha256=sha(read(H/'callback-source-closure.json')),static_review_sha256=sha(read(H/'static-review.json')),
    code_hashes={n:m[n]for n in codes},entry_guard_resource_unchanged=True,changed_from390=overlays,added=['snapshot_dispatch_successor.json'],
    source_only=True,imports=False,collection_or_tests=False,device_or_body_probes=False,actual_authority=False,capacity_denial_preserved=True,readiness_false=True)
(H/'handoff.json').write_bytes(canonical(handoff))
print(json.dumps(dict(passed=True,handoff_sha256=sha(read(H/'handoff.json')),**handoff),sort_keys=True))
