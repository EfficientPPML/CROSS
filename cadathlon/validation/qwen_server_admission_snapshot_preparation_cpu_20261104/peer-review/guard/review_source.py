"""Stdlib source-only peer review, without importing any candidate source."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).parent
ROOT=Path('/tmp/qwen-server-admission-snapshot-source-20261104')
HAND=Path('/tmp/qwen-server-admission-snapshot-handoff-20261104')
PURE=Path('/tmp/qwen-server-public-packet-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
mapraw=(HAND/'sources.json').read_bytes()
assert sha(mapraw)=='7676af6f7b07ba8aa0c0a1613e63b8b75007946b374cc616b82ff4527b11c2bf'
mapping=json.loads(mapraw)
assert len(mapping)==4
raw={n:(ROOT/n).read_bytes()for n in mapping}
assert {n:sha(v)for n,v in raw.items()}==mapping
assert sha(raw['qwen_admission_snapshot.py'])=='69a9bc77612aa9e21f7afa178a43741b0bda9475de35e5bbc4e750e9fd1bf6d0'
archive=(HAND/'sources.tar.gz').read_bytes()
assert sha(archive)=='898365d5e5a9969e61541177a4270da5c49d9be9e06f312146033ce9560e1e90'
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
    members=tar.getmembers()
    assert all(m.isfile()for m in members)
    assert {m.name for m in members}==set(mapping)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==mapping
refs=json.loads(raw['source_refs.json'])
pure={name:(PURE/(name+'.py')).read_bytes()for name in refs['pure_helper_modules']}
assert len(pure)==14
for name,value in pure.items():
    assert sha(value)==refs['references'][str(PURE/(name+'.py'))]['sha256']
tree=ast.parse(raw['qwen_admission_snapshot.py'])
aliases={a.asname or a.name:a.name for node in tree.body if isinstance(node,ast.Import)for a in node.names}
fields_node=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_FIELDS'for t in n.targets))
fields={}
for row in fields_node.elts:
    cls,names=row.elts
    assert isinstance(cls,ast.Attribute)and isinstance(cls.value,ast.Name)
    name=aliases[cls.value.id]+'.'+cls.attr
    fields[name]=ast.literal_eval(names)
assert len(fields)==13
assert {n:list(v)for n,v in fields.items()}==refs['complete_closed_field_coverage']
trees={name:ast.parse(value)for name,value in pure.items()}
for name,wanted in fields.items():
    module,klass=name.rsplit('.',1)
    cls=next(n for n in trees[module].body if isinstance(n,ast.ClassDef)and n.name==klass)
    actual=tuple(n.target.id for n in cls.body if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name))
    assert actual==wanted,(name,actual,wanted)
    assert all(n.value is None for n in cls.body if isinstance(n,ast.AnnAssign))
defaults=[]
for module,t in trees.items():
    for n in ast.walk(t):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            vals=n.args.defaults+[v for v in n.args.kw_defaults if v is not None]
            if vals:defaults.append({'module':module,'function':n.name,'expressions':[ast.unparse(v)for v in vals]})
            assert all(isinstance(v,ast.Constant)and v.value is None or isinstance(v,ast.BinOp)and isinstance(v.op,ast.LShift)and all(isinstance(x,ast.Constant)and type(x.value)is int for x in(v.left,v.right))for v in vals)
for n,value in raw.items():assert(ROOT/n).read_bytes()==value
for n,value in pure.items():assert(PURE/(n+'.py')).read_bytes()==value
report={
 'schema':'qwen-admission-snapshot-source-peer-review-v1',
 'review_completed':True,'source_checks_passed':True,'no_unresolved_blockers':False,
 'guard_sha256':sha(raw['qwen_admission_snapshot.py']),
 'source_manifest_sha256':sha(mapraw),'source_archive_sha256':sha(archive),
 'source_hashes':mapping,'pure_source_hashes':{n+'.py':sha(v)for n,v in pure.items()},
 'closed_types':{n:list(v)for n,v in fields.items()},'explicit_source_defaults':defaults,
 'checks':{'four_current_and_archive_exact':True,'fourteen_pure_references_exact':True,'thirteen_instance_field_sets_exact':True,'all_reviewed_bytes_unchanged_at_end':True},
 'findings':[
  {'id':'SNAPSHOT-REENTRY','status':'source finding; no execution','location':'AdmissionSnapshot.check status guard before checking/try','detail':"When registry status is checking, a nested check raises before assigning failed. If the trusted checker catches that rejection, the outer check can complete and overwrite the registry with active. This does not satisfy the intended sticky failure on every reentry attempt.",'required_delta':'Poison the original state on reentry and verify the original in-progress state is still valid before restoring active; retain exact original graph.'},
  {'id':'SNAPSHOT-CAPTURE-RETENTION','status':'source finding; no execution','location':'capture_admitted between graph capture and _REGISTRY insertion','detail':'_ISSUED is sticky, but full validator/final callback/graph-comparison failures before the guard is registered do not retain the captured graph in a process-owned failure registry. The caller may retain the packet, but the helper itself does not yet meet its conservative failure-retention intention.','required_delta':'Retain original packet/graph/bindings/checker on post-capture failure without granting an active guard.'},
 ],
 'source_conclusions':[
  'All13 explicit graph field sets match the exact held packet27 types. The reachable stored values are exact admitted dataclasses, immutable tuples and exact byte/string values; computed store/ID properties are not retained fields and remain external integration responsibilities.',
  'Graph checks reject replaced instance dictionaries, fields, equal replacement objects, tuple members and unknown mutable payload types. Original bytes/strings cannot change in place; containing field/tuple identity supplies their binding.',
  'Selected module objects, file/spec-origin identities, existing selected function/class exports, complete class descriptor identities, function code/defaults and keyword-default identities are checked before object field access.',
  'Source-declared function defaults are None or immutable integer ceilings; selected class defaults are None/False where present. No concrete valid-admission default incompatibility was identified by source read. Generated dataclass behavior and actual import assembly remain unexecuted.',
  'PID, thread identifier and thread-object checks prevent ordinary transfer to another process/thread. A failed ordinary check marks the original state failed and retains references; clean retirement checks once then deletes the registry entry and does not permit a second capture.',
  'The initial graph precedes full packet validation and is compared afterward, preventing a final binding callback from swapping unvalidated proof fields before publication. An original trusted check_bindings callback and independently authenticated initial source closure remain prerequisites.',
  'The guard is process-local only, issues no grant, leaves full public validators unchanged and does not replace live held-file, source/origin/audit, backend/security, capacity or original-owner retirement checks.',
 ],
 'limits':[
  'The snapshot does not recursively freeze pure-module mutable global values, function __globals__ dictionaries, closure cells or mutable contents behind class metadata values; source/callback trust must explicitly exclude those mutations. It does not defend arbitrary guard-private-global/interpreter mutation.',
  'Existing exported bindings are checked, not an exact entire module-global name set. This is an identity snapshot after a trusted full validation boundary, not independent validation of callable code against source bytes.',
  'CALL_HIERARCHY records the earlier backend/loader source trace before process integration. It supplies no timings, speedup or memory-capacity authority; process-local optimization and terminal full reauthentication remain unintegrated.',
  'The8192 object inventory bound was source-inspected only; no actual admission graph, proof instance or generated dataclass/runtime import was constructed.',
 ],
 'candidate_imported':False,'collection_or_tests':False,'proof_or_body_reads':False,
 'runtime_probe':False,'capacity_observation':False,'process_integration':False,'execution_authorized':False,
 'review_script_sha256':sha(Path(__file__).read_bytes()),
}
dest=HERE/'review.json'
with dest.open('xb')as stream:stream.write((json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps({'review_completed':True,'source_checks_passed':True,'unresolved_source_findings':2,'receipt':str(dest),'sha256':sha(dest.read_bytes())}))
