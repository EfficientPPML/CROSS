"""Source-only v2 delta review; no candidate import or execution."""
import ast
import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).parent
OLD=Path('/tmp/qwen-server-admission-snapshot-source-20261104')
NEW=Path('/tmp/qwen-server-admission-snapshot-source-v2-20261104')
HAND=Path('/tmp/qwen-server-admission-snapshot-handoff-v2-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
prior=(HERE/'review.json').read_bytes()
assert sha(prior)=='b44310718a631c1231a623f6f503a168967a89e7f5e3d9c44816447b92540398'
oldmap=json.loads(prior)['source_hashes']
old={n:(OLD/n).read_bytes()for n in oldmap}
assert{n:sha(v)for n,v in old.items()}==oldmap
mapraw=(HAND/'sources.json').read_bytes()
assert sha(mapraw)=='dd990a97b71b9e9858843ee4156c30d6c2b4522b3bd7b59b5324af9e50a10bf8'
mapping=json.loads(mapraw);assert len(mapping)==5
new={n:(NEW/n).read_bytes()for n in mapping}
assert{n:sha(v)for n,v in new.items()}==mapping
assert{p.name for p in NEW.iterdir()}==set(mapping)
assert all(p.is_file()and not p.is_symlink()for p in NEW.iterdir())
archive=(HAND/'sources.tar.gz').read_bytes()
assert sha(archive)=='b211059616c5923c2bc15acb0aa5b387d6afa31bbe7b4394773dcc0548b75ce9'
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
    members=tar.getmembers()
    assert len(members)==len(mapping)and all(m.isfile()for m in members)
    assert{m.name:sha(tar.extractfile(m).read())for m in members}==mapping
assert set(new)-set(old)=={'successor_history.json'}
assert {n for n in old if old[n]!=new[n]}=={'qwen_admission_snapshot.py','DESIGN.md'}
patch=''.join(difflib.unified_diff(old['qwen_admission_snapshot.py'].decode().splitlines(True),new['qwen_admission_snapshot.py'].decode().splitlines(True),fromfile='v1/qwen_admission_snapshot.py',tofile='v2/qwen_admission_snapshot.py')).encode()
assert patch==(HAND/'guard.patch').read_bytes()
trees=[ast.parse(r['qwen_admission_snapshot.py'])for r in(old,new)]
def defs(t):return{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
before,after=map(defs,trees)
assert before.keys()==after.keys()
assert{n for n in before if before[n]!=after[n]}=={'AdmissionSnapshot','capture_admitted'}
classes=[next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=='AdmissionSnapshot')for t in trees]
methods=[{n.name:ast.dump(n,include_attributes=False)for n in cls.body if isinstance(n,ast.FunctionDef)}for cls in classes]
assert methods[0].keys()==methods[1].keys()
assert{n for n in methods[0]if methods[0][n]!=methods[1][n]}=={'check'}
def assignments(t):return{n.targets[0].id:ast.dump(n.value,include_attributes=False)for n in t.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
a,b=map(assignments,trees)
assert set(b)-set(a)=={'_CAPTURE_ATTEMPT'}and all(a[k]==b[k]for k in a)
assert b['_CAPTURE_ATTEMPT']==ast.dump(ast.Constant(value=None),include_attributes=False)
source=new['qwen_admission_snapshot.py'].decode()
assert source.index('_CAPTURE_ATTEMPT=(packet,None,None,None)')<source.index('checker=_callback(check_bindings)')
assert source.index('_CAPTURE_ATTEMPT=(packet,bindings,graph,checker)')<source.index('_FULL_PACKET_CHECK(packet)')
assert source.index('guard.check(packet)')<source.index('_CAPTURE_ATTEMPT=None  #')
assert "if _REGISTRY.get(id(self))is not checking:"in source
for n,v in old.items():assert(OLD/n).read_bytes()==v
for n,v in new.items():assert(NEW/n).read_bytes()==v
report={
 'schema':'qwen-admission-snapshot-v2-source-peer-review-v1','passed':True,
 'guard_sha256':mapping['qwen_admission_snapshot.py'],'source_manifest_sha256':sha(mapraw),'source_archive_sha256':sha(archive),
 'source_hashes':mapping,'parent_receipt_sha256':sha(prior),'guard_patch_sha256':sha(patch),
 'checks':{'five_current_and_tar_exact':True,'old_four_preserved':True,'only_declared_guard_design_history_delta':True,'field_module_and_helper_binding_code_unchanged':True,'same_retire_and_default_denial':True,'all_reviewed_bytes_unchanged_at_end':True},
 'resolved_source_findings':[
  {'id':'SNAPSHOT-REENTRY','resolution':'A nested check observing checking writes failed before raising. The outer invocation retains its exact checking tuple and rejects if the registry no longer contains that tuple, so a caught inner rejection cannot restore active.'},
  {'id':'SNAPSHOT-CAPTURE-RETENTION','resolution':'The initial packet is retained before the first fallible callback. The completed original graph/bindings/checker are retained before full validation. Attempt references clear only after registry insertion and a successful original guard check; failed capture leaves the attempt retained.'},
 ],
 'no_additional_concrete_blocker_in_delta':True,
 'limitations':[
  'Source-only correction, not a runtime negative/successor replay or actual proof admission.',
  'Prior exact13 graph coverage and module/class/function binding review applies because those ASTs and14-source reference inventory are unchanged.',
  'Mutable pure-helper globals, closure-cell contents and arbitrary snapshot-private-state/interpreter mutation remain outside the declared trusted-source callback contract.',
  'Full process integration, terminal full reauthentication, independently retained check/retire callbacks and live file/resource/backend/ownership checks remain required; this guard grants no authority.',
 ],
 'candidate_imports':False,'tests_or_collection':False,'runtime_probes':False,'proof_or_body_reads':False,'integration':False,'execution_authorized':False,
 'review_script_sha256':sha(Path(__file__).read_bytes()),
}
out=HERE/'review-v2.json'
with out.open('xb')as f:f.write((json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps({'passed':True,'receipt':str(out),'sha256':sha(out.read_bytes()),'scope':'source only'}))
