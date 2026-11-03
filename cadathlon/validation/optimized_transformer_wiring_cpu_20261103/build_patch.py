"""Build an incremental wiring-only patch after the three recorded prerequisites."""
from pathlib import Path
import ast,difflib,hashlib,json,subprocess
LAB=Path(__file__).resolve().parent;before=LAB/'before';after=LAB/'checkout'
paths=sorted({p.relative_to(root) for root in (before,after) for p in root.rglob('*.py') if '__pycache__' not in p.parts})
changes={};patch=[]
for rel in paths:
 old=before/rel;new=after/rel
 a=old.read_text() if old.exists() else '';b=new.read_text() if new.exists() else ''
 if a==b:continue
 changes[str(rel)]={'before_sha256':hashlib.sha256(a.encode()).hexdigest() if old.exists() else None,'after_sha256':hashlib.sha256(b.encode()).hexdigest() if new.exists() else None}
 patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+str(rel) if old.exists() else '/dev/null',tofile='b/'+str(rel) if new.exists() else '/dev/null'))
path=LAB/'model_wiring_followup.patch';path.write_text(''.join(patch))
subprocess.run(['git','apply','--check',str(path)],cwd=before,check=True)
# Bootstrap arithmetic/checkpoints and existing adapter bytes are outside scope.
for rel in ['jaxite_word/bootstrapping.py',*[f'cadathlon/agent_pset/{p}/src/{m}_{kind}_reference.py' for m,p in [('bert','p2_bert'),('qwen','p4_qwen')] for kind in ['bounded','packed']]]:
 assert (before/rel).read_bytes()==(after/rel).read_bytes(),rel
get=lambda root:next(n for n in ast.parse((root/'cadathlon/common/encrypted_reference.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='reference_ring_profile')
assert ast.dump(get(before))==ast.dump(get(after))
(LAB/'patch_sources.json').write_text(json.dumps({'scope':__doc__,'changes':changes,'patch_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bootstrap_and_existing_adapters_unchanged':True,'ring_profile_unchanged':True},indent=2)+'\n')
print(path,len(changes),'files')
