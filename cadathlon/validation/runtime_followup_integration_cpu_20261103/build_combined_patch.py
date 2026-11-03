"""Build the reviewed combined patch against the still-frozen live runtime."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess
import tempfile

LAB=Path(__file__).resolve().parent;ROOT=LAB/'checkout';LIVE=Path('/home/jianming_gatech/CROSS')
paths=set()
for folder in ('jaxite_word','cadathlon/common','cadathlon/tests'):
    paths.update(p.relative_to(ROOT) for p in (ROOT/folder).glob('*.py'))
for group in ('agent_pset','non_agent_pset'):
    for problem in (ROOT/'cadathlon'/group).iterdir():
        if problem.is_dir():
            paths.update(p.relative_to(ROOT) for p in (problem/'src').glob('*.py'))
paths.update(Path(p) for p in ('cadathlon/validation/check_transformer_suite.py',
                             'cadathlon/validation/run_bounded_qwen2_proof.py'))
changes={};patch=[]
for rel in sorted(paths):
    before=LIVE/rel;after=ROOT/rel
    a=before.read_bytes() if before.exists() else None;b=after.read_bytes()
    if a==b:
        continue
    changes[str(rel)]={'before_sha256':hashlib.sha256(a).hexdigest() if a is not None else None,
                       'after_sha256':hashlib.sha256(b).hexdigest()}
    if a is not None:
        destination=LAB/'before/sources'/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(a)
    patch.extend(difflib.unified_diff((a or b'').decode().splitlines(True),b.decode().splitlines(True),
                 fromfile='a/'+str(rel) if a is not None else '/dev/null',tofile='b/'+str(rel)))
(LAB/'combined.patch').write_text(''.join(patch))
subprocess.run(['git','apply','--check',str(LAB/'combined.patch')],cwd=LIVE,check=True)
with tempfile.TemporaryDirectory(prefix='runtime-followup-patch-') as temporary:
    base=Path(temporary);shutil.copytree(LAB/'before/sources',base,dirs_exist_ok=True)
    subprocess.run(['git','apply',str(LAB/'combined.patch')],cwd=base,check=True)
    for rel,entry in changes.items():
        assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==entry['after_sha256']
(LAB/'changed_sources.json').write_text(json.dumps(changes,indent=2)+'\n')
print(json.dumps({'changed_files':len(changes),'patch_sha256':hashlib.sha256((LAB/'combined.patch').read_bytes()).hexdigest(),
                  'all_changed_sources_reconstructed':True,'applies_to_live_but_unapplied':True},indent=2))
