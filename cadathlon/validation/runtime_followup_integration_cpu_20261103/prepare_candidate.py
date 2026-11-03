"""Compose the three reviewed follow-ups after the frozen five-candidate base."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

LIVE=Path('/home/jianming_gatech/CROSS')
BASE=Path('/tmp/cross-all-candidates-20261103/checkout')
LAB=Path('/tmp/runtime-followup-integration-20261103');ROOT=LAB/'checkout'
LAB.mkdir(exist_ok=False)
shutil.copytree(BASE,ROOT,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','run'))
prerequisites=[
 'all_candidates_cpu_20261103/combined.patch',
 'rescale_scoped_ntt_cpu_20261103/rescale_scoped_ntt_followup.patch',
 'hemul_nested_rescale_cpu_20261103/hemul_nested_rescale_followup.patch',
 'lazy_bconv_cpu_20261103/candidate.patch',
]
records=[]
for index,rel in enumerate(prerequisites):
    source=LIVE/'cadathlon/validation'/rel
    target=LAB/'prerequisites'/f'{index+1}_{Path(rel).name}'
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(source,target)
    records.append({'source':str(source.relative_to(LIVE)),'copy':str(target.relative_to(LAB)),
                    'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    if index:
        subprocess.run(['git','apply','--check',str(target)],cwd=ROOT,check=True)
        subprocess.run(['git','apply',str(target)],cwd=ROOT,check=True)
(LAB/'prerequisites.json').write_text(json.dumps(records,indent=2)+'\n')
# This composition must leave all arithmetic in the core bootstrap untouched.
assert (ROOT/'jaxite_word/bootstrapping.py').read_bytes()==(LIVE/'jaxite_word/bootstrapping.py').read_bytes()
print(json.dumps({'checkout':str(ROOT),'prerequisites':records},indent=2))
