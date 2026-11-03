"""Reconstruct the reviewed runtime/collector candidates without editing live code."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess

LIVE = Path('/home/jianming_gatech/CROSS')
LAB = Path('/tmp/cross-all-candidates-20261103')
ROOT = LAB / 'checkout'
ROOT.mkdir(parents=True, exist_ok=False)
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=LIVE).decode().split('\0')
for rel in filter(None, paths):
    src = LIVE / rel
    if not src.is_file():
        raise RuntimeError(f'Missing tracked file: {rel}')
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
patches = [
    'combined_runtime_candidate_cpu_20261103/combined_runtime.patch',
    'hemul_scoped_ntt_cpu_20261103/hemul_scoped_ntt_followup.patch',
    'guarded_dispatch_policy_cpu_20261103/candidate.patch',
    'optimized_transformer_wiring_cpu_20261103/model_wiring_followup.patch',
    'transformer_owned_key_collector_cpu_v2_20261103/candidate.patch',
]
dependencies = []
for i, rel in enumerate(patches):
    path = LIVE / 'cadathlon/validation' / rel
    saved = LAB / 'prerequisites' / f'{i+1}_{Path(rel).name}'
    saved.parent.mkdir(exist_ok=True)
    shutil.copy2(path, saved)
    dependencies.append({'path': str(path.relative_to(LIVE)), 'copy': str(saved.relative_to(LAB)),
                         'sha256': hashlib.sha256(saved.read_bytes()).hexdigest()})
    subprocess.run(['git', 'apply', '--check', str(saved)], cwd=ROOT, check=True)
    subprocess.run(['git', 'apply', str(saved)], cwd=ROOT, check=True)

changed = {}
patch = []
for src in sorted(ROOT.rglob('*.py')):
    rel = src.relative_to(ROOT)
    if '__pycache__' in rel.parts or (len(rel.parts)>2 and rel.parts[:2] == ('cadathlon','validation') and len(rel.parts)>3):
        continue
    old = LIVE / rel
    before = old.read_bytes() if old.exists() else None
    after = src.read_bytes()
    if before == after:
        continue
    changed[str(rel)] = {'before_sha256': hashlib.sha256(before).hexdigest() if before is not None else None,
                         'after_sha256': hashlib.sha256(after).hexdigest()}
    if before is not None:
        saved = LAB / 'changed_base_sources' / rel
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(before)
    patch.extend(difflib.unified_diff((before or b'').decode().splitlines(True), after.decode().splitlines(True),
                 fromfile='a/'+str(rel) if before is not None else '/dev/null', tofile='b/'+str(rel)))
(LAB/'combined.patch').write_text(''.join(patch))
subprocess.run(['git','apply','--check',str(LAB/'combined.patch')],cwd=LIVE,check=True)
(LAB/'dependencies.json').write_text(json.dumps(dependencies,indent=2)+'\n')
(LAB/'changed_sources.json').write_text(json.dumps(changed,indent=2)+'\n')
print(json.dumps({'checkout':str(ROOT),'copied_tracked_files':len(list(filter(None,paths))),
                  'changed_files':len(changed),'dependencies':dependencies},indent=2))
