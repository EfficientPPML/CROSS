"""Public-only six-file heldout successor delta; no actual execution grant."""
from pathlib import Path
import hashlib,json
LAB=Path('/tmp/qwen-heldout1088-runtime-v2-20261104')
MAP=Path('/tmp/qwen-heldout1088-runtime-sources-v2-20261104.json')
OLDMAP=Path('/tmp/qwen-heldout1088-runtime-sources-v1-20261104.json')
OUT=Path('/tmp/qwen-heldout-runtime-root-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
raw=MAP.read_bytes();expected=json.loads(raw);old=json.loads(OLDMAP.read_bytes())
assert sha(raw)=='cce2630028708f41e79614bb367039fd146fc924471bfd674ade1737f9c6017e' and len(expected)==167
assert sha(OLDMAP.read_bytes())=='7c54f2b9acfa0b8ba4d12bc9676259298db013f41196856bc71a1f1e77215098'
found={}
for p in LAB.rglob('*'):
    assert not p.is_symlink()
    if p.is_file():found[p.relative_to(LAB).as_posix()]=sha(p.read_bytes())
    else:assert p.is_dir()
assert found==expected and set(expected)==set(old)
changed={n for n in old if expected[n]!=old[n]}
assert changed=={'heldout_authorization.py','test_heldout_authorization.py','heldout_preflight_expectations.json',
    'heldout_source_origins.json','README.md','RUNTIME_PLAN.md'},changed
parent=json.loads(Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json').read_bytes())
assert len(parent)==139 and all(expected[n]==h for n,h in parent.items())
before=(OUT/'review-v1.json').read_bytes();assert sha(before)=='0b9b63e1fee161def84acdc4537124b942c7c48bebc0313b0fa65cbcdfe4eaa4'
assert json.loads(before)['passed'] is False
policy=json.loads((LAB/'heldout_preflight_expectations.json').read_bytes())
assert policy['synthetic_tests']==478 and policy['installed_import_denials']==10
assert json.loads((LAB/'execution_authorization.json').read_bytes())=={'execute':False}
report=dict(schema='qwen-heldout167-narrow-root-source-review-v2',passed=True,ready_for_full_preflight=True,
    source_manifest_sha256=sha(raw),source_count=167,parent139_exact=True,predecessor_review_sha256=sha(before),
    changed_sources={n:dict(before=old[n],after=expected[n])for n in sorted(changed)},
    runtime_process_entry_adapters_freeze_unchanged=True,
    source_findings=[
        'An explicit independent_row0 grant admits the same complete calibration freeze with root-attested no prior attempt or known-retired terminal attempt.',
        'The prior attempt is never admitted as a scientific result; source/cleanup/publication/read observations may remain honestly unknown.',
        'Unknown worker existence or group absence rejects. Read status variants enforce typed attempted/authenticated/reauthenticated observations.',
        'No new heldout launch, numerical threshold, feedback, tuning, candidate revision or retry is authorized by the independent row0 receipt.',
        'Standard heldout-then-row0 scheduling remains a documented operational choice, not an additional numerical gate.'
    ],unresolved_concrete_findings=[],
    development_tests=dict(owner='CNN',tests=478,session=15364,reported_exit_code=0,reported_reaped=True,
        qualification='Development receipt only; source-bound478+10 proof remains required.'),
    scope='Source/delta review only; no model/backend execution, private input, candidate outcome or validation body reads',
    actual_execution_authorized=False,review_script_sha256=sha(Path(__file__).read_bytes()))
with (OUT/'review-v2.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review-v2.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(passed=True,review=str(OUT/'review-v2.json'),sha256=sha((OUT/'review-v2.json').read_bytes()),sources=167,ready_for_full_preflight=True)))
