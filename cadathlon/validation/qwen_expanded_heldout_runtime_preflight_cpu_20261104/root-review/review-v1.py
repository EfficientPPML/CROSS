"""Public-only narrow heldout167 predecessor review; no execution authority."""
from pathlib import Path
import ast,hashlib,json
LAB=Path('/tmp/qwen-heldout1088-runtime-20261104')
MAP=Path('/tmp/qwen-heldout1088-runtime-sources-v1-20261104.json')
PARENT=Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json')
OUT=Path('/tmp/qwen-heldout-runtime-root-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
raw=MAP.read_bytes();expected=json.loads(raw)
assert sha(raw)=='7c54f2b9acfa0b8ba4d12bc9676259298db013f41196856bc71a1f1e77215098' and len(expected)==167
actual={}
for p in LAB.rglob('*'):
    assert not p.is_symlink()
    if p.is_file():actual[p.relative_to(LAB).as_posix()]=sha(p.read_bytes())
    else:assert p.is_dir()
assert actual==expected
parent=json.loads(PARENT.read_bytes())
assert sha(PARENT.read_bytes())=='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
assert len(parent)==139 and all(actual[n]==h for n,h in parent.items())
names=('heldout_entry.py','heldout_process.py','heldout_runtime.py','heldout_adapters.py','heldout_dispatch.py',
    'heldout_authorization.py','heldout_input_gate.py','freeze_admission.py','heldout_protected_paths.json','heldout_runtime_policy.json')
reviewed={n:actual[n]for n in names}
adapter=ast.parse((LAB/'heldout_adapters.py').read_bytes())
cls=next(n for n in adapter.body if isinstance(n,ast.ClassDef)and n.name=='HeldoutAdapters')
assert {n.name for n in cls.body if isinstance(n,ast.FunctionDef)}=={'__init__','_guard','binding','run'}
assert isinstance(cls.bases[0],ast.Attribute) and cls.bases[0].value.id=='measured' and cls.bases[0].attr=='ContextAdapters'
state=Path('/tmp/qwen-heldout-read-state-root-review-20261104')
negative=json.loads((state/'negative-v1/review.json').read_bytes());fixed=json.loads((state/'successor-v1/review.json').read_bytes())
assert negative['expected_failures']==3 and negative['passing_control']==1 and fixed['passed'] is True and fixed['tests']==4
assert negative['source_hashes_before']['test_read_state.py']==fixed['identical_test_sha256']
assert json.loads((LAB/'execution_authorization.json').read_bytes())=={'execute':False}
report=dict(schema='qwen-heldout167-narrow-root-source-review-v1',source_review_completed=True,passed=False,
    ready_for_full_preflight=False,source_manifest_sha256=sha(raw),source_count=167,parent139_exact=True,
    reviewed_sources=reviewed,unchanged_numeric_methods_inherited=True,
    scope='Public source/AST and existing synthetic receipt read only; no model/backend execution, private input, candidate outcome or heldout body reads',
    source_findings=[
        'New heldout/row0 adapter overrides workload identity and aggregation only; measurement, metrics, domains and independent embeddings are inherited unchanged.',
        'Separate runtime reauthenticates freeze/grant before validation body and model access; row0 constructs no ValidationReader.',
        'PASS retains1120/2 actual native calls. Known invalid native/exact comparisons may complete a scientific failure with honest partial counters; no feedback/revision is authorized.',
        'Confirmed validation-read state survives later exceptions and final recheck failures; process-owned telemetry cannot be overwritten by numerical result fields.',
        'The actual candidate private and public outcome roots are declared protected source metadata. Lexical denial avoids inspecting those roots during preparation.'
    ],
    repaired_read_state_negative_sha256=sha((state/'negative-v1/review.json').read_bytes()),
    repaired_read_state_successor_sha256=sha((state/'successor-v1/review.json').read_bytes()),
    unresolved_concrete_findings=[dict(issue='The current row0 authorization gate requires a complete scientific heldout result, adding a prerequisite absent from the frozen protocol.',
        resolution='Separate successor must admit independently root-granted row0 from the same firstcalPASS freeze, with honest none/retired prior-attempt state and no scientific-result relabeling, tuning or retry.')],
    development439_note='Owner reports439 development checks; metadata/docs changed afterward. This is not a source-bound439 proof or actual execution authorization.',
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700,exist_ok=True)
with (OUT/'review-v1.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review-v1.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(review=str(OUT/'review-v1.json'),sha256=sha((OUT/'review-v1.json').read_bytes()),sources=167,passed=False,remaining_issue='row0 authorization prerequisite')))
