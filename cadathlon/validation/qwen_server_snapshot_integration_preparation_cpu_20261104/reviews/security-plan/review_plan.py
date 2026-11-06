"""Bounded source-only integration-plan review; no proposed code execution."""
import ast
import hashlib
import json
from pathlib import Path
P=Path('/tmp/qwen-server-snapshot-integration-plan-20261104')
H=Path('/tmp/qwen-server-snapshot-integration-plan-handoff-20261104.json')
OUT=Path('/tmp/qwen-server-snapshot-integration-security-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return Path(p).read_bytes()
def parse(p):return json.loads(read(p))
h=parse(H);m=parse(P/'sources.json');assert len(m)==6
assert sha(read(P/'sources.json'))==h['map_sha256']
assert {p.name:sha(p.read_bytes())for p in P.iterdir()if p.is_file()and p.name!='sources.json'}==m
assert sha(read(P/'PLAN.md'))==h['plan_sha256']=='28687cde76bdf62272c99af08286980bc3c92be96fcf2f111b3b58fbc908aa1b'
refs=parse(P/'source_refs.json')['files']
for name,row in refs.items():
    b=read(name);assert len(b)==row['bytes']and sha(b)==row['sha256']
compat=parse(P/'compatibility.json');assert len(compat['helpers'])==14
for name,row in compat['helpers'].items():
    assert row['exact']is True and row['server_sha256']==row['tested_guard_dependency_sha256']
    assert refs['/tmp/qwen-server-actual-runtime-20261104/'+name]['sha256']==row['server_sha256']
guard=Path('/tmp/qwen-server-admission-snapshot-source-v2-20261104/qwen_admission_snapshot.py')
assert sha(read(guard))==h['guard_sha256']=='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
gt=ast.parse(read(guard));fields=next(n.value for n in gt.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_FIELDS')
assert isinstance(fields,ast.Tuple)and len(fields.elts)==13
modules=next(n.value for n in gt.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_MODULES')
assert len(modules.elts)==14
key=ast.parse(read('/tmp/qwen-server-actual-runtime-20261104/qwen_key_contract.py'))
ids=next(n for n in key.body if isinstance(n,ast.ClassDef)and n.name=='ExpectedIds')
id_fields=[n.target.id for n in ids.body if isinstance(n,ast.AnnAssign)]
assert id_fields==['context_id','public_manifest_sha256','keygen_source_sha256','model_id','producer_source_manifest_sha256','consumer_source_manifest_sha256','private_manifest_sha256']
server=Path('/tmp/qwen-server-actual-runtime-20261104')
resource=ast.parse(read(server/'qwen_server_resource.py'))
deny=next(n for n in resource.body if isinstance(n,ast.FunctionDef)and n.name=='require_complete_capacity')
assert len(deny.body)==1 and isinstance(deny.body[0],ast.Raise)and deny.body[0].exc.func.id=='PermissionError'
graph=parse(P/'callgraph.json')
assert set(graph['full_boundaries_retained'])=={'admit','capture','entry','post_import','enable_public_keys','enable_server_bodies','post_stage','terminal_cleanup'}
assert set(graph['proposed_runtime_overlays'])=={'qwen_server_process.py','qwen_server_backend.py','qwen_server_entry.py:ROOT_MODULES'}
assert graph['literal_addition']=='qwen_admission_snapshot.py'
assert graph['measured_acceleration']is False and graph['actual_authority']is False
assert all(sha(read(n))==row['sha256']for n,row in refs.items())
report=dict(schema='qwen-server-snapshot-integration-security-plan-review-v1',passed=True,material_blockers=[],
    plan_sha256=h['plan_sha256'],handoff_sha256=sha(read(H)),plan_map_sha256=h['map_sha256'],
    guard_sha256=h['guard_sha256'],plan_files=6,public_reference_count=len(refs),helper14_exact=True,
    selected_source_pins={n:refs[str(server/n)]['sha256']for n in('qwen_server_process.py','qwen_server_backend.py','qwen_server_entry.py','qwen_server_resource.py','qwen_finalizer.py')},
    accepted_scope=[
        'Literal879071 guard plus narrow process/backend/entry selection successor; no pure-validator or arithmetic changes.',
        'Nonrecursive source/origin/environment binding callback explicitly excludes runtime/guard/packet/phase and proof bodies.',
        'Original guard methods/source closure and derived7-ID bytes/input digest retained independently and checked before use.',
        'Complete proof/source/public-file revalidation preserved at finite process boundaries and terminal.',
        'All335 live reader, keyless, security, context ownership, audit transitions and capacity denial remain selected.',
        'Attempt-all cleanup and primary-error policy remain; guard retires only after successful source/audit/evidence closure.',
        'Failed capture/runtime/retirement retains original owners and guard state; late failure remains non-PASS/exit70.'],
    implementation_obligations=[
        'Capture complete current parent source inventory before edits; this plan pins only listed source slices.',
        'Mechanically list callback closure and bind original callbacks/expected map/blob objects independently; no mutable rebase.',
        'Make caught runtime binding/state rejection sticky, and keep the callback valid after audit restoration.',
        'Use original ExpectedIds/digest derivation with guard brackets and preserve actual request-file digest comparison.',
        'Test integrated recursion/count/ownership/replacement/late-retirement paths separately;18 guard tests do not prove integration.'],
    limitations=['No current full387 source-proof claim; current process/backend/entry slices match their exact listed pins.',
        'No timing estimate, scientific proof execution, body-read authority or capacity admission follows from this plan.',
        'No imports, tests, implementation edits, probes or actual grants performed.'],
    current_server_modified=False,implementation_review=False,actual_authority=False)
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'review':str(OUT/'review.json'),'sha256':sha(read(OUT/'review.json')),'public_refs':len(refs),'material_blockers':[]},sort_keys=True))
