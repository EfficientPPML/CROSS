"""Root source/AST verification of held29; no candidate imports or body reads."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/tmp/qwen-model-stage-aggregate-20261104')
PARENT=Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104')
PLAN=Path('/tmp/qwen-model-stage-aggregate-plan-v2-20261104')
MAP=Path('/tmp/qwen-model-stage-aggregate-review-sources-v1-20261104.json')
PEER=Path('/tmp/qwen-model-stage-transformers-review-20261104/review.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def dump(n):return ast.dump(n,include_attributes=False)
assert sha(MAP)=='613be1bd3e69ea5284882a2241913c02bbbbcdf11665e04fc42db0801d30ee6a'
expected=read(MAP);assert len(expected)==29
assert not any(p.is_symlink()for p in SOURCE.rglob('*'))
assert {p.relative_to(SOURCE).as_posix():sha(p)for p in SOURCE.rglob('*')if p.is_file()}==expected
selection=read(PLAN/'source_selection.json');exact=0
for name,row in selection['base_modules'].items():
    assert sha(Path(row['path']))==row['sha256']
    if name!='qwen_public_parameters.py':assert expected[name]==row['sha256'];exact+=1
for name,row in selection['base_public_metadata'].items():
    assert expected['public_inputs/'+name]==sha(Path(row['path']))==row['sha256'];exact+=1
assert exact==19
for name in ('admission.schema.json','aggregate_manifest.schema.json','unadmitted_aggregate_manifest.json'):
    assert (SOURCE/name).read_bytes()==(PLAN/name).read_bytes()
before=ast.parse((PARENT/'qwen_public_parameters.py').read_bytes())
after=ast.parse((SOURCE/'qwen_public_parameters.py').read_bytes())
old={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name in ('flat_name','ParameterCatalog','_FdReader','_hash_fd','_header','open_source_bound_parameters'):
    assert dump(old[name])==dump(new[name])
changed={'_HeldFile':{'__init__','check'},'_MatrixLease':{'__init__'},'PublicParameterFiles':{'__init__','check','matrix','gamma'}}
unchanged_methods=[]
for cls,allowed in changed.items():
    a={n.name:n for n in old[cls].body if isinstance(n,ast.FunctionDef)}
    b={n.name:n for n in new[cls].body if isinstance(n,ast.FunctionDef)}
    for name,node in a.items():
        if name not in allowed:assert dump(node)==dump(b[name]);unchanged_methods.append(cls+'.'+name)
assert set(new)-set(old)=={'open_identity_bound_parameters','open_actual_admitted_parameters'}
for file,entry in [('model_admission.py','create_actual_model_admission'),('model_store_validation.py','create_actual_model_validator'),('qwen_public_parameters.py','open_actual_admitted_parameters')]:
    fn=next(n for n in ast.parse((SOURCE/file).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==entry)
    assert len(fn.body)==1 and isinstance(fn.body[0],ast.Raise)and fn.body[0].exc.func.id=='PermissionError'
pure=ast.parse((SOURCE/'model_admission.py').read_bytes());imports=set()
for n in ast.walk(pure):
    if isinstance(n,ast.Import):imports.update(a.name for a in n.names)
    elif isinstance(n,ast.ImportFrom):imports.add(n.module)
assert imports=={'dataclasses','pathlib','re','qwen_file_contract','qwen_public_parse','public_source_pins'}
assert sha(PEER)=='7e88158249379f715750c2701faeef749a19f9316ad1a7fdf30cafbeee475f75'
auth=read(SOURCE/'execution_authorization.json')
assert all(auth[n]is False for n in ('actual_model_validation_enabled','actual_parameter_factory_enabled','backend_context_enabled','model_body_reads_authorized','model_forward_enabled','tests_executed'))
report=dict(schema='qwen-model-stage-root-source-review-v1',passed=True,source_count=29,exact_predecessor_copies=19,
    source_manifest_sha256=sha(MAP),peer_review_sha256=sha(PEER),
    modules={n:expected[n]for n in ('model_admission.py','model_store_validation.py','qwen_public_parameters.py')},
    unchanged_reader_methods=unchanged_methods,script_sha256=sha(Path(__file__)),
    source_plan_review_sha256=sha(HERE/'clarification-review.json'),
    scope='Source/AST only: pure fixed-content evidence gate, pre-body held-identity overlay, and full catalog recording visit/cleanup. No tests or existing tensor reads.',
    required_future_wiring='Actual model admission and matching model_id must precede RNG or key-loader entry; parameter factory inside a loader cannot be the outer actual gate.',
    source_map_serialization='The new intent/pins require canonical compact source-map bytes; future actual source/launch evidence must bind that exact encoding, not assume a pretty-printed map has the same digest.',
    tests_executed=0,actual_execution_authorized=False,next='Author bounded focused synthetic tests in a separate successor; review before collection/execution.')
(HERE/'implementation-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
