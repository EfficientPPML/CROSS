"""Bounded stdlib source/AST readback; never imports Q5 or numeric modules."""
from pathlib import Path
import ast,hashlib,json,tarfile
R=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-20261104')
H=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-handoff-20261104')
O=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def canonical(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def inventory(p):
    assert not any(x.is_symlink() for x in p.rglob('*'))
    return {x.relative_to(p).as_posix():sha(x.read_bytes()) for x in sorted(p.rglob('*')) if x.is_file()}
def tar_map(p):
    with tarfile.open(p,'r:gz') as t:
        members=t.getmembers();assert all(x.isfile() for x in members)
        assert len(members)==len({x.name for x in members})
        return {x.name:sha(t.extractfile(x).read())for x in members}
def tree(n):return ast.parse((R/n).read_bytes())
def function(t,name):return next(x for x in ast.walk(t)if isinstance(x,ast.FunctionDef)and x.name==name)
assert sha((H/'handoff.json').read_bytes())=='d49dbcd4092b49b0df16cfc94e89fc34be126eb29ef800c46929cbf5ae78eb6a'
assert sha((H/'sources.json').read_bytes())=='b787490da5ad804dacdeca7c19cbb39a99868ab80ea762a607e1a6457cfb3180'
assert sha((H/'sources.tar.gz').read_bytes())=='6cf407cc9c5b8b310acc282ab55d8c0a4ac9d59d8d89e845649af7ec74ab89b8'
m=json.loads((H/'sources.json').read_bytes());assert len(m)==382 and inventory(R)==tar_map(H/'sources.tar.gz')==m
parent=json.loads((R/'parent250_sources.json').read_bytes())
assert len(parent)==250 and {n[7:]:d for n,d in m.items()if n.startswith('parent/')}==parent
copies=json.loads((R/'copy_origins.json').read_bytes());assert len(copies)==339
for name,row in copies.items():assert m[name]==row['sha256']
core=json.loads((R/'core_source_manifest.json').read_bytes());assert len(core)==34
for name,digest in core.items():assert m[name]==digest
oldroot=Path('/tmp/qwen-toy5-actual-key-io-layer-20261104');oldh=Path('/tmp/qwen-toy5-actual-key-io-layer-handoff-20261104')
assert sha((oldh/'sources.json').read_bytes())=='58ca1429aaf242555c8fd49aeb7ae9b2dcc26b12eb161e3438982d46c63d153b'
assert sha((oldh/'sources.tar.gz').read_bytes())=='5111a536df6b666a7c96ea1f4e541cda385c41cc0394b4a1fa2694ac3c1792bf'
oldmap=json.loads((oldh/'sources.json').read_bytes());assert len(oldmap)==18 and inventory(oldroot)==tar_map(oldh/'sources.tar.gz')==oldmap
selection=json.loads((R/'source_selection.json').read_bytes())
assert selection['actual_runtime_issuer'] is False and selection['genuine_backend_imports_enabled'] is False
assert selection['execution_grants'] is None and set(selection['role_modules'])=={'setup','server','client'}
for role,modules in selection['role_modules'].items():
    for name,path in modules.items():
        assert path in m and path.endswith('.py') and not path.startswith(('parent/','reference/frozen/'))
        if name.startswith('reference'):assert role=='client'
    assert 'q5_toy_weights' in modules if role=='setup' else 'q5_toy_weights' not in modules
entry=tree('q5_actual_entry.py');dispatch=function(entry,'dispatch')
assert len(dispatch.body)==1 and isinstance(dispatch.body[0],ast.Raise)
assert not any(isinstance(n,(ast.Import,ast.ImportFrom))for n in ast.walk(dispatch))
proc=tree('q5_actual_process.py')
runtime=next(n for n in proc.body if isinstance(n,ast.ClassDef)and n.name=='RuntimeAdmission')
check=next(n for n in runtime.body if isinstance(n,ast.FunctionDef)and n.name=='check')
assert isinstance(check.body[-1],ast.Raise)
assert isinstance(function(proc,'run_actual').body[0],ast.Raise)
deps=tree('q5_actual_dependencies.py');arithmetic=function(deps,'arithmetic')
table=ast.literal_eval(next(n.value for n in arithmetic.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='table'for t in n.targets)))
six={name:mod for name,mod in table};assert len(six)==6
exercise=function(tree('q5_checkpoint_arithmetic.py'),'exercise')
required=ast.literal_eval(next(n.value for n in exercise.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='required'for t in n.targets)))
assert set(six)==required
definitions={}
for name,mod in six.items():
    path='common/'+mod+'.py';original='parent/runtime/cadathlon/common/'+mod+'.py'
    assert (R/path).read_bytes()==(R/original).read_bytes()
    definition=next(n for n in tree(path).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
    definitions[name]=dict(path=path,source_sha256=m[path],definition_ast_sha256=sha(ast.dump(definition,include_attributes=False).encode()))
assert m['q5_checkpoint_arithmetic.py']==parent['q5_checkpoint_arithmetic.py']
assert m['runtime/jaxite_word/ckks_ctx.py']=='bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
assert m['runtime/jaxite_word/key_gen.py']=='742e6af9e1ce9025416fa2ef68ae037bca7e772d48fbbd6ca387cc79820119c8'
codec=function(tree('runtime/jaxite_word/ckks_ctx.py'),'decrypt_slots')
assert [a.arg for a in codec.args.args]==['self','ciphertext','scale','slots_to_decode']
assert [a.arg for a in codec.args.kwonlyargs]==['validate_approximation'] and ast.literal_eval(codec.args.kw_defaults[0]) is True
roles=tree('q5_checkpoint_roles.py');client=function(roles,'client_bound')
decoder_calls=[n for n in ast.walk(client)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='decode']
assert len(decoder_calls)==1
d=decoder_calls[0];assert [ast.unparse(a)for a in d.args]==['ctx','value']
k={v.arg:ast.unparse(v.value)for v in d.keywords}
assert k=={'scale':"artifact['metadata']['scale']",'slots_to_decode':'65536','validate_approximation':'True'}
initial_scan=next(n for n in ast.walk(client)if isinstance(n,ast.For)and ast.unparse(n.target)=='(index, artifact)'and any(isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=='read'for x in ast.walk(n))and n.lineno<170)
private_with=next(n for n in ast.walk(client)if isinstance(n,ast.With)and any(isinstance(i.context_expr,ast.Call)and isinstance(i.context_expr.func,ast.Name)and i.context_expr.func.id=='private_loader'for i in n.items))
assert initial_scan.end_lineno<private_with.lineno
targets=['SOURCE_DESIGN.md','source_selection.json','RESOURCE_STATUS.json','q5_actual_entry.py','q5_actual_bootstrap.py','q5_actual_dependencies.py','q5_actual_process.py','q5_role_authority.py','q5_checkpoint_roles.py','q5_key_admission.py','q5_key_loader.py','q5_fixture_assets.py','qwen_backend_identity.py','q5_checkpoint_arithmetic.py','runtime/jaxite_word/ckks_ctx.py','core_source_manifest.json',*[v['path']for v in definitions.values()]]
receipt=dict(schema='qwen-toy5-disabled-runtime-cnn-source-review-v1',source_only=True,passed_within_scope=True,
    source_count=382,source_manifest_sha256=sha((H/'sources.json').read_bytes()),source_archive_sha256=sha((H/'sources.tar.gz').read_bytes()),handoff_sha256=sha((H/'handoff.json').read_bytes()),
    current_archive_exact=True,parent250_exact=True,copied339_metadata_exact=True,core34_exact=True,old18_current_archive_preserved=True,
    reviewed_sources={n:m[n]for n in targets},original_six_definitions=definitions,
    concrete_findings=[],
    implemented_boundaries=[
      'New entry dispatch denies immediately before helper/source/body/RNG; normal exactint0/1 and escapedBaseException70 guard is declared, with argparse outside.',
      'Phase schemas bind separate setup/server/client source IDs, external setup/server terminals and disjoint borrowed/output roots; all remain preparation-only with actual_execution_enabled false.',
      'Closed selector distinguishes active source names from historical parent/reference-frozen copies; fixed34 core is delegated behind the original finder. Numeric imports remain prohibited.',
      'ActualBackend names exact six original250 common definitions and defaultbc978 unbound codecs; reviewed selected call forms match current source signatures.',
      'Setup composes fixed public toy generation,203 public arrays/one secret and one freshL49 input behind unavailable runtime checks; same-setup key identity is not independent readiness.',
      'Server requests body admission after key scope and before public assets/input/graph; original owned consumers and guards retire before loader exit.',
      'Client performs47 full artifact reads before invoking private_loader, including before private_manifest metadata open. Default decoder uses actual scale/full65536/validate_approximation=True and immutable copied snapshot; no retry after approximation failure.',
      'Captured original finish/close/codec/owner callbacks and late guards cross the entered key loader on failure. Report publication precedes evidence close and still requires independent normal terminal/outer/root admission.'
    ],
    incomplete_actual_implementations=[
      'RuntimeAdmission has no issuer; check() always raises after phase validation. Exact class construction is not authority.',
      'All runtime scope/source/security/common-definition/load-handle methods remain failing declarations. This review does not treat their names as implemented identity or audit enforcement.',
      'Installed/native dependency policy, audit/device permission transitions, current capacity/resource leases, source-import activation and per-phase role grants remain unavailable.',
      'No executable actual closure follows from the selected callgraph. Caller/global/native containment and callbacks are not dynamically tested here.'
    ],
    review_limits=[
      'Root separately reviews CT/NTT reconstruction, opaque transport, asset-reader internals and cleanup mechanics; this receipt covers role ordering/selection/caller signatures only.',
      'Old18 review and original250 synthetic evidence do not prove this382 actual composition. No new fixture/import/recording or native execution occurred.',
      'Source hashes and AST equality are public source checks, not mathematical/HE/capacity/first-divergence or actual body-integrity claims.'
    ],candidate_imports=False,collection=False,tests=False,backend_or_capacity_probes=False,borrowed_body_reads=False,actual_ready=False,execution_authorized=False)
assert inventory(R)==m and inventory(oldroot)==oldmap
(O/'review.json').write_bytes(canonical(receipt))
print(json.dumps(dict(passed_within_scope=True,review_sha256=sha((O/'review.json').read_bytes()),source_count=382,actual_ready=False,candidate_imports=False,tests=False)))
