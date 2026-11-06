"""Stdlib source-only harness preparation; never imports or launches candidates."""
from pathlib import Path
import ast, difflib, hashlib, json, re, shutil, tarfile

OUT=Path('/tmp/qwen-toy5-role-focused-proof-20261104/preparation-v1')
SOURCE=Path('/tmp/qwen-toy5-role-focused-tests-20261104')
HANDOFF=Path('/tmp/qwen-toy5-role-focused-test-handoff-20261104')
TEMPLATE=Path('/tmp/qwen-device-component-focused-proof-20261104/preparation-v1')
PYTHON='/home/jianming_gatech/miniconda3/envs/jaxite/bin/python'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canonical=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda value:(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def write(name,value):(OUT/name).write_bytes(canonical(value))
raw=(HANDOFF/'sources.json').read_bytes()
assert sha(raw)=='43a7e90b04a2b99edfb953a7724c3a50580dcff844c2da987a32ec1da7a41df9'
expected=json.loads(raw);assert compact(expected)==raw and len(expected)==427
assert not any(p.is_symlink()for p in SOURCE.rglob('*'))
current={p.relative_to(SOURCE).as_posix():sha(p.read_bytes())for p in sorted(SOURCE.rglob('*'))if p.is_file()}
assert current==expected
assert sha((HANDOFF/'sources.tar.gz').read_bytes())=='269a3661b3a84309bed36c7527c7ccedf06c086b1dd9e907e269b0a03b87380b'
with tarfile.open(HANDOFF/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==427 and all(m.isfile()for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==expected
execution=OUT/'execution'
if execution.exists():
    assert {p.relative_to(execution).as_posix():sha(p.read_bytes())for p in execution.rglob('*')if p.is_file()}==expected
else: execution.mkdir()
for name,digest in expected.items():
    target=execution/name;target.parent.mkdir(parents=True,exist_ok=True)
    body=(SOURCE/name).read_bytes();assert sha(body)==digest;target.write_bytes(body)
(OUT/'sources-before.json').write_bytes(raw)
shutil.copyfile(HANDOFF/'sources.tar.gz',OUT/'sources.tar.gz')
shutil.copyfile(HANDOFF/'handoff.json',OUT/'source-handoff.json')
selection=json.loads((SOURCE/'selected_tests.json').read_bytes())
artifacts=json.loads((SOURCE/'ARTIFACT_SELECTION.json').read_bytes())
numpy=json.loads((SOURCE/'numpy-pins.json').read_bytes())
helpers=json.loads((SOURCE/'helper-selection.json').read_bytes())
runtime_raw=(SOURCE/'runtime-map.json').read_bytes();runtime=json.loads(runtime_raw)
assert len(runtime)==389 and sha(runtime_raw)=='d82d68101c1f3ddc7f18b6ae4492aa7618d779012babaf9ed85355dfab927fea'
assert {n.removeprefix('runtime/'):v for n,v in expected.items()if n.startswith('runtime/')}==runtime
nodes=[]
for name in selection['files']:
    tree=ast.parse((SOURCE/name).read_text())
    nodes.extend(name+'::'+n.name for n in tree.body if isinstance(n,ast.FunctionDef)and n.name.startswith('test_'))
assert nodes==selection['nodes']and len(nodes)==55
assert len(artifacts['children'])==56 and artifacts['regular_file_limit']==392
child_tree=ast.parse((SOURCE/'case_child.py').read_text())
assignment=next(n for n in child_tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='NUMPY_ALLOWED'for t in n.targets))
numpy_cases=sorted(ast.literal_eval(assignment.value.comparators[0]));assert len(numpy_cases)==10
for path,digest in numpy['sources'].items():assert sha(Path(path).read_bytes())==digest
assert len(numpy['sources'])==4 and len(numpy['origins'])==3
xml_cases=[dict(classname=n.split('::')[0].removesuffix('.py'),name=n.split('::')[1])for n in nodes]
forbidden=sorted(set(helpers['flat_core_names_denied'])|set(helpers['helpers'])|set(helpers['denied_unused_top_level'])|{'numpy','jax','jaxlib','jax_plugins','jaxite_word','libtpu','torch','torch_xla','tensorflow','transformers','flax','safetensors','tokenizers'})
config=dict(schema='qwen-toy5-role-focused-proof-v1',synthetic_only=True,actual_execution_authorized=False,
    source_root=str(SOURCE),proof_root=str(OUT),python=PYTHON,cpu_affinity=[108,109,110,111],
    source_count=427,runtime_count=389,source_manifest_sha256=sha(raw),
    source_manifest_encoding='compact sorted JSON with separators comma/colon and trailing newline',
    source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
    selected_files=selection['files'],selected_nodes=nodes,selected_nodes_sha256=sha(canonical(nodes)),
    selected_xml_cases=xml_cases,selected_xml_cases_sha256=sha(canonical(xml_cases)),
    selection_sha256=sha((SOURCE/'selected_tests.json').read_bytes()),runtime_map_sha256=sha(runtime_raw),
    runtime_source_expectations=runtime,runtime_module_origins={},
    artifact_plan=artifacts,artifact_plan_sha256=sha((SOURCE/'ARTIFACT_SELECTION.json').read_bytes()),
    expected_children=56,expected_regular_child_files=392,max_regular_bytes=392*1048576,
    numpy_pins=numpy,numpy_pins_sha256=sha((SOURCE/'numpy-pins.json').read_bytes()),numpy_allowed_cases=numpy_cases,
    worker_forbidden_roots=forbidden,selected_helpers=sorted(helpers['helpers']),
    owned_tmpdir=str(OUT/'tmp'),pytest_basetemp=str(OUT/'pytest-owned'),
    pytest_config='/dev/null',actual_role=False,backend_context=False,capacity_observation=False,
    scope='55 focused tests in56 owned children: pure synthetic authority, recording constructors/owners/role bodies, tiny owned public I/O and disabled-entry negatives. Exact389 runtime retained. No actual authority, RNG/keygen, learned bodies, native backend/context, device/capacity, HE, first-divergence or performance claim. NumPy only in10 declared child cases;4 installed source hashes and observed3 module origins, not full native dependency proof.')
write('run_config.json',config)
worker_sha=sha((OUT/'worker_driver.py').read_bytes());inspector_sha=sha((OUT/'inspect_children.py').read_bytes())
driver=(OUT/'executed_driver.py').read_text()
for key,value in [('CONFIG_SHA256',sha(canonical(config))),('WORKER_SHA256',worker_sha),('INSPECTOR_SHA256',inspector_sha)]:
    driver,count=re.subn(r"^"+key+r" = '[^']*'$",key+" = "+repr(value),driver,flags=re.M)
    assert count==1
(OUT/'executed_driver.py').write_text(driver)
for name in ('executed_driver.py','worker_driver.py','inspect_children.py','prepare_source.py'):
    ast.parse((OUT/name).read_text(),filename=str(OUT/name))
for before_name,current_name in [('executed_driver.py','executed_driver.py'),('worker_driver.py','worker_driver.py')]:
    before=(TEMPLATE/before_name).read_bytes();(OUT/('before_'+before_name)).write_bytes(before)
    patch=''.join(difflib.unified_diff(before.decode().splitlines(True),(OUT/current_name).read_text().splitlines(True),fromfile='admitted-device50/'+before_name,tofile='unrun-q5-427/'+current_name))
    (OUT/(current_name+'.patch')).write_text(patch)
provenance=dict(schema='qwen-toy5-role-harness-source-provenance-v1',
    template={name:dict(path=str(TEMPLATE/name),sha256=sha((TEMPLATE/name).read_bytes()))for name in ('executed_driver.py','worker_driver.py')},
    changes=['schema/config/source roots/counts/static55 and389 runtime pins','matching genuine NumPy interpreter and4 source hashes; numeric imports forbidden in enclosing worker;10 child exceptions already source-bound','explicit -c /dev/null and exact6 files ordered55 before fixtures','fixed392 receipt/log selection with metadata-only fixture roots and unfollowed aliases','immediate Popen ownership retained, dedicated direct-wait flag/startticks and secondary-preserving final owner publication added','separate pre-exit collector/worker records; actual collector normal exit/reap requires external root/tool','pinned stdlib inspector is separate source, never a runtime import'],
    ownership_AST_identical=False,
    ownership_statement='Derived from admitted device50 immediate-ownership/nested-direct-wait pattern; augmented to preserve explicit normal-wait evidence and primary through final observation/publication. Review required; no executed-equivalence claim.',
    source_imported=False,collection=False,executed=False,actual_authority=False,
    authoring_history='development/ast-v1 preserves a missing lexical space caught by stdlib AST parsing before any candidate import, collection or run; only worker spacing and source preparation restart handling corrected.',
    root_source_review=dict(path='/tmp/qwen-toy5-role-focused-root-source-review-20261104/review.json',sha256='1bb47f6046cd468ed4117fa7bf91f99a04293bcb33ec0860d619edeb2983ae5f'))
write('driver_provenance.json',provenance)
argv=['env','-i','PATH=/usr/bin:/bin','HOME=/home/jianming_gatech','LANG=C.UTF-8','TZ=UTC','PYTHONDONTWRITEBYTECODE=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c','108-111',PYTHON,'-I','-B',str(OUT/'executed_driver.py')]
proposal=dict(schema='qwen-toy5-role-focused-launch-proposal-v1',held_unrun=True,launch_authorized=False,
    argv=argv,proof_root=str(OUT),source_root=str(SOURCE),source_count=427,runtime_count=389,
    static_cases=55,expected_children=56,expected_regular_child_files=392,
    source_manifest_sha256=sha(raw),source_archive_sha256=config['source_archive_sha256'],
    driver_sha256=sha((OUT/'executed_driver.py').read_bytes()),worker_sha256=worker_sha,
    inspector_sha256=inspector_sha,config_sha256=sha(canonical(config)),
    candidate_imported=False,collection=False,execution=False,backend_or_capacity=False,
    fixture_bodies_opened_by_collector=False,aliases_followed=False,
    current_execution_archive_equal=True,resource_check_required_by_root=True)
write('launch-proposal.json',proposal)
write('static-review.json',dict(passed=True,scope='stdlib source-only current/execution/tar and AST review',
    source_count=427,runtime_count=389,static_nodes=55,prospective_children=56,prospective_regular_files=392,
    numpy_source_hashes=4,source_manifest_sha256=sha(raw),source_archive_sha256=config['source_archive_sha256'],
    candidates_imported=False,collection=False,tests_executed=False))
(OUT/'README.md').write_text('''# Held, unrun Q5 role-focused preparation\n\nThis enclosing source-only harness selects 55 tests and 56 direct children against the exact nested389 runtime in held427. The separate root source review covers the test closure; it does not grant execution. CPU108–111 remains conditional on root resource review. The launch proposal uses the pinned jaxite Python interpreter, env-i, -I -B, plugin autoload disabled, -c /dev/null, and exact ordered prefixture collection. No inherited suite is selected.\n\nThe worker itself forbids NumPy and backend imports. Ten declared child cases permit genuine NumPy for tiny owned public files or float64 snapshot checks. Four installed source files and the three observed module origins are checked; this does not attest every native NumPy dependency. No RNG/key generation, real key/context/device/backend, borrowed body, actual role or HE is authorized. The runtime remains default-deny.\n\nThe collector owns its Popen immediately, never uses a healthy timeout, and always attempts the original direct wait after group cleanup. A dedicated flag is set only after wait returns normally. It records separate worker early/pre-exit/owner-terminal and collector pre-exit receipts. Root/tool must independently record the actual collector exit/reap/group absence; the collector cannot certify its own reap. Primary exceptions survive final observation/publication errors.\n\nThe inspector reads only the seven fixed public receipt/log names per child (392 expected; each at most1MiB) and joins original PID, PGID, start ticks, case, variant, source, expected exit and direct wait records. It never follows pytest aliases. Child fixture roots owned_io and owned_assets receive bounded lstat metadata only; no fixture body is opened, hashed or copied. The deliberately preserved FIFO weight_00.npy is never opened by collection. Unexpected/missing evidence rejects readiness; original files remain available.\n\nSource/execution/archive maps and NumPy source hashes are checked before/after a future run. This preparation has only stdlib source copying/hashing/AST parsing, with no candidate imports, collection, tests, runtime probes or launch. The driver differs explicitly from the admitted device50 template; driver_provenance.json and exact patches describe that delta. All prior proofs remain unchanged.\n''')
files=['executed_driver.py','worker_driver.py','inspect_children.py','run_config.json','launch-proposal.json',
       'source-handoff.json','sources-before.json','sources.tar.gz','driver_provenance.json','static-review.json',
       'README.md','prepare_source.py','before_executed_driver.py','before_worker_driver.py',
       'executed_driver.py.patch','worker_driver.py.patch',
       'development/ast-v1/history.json','development/ast-v1/executed_driver.py.txt',
       'development/ast-v1/worker_driver.py.txt','development/ast-v1/inspect_children.py.txt',
       'development/ast-v1/prepare_source.py.txt','development/ast-v1/run_config.json.txt']
pins={name:dict(path=str(OUT/name),sha256=sha((OUT/name).read_bytes()),bytes=(OUT/name).stat().st_size)for name in files}
write('handoff.json',dict(schema='qwen-toy5-role-focused-harness-handoff-v1',held=True,unrun=True,
    source_count=427,runtime_count=389,static_nodes=55,prospective_children=56,prospective_regular_files=392,
    proof_root=str(OUT),source_root=str(SOURCE),cpu_affinity=[108,109,110,111],python=PYTHON,
    source_manifest_sha256=sha(raw),source_archive_sha256=config['source_archive_sha256'],files=pins,
    candidate_imports=False,collection=False,tests_executed=False,launch_authorized=False,
    no_fixture_body_capture=True,no_alias_traversal=True,
    next='Root/peer source/harness review and explicit later launch authorization; root owns resource decision.'))
print(json.dumps(dict(held=str(OUT),handoff_sha256=sha((OUT/'handoff.json').read_bytes()),
    driver_sha256=proposal['driver_sha256'],worker_sha256=worker_sha,inspector_sha256=inspector_sha,
    config_sha256=proposal['config_sha256'],proposal_sha256=sha((OUT/'launch-proposal.json').read_bytes()),
    source_count=427,static_nodes=55,prospective_children=56),sort_keys=True))
