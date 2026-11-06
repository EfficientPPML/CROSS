from pathlib import Path
import ast,hashlib,json
OUT=Path(__file__).parent;SOURCE=Path('/tmp/qwen-full28-device-focused-tests-20261104/cases_loader_engine.py');RUNTIME=Path('/tmp/qwen-server-device-integration-runtime-v4-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pin(p):
 p=Path(p);assert p.is_file()and not p.is_symlink();raw=p.read_bytes();return dict(path=str(p),sha256=sha(raw),bytes=len(raw))
def write(name,v):(OUT/name).write_bytes(enc(v))
tree=ast.parse(SOURCE.read_text(),filename=str(SOURCE));cases=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CASES'for t in n.targets)))
plan=Path('/tmp/qwen-full28-device-focused-test-plan-v2-20261104/nodes.json');nodes=[r for r in json.loads(plan.read_bytes())['nodes']if r['group']in('L','E')]
assert tuple(r['case']for r in nodes)==cases and len(cases)==8
assert not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name.startswith('test_')for n in tree.body)
assert {n.name for n in tree.body if isinstance(n,ast.FunctionDef)}=={'_setup','_start','_dependency_for','_loader_case','_engine_case','run'}
refs=[pin(plan),pin('/tmp/qwen-server-device-integration-handoff-v4-20261104/sources.json'),pin('/tmp/qwen-full28-native403-security-review-20261104/review.json')]
for name in ('qwen_key_loader.py','qwen_server_backend.py','qwen_stage_composition.py','qwen_backend_composition.py','qwen_server_device_owner.py','qwen_file_contract.py','qwen_key_contract.py','qwen_key_adapter.py','qwen_key_files_admission.py','qwen_key_readers.py','qwen_publisher.py','qwen_finalizer.py'):
 refs.append(pin(RUNTIME/name))
write('source_refs.json',refs);write('selected_nodes.json',nodes)
seams={
 'schema':'qwen-full28-loader-engine-recording-seams-v1',
 'runtime_control_real':['ActualServerDependencies','ActualServerFactories','ServerBackend constructor/check/callbacks','_load_selected and its original attempt-all finally','_server_selected and captured original stage callbacks','BorrowedServerEngine constructor/check/run and original Finalizer','DeviceOwner registration/post_keys/pre_evaluation','real request/ciphertext-binding/artifact/metadata/snapshot functions','real Admission/FilesAdmission and ExpectedIds structural checks'],
 'public_metadata':['Real PublicProfile from already retained runtime-relative key_adapter_source/public_inputs profile/plan bytes; no new filesystem read.','PublicContract.check/profile/json/check_identity are explicitly recorded synthetic metadata/identity, not genuine proof admission.','PublicKeyBinding.check is recorded for a synthetic335 descriptor manifest; expected manifest digest/context proof is not claimed.','ParameterCatalog.from_public returns an opaque recorded catalog; matrix/gamma methods always raise if requested.'],
 'io':['Real ReadDirectory class retained; constructor/check/close methods are recording-only and never allocate an FD.','read_public/map_array selected aliases return335 tiny descriptor/owner records with explicit fakeFD state. No NPY/header/mmap/key body operation.','os.fstat/stat/listdir and Path.lstat delegate all calls except the exact fakeFD/named synthetic public root/parent/manifest. No proc/sys/dev access.','RequestFiles/open_identity_bound_parameters factories return recording owners. Exact Publisher class retained with recorded constructor/write/finish/close;38 metadata artifacts only, no residue serialization.'],
 'numeric':['Recorded core namespaces/security/context/alias methods; no real core import, RNG, NTT or key generation.','Owned recording_jax Array instances, declared CT shape/moduli metadata and per-instance transpose only; no numerical buffers allocated.','Owned temporary recording_jax.numpy.asarray and clear_caches fields installed before Backend captures them; cleanup belongs to shared fixture temporary-module restoration.','_runtime_dependencies returns exact real Dependencies populated with tiny recorded constructors; real _runtime_dependencies/_materialize never invoked.'],
 'signals':'Signal mask/query/set operations are explicitly recorded, so original loader/cleanup branches execute without modifying real process signals.',
 'faults':['Wrong-device recording array fails real registration; DeviceOwner retains failed graph while pre-yield known-ready loader disposes/closes. No false entered-loader-quarantine assertion.','Actual-only registration accessor replacement requires zero replacement calls before yield.','Factory.before_evaluation replacement in recorded numerical-owner constructor requires zero sentinel/evaluate/publication.','Original recorded native.synchronize raises only at pre_evaluation; original entered engine/stage/loader quarantine required,38publication counter0.'],
 'retirement':['Successful actual E case calls shared finish_runtime_owner after real loader/context retirement; four ordered observations retained.','L partial/None routes and failed E cases explicitly do not claim clean device/process retirement. Shared f.close uses original failure/partial handling; registries are never reset.','Recording None control uses original RecordingDependencies/RecordingStageFactories and raw recording numerical dependencies; actual device registration remains absent.'],
 'imports_tests_probes_performed':False,
}
write('SEAMS.json',seams)
(OUT/'README.md').write_text('''# Eight held loader/evaluation test-source cases

Only cases_loader_engine.py was authored. The shared fixture, child, parent, runtime403 and plan59 remain owned by CNN/root and are not edited here. `run(f, case)` dispatches exactly the accepted four L and four E identifiers; this module defines no pytest collection functions.

The fixture API is peer-declared: prepare_owner(before_capture), start, activate_runtime, runtime.allow_public_key_reads, finish_runtime_owner, rebuild_request, patch/mutate, module, enable_numpy, array, and the stated exact synthetic graph fields. The shared fixture was not imported or executed. All named seams are in SEAMS.json. Integration review must verify their final signatures before any collection or run.

AST/source inspection only. Eight cases are source proposals, not passing test results. No real keys, model matrices, gamma containers, ciphertext residues, native/backend context, device/host capacity or actual authority is exercised. Previous36/30 results are historical only. The future harness must use fresh owned children and preserve failures without retry.
''')
write('handoff.json',dict(schema='qwen-full28-loader-engine-exclusive-source-handoff-v1',source=pin(SOURCE),entry='run(f, case)',case_count=8,selected_nodes=pin(OUT/'selected_nodes.json'),seams=pin(OUT/'SEAMS.json'),runtime_source_map=pin('/tmp/qwen-server-device-integration-handoff-v4-20261104/sources.json'),exclusive_file_only=True,shared_fixture_review_pending=True,source_only=True,candidate_imports=False,collection=False,tests=False,probes=False,actual_authority=False))
write('files.json',{p.name:dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size)for p in sorted(OUT.iterdir())if p.is_file()})
print(json.dumps(dict(source=pin(SOURCE),handoff=pin(OUT/'handoff.json'),nodes=8,execution=False)))
