"""Copy only declared public Q5 preparation artifacts; never execute their code."""
import hashlib, json, pathlib, stat

BUILD = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path('/tmp/qwen_toy5_checkpoint_preparation_cpu_20261104')
OUT.mkdir(exist_ok=False)
origins = {}
def sha(b): return hashlib.sha256(b).hexdigest()
def dump(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
def copy(src, dst):
    src, dst = pathlib.Path(src), OUT/dst
    assert dst.relative_to(OUT).as_posix() not in origins
    a=src.lstat(); assert stat.S_ISREG(a.st_mode) and not src.is_symlink()
    b=src.read_bytes(); z=src.lstat()
    assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
    dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(b)
    origins[dst.relative_to(OUT).as_posix()]={'source':str(src),'sha256':sha(b),'bytes':len(b)}
def group(root, names, dest):
    for n in names: copy(pathlib.Path(root)/n, str(pathlib.PurePosixPath(dest)/n))

base='/tmp/qwen-toy5-checkpoint-focused-proof-20261104'
common=['README.md','driver_provenance.json','executed_driver.py','executed_driver.py.patch','execution-after.json','handoff.json','launch-proposal.json','live-observation.json','owned-pytest-metadata.json','report.json','run_config.json','source-handoff.json','sources-after.json','sources-before.json','sources.tar.gz','tests.log','tests.xml','worker-audit.json','worker_driver.py','worker_driver.py.patch']
group(base+'/preparation-v1',common,'proof/v1')
group(base+'/preparation-v2',common+['predecessor-root-terminal-review.json','run_config.json.patch','source-correction.json'],'proof/v2')
for v in (1,2):
    suffix='' if v==1 else '-v2'
    copy('/tmp/qwen-toy5-checkpoint-test-sources-v%d-20261104.json'%v,'source/v%d/sources.json'%v)
    root='/tmp/qwen-toy5-checkpoint-test-source-handoff%s-20261104'%suffix
    names=(['handoff.json','capture-authoring-history.json','capture_source.py','static-review.json','capture_source_initial_stat_assertion.py'] if v==1 else ['handoff.json','before_q5_test_fixtures.py','correction.json','semantic_phase_filter.patch','prepare_successor.py','parent261-sources.json'])
    group(root,names,'source/v%d/history'%v)
    copy('/tmp/qwen-toy5-checkpoint-harness-build%s-20261104/prepare.py'%suffix,'harness-build/v%d/prepare.py'%v)
    review='/tmp/qwen-toy5-checkpoint-root-review%s-20261104'%suffix
    group(review,['direct-owner.json','preexecution-review.json','root-process-observation.json','run-grant.json','terminal-review.json']+(['pytest-ancestor-config-observation.json'] if v==1 else []),'reviews/root-v%d'%v)
    for label in ('preexecution','terminal'):
        copy('/tmp/root_q5_checkpoint_%s%s_20261104.py'%(label,'' if v==1 else '_v2'),'reviews/root-v%d/%s.py'%(v,label))
copy('/tmp/root_q5_checkpoint_grant_20261104.py','reviews/root-v1/grant.py')
for src,dst,names in [
 ('/tmp/qwen-toy5-checkpoint-security-source-review-20261104','reviews/security-source',['review.json','review_source.py']),
 ('/tmp/qwen-toy5-harness-transformers-review-20261104','reviews/transformer-harness',['review.json','review_source.py']),
 ('/tmp/qwen-toy5-terminal-transformers-review-20261104','reviews/transformer-terminal',['review.json','review_public.py'])]: group(src,names,dst)

parent='/tmp/qwen-toy5-checkpoint-component-20261104'
parent_h='/tmp/qwen-toy5-checkpoint-source-handoff-20261104'
group(parent_h,['sources.json','handoff.json'],'source/parent250')
new=['q5_actual_metadata_replay.py','q5_checkpoint_arithmetic.py','q5_checkpoint_contract.py','q5_checkpoint_entry.py','q5_checkpoint_oracle.py','q5_checkpoint_roles.py','q5_checkpoint_transport.py']
parent_readable=new+['SOURCE_DESIGN.md','source_selection.json','copy_origins.json','cleanup_origin.py','execution_authorization.json','design/DESIGN.md','design/checkpoint_plan.json','design/resource_plan.json','provenance/arithmetic-from-toy4.patch.txt','provenance/dense47-proposed-mapping.json','provenance/root-design-review.json','history/oracle-before-static-syntax-fix.py.txt','history/static-syntax-correction.json','reference/runtime_pins.json']
group(parent,parent_readable,'source/parent250/readable')
test_readable=['FOCUSED_TEST_PLAN.md','conftest.py','numpy_pins.json','q5_test_fixtures.py','selected_tests.json','test_provenance.json','test_q5_mapping.py','test_q5_metadata.py','test_q5_checkpoint_state.py','test_q5_entry.py']
group('/tmp/qwen-toy5-checkpoint-tests-v2-20261104',test_readable,'source/v2/readable')
copy('/tmp/qwen-toy5-checkpoint-tests-20261104/q5_test_fixtures.py','source/v1/readable/q5_test_fixtures.py')
copy(BUILD/'assemble.py','capsule-tools/assemble.py')
copy(BUILD/'verify.py','verify.py')
copy(BUILD/'verify_initial_grant_schema.py','capsule-tools/verify_initial_grant_schema.py')
copy(BUILD/'verifier-authoring-history.json','capsule-tools/verifier-authoring-history.json')

summary={
 'schema':'qwen-toy5-checkpoint-preparation-capsule-v1',
 'scope':'Public synthetic CPU preparation; no encrypted execution or actual role authority.',
 'parent_sources':250,'test_sources_per_version':261,'selected_tests':48,
 'test_families':{'mapping':7,'metadata':21,'recording_state':12,'entry':8},
 'v1':{'passed':False,'pass':42,'fail':6,'session':35774,'terminal_chunk':'07788f','exit_code':1,'reaped':True},
 'v2':{'passed':True,'pass':48,'fail':0,'session':32409,'terminal_chunk':'d48ad9','exit_code':0,'reaped':True},
 'changed_source_files':['q5_test_fixtures.py'],'unchanged_runtime_files':250,
 'logical_checkpoints':47,'synthetic_refresh_returns':19,'planned_new_exports':46,
 'nominal_planned_payload_bytes':1354760192,'capacity_measured':False,
 'independent_live_observation':False,'genuine_numpy':True,
 'numpy_scope':'Four Python source hashes before/after; root numpy module file/spec origin only; not whole installed binary closure.',
 'refresh_slots':'ideal identity','refresh_metadata':'synthetic observed-return fixtures',
 'ledger':'recording metadata/retained-object controls, no ciphertext/backend',
 'actual_metadata_captured':False,'actual_transport_exercised':False,'actual_loader_exercised':False,
 'actual_role_authorized':False,'encrypted_first_divergence_claim':False,'speed_claim':False,
 'full_model_claim':False,'prior140_PASS_inherited':False,
 'source_archives':['proof/v1/sources.tar.gz','proof/v2/sources.tar.gz'],
 'parent250_reconstruction':'Each source archive contains runtime/ with the exact parent250 map; standalone parent tar is referenced by original handoff but not redundantly copied.',
 'public_predecessors':['../optimized_transformer_tpu_sweep_20261103/checkpoints/qwen_test5/README.md','../qwen_toy5_public_replay_cpu_20261104/README.md']}
dump(OUT/'summary.json',summary)
(OUT/'README.md').write_text('''# Qwen toy5 checkpoint preparation — CPU synthetic proof

The corrected focused proof passed **48 tests**. Its predecessor is preserved with **42 passes and 6 failures**. Both versions retain the exact 250-file runtime and the same 48 tests and expected checkpoint indices; the sole test-source change selects the six named layer-end phases rather than every refresh bookkeeping phase.

The selected cases comprise 7 literal/independent-dense mapping and diagnostic checks for all 47 checkpoints, 21 binary64 metadata checks consuming 19 synthetic refresh-return observations (including level 26→25 alignment), 12 recording Ledger state/ownership checks, and 8 immediate-denial/terminal checks. They use fixed public seed 9105 toy arithmetic with genuine NumPy. Refresh slots are ideal identity and returned metadata is synthetic. The original mathematical graph and default decoder are retained; the earlier 140-pair replay result is not inherited.

The first fixture captured 85 points: 47 intended points plus 38 start/complete bookkeeping phases from 19 refreshes. The corrected fixture filters by the six original semantic labels, not by expected numeric indices. All runtime files, expected indices and test assertions are unchanged. [History](HISTORY.md) records this observed failure and the separate earlier source-capture/static authoring corrections.

Root exclusively owned both runs: session 35774 ended with status 1 and session 32409 with status 0, both directly reaped. Harness records identify the worker/outer processes; independent root observations arrived after exit, so there is **no independent live observation claim**. Exact owner, grant, terminal, source and peer receipts are retained under `reviews/`. The independent terminal peer review adds a public readback, without rerunning tests or making a new process observation.

The two exact 261-file source archives under `proof/v1` and `proof/v2` each contain the unchanged `runtime/` subtree matching `source/parent250/sources.json`. The original 250-file handoff retains its original tar digest and its source-only status at the time it was written. Selected runtime/test files are also readable under `source/`; historical preparation and review flags are not rewritten after the run.

NumPy coverage binds four Python source files before/after and the imported root module's file/spec origin. It does not authenticate the full NumPy native-library installation or force unused header modules to import. Both worker audits contain no backend imports or denials. Version 2 uses explicit `-c /dev/null`; version 1 retains the root's ancestor-configuration observation rather than claiming its flags disabled config discovery.

No RNS export, transport, real public/private loader, default decoder execution, HE context, key generation, learned model, actual encrypted metadata, private body, capacity check, or actual role was exercised. The Ledger checks use recording objects. The 46 planned exports and 1,354,760,192 nominal payload bytes remain storage design estimates. This proof does not identify the first encrypted divergence, establish encrypted accuracy, demonstrate acceleration, or validate the full Qwen model. Actual dispatch remains disabled.

The original [TPU toy5 failure](../optimized_transformer_tpu_sweep_20261103/checkpoints/qwen_test5/README.md) and [public polynomial replay](../qwen_toy5_public_replay_cpu_20261104/README.md) are separately preserved. This capsule does not replace either result or claim whole-sweep completion.

Run `python -I -B verify.py` for portable hash/archive/XML/receipt checks. Optional `python -I -B verify.py --current` additionally hashes only declared current public source files, execution source copies and the four declared NumPy Python files. Neither mode imports the tested code, reruns tests/review scripts/numerics, accesses borrowed stores, or launches a role. `WHITELIST.json` is the closed file list; `copy_origins.json` binds every copied file to its original path, digest and size. `SHA256SUMS` indexes every file except itself. These are integrity/readback checks, not signatures or execution authority.
''')
(OUT/'HISTORY.md').write_text('''# Preserved history

1. The original seven-module runtime was source-authored only. Its AST syntax correction and original pre-fix oracle text remain in `source/parent250/readable/history/`; no runtime failure was observed for that authoring correction.
2. The initial source-capture helper compared full stat results, including access time after reading. It failed before source-map/archive publication. The stable identity comparison was corrected; both helper versions and its exact authoring-history receipt remain under `source/v1/history/`. This was a capture-tool failure, not a test or numerical failure.
3. The root-authorized first focused run selected 48 tests and returned 42 passes/6 failures. The fixture included all 38 internal refresh start/complete phase events. The exact failed proof, logs, XML, source archive, harness, root terminal and owner receipts remain under `proof/v1` and `reviews/root-v1`.
4. The separate successor changes only the fixture's semantic phase selection, retaining all expected indices/assertions and runtime250. Exact before-source, patch, correction and source-authoring script are under `source/v2/history/`. The harness successor also adds explicit `-c /dev/null` and derived source/config pins; both build scripts and exact patches are retained.
5. Root separately authorized and owned the corrected run. All 48 tests passed; root independently checked exact source/execution/start/end/archive, XML, audit and terminal/absence evidence. The terminal peer readback is separate from the pre-execution source/harness reviews. No tests were rerun while assembling this capsule.

The original handoffs retain their historical unexecuted/preparation-only flags. The version 1 failure is not hidden or reclassified. Neither root late observation establishes an independent live identity observation.
''')
dump(OUT/'copy_origins.json',{'schema':'explicit-public-copy-origins-v1','files':origins})
generated=['summary.json','README.md','HISTORY.md','copy_origins.json','WHITELIST.json','SHA256SUMS']
names=sorted(list(origins)+generated)
dump(OUT/'WHITELIST.json',{'schema':'closed-public-whitelist-v1','files':names})
index=''.join(sha((OUT/n).read_bytes())+'  '+n+'\n' for n in names if n!='SHA256SUMS')
(OUT/'SHA256SUMS').write_text(index)
print(json.dumps({'capsule':str(OUT),'files':len(names),'indexed_files':len(names)-1,'copied':len(origins),'index_sha256':sha(index.encode())},sort_keys=True))
