"""Copy only explicit public source/evidence into a fresh review capsule."""
from pathlib import Path
import hashlib, json, stat, tarfile

OUT = Path('/tmp/qwen_server_snapshot_integration_preparation_cpu_20261104')
BUILD = Path(__file__).resolve().parent
H = lambda b: hashlib.sha256(b).hexdigest()
def canonical(v): return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def document(p): return json.loads(Path(p).read_bytes())
assert not OUT.exists()
OUT.mkdir()
origins = {}
def put(dest, raw, origin):
    assert dest not in origins and not Path(dest).is_absolute() and '..' not in Path(dest).parts
    target = OUT / dest
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    origins[dest] = dict(origin, sha256=H(raw), bytes=len(raw))
def copy(source, dest, expected=None):
    source = Path(source)
    assert source.is_file() and not source.is_symlink()
    raw = source.read_bytes()
    if expected is not None: assert H(raw) == expected, str(source)
    put(dest, raw, dict(source=str(source)))
def group(root, prefix, names):
    for name in names: copy(Path(root)/name, prefix+'/'+name)
def archive(path, expected):
    assert H(Path(path).read_bytes()) == expected
    with tarfile.open(path, 'r:gz') as tf:
        members = tf.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        return {m.name:tf.extractfile(m).read() for m in members}
def fragment(arc, digest, member, raw, dest):
    put(dest, raw, dict(source_archive=str(arc), source_archive_sha256=digest, member=member))

R0=Path('/tmp/qwen-server-snapshot-runtime-handoff-20261104')
R1=Path('/tmp/qwen-server-snapshot-runtime-handoff-v2-20261104')
T0=Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104')
T1=Path('/tmp/qwen-server-snapshot-integration-test-handoff-v2-20261104')
P=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104')
source_specs = {
    '387': (R0/'parent387-sources.json',R0/'parent387-sources.tar.gz','858ce5f5985b78e8507990685b075d86f6d6e2fbbcf48bfa86294b921aac8348'),
    '390': (R0/'sources.json',R0/'sources.tar.gz','9bd175e441360b4dcdff986c435ac8bfc2f728b824502df0c12de80d959f4a1b'),
    '391': (R1/'sources.json',R1/'sources.tar.gz','1cd11ef416476e61df6c630dd5c93a86cbc5c7cc3eafb8ced5b41ab3a3e66c04'),
}
all_sources={}
for count,(mp,arc,digest) in source_specs.items():
    files=archive(arc,digest);mapping=document(mp)
    assert len(mapping)==int(count) and {n:H(b) for n,b in files.items()}==mapping
    all_sources[count]=files
    copy(mp,'lineage/'+count+'/sources.json')
for count,root,expected in [
    ('404',T0,'bbc4c0bd3ce4bc6a1cd0efa10a7afc799e280887132869b846849fee10b2e713'),
    ('405',T1,'7814e934402820f57da5084c6cefc4d0a924cc5e68a06c219bdeacd47d03ee59'),
    ('405-pre-receipt',T1/'before-root-failure-receipts','8fb5450802f5051238cdb7fe1b16a0137b4fe14dcdf662427052bc88a2d720ec'),
]:
    files=archive(root/'sources.tar.gz',expected);mapping=document(root/'sources.json')
    assert len(files)==(404 if count=='404' else 405) and {n:H(b) for n,b in files.items()}==mapping
    all_sources[count]=files
    group(root,'sources/'+count,['sources.json','sources.tar.gz','handoff.json'])
assert {n[8:]:b for n,b in all_sources['405'].items() if n.startswith('runtime/')} == all_sources['391']
assert {n[8:]:b for n,b in all_sources['404'].items() if n.startswith('runtime/')} == all_sources['391']
lineage={}
for count in ['387','390','391']:
    overrides={}
    arc=source_specs[count][1];digest=source_specs[count][2]
    for name,raw in all_sources[count].items():
        if all_sources['391'].get(name)!=raw:
            dest='lineage/'+count+'/overrides/'+name
            fragment(arc,digest,name,raw,dest);overrides[name]=dest
    lineage[count]=dict(source_count=int(count),source_map='lineage/'+count+'/sources.json',
        base_archive='sources/405/sources.tar.gz',base_prefix='runtime/',overrides=overrides,
        historical_archive_sha256=digest,historical_archive_bundled=False,
        historical_archive_hash_offline_verified=False,
        offline_scope='Reconstructs exact file bytes/map; omitted historical compressed archive digest is receipt provenance only.')

group(R0,'lineage/390',['callback-source-closure.json','capture_parent.py','freeze_source.py','handoff.json','parent-capture.json','source.patch','static-review.json'])
group(R1,'lineage/391',['callback-source-closure.json','freeze_source.py','handoff.json','source.patch','static-review.json'])
group(T0,'sources/404',['capture_source.py'])
group(T1,'sources/405',['capture_source.py','append_failure_receipts.py','source.patch'])
group(T1/'before-root-failure-receipts','sources/405-pre-receipt',['fixture_successor_history.json','source.patch'])
for name in ['integration_cases.py','integration_fixture.py','integration_checks.py','integration_child.py','test_snapshot_integration.py','conftest.py','pytest.ini','runtime_sources.json','SEAMS.json','entry_fixture_provenance.json','TEST_SCOPE.md','selected_tests.json','origin.json','fixture_successor_history.json']:
    copy(Path('/tmp/qwen-server-snapshot-integration-tests-v2-20261104')/name,'selected-tests/'+name,document(T1/'sources.json')[name])
for name in ['qwen_server_process.py','qwen_server_backend.py','qwen_server_entry.py','qwen_admission_snapshot.py','qwen_server_resource.py','qwen_server_entry_status.json','cleanup_origin.py','qwen_finalizer.py','SNAPSHOT_INTEGRATION.md','snapshot_integration.json','snapshot_dispatch_successor.json']:
    fragment(R1/'sources.tar.gz',source_specs['391'][2],name,all_sources['391'][name],'selected-runtime/'+name)

proof_common=['README.md','before_executed_driver.py','before_worker_driver.py','driver_provenance.json','executed_driver.py','executed_driver.py.patch','execution-after.json','handoff.json','launch-proposal.json','live-observation.json','prepare_source.py','run_config.json','source-handoff.json','sources-after.json','sources-before.json','tests.log','tests.xml','worker-audit.json','worker_driver.py','worker_driver.py.patch']
group(P/'preparation-v1','proof/v1',proof_common)
group(P/'preparation-v2','proof/v2',proof_common+['owned-pytest-metadata.json','preserved-v1-files.json','report.json','run_config.json.patch','test-child-terminals.json'])

reviews=[
 ('/tmp/qwen-server-snapshot-integration-root-test-review-20261104','reviews/root-tests', ['direct-owner-failed-v1.json','direct-owner-v2.json','failed-terminal-v1.json','live-observation.json','live-v2-observation.json','preexecution-review.json','preexecution-v2-review.json','review.json','run-grant.json','run-v2-grant.json','source-v2-review.json','terminal-v2-review.json']),
 ('/tmp/qwen-server-snapshot-root-source-review-v2-20261104','reviews/root-source391',['review.json']),
 ('/tmp/qwen-server-snapshot-integration-security-review-20261104','reviews/security-plan',['review.json','review_plan.py']),
 ('/tmp/qwen-server-snapshot-integration-security-review-20261104/source390','reviews/security-source390',['review.json','review_source.py']),
 ('/tmp/qwen-server-snapshot-integration-security-review-20261104/source391','reviews/security-source391',['review.json','review_source.py']),
 ('/tmp/qwen-server-snapshot-integration-test-security-review-20261104','reviews/security-tests',['review.json','review_source.py','source-v2-review.json','review_source_v2.py','harness-review.json','review_harness.py','harness-v2-review.json','review_harness_v2.py']),
 ('/tmp/qwen-server-snapshot-integration-failed-security-review-20261104','reviews/security-failed',['review.json','review_public.py','owned-public-records.json','failed-child-logs.json','process-absence.json']),
 ('/tmp/qwen-server-snapshot-integration-terminal-security-review-20261104','reviews/security-terminal',['review.json','review_public.py','process-absence.json','owned-public-record-pins.json']),
 ('/tmp/qwen-server-constructor-import-security-review-20261104','reviews/constructor-import',['review.json','review_source.py']),
]
for root,prefix,names in reviews:group(root,prefix,names)
for name in ['root_snapshot_integration_source_review_20261104.py','root_snapshot_integration_source_v2_20261104.py','root_snapshot_integration_preexecution_20261104.py','root_snapshot_integration_harness_v2_20261104.py','root_snapshot_integration_failed_terminal_20261104.py','root_snapshot_integration_terminal_v2_20261104.py']:
    copy(Path('/tmp')/name,'reviews/root-scripts/'+name)
group('/tmp/qwen-server-snapshot-integration-plan-20261104','plans/integration',['PLAN.md','sources.json','source_refs.json','callgraph.json','compatibility.json','review_limits.json','prepare_plan.py'])
group('/tmp/qwen-server-snapshot-integration-test-plan-20261104','plans/focused-proof',['PLAN.md','files.json','scope.json','source_refs.json'])
group('/tmp/qwen-server-snapshot-test-architecture-20261104','plans/architecture',['PLAN.md','files.json','source_refs.json'])

children={}
for version,record_path,regular_count in [
    ('v1',Path('/tmp/qwen-server-snapshot-integration-failed-security-review-20261104/owned-public-records.json'),100),
    ('v2',Path('/tmp/qwen-server-snapshot-integration-terminal-security-review-20261104/owned-public-record-pins.json'),150),
]:
    records=document(record_path);regular=records['regular_files'];assert len(regular)==regular_count
    base=P/('preparation-'+version)
    assert len(records['unfollowed_aliases'])==15
    for name,item in regular.items():
        assert name.startswith('pytest-owned/') and len(Path(name).parts)==3
        assert Path(name).name in ['parent-terminal.json','child-terminal.json','result.json','child.log','tiny-public-proof.json']
        assert item['bytes']<=131072
        source=base/name
        assert all(not q.is_symlink() for q in [base,source.parent,source])
        assert source.stat().st_size==item['bytes']
        copy(source,'children/'+version+'/'+name,item['sha256'])
    assert sum(Path(n).name=='tiny-public-proof.json' for n in regular)==6
    children[version]=dict(pin_table='reviews/security-failed/owned-public-records.json' if version=='v1' else 'reviews/security-terminal/owned-public-record-pins.json',
        regular_count=regular_count,child_artifacts=regular_count-6,tiny_synthetic_fixtures=6,
        aliases=15,aliases_copied_or_followed=False)

copy(BUILD/'assemble.py','capsule-tools/assemble.py')
copy(BUILD/'verify.py','verify.py')
summary=dict(schema='qwen-server-snapshot-integration-preparation-capsule-v1',
    final=dict(source_count=405,runtime_count=391,tests=36,passed=36,failures=0,errors=0,skipped=0,
        owner_session=46330,launch_tool='5005f4',terminal_tool='98ea2c',exit_code=0,reaped=True,
        report_sha256='98c2642ab0cdc5dd5e7871c3d55bd6b8ff7b1da3e7b76e6e3de29db06f56afbb'),
    failed=dict(source_count=404,runtime_count=391,tests=36,passed=11,failures=25,errors=0,skipped=0,
        owner_session=31389,launch_tool='4c43f9',terminal_tool='915588',exit_code=1,reaped=True,
        report_present=False,cause='Bound fixture callbacks rejected before25 direct assertions;6 process and5 terminal cases passed.'),
    runtime_lineage=lineage,children=children,
    literal_guard_sha256='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395',
    definition_only_entry=True,genuine_cli_bootstrap=False,
    scientific_admission=False,capacity_measurement=False,native_backend=False,HE=False,
    actual_server_role=False,borrowed_model_key_ciphertext_reads=False,performance_claim=False,
    temporary_numpy_jax_names='Owned ModuleType recordings only, removed; no genuine numeric backend imported.',
    verifier_scope='Stdlib public file hashes, source-map reconstruction, XML/receipt joins and optional declared public source/archive hashes only; no imports/tests/probes or proof replay.')
(OUT/'summary.json').write_bytes(canonical(summary))
(OUT/'README.md').write_text('''# Qwen server snapshot integration: public synthetic preparation

The corrected 405-source preparation passed all 36 selected cases. The earlier 404-source run is preserved as 25 fixture-setup failures and 11 passing cases. Root owned both launches and direct waits; independent root live observations, terminal receipts, and peer post-exit checks are included. The source runtime remains the exact 391-file successor and the snapshot guard remains literal `8790711a…`.

The cases exercise retained runtime/control and backend-constructor paths, snapshot and public-export rejection before replacement invocation, sticky rejection after restoration, once-derived metadata, repeated hot checks without extra full validations, current capacity denial, scope forwarding, full boundary rereads, and cleanup ordering. Admission, resource, core, audit, loader, stage and evidence seams are explicitly recorded in `selected-tests/SEAMS.json`. The entry fixture removes only its terminal dispatch node and retains real source-file/origin checks; it is **not genuine CLI bootstrap**. Temporary NumPy/JAX names are owned recording modules, not imported native packages. No scientific admission, capacity measurement, existing model/key/ciphertext body read, actual role, HE, speed improvement, or full-model result follows.

`sources/` contains exact failed404, final405 and pre-receipt405 archives. Final405 contains runtime391. `lineage/` provides all387/390/391 maps plus sparse old source bytes: the verifier reconstructs each full historical file map. The omitted standalone387/390/391 compressed tar digests are historical receipt provenance only and cannot be independently rehashed offline. The included source-review and root/peer receipts preserve their original limited or failed status.

`children/v1` retains94 child artifacts plus6 tiny newly owned synthetic proof fixtures; `children/v2` retains144 child artifacts plus6 such fixtures. The30 total pytest aliases are metadata only and were neither copied nor traversed. Tiny fixture content is explicitly synthetic and supplies no actual admission evidence. The failed run has no final report because the strict outer inspector rejected missing positive records; raw XML, logs and all36 early ownership records remain present.

Run `python3 -I -B verify.py` for portable public checks. `--current` additionally checks only declared public source files and archives. Neither mode imports candidate code, reruns tests/validators, follows private descriptors, checks device capacity, or reads borrowed data. `WHITELIST.json`, `copy_origins.json`, and `SHA256SUMS` define the closed capsule. Source authoring and earlier unrun labels are preserved as historical records, not claims that these completed runs remain unexecuted.
''')
(OUT/'HISTORY.md').write_text('''# Preserved sequence and limits

1. Root captured the387-file server parent before authoring. The source-only390 integration added retained snapshot checks but review found a missing transitive cleanup_origin callback binding and public dispatcher lookups that could invoke a replaced export before rejection. Neither390 nor its source review is an integration PASS.
2. The391 successor adds literal shared cleanup_origin4543 to the21-file callback binding and retains original dispatcher defaults at ordinary caller boundaries. The private registry/default/interpreter bookkeeping remains trusted; arbitrary Python mutation containment is not claimed. Capacity denial and false readiness remain unchanged.
3. Architecture and constructor import tracing preceded the404 focused test closure. Earlier source reviews missed that the direct fixture supplied bound methods where the runtime requires FunctionType. Root run31389 returned1:25 setup failures,6 process passes and5 terminal passes. Intended direct mutation assertions were not reached. The outer artifact inspector then stopped without a final report. Runtime391 was not changed in response.
4. The405 test-only successor replaces those two arguments with local plain forwarding functions. All391 runtime bytes and all36 assertions remain unchanged. A first405 checkpoint was captured before root failure receipts arrived; its full map, tar, handoff and history remain under sources/405-pre-receipt. Final405 changes only that history record relative to the first405 checkpoint.
5. Harness-v2 changes only the driver config pin and7 config path/count/pin fields; worker and ownership/cleanup/artifact logic remain exact. Root run46330 returned0 with36PASS. Independent source, preexecution, live, terminal and peer receipts are included. This is one corrected successor execution, not a relabeling of the failed run.

The proof uses finite validation counters, not a timing comparison. Reported elapsed seconds are raw execution metadata only. Evidence-close-only failure tests establish terminal70, retained owners and an unretired guard; they do not falsely claim the guard itself entered a sticky failed state without a guard rejection. Actual full proof validation, science,335-key reading, body integrity, native initialization and server capacity remain outside this proof.
''')
(OUT/'copy_origins.json').write_bytes(canonical(dict(schema='explicit-public-copy-origins-v1',files=origins)))
generated=['summary.json','README.md','HISTORY.md','copy_origins.json','WHITELIST.json','SHA256SUMS']
names=sorted(set(origins)|set(generated))
assert len(names)==len(origins)+len(generated)
(OUT/'WHITELIST.json').write_bytes(canonical(dict(schema='closed-public-capsule-whitelist-v1',files=names)))
indexed=sorted(set(names)-{'SHA256SUMS'})
(OUT/'SHA256SUMS').write_text(''.join(H((OUT/n).read_bytes())+'  '+n+'\n' for n in indexed))
assert {p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}==set(names)
print(json.dumps(dict(capsule=str(OUT),copied_files=len(origins),files=len(names),indexed=len(indexed),index_sha256=H((OUT/'SHA256SUMS').read_bytes()),whitelist_sha256=H((OUT/'WHITELIST.json').read_bytes()))))
