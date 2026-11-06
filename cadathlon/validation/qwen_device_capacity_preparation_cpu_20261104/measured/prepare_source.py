"""Source-only adaptation of the admitted snapshot owned-process harness."""
import ast
import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile

OUT=Path(__file__).resolve().parent
BASE=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v2')
SOURCE=Path('/tmp/qwen-device-component-tests-v2-20261104')
HAND=Path('/tmp/qwen-device-component-test-handoff-v2-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
encode=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
mapraw=(HAND/'sources.json').read_bytes();mapping=json.loads(mapraw)
assert len(mapping)==50 and sha(mapraw)=='1725249e18cf6298eda8516726955efe879635c0132579b65543ad3717493e74'
archive=(HAND/'sources.tar.gz').read_bytes();assert sha(archive)=='1645d2fa604ad76afb41b4fdc9ea4ab88c51bd665eed3ed287ca79147923e5aa'
old_driver=(BASE/'executed_driver.py').read_bytes();old_worker=(BASE/'worker_driver.py').read_bytes()
assert sha(old_driver)=='3d70d7531b9284a35da14c08d0d6cb35ecaa5148318a7c5a70e0a0133862a0aa'
assert sha(old_worker)=='55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be'
for n,b in [('before_executed_driver.py',old_driver),('before_worker_driver.py',old_worker),('sources-before.json',mapraw),('sources.tar.gz',archive),('source-handoff.json',(HAND/'handoff.json').read_bytes())]:
    with(OUT/n).open('xb')as stream:stream.write(b)
execution=OUT/'execution';execution.mkdir()
for n,d in mapping.items():
    src=SOURCE/n;assert not src.is_symlink();b=src.read_bytes();assert sha(b)==d
    p=execution/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
    assert{m.name:sha(tar.extractfile(m).read())for m in tar.getmembers()}==mapping
selection=json.loads((SOURCE/'selected_tests.json').read_bytes());nodes=selection['static_nodeids']
assert len(nodes)==30 and len(set(nodes))==30
runtime=json.loads((SOURCE/'runtime_sources.json').read_bytes());assert len(runtime)==34
assert all(mapping['runtime/'+n]==d for n,d in runtime.items())
cases=[dict(classname='test_device_component',name=n.split('::',1)[1])for n in nodes]
config=json.loads((BASE/'run_config.json').read_bytes())
config.update(schema='qwen-device-component-focused-proof-v1',source_count=50,runtime_count=34,
    source_root=str(SOURCE),proof_root=str(OUT),owned_tmpdir=str(OUT/'tmp'),pytest_basetemp=str(OUT/'pytest-owned'),
    source_manifest_sha256=sha(mapraw),source_archive_sha256=sha(archive),selection_sha256=mapping['selected_tests.json'],
    selected_files=['test_device_component.py'],selected_nodes=nodes,selected_nodes_sha256=sha(encode(nodes)),
    selected_xml_cases=cases,selected_xml_cases_sha256=sha(encode(cases)),pytest_ini_sha256=mapping['pytest.ini'],
    expected_child_records=30,expected_parent_records=30,expected_regular_child_files=120,
    owned_temp_max_files=256,runtime_source_expectations=runtime,
    scope='Exactly30 fresh-child synthetic device policy/state cases: real device6 and resource912d validators, two recording phase checks, source-owned process hooks and temporary JAX-shaped double. No genuine backend/device/capacity/scientific/role execution.',
    child_origin_evidence='Source-bound child source_binding assertions on loaded local file/spec and original resource functions; no observed-origin table.',
    temporary_jax_double='Owned recording_jax.py injected only in child; removed and restored before successful result; never installed JAX import/discovery.')
(OUT/'run_config.json').write_bytes(encode(config))
worker=old_worker.decode().replace('Unrun snapshot recording-state worker; exact prefixture selection only.','Unrun synthetic device-state worker; exact prefixture selection only.')
worker=worker.replace("assert config['runtime_source_expectations']=={}and config['runtime_module_origins']=={}", """assert config['runtime_module_origins']=={}
def runtime_hashes():
    selected_runtime=source/'runtime'
    assert not any(p.is_symlink()for p in selected_runtime.rglob('*'))
    return {p.relative_to(selected_runtime).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(selected_runtime.rglob('*'))if p.is_file()}
runtime_before=runtime_hashes()
assert len(runtime_before)==34 and runtime_before==config['runtime_source_expectations']""")
worker=worker.replace('snapshot proof forbids','device proof forbids').replace('exact snapshot cases required before fixtures','exact device component cases required before fixtures')
worker=worker.replace("selected==['test_snapshot_guard.py']", "selected==['test_device_component.py']")
worker=worker.replace("runtime_sources_before={},runtime_sources_after={},runtime_module_origins={},", "runtime_sources_before=runtime_before,runtime_sources_after=runtime_hashes(),runtime_module_origins={},")
worker=worker.replace("temporary_root=str(temporary),synthetic_only=True,scientific_admission_tested=False)", "temporary_root=str(temporary),synthetic_only=True,scientific_admission_tested=False,device_component_recording_only=True)")
ast.parse(worker)
(OUT/'worker_driver.py').write_text(worker)
driver=old_driver.decode()
driver=driver.replace('Held unrun public/recording snapshot proof driver; never auto-discovers tests.','Held unrun synthetic device component driver; never auto-discovers tests.')
driver=driver.replace("CONFIG_SHA256 = '73f36f9634f726699a8c70ec826f2020b59713dc1fc9960b85859d27bb281ab0'",'CONFIG_SHA256 = '+repr(sha(encode(config))))
driver=driver.replace("WORKER_SHA256 = '55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be'",'WORKER_SHA256 = '+repr(sha(worker.encode())))
driver=driver.replace("name.startswith('test_snapshot_guard_')", "name.startswith('test_device_component_')")
driver=driver.replace('snapshot proof pins remain unset; launch disabled','device proof pins remain unset; launch disabled')
driver=driver.replace("'qwen-admission-snapshot-focused-proof-v2'", "'qwen-device-component-focused-proof-v1'")
old_start=driver.index('    expected = [n.split(')
old_stop=driver.index('\n\n\ndef main():',old_start)
new_inspection="""    expected = [n.split('[', 1)[1][:-1] for n in config['selected_nodes']]
    records = {}
    regular = [out / name for name, item in metadata.items() if item['kind'] == 'regular' and name.startswith('pytest-owned/')]
    assert type(config['expected_regular_child_files']) is int and config['expected_regular_child_files'] == 120
    assert len(regular) == 120
    assert all(p.name in ('parent-terminal.json', 'child-terminal.json', 'result.json', 'child.log') for p in regular)
    early_resource_rejections = {'document_role_order', 'document_noncanonical', 'document_source_phase_join'}
    no_recording_jax = {n for n in expected if n.startswith('document_')} | {'binding_failure', 'lease_failure', 'reentry_caught', 'reentry_uncaught', 'initialization_failure'}
    for path in sorted(p for p in regular if p.name == 'child-terminal.json'):
        terminal = json.loads(path.read_bytes())
        result_path, log_path = path.parent / 'result.json', path.parent / 'child.log'
        parent_path = path.parent / 'parent-terminal.json'
        assert all(p in regular for p in (result_path, log_path, parent_path))
        result = json.loads(result_path.read_bytes()); parent = json.loads(parent_path.read_bytes())
        assert type(parent) is dict and set(parent) == {'schema', 'pid', 'owned_session_pgid', 'observed_pgid', 'exit_code', 'reaped', 'group_absent', 'scenario', 'child_result_read'}
        assert parent['schema'] == 'qwen-device-component-owned-child-v1' and parent['child_result_read'] is False
        scenario = terminal['scenario']
        assert scenario in expected and scenario not in records and result['scenario'] == parent['scenario'] == scenario
        pid, pgid = result['pid'], result['pgid']
        assert type(pid) is int and pid > 0 and type(pgid) is int and pgid == pid
        for owner in (parent, terminal):
            assert type(owner['pid']) is int and owner['pid'] == pid
            assert owner['observed_pgid'] is None or (type(owner['observed_pgid']) is int and owner['observed_pgid'] == pid)
            assert type(owner['exit_code']) is int and owner['exit_code'] == 0
            assert owner['reaped'] is True and owner['group_absent'] is True
        assert type(parent['owned_session_pgid']) is int and parent['owned_session_pgid'] == pid
        assert type(terminal['child_reported_pgid']) is int and terminal['child_reported_pgid'] == pid
        assert parent['observed_pgid'] == terminal['observed_pgid']
        assert terminal['runtime_scope'] == 'synthetic device policy/state only'
        for name in ('assertions_completed', 'recording_phase_validation', 'resource_validator_is_real', 'original_phase_identities_restored', 'original_resource_identities_preserved', 'profile_observer_restored', 'temporary_jax_removed'):
            assert result[name] is True
        for name in ('genuine_backend_imported', 'genuine_capacity_measured', 'actual_policy_or_grant', 'scientific_admission_tested', 'device_sources_modified'):
            assert result[name] is False
        assert result['backend_denials'] == []
        assert type(result['temporary_recording_jax_used']) is bool
        assert result['temporary_recording_jax_used'] is (scenario not in no_recording_jax)
        for name in ('resource_calls', 'logical_calls'):
            assert type(result[name]) is int and result[name] >= 0
            assert (result[name] > 0) is (scenario not in early_resource_rejections)
        for group in (False, True):
            try: (os.killpg if group else os.kill)(pid, 0)
            except ProcessLookupError: pass
            else: raise ValueError('owned device test child or group remains live')
        records[scenario] = dict(terminal=terminal, result=result, parent_terminal=parent,
            parent_terminal_path=parent_path.relative_to(out).as_posix(), parent_terminal_sha256=sha(parent_path.read_bytes()),
            terminal_path=path.relative_to(out).as_posix(), result_path=result_path.relative_to(out).as_posix(),
            log_path=log_path.relative_to(out).as_posix(), log_sha256=sha(log_path.read_bytes()))
    assert set(records) == set(expected) and len(records) == 30
    assert type(config['expected_parent_records']) is int and config['expected_parent_records'] == 30
    for name in ('parent-terminal.json', 'child-terminal.json', 'result.json', 'child.log'):
        assert sum(p.name == name for p in regular) == 30
    return metadata, [records[name] for name in expected]"""
driver=driver[:old_start]+new_inspection+driver[old_stop:]
driver=driver.replace("and audit['temporary_root'] == str(temporary) and audit['scientific_admission_tested'] is False", "and audit['temporary_root'] == str(temporary) and audit['scientific_admission_tested'] is False\n              and audit['device_component_recording_only'] is True")
driver=driver.replace("        snapshot_guard_recording_only=True, full_scientific_admission_tested=False,", "        device_component_recording_only=True, full_scientific_admission_tested=False,\n        actual_device_initialized=False, genuine_capacity_measured=False,\n        temporary_recording_jax_children=sum(r['result']['temporary_recording_jax_used'] for r in child_records),\n        child_origin_evidence=config['child_origin_evidence'], observed_child_module_origin_table=False,")
ast.parse(driver)
(OUT/'executed_driver.py').write_text(driver)
def ownership(raw):
    return ast.dump(next(n for n in ast.walk(ast.parse(raw))if isinstance(n,ast.With)and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='Popen'for c in ast.walk(n))),include_attributes=False)
assert ownership(old_driver)==ownership(driver)
def alias_walk(raw):
    f=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name=='inspect_owned')
    return [ast.dump(n,include_attributes=False)for n in f.body[:3]]
assert alias_walk(old_driver)==alias_walk(driver.replace("name.startswith('test_device_component_')","name.startswith('test_snapshot_guard_')"))
for name,before,after in [('executed_driver.py',old_driver.decode(),driver),('worker_driver.py',old_worker.decode(),worker)]:
    (OUT/(name+'.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='snapshot-preparation-v2/'+name,tofile='device-preparation-v1/'+name)))
provenance=dict(schema='qwen-device-focused-harness-source-origin-v1',base=str(BASE),base_driver_sha256=sha(old_driver),base_worker_sha256=sha(old_worker),
    immediate_owned_Popen_cleanup_AST_exact=True,owned_alias_walk_only_prefix_adaptation=True,
    root_owned_snapshot_terminal='/tmp/qwen-server-admission-snapshot-root-review-v2-20261104/terminal-review.json',
    root_owned_snapshot_terminal_sha256=sha(Path('/tmp/qwen-server-admission-snapshot-root-review-v2-20261104/terminal-review.json').read_bytes()),
    declared_adaptations=['Exact50/34/30 source/config/node/receipt bindings and fresh output','Real runtime34 hashes recorded before/after in worker; no parent candidate imports','Typed30 early+positive receipt joins and120 regular child files','Child resource-call counters and temporary-double disclosure/restoration checked','Only owned pytest alias prefix adapted; no assumed alias count','Origin evidence is source-bound child assertions, not an observed module-origin table'],
    candidate_imported=False,collection=False,executed=False,actual_authority=False)
(OUT/'driver_provenance.json').write_bytes(encode(provenance))
(OUT/'README.md').write_text('''# Unrun device50 / 30 focused harness

This held preparation selects exactly50 source files,34 runtime files and30
fresh-child cases. The real device6/resource912d validators remain byte-exact.
Two phase check methods and owned process/JAX-shaped fixtures are explicit test
seams. No genuine backend/device/host-capacity/scientific/actual-role claim follows.

The original snapshot preparation-v2 owned Popen/direct-wait/exception-cleanup
AST is unchanged. Source/current/execution/archive checks precede the isolated
worker. The worker records all34 runtime hashes before and after its exact
prefixture selection; parent numeric/backend module origins remain empty.
Child origins are asserted by their source-bound source_binding implementation
and completed/restored assertions, not by a fabricated observed-origin table.

The driver requires30 early parent receipts and30 positive terminals/results/logs,
exactly120 regular child files, typed identity joins, restored original phase and
resource identities, explicit temporary-double removal, real resource-call
counters, and no genuine backend denials/imports. It records bounded owned pytest
aliases without following them and without assuming30 aliases. CPU112-115,
fresh TMPDIR/basetemp, explicit pinned pytest.ini, no plugin autoload/cache and
no healthy deadline remain. No collection, import or launch has occurred.

Root must review this source and authorize any one run separately. All parent
snapshot/device sources and proofs are preserved; this is not a run grant.
''')
argv=['env','-i','PATH=/usr/bin:/bin','HOME=/home/jianming_gatech','LANG=C.UTF-8','TZ=UTC','PYTHONDONTWRITEBYTECODE=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','taskset','-c','112-115','/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(OUT/'executed_driver.py')]
proposal=dict(schema='qwen-device-focused-launch-proposal-v1',proof_root=str(OUT),source_root=str(SOURCE),source_count=50,runtime_count=34,static_cases=30,
    source_manifest_sha256=sha(mapraw),source_archive_sha256=sha(archive),config_sha256=sha(encode(config)),driver_sha256=sha(driver.encode()),worker_sha256=sha(worker.encode()),
    argv=argv,expected_early_records=30,expected_positive_records=30,expected_regular_child_files=120,alias_count_not_assumed=True,
    held_unrun=True,current_execution_archive_equal=True,owned_Popen_cleanup_AST_exact=True,
    candidate_imported=False,collection=False,execution=False,capacity_observed=False,actual_policy_instances=False,launch_authorized=False,
    observed_child_module_origin_table=False,origin_evidence='source-bound completed/restored child assertions')
(OUT/'launch-proposal.json').write_bytes(encode(proposal))
files={p.relative_to(OUT).as_posix():sha(p.read_bytes())for p in sorted(OUT.rglob('*'))if p.is_file()and 'execution'not in p.relative_to(OUT).parts}
(OUT/'handoff.json').write_bytes(encode({**proposal,'files':files}))
assert{p.relative_to(execution).as_posix():sha(p.read_bytes())for p in execution.rglob('*')if p.is_file()}==mapping
assert all(sha((SOURCE/n).read_bytes())==d for n,d in mapping.items())
assert not(OUT/'tmp').exists()and not(OUT/'pytest-owned').exists()
print(json.dumps({**proposal,'handoff_sha256':sha((OUT/'handoff.json').read_bytes())},sort_keys=True))
