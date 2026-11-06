"""Author an unrun fixed-source harness from the admitted device50 template."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

H=Path(__file__).resolve().parent
B=Path('/tmp/qwen-device-component-focused-proof-20261104/preparation-v1')
S=Path('/tmp/qwen-server-snapshot-integration-tests-20261104')
SH=Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
def canon(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def compact(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def inventory(p):
    out={}
    for f in sorted(p.rglob('*')):
        assert not f.is_symlink()
        if f.is_file():out[f.relative_to(p).as_posix()]=sha(f.read_bytes())
    return out
mapping=json.loads((SH/'sources.json').read_bytes());selection=json.loads((S/'selected_tests.json').read_bytes())
assert len(mapping)==404 and inventory(S)==mapping
assert sha((SH/'sources.json').read_bytes())=='9fa73f2f59e6bef71a2de475a09e0aaccb25d56375f7a06452cac7b75e4afc1c'
assert(SH/'sources.json').read_bytes()==compact(mapping)
assert sha((SH/'sources.tar.gz').read_bytes())=='bbc4c0bd3ce4bc6a1cd0efa10a7afc799e280887132869b846849fee10b2e713'
nodes=selection['static_nodeids'];assert len(nodes)==36 and selection['selected_files']==['test_snapshot_integration.py']
xml_cases=[dict(classname='test_snapshot_integration',name=n.split('::')[1])for n in nodes]
runtime=json.loads((S/'runtime_sources.json').read_bytes());assert len(runtime)==391
assert not(H/'execution').exists();(H/'execution').mkdir()
for name,digest in mapping.items():
    data=(S/name).read_bytes();assert sha(data)==digest
    target=H/'execution'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
for name,dest in(('sources.json','sources-before.json'),('sources.tar.gz','sources.tar.gz'),('handoff.json','source-handoff.json')):
    shutil.copyfile(SH/name,H/dest)
before_driver=(B/'executed_driver.py').read_text();before_worker=(B/'worker_driver.py').read_text()
assert sha(before_driver.encode())=='45d8dbafdf3c7758960b976a8d254b11095336dbcae806014e164314ad76eef9'
assert sha(before_worker.encode())=='e1cc022458245c3bc84a703bd0ef5c8b70522981d9f378ec0819536a141899d9'
(H/'before_executed_driver.py').write_text(before_driver);(H/'before_worker_driver.py').write_text(before_worker)
config=json.loads((B/'run_config.json').read_bytes())
config.update(schema='qwen-snapshot-integration-focused-proof-v1',source_root=str(S),proof_root=str(H),
    source_count=404,runtime_count=391,source_manifest_sha256=sha((SH/'sources.json').read_bytes()),
    source_archive_sha256=sha((SH/'sources.tar.gz').read_bytes()),source_manifest_encoding='canonical sorted compact JSON plus newline',
    runtime_source_expectations=runtime,selected_files=selection['selected_files'],selected_nodes=nodes,
    selected_nodes_sha256=sha(canon(nodes)),selected_xml_cases=xml_cases,selected_xml_cases_sha256=sha(canon(xml_cases)),
    selection_sha256=sha((S/'selected_tests.json').read_bytes()),pytest_ini_sha256=sha((S/'pytest.ini').read_bytes()),
    pytest_basetemp=str(H/'pytest-owned'),owned_tmpdir=str(H/'tmp'),expected_child_records=36,expected_parent_records=36,
    expected_regular_child_files=144,expected_owned_public_fixture_files=6,
    child_origin_evidence='Real source-bound entry.snapshot/origins and exact nested391; explicit definition-only __main__ and temporary source-owned namespaces. No installed-origin table.',
    temporary_jax_double='Owned ModuleType numpy/jax names only in selected children; source-defined recording config/platform only for process cases. Removed before result.',
    scope='Exactly36 fresh-child synthetic retained server integration cases with explicit admission/core/audit/loader seams; real runtime/control/backend constructor and Finalizer in their declared groups. Definition-only entry, no CLI bootstrap/science/body/capacity/native/actual role.',
    entry_fixture_provenance_sha256=sha((S/'entry_fixture_provenance.json').read_bytes()),seams_sha256=sha((S/'SEAMS.json').read_bytes()))
(H/'run_config.json').write_bytes(canon(config));config_hash=sha((H/'run_config.json').read_bytes())
worker=before_worker.replace('Unrun synthetic device-state worker','Unrun synthetic retained-runtime integration worker')
worker=worker.replace("len(runtime_before)==34","len(runtime_before)==391")
worker=worker.replace('device proof forbids','snapshot integration proof forbids').replace('exact device component cases','exact snapshot integration cases')
worker=worker.replace("['test_device_component.py']","['test_snapshot_integration.py']")
worker=worker.replace('device_component_recording_only=True','snapshot_integration_recording_only=True')
(H/'worker_driver.py').write_text(worker);worker_hash=sha(worker.encode())
driver=before_driver.replace('Held unrun synthetic device component driver','Held unrun synthetic retained-runtime integration driver')
driver=driver.replace("CONFIG_SHA256 = '1b3cd9318ecf5a85731da977f473ab77073fe94c3d77dfad4c69aba34bf5d154'",f'CONFIG_SHA256 = {config_hash!r}')
driver=driver.replace("WORKER_SHA256 = 'e1cc022458245c3bc84a703bd0ef5c8b70522981d9f378ec0819536a141899d9'",f'WORKER_SHA256 = {worker_hash!r}')
driver=driver.replace("map_canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\\n').encode()","map_canonical = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\\n').encode()")
driver=driver.replace('qwen-device-component-focused-proof-v1','qwen-snapshot-integration-focused-proof-v1')
driver=driver.replace('device proof pins remain unset','snapshot integration proof pins remain unset')
start=driver.index('    expected = [n.split(');end=driver.index('    return metadata, [records[name] for name in expected]',start)
replacement='''    expected = [n.split('[', 1)[1][:-1] for n in config['selected_nodes']]
    records = {}
    all_regular = [out / name for name, item in metadata.items() if item['kind'] == 'regular' and name.startswith('pytest-owned/')]
    fixture_files = [p for p in all_regular if p.name == 'tiny-public-proof.json']
    regular = [p for p in all_regular if p not in fixture_files]
    assert config['expected_regular_child_files'] == 144 and len(regular) == 144
    assert config['expected_owned_public_fixture_files'] == 6 and len(fixture_files) == 6
    assert all(p.name in ('parent-terminal.json', 'child-terminal.json', 'result.json', 'child.log') for p in regular)
    fixture_scenarios = []
    for path in sorted(p for p in regular if p.name == 'child-terminal.json'):
        terminal = json.loads(path.read_bytes())
        result_path, log_path = path.parent / 'result.json', path.parent / 'child.log'
        parent_path = path.parent / 'parent-terminal.json'
        assert all(p in regular for p in (result_path, log_path, parent_path))
        result = json.loads(result_path.read_bytes()); parent = json.loads(parent_path.read_bytes())
        assert type(parent) is dict and set(parent) == {'schema', 'pid', 'owned_session_pgid', 'observed_pgid', 'exit_code', 'reaped', 'group_absent', 'scenario', 'child_result_read'}
        assert parent['schema'] == 'qwen-snapshot-integration-owned-child-v1' and parent['child_result_read'] is False
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
        assert terminal['scope'] == 'synthetic retained-runtime integration only'
        for name in ('assertions_completed', 'original_main_restored', 'temporary_names_removed', 'profile_restored', 'declared_exports_restored', 'runtime_sources_unchanged', 'definition_only_entry'):
            assert result[name] is True
        for name in ('genuine_cli_bootstrap', 'genuine_backend_imported', 'scientific_admission', 'capacity_measurement', 'model_key_ciphertext_body_reads', 'actual_server_execution'):
            assert result[name] is False
        assert result['source_count'] == 391 and result['backend_denials'] == []
        assert result['entry_dispatch_excluded_sha256'] == config['excluded_entry_dispatch_ast_sha256']
        fixture = path.parent / 'tiny-public-proof.json'
        if scenario.startswith('process_'):
            assert fixture in fixture_files and fixture.stat().st_size <= 1024
            fixture_scenarios.append(scenario)
        else: assert fixture not in all_regular
        for group in (False, True):
            try: (os.killpg if group else os.kill)(pid, 0)
            except ProcessLookupError: pass
            else: raise ValueError('owned integration test child or group remains live')
        records[scenario] = dict(terminal=terminal, result=result, parent_terminal=parent,
            parent_terminal_path=parent_path.relative_to(out).as_posix(), parent_terminal_sha256=sha(parent_path.read_bytes()),
            terminal_path=path.relative_to(out).as_posix(), result_path=result_path.relative_to(out).as_posix(),
            log_path=log_path.relative_to(out).as_posix(), log_sha256=sha(log_path.read_bytes()))
    assert set(records) == set(expected) and len(records) == 36
    assert set(fixture_scenarios) == {n for n in expected if n.startswith('process_')}
    assert type(config['expected_parent_records']) is int and config['expected_parent_records'] == 36
    for name in ('parent-terminal.json', 'child-terminal.json', 'result.json', 'child.log'):
        assert sum(p.name == name for p in regular) == 36
'''
driver=driver[:start]+replacement+driver[end:]
driver=driver.replace("name.startswith('test_device_component_')","name.startswith('test_snapshot_integration_')")
driver=driver.replace("audit['device_component_recording_only']","audit['snapshot_integration_recording_only']")
driver=driver.replace('device_component_recording_only=True','snapshot_integration_recording_only=True')
driver=driver.replace("temporary_recording_jax_children=sum(r['result']['temporary_recording_jax_used'] for r in child_records),","definition_only_entry=True, genuine_cli_bootstrap=False,\n        retained_child_artifacts=144, owned_public_fixture_files=6,")
# Bind the independently source-derived excluded node as well as its full file.
config['excluded_entry_dispatch_ast_sha256']=json.loads((S/'entry_fixture_provenance.json').read_bytes())['excluded_dispatch_ast_sha256']
old_config_hash=config_hash;(H/'run_config.json').write_bytes(canon(config));config_hash=sha((H/'run_config.json').read_bytes())
driver=driver.replace(f'CONFIG_SHA256 = {old_config_hash!r}',f'CONFIG_SHA256 = {config_hash!r}')
(H/'executed_driver.py').write_text(driver)
old_ast=ast.parse(before_driver);new_ast=ast.parse(driver)
def ownership(tree):
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    with_node=next(n for n in main.body if isinstance(n,ast.With))
    assert isinstance(with_node.body[0],ast.Assign)and isinstance(with_node.body[1],ast.Try)
    return with_node.body[1]
assert ast.dump(ownership(old_ast),include_attributes=False)==ast.dump(ownership(new_ast),include_attributes=False)
ast.parse(worker)
assert "separators=(',', ':')"in driver and config['cpu_affinity']==[112,113,114,115]
assert '--import-mode=importlib'in worker and "'-c',str(source/'pytest.ini')"in worker
assert inventory(S)==inventory(H/'execution')==mapping
with tarfile.open(H/'sources.tar.gz','r:gz')as t:
    assert {m.name:sha(t.extractfile(m).read())for m in t.getmembers()}==mapping
for name,before,after in(('executed_driver.py',before_driver,driver),('worker_driver.py',before_worker,worker)):
    (H/(name+'.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='device50/'+name,tofile='snapshot404/'+name)))
provenance=dict(schema='qwen-snapshot-integration-harness-source-lineage-v1',base=str(B),
    base_driver_sha256=sha(before_driver.encode()),base_worker_sha256=sha(before_worker.encode()),
    changes=['Exact404/391 compact source map, static36 selected XML nodes and true scope.','Owned small-artifact inspection accepts144 child evidence files plus6 named tiny newly owned public proof fixtures; no fixture is an actual proof.','Entry fixture node pin, CPU112-115 unchanged, explicit pytest config/owned basetemp.'],
    immediate_Popen_ownership_try_AST_exact=True,current_execution_tar_equal=True,
    imports=False,collection=False,execution=False,launch_authorized=False)
(H/'driver_provenance.json').write_bytes(canon(provenance))
(H/'README.md').write_text('Held unrun404/36 harness; no launch authority. Exact391 nested source remains unchanged. Reuses admitted device50 Popen cleanup, fixed112-115 and explicit config. Source map encoding is compact sorted JSON plus newline. Retains144 result/log/early-parent/child-terminal artifacts; hashes6 tiny owned public fixtures separately. Optional owned pytest aliases are inspected without traversal. No genuine backend, CLI bootstrap, science, body or capacity claim. Source/driver review and explicit run grant remain required.\n')
argv=['env','-i','PATH=/usr/bin:/bin','HOME=/home/jianming_gatech','LANG=C.UTF-8','TZ=UTC','PYTHONDONTWRITEBYTECODE=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','taskset','-c','112-115','/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(H/'executed_driver.py')]
proposal=dict(schema='qwen-snapshot-integration-focused-launch-proposal-v1',proof_root=str(H),source_root=str(S),argv=argv,
    source_count=404,runtime_count=391,static_cases=36,source_manifest_sha256=config['source_manifest_sha256'],source_archive_sha256=config['source_archive_sha256'],
    driver_sha256=sha(driver.encode()),worker_sha256=sha(worker.encode()),config_sha256=config_hash,
    expected_early_records=36,expected_positive_records=36,expected_regular_child_artifacts=144,expected_owned_public_fixture_files=6,
    alias_count_not_assumed=True,source_manifest_encoding=config['source_manifest_encoding'],
    observed_child_module_origin_table=False,origin_evidence=config['child_origin_evidence'],
    current_execution_archive_equal=True,owned_Popen_cleanup_AST_exact=True,held_unrun=True,launch_authorized=False,
    candidate_imported=False,collection=False,execution=False,actual_policy_instances=False,capacity_observed=False)
(H/'launch-proposal.json').write_bytes(canon(proposal))
names=['README.md','before_executed_driver.py','before_worker_driver.py','driver_provenance.json','executed_driver.py','executed_driver.py.patch','launch-proposal.json','prepare_source.py','run_config.json','source-handoff.json','sources-before.json','sources.tar.gz','worker_driver.py','worker_driver.py.patch']
handoff=dict(proposal,files={n:sha((H/n).read_bytes())for n in names})
(H/'handoff.json').write_bytes(canon(handoff))
print(json.dumps(dict(passed=True,handoff_sha256=sha((H/'handoff.json').read_bytes()),proposal_sha256=sha((H/'launch-proposal.json').read_bytes()),**proposal),sort_keys=True))
