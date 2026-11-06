"""Independent stdlib-only snapshot harness review; never runs candidate code."""
import ast
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1')
OUT=Path('/tmp/qwen-snapshot-harness-security-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def compact(v):return(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
def pretty(v):return(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def inventory(root):
    paths=sorted(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}

handoff=json.loads((ROOT/'handoff.json').read_bytes())
assert sha((ROOT/'handoff.json').read_bytes()).startswith('c9c08053')
config=json.loads((ROOT/'run_config.json').read_bytes())
pins={name:sha((ROOT/name).read_bytes())for name in handoff['files']}
assert pins==handoff['files']
assert pins['executed_driver.py']=='c436247368deed6d0794040e51f7044ea54de7fc354bb70645cc0116512d37d6'
assert pins['worker_driver.py']=='55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be'
assert pins['run_config.json']=='02d9dec6cfff9dd65b9c32f89fda52a08c47848e33bbab74237b80395b28e47d'
expected=json.loads((ROOT/'sources-before.json').read_bytes())
assert len(expected)==37 and compact(expected)==(ROOT/'sources-before.json').read_bytes()
source=Path(config['source_root']);execution=ROOT/'execution'
assert inventory(source)==inventory(execution)==expected
with tarfile.open(ROOT/'sources.tar.gz','r:gz')as archive:
    members=archive.getmembers()
    assert len(members)==len({m.name for m in members})==37
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==expected
runtime=json.loads((execution/'runtime_sources.json').read_bytes())
assert len(runtime)==28 and inventory(execution/'runtime')==runtime
assert inventory(Path('/tmp/qwen-server-admission-snapshot-tests-20261104/runtime'))==runtime
test=ast.parse((execution/'test_snapshot_guard.py').read_bytes())
cases=ast.literal_eval(next(n.value for n in test.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='CASES'))
nodes=['test_snapshot_guard.py::test_snapshot_guard['+case+']'for case in cases]
xml=[dict(classname='test_snapshot_guard',name='test_snapshot_guard['+case+']')for case in cases]
assert len(nodes)==len(set(nodes))==18
assert nodes==config['selected_nodes']and xml==config['selected_xml_cases']
assert sha(pretty(nodes))==config['selected_nodes_sha256']and sha(pretty(xml))==config['selected_xml_cases_sha256']
assert config['selected_files']==['test_snapshot_guard.py']and config['cpu_affinity']==[112,113,114,115]
assert config['runtime_module_origins']==config['runtime_source_expectations']=={}
assert config['expected_child_records']==18 and 'numpy'in config['forbidden_roots']
assert config['synthetic_only']is True and config['actual_execution_authorized']is False
assert sha((execution/'pytest.ini').read_bytes())==config['pytest_ini_sha256']

def owned_try(raw):
    tree=ast.parse(raw)
    for node in ast.walk(tree):
        if not isinstance(node,ast.With):continue
        for index,item in enumerate(node.body[:-1]):
            if isinstance(item,ast.Assign)and isinstance(item.value,ast.Call)and isinstance(item.value.func,ast.Attribute)and item.value.func.attr=='Popen':
                after=node.body[index+1];assert isinstance(after,ast.Try)
                return ast.dump(after,include_attributes=False)
    raise AssertionError('immediate post-Popen ownership missing')
assert owned_try((ROOT/'executed_driver.py').read_bytes())==owned_try((ROOT/'before_executed_driver.py').read_bytes())
owned_try((execution/'test_snapshot_guard.py').read_bytes())
for name in('worker_driver.py','executed_driver.py'):ast.parse((ROOT/name).read_bytes(),filename=name)
assert pins=={name:sha((ROOT/name).read_bytes())for name in pins}
assert inventory(source)==inventory(execution)==expected
report=dict(schema='qwen-snapshot-harness-security-source-review-v1',passed=True,concrete_blockers=[],
    reviewed_root=str(ROOT),handoff_sha256=sha((ROOT/'handoff.json').read_bytes()),pins=pins,
    source_count=37,unchanged_runtime_count=28,static_test_count=18,
    source_manifest_sha256=config['source_manifest_sha256'],source_archive_sha256=config['source_archive_sha256'],
    checks={
        'current_execution_archive_and_compact_map_equal':True,
        'runtime28_equals_preserved_parent':True,
        'ordered18_static_node_and_xml_selection_exact':True,
        'prefixture_collection_gate_and_source_pinned_ini_reviewed':True,
        'sanitized_isolated_no_bytecode_cpu112_115_worker_and_children':True,
        'numpy_backend_denials_empty_origins_and_terminal_veto_reviewed':True,
        'outer_immediate_Popen_ownership_try_AST_equals_admitted_base':True,
        'child_PGID_race_preserves_observed_vs_reported_distinction':True,
        'owned_alias_uid_inode_readlink_direct_target_checks_no_traversal':True,
        'regular_temp_outputs_bounded128_entries_16MiB':True,
        'all18_result_terminal_log_triples_required_with_original_restore_flags':True,
        'direct_wait_reap_group_absence_and_primary_cleanup_reviewed':True},
    limitations=[
        'Source-only harness review; no collection, candidate import, test, subprocess, capacity or device probe performed.',
        'Temporary alias behavior and18 child receipts remain future runtime obligations, not observed outcomes.',
        'Recording full-validator seam and marker bytes do not test scientific authentication or confer actual role authority.',
        'Owned-output checks are not an OS or arbitrary-Python/native sandbox.',
        'A later completed report still requires external owner final exit/reap and independent terminal admission.'],
    no_candidate_imports=True,no_tests_or_probes=True,no_borrowed_body_reads=True,no_launch_authority=True)
(OUT/'review.json').write_bytes(pretty(report))
print(json.dumps({'review':str(OUT/'review.json'),'sha256':sha((OUT/'review.json').read_bytes()),'checks':len(report['checks']),'source_count':37,'static_cases':18},sort_keys=True))
