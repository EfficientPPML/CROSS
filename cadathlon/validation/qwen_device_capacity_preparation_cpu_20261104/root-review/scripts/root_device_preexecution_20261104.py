from pathlib import Path
import ast, datetime, hashlib, json, tarfile

sha = lambda b: hashlib.sha256(b).hexdigest()
pretty = lambda v: (json.dumps(v, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
proof = Path('/tmp/qwen-device-component-focused-proof-20261104/preparation-v1')
review = Path('/tmp/qwen-device-component-root-review-20261104')
review.mkdir(exist_ok=True)
pins = {
    'handoff.json': 'b46829a7876dc45f445d36dbb05271ca579548ec3c26fed2016d441f4aa350a9',
    'executed_driver.py': '45d8dbafdf3c7758960b976a8d254b11095336dbcae806014e164314ad76eef9',
    'worker_driver.py': 'e1cc022458245c3bc84a703bd0ef5c8b70522981d9f378ec0819536a141899d9',
    'run_config.json': '1b3cd9318ecf5a85731da977f473ab77073fe94c3d77dfad4c69aba34bf5d154',
    'launch-proposal.json': 'c38bfffd3d613cf813f127d883fca522a10e1b626d2c2d72fd7fef54b1d46109',
    'source-handoff.json': '3d5a78248daaf74b9039f3d4d1bd60381327728a18df9ff4e49167b6ccd25ed8',
}
for name, digest in pins.items():
    assert sha((proof / name).read_bytes()) == digest
handoff = json.loads((proof / 'handoff.json').read_bytes())
for name, digest in handoff['files'].items():
    assert sha((proof / name).read_bytes()) == digest
cfg = json.loads((proof / 'run_config.json').read_bytes())
raw = (proof / 'sources-before.json').read_bytes()
mapping = json.loads(raw)
assert raw == pretty(mapping) and sha(raw) == cfg['source_manifest_sha256'] == '1725249e18cf6298eda8516726955efe879635c0132579b65543ad3717493e74'
assert len(mapping) == cfg['source_count'] == 50

def inventory(root):
    assert root.resolve() == root and not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}

source = Path(cfg['source_root'])
assert inventory(source) == inventory(proof / 'execution') == mapping
archive = proof / 'sources.tar.gz'
assert sha(archive.read_bytes()) == cfg['source_archive_sha256'] == '1645d2fa604ad76afb41b4fdc9ea4ab88c51bd665eed3ed287ca79147923e5aa'
with tarfile.open(archive) as tar:
    members = tar.getmembers()
    assert len(members) == len({m.name for m in members}) == 50
    assert all(m.isfile() and m.name in mapping and '..' not in Path(m.name).parts and not Path(m.name).is_absolute() for m in members)
    assert {m.name: sha(tar.extractfile(m).read()) for m in members} == mapping
runtime = json.loads((source / 'runtime_sources.json').read_bytes())
assert len(runtime) == cfg['runtime_count'] == 34
assert inventory(source / 'runtime') == runtime == cfg['runtime_source_expectations']
assert cfg['runtime_module_origins'] == {}
old = Path('/tmp/qwen-device-component-tests-20261104')
old_map = json.loads(Path('/tmp/qwen-device-component-test-sources-v1-20261104.json').read_bytes())
assert len(old_map) == 49 and inventory(old) == old_map
assert {n for n in old_map if old_map[n] != mapping[n]} == {'device_child.py', 'TEST_SCOPE.md'}
assert set(mapping) - set(old_map) == {'successor_history.json'}
assert inventory(old / 'runtime') == runtime
tree = ast.parse((source / 'device_cases.py').read_bytes())
cases = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CASES' for t in n.targets)))
nodes = ['test_device_component.py::test_device_component[' + n + ']' for n in cases]
xml = [dict(classname='test_device_component', name=n.split('::')[1]) for n in nodes]
assert len(nodes) == 30 and len(set(nodes)) == 30 and nodes == cfg['selected_nodes']
assert xml == cfg['selected_xml_cases'] and cfg['selected_files'] == ['test_device_component.py']
assert sha(pretty(nodes)) == cfg['selected_nodes_sha256'] and sha(pretty(xml)) == cfg['selected_xml_cases_sha256']
assert sha((source / 'selected_tests.json').read_bytes()) == cfg['selection_sha256']
assert sha((source / 'pytest.ini').read_bytes()) == cfg['pytest_ini_sha256'] == '77d704711a449699ac290d3ea0d3acf1f07c664badbf3fe2803095447b374c49'
assert cfg['cpu_affinity'] == [112, 113, 114, 115]
assert cfg['synthetic_only'] is True and cfg['actual_execution_authorized'] is False
assert cfg['expected_child_records'] == cfg['expected_parent_records'] == 30
assert cfg['expected_regular_child_files'] == 120
assert Path(cfg['owned_tmpdir']) == proof / 'tmp' and Path(cfg['pytest_basetemp']) == proof / 'pytest-owned'
for name in mapping:
    if name.endswith('.py'):
        ast.parse((source / name).read_bytes(), filename=name)
for name in ('executed_driver.py', 'worker_driver.py'):
    ast.parse((proof / name).read_bytes(), filename=name)

def owner_block(path):
    return ast.dump(next(n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n, ast.Try) and n.body and isinstance(n.body[0], ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'child_pgid' for t in n.body[0].targets)), include_attributes=False)

assert owner_block(proof / 'executed_driver.py') == owner_block(proof / 'before_executed_driver.py')
absent = ['tests.log', 'tests.xml', 'report.json', 'worker-audit.json', 'live-observation.json', 'sources-after.json', 'execution-after.json', 'pytest-owned', 'tmp', 'owned-pytest-metadata.json', 'test-child-terminals.json']
assert not any((proof / n).exists() or (proof / n).is_symlink() for n in absent)
record = dict(schema='qwen-device-root-preexecution-review-v1', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    passed=True, source_count=50, runtime_files=34, static_cases=30, harness_pins=pins,
    source_manifest_sha256=sha(raw), source_archive_sha256=sha(archive.read_bytes()),
    source_current_execution_archive_equal=True, predecessor49_preserved=True, runtime34_unchanged=True,
    static_nodes=nodes, expected_regular_child_files=120, expected_early_parent_records=30,
    root_source_reads=['16a3fd', 'a264e6', 'aebc76', '026d94', 'c8a284'],
    root_harness_reads=['991ace', '9ccf4f'], owned_Popen_block_exact=True,
    child_origin_evidence=cfg['child_origin_evidence'], observed_child_origin_table=False,
    prior_outputs_absent=absent, source_only=True, candidate_imported=False, collection=False,
    execution_authorized=False, scope=cfg['scope'], script_sha256=sha(Path(__file__).read_bytes()))
out = review / 'preexecution-review.json'
with out.open('xb') as stream:
    stream.write(pretty(record))
print(json.dumps(dict(path=str(out), sha256=sha(out.read_bytes()), static_cases=30, passed=True)))
