"""Author's static source capture. Does not import or call candidate modules."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/tmp/qwen-keygen-authority-boundary-20261104')
OUT = Path('/tmp/qwen-keygen-authority-boundary-handoff-20261104')
MAP = Path('/tmp/qwen-keygen-authority-boundary-sources-v1-20261104.json')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def current():
    result = {}
    for path in sorted(ROOT.rglob('*')):
        assert not path.is_symlink(), path
        if path.is_file():
            result[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
        else:
            assert path.is_dir(), path
    return result


start = current()
assert len(start) == 28
copies = json.loads((ROOT/'selected_copies.json').read_bytes())
assert len(copies) == 20
for name, origin in copies.items():
    assert start[name] == origin['sha256'] == digest(Path(origin['path']).read_bytes())

parent_checks = []
for base, map_name, count in (
        ('/tmp/qwen-key-generation-producer-v2-20261104', 'parent71_sources.json', 71),
        ('/tmp/qwen-model-stage-aggregate-20261104', 'model29_sources.json', 29)):
    source_map = json.loads((ROOT/map_name).read_bytes())
    assert len(source_map) == count
    for name, expected in source_map.items():
        path = Path(base)/name
        assert not path.is_symlink() and path.is_file()
        assert digest(path.read_bytes()) == expected, name
    parent_checks.append(dict(root=base, source_count=count,
                              inventory_file_sha256=start[map_name]))

modules = {}
for name in start:
    if name.endswith('.py'):
        modules[name] = ast.parse((ROOT/name).read_text(), filename=name)
assert len(modules) == 7
code = modules['keygen_authority.py']
literal = {}
for node in code.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id.endswith('_FIELDS') or node.targets[0].id in ('CHECK_NAMES', 'SCOPE_FALSE'):
            literal[node.targets[0].id] = ast.literal_eval(node.value)
entry = next(node for node in code.body if isinstance(node, ast.FunctionDef) and node.name == 'create_actual_keygen_process')
assert len(entry.body) == 1 and isinstance(entry.body[0], ast.Raise)
assert isinstance(entry.body[0].exc, ast.Call) and isinstance(entry.body[0].exc.func, ast.Name)
assert entry.body[0].exc.func.id == 'PermissionError'

schemas = json.loads((ROOT/'boundary.schema.json').read_bytes())['$defs']
mapping = dict(actual_intent='INTENT_FIELDS', expected_launch_pins='LAUNCH_PIN_FIELDS',
               grant='GRANT_FIELDS', expected_ready_pins='READY_PIN_FIELDS',
               positive_report='REPORT_FIELDS', outer='OUTER_FIELDS', root_readiness='ROOT_FIELDS')
for name, field in mapping.items():
    row = schemas[name]
    assert row['type'] == 'object' and row['additionalProperties'] is False
    assert set(row['required']) == set(row['properties']) == literal[field]
report = schemas['positive_report']['properties']
assert set(report['checks']['properties']) == set(literal['CHECK_NAMES'])
assert set(report['scope']['properties']) == set(literal['SCOPE_FALSE']) | {'fresh_key_generation', 'new_owned_key_readback'}
assert report['backend']['properties']['observed_device_platforms']['items'] == {'const': 'cpu'}

pins = next(ast.literal_eval(n.value) for n in modules['keygen_boundary_pins.py'].body
            if isinstance(n, ast.Assign) and n.targets[0].id == 'REQUIRED_ACTUAL_SOURCES')
assert len(pins) == 58
parent = json.loads((ROOT/'parent71_sources.json').read_bytes())
model = json.loads((ROOT/'model29_sources.json').read_bytes())
for name, expected in pins.items():
    assert (model if name == 'model_admission.py' else parent)[name] == expected

refs = json.loads((ROOT/'source_refs.json').read_bytes())
for row in refs.values():
    assert digest(Path(row['path']).read_bytes()) == row['sha256']
assert current() == start
map_raw = canonical(start)
with MAP.open('xb') as stream:
    stream.write(map_raw)
with (OUT/'sources.tar.gz').open('xb') as stream:
    with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name in start:
                archive.add(ROOT/name, arcname=name, recursive=False)
with tarfile.open(OUT/'sources.tar.gz', mode='r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len(start)
    assert all(m.isfile() for m in members)
    archived = {m.name: digest(archive.extractfile(m).read()) for m in members}
    assert archived == start
assert current() == start
handoff = dict(schema='qwen-keygen-authority-source-handoff-v1', source_root=str(ROOT),
    source_count=28, source_manifest=str(MAP), source_manifest_sha256=digest(map_raw),
    source_manifest_encoding='sorted compact UTF-8 JSON plus one newline',
    source_archive=str(OUT/'sources.tar.gz'), source_archive_sha256=digest((OUT/'sources.tar.gz').read_bytes()),
    source_map=start, exact_copied_files=20, unchanged_parents=parent_checks,
    parsed_python_sources=7, public_argument_schema_count=7, required_future_prerequisite_sources=58,
    actual_entry_ast_immediate_denial=True, static_schema_field_sets_match=True,
    static_source_checks_only=True, candidate_imports=False, tests_or_collection=False,
    backend_imports=False, body_reads=False, rng=False, key_generation=False,
    grant_issued=False, process_entry_or_output_implemented=False,
    actual_memory_policy_unresolved=True, actual_parent_binding_unimplemented=True,
    actual_role_maps_and_grants_pending=True,
    evidence_scope='Author static source/metadata capture only; no candidate execution or test result',
    capture_script_sha256=digest(Path(__file__).read_bytes()))
with (OUT/'handoff.json').open('xb') as stream:
    stream.write(canonical(handoff))
print(json.dumps({k: handoff[k] for k in ('source_count', 'source_manifest_sha256', 'source_archive_sha256')}))
print('handoff_sha256='+digest((OUT/'handoff.json').read_bytes()))
