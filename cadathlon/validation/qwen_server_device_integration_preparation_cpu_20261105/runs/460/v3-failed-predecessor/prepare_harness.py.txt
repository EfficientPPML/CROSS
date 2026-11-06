"""Stdlib-only source preparation; never imports candidate or launches tests."""
import ast
import difflib
import hashlib
import json
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath

old = Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v2')
new = old.parent / 'preparation-v3'
source = Path('/tmp/qwen-full28-device-focused-tests-v2-20261105')
source_map = Path('/tmp/qwen-full28-device-focused-test-sources-v2-20261105.json')
source_handoff = Path('/tmp/qwen-full28-device-focused-test-handoff-v2-20261105/handoff.json')
review = Path('/tmp/qwen-full28-device-focused-root-source-review-20261105/correction459-review.json')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
pretty = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
compact = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def inventory(root):
    paths = sorted(root.rglob('*'))
    assert all(not p.is_symlink() for p in paths)
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in paths if p.is_file()}

old_handoff = json.loads((old / 'handoff.json').read_bytes())
old_config = json.loads((old / 'run_config.json').read_bytes())
old_files = {name: (old / name).read_bytes() for name in (
    'executed_driver.py', 'worker_driver.py', 'inspect_children.py', 'run_config.json',
    'launch-proposal.json', 'handoff.json', 'driver_provenance.json', 'sources-before.json',
    'source-handoff.json', 'prepare_harness.py')}
for name, raw in old_files.items():
    if name != 'handoff.json':
        assert sha(raw) == old_handoff['pins'][name]
assert sha((old / 'sources.tar.gz').read_bytes()) == old_handoff['source_archive_sha256']
raw_map = source_map.read_bytes()
expected = json.loads(raw_map)
handoff_source = json.loads(source_handoff.read_bytes())
assert raw_map == compact(expected) and len(expected) == 459
assert sha(raw_map) == '66126ceea29081ca2e1cb7781cc4bee4804e4c3faced8d72e7f4e4b85ebec318'
assert inventory(source) == expected
assert sha(review.read_bytes()) == '4cdffb13008f92156c729c9bdad48ca96a3b1b9f8115a30628eb311bb13e94d7'
archive_path = Path(handoff_source['source_archive'])
assert sha(archive_path.read_bytes()) == handoff_source['source_archive_sha256']
assert not new.exists()
new.mkdir()
execution = new / 'execution'
execution.mkdir()
for name, digest in expected.items():
    rel = PurePosixPath(name)
    assert str(rel) == name and not rel.is_absolute() and '..' not in rel.parts
    original = source / name
    assert stat.S_ISREG(original.lstat().st_mode)
    target = execution / name
    target.parent.mkdir(parents=True, exist_ok=True)
    data = original.read_bytes()
    assert sha(data) == digest
    target.write_bytes(data)
(new / 'sources-before.json').write_bytes(raw_map)
(new / 'source-handoff.json').write_bytes(source_handoff.read_bytes())
shutil.copyfile(archive_path, new / 'sources.tar.gz')
(new / 'worker_driver.py').write_bytes(old_files['worker_driver.py'])

left = """            # The only selected target is one lexical sibling basename. Never
            # resolve an alias or follow a target symlink during validation.
            assert type(target)is str and 0<len(target)<=255 and target not in('.', '..')
            assert '/'not in target and '\\\\'not in target and Path(target).name==target
            named=child.parent/target;assert named.parent==base and named!=child
"""
right = """            # Pytest can use an absolute lexical sibling or a single basename.
            # Never resolve the alias or follow a target symlink during validation.
            assert type(target)is str and 0<len(target)<=4096 and '\\\\'not in target
            lexical=Path(target)
            assert str(lexical)==target and '..'not in lexical.parts and '.'not in lexical.parts
            if lexical.is_absolute():
                named=lexical
            else:
                assert target not in('.', '..')and '/'not in target and lexical.name==target
                named=child.parent/target
            assert 0<len(named.name)<=255 and named.parent==base and named!=child
"""
inspector = old_files['inspect_children.py'].decode()
assert inspector.count(left) == 1
inspector = inspector.replace(left, right)
(new / 'inspect_children.py').write_text(inspector)
inspector_sha = sha(inspector.encode())

config = dict(old_config)
for key in ('proof_root', 'owned_tmpdir', 'pytest_basetemp'):
    config[key] = config[key].replace(str(old), str(new))
config.update(source_root=str(source), source_count=459,
              source_manifest_sha256=sha(raw_map),
              source_archive_sha256=handoff_source['source_archive_sha256'],
              external_source_review=dict(path=str(review), sha256=sha(review.read_bytes())))
(new / 'run_config.json').write_bytes(pretty(config))
config_sha = sha((new / 'run_config.json').read_bytes())
driver = old_files['executed_driver.py'].decode()
assert driver.count('458') == 4
driver = driver.replace('458', '459').replace(old_handoff['config_sha256'], config_sha).replace(old_handoff['inspector_sha256'], inspector_sha)
(new / 'executed_driver.py').write_text(driver)
proposal = json.loads(old_files['launch-proposal.json'])
proposal['argv'] = [part.replace(str(old), str(new)) for part in proposal['argv']]
proposal.update(driver_sha256=sha(driver.encode()), config_sha256=config_sha,
                inspector_sha256=inspector_sha, source_manifest_sha256=sha(raw_map),
                source_archive_sha256=handoff_source['source_archive_sha256'])
(new / 'launch-proposal.json').write_bytes(pretty(proposal))

history = new / 'v2-failed-predecessor'
history.mkdir()
for name, data in old_files.items():
    (history / (name + '.txt')).write_bytes(data)
for name in ('inspect_children.py', 'executed_driver.py', 'run_config.json', 'launch-proposal.json'):
    (history / (name + '.patch')).write_text(''.join(difflib.unified_diff(
        old_files[name].decode().splitlines(True), (new / name).read_text().splitlines(True),
        fromfile='failed-preparation-v2/' + name, tofile='unrun-preparation-v3/' + name)))
external = {}
for label, path, pin in (
    ('root459_source', review, '4cdffb13008f92156c729c9bdad48ca96a3b1b9f8115a30628eb311bb13e94d7'),
    ('root_failed_terminal', Path('/tmp/qwen-full28-device-focused-root-harness-review-20261105/failed-terminal-review.json'), '9d824d2c3c8ee6f66eda409cad946b87302b0f5bd518d3ce223a61d0e7d711e9'),
    ('cnn_failed_peer', Path('/tmp/qwen-full28-device-failed-security-peer-20261105/review.json'), 'e4887cf3d92cd834b9d379907e08011e8305517a41703c1250796b1965c2b9e8'),
):
    data = path.read_bytes()
    assert sha(data) == pin
    (history / (label + '.json')).write_bytes(data)
    external[label] = dict(path=str(path), sha256=pin)

provenance = dict(
    schema='full28-focused-harness-source459-alias-successor-v3',
    predecessor_root=str(old), predecessor_handoff_sha256=sha(old_files['handoff.json']),
    predecessor_source458_status='FAILED_FIRST_RUN_59_CHILD_SETUP_FAILURES_BEFORE_CANDIDATE_IMPORT',
    predecessor_worker_exit_code=1, predecessor_collector_exit_code=70,
    predecessor_failure='Function-local importlib shadowing; no result files. Strict inspector then failed closed on missing results.',
    source_successor='Exact reviewed459 removes only redundant local import; all403 runtime bytes and59 assertions unchanged.',
    inspector_correction='Accept canonical lexical absolute sibling or one basename, require same parent, numeric selected prefix, lstat directory/UID, unchanged alias lstat/readlink. No resolve/followed target tests.',
    optional_missing_result_reporting_added=False,
    missing_result_behavior='Unchanged fail-closed infrastructure70; no successful admission of missing results.',
    changed_execution_files=['inspect_children.py', 'executed_driver.py', 'run_config.json'],
    driver_delta='Config/inspector pins and four458-to459 count/doc literals only.',
    changed_config_fields=sorted(k for k in config if config[k] != old_config[k]),
    worker_unchanged=True, owned_popen_wait_finally_unchanged=True,
    selected_nodes59_unchanged=True, runtime403_unchanged=True,
    external_reviews=external, old_harness_files={k:sha(v) for k,v in old_files.items()},
    current_status='HELD_UNRUN_SOURCE_ONLY', imports_collection_tests=False,
    actual_execution_authorized=False, previous_v1_status='PRESERVED_UNRUN',
)
(new / 'driver_provenance.json').write_bytes(pretty(provenance))
(new / 'prepare_harness.py').write_bytes(Path(__file__).read_bytes())

for name in ('inspect_children.py', 'executed_driver.py', 'worker_driver.py'):
    ast.parse((new / name).read_bytes())
tree = ast.parse(inspector)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inspect')
loop = next(n for n in function.body if isinstance(n, ast.For))
branch = next(n for n in loop.body if isinstance(n, ast.If))
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('resolve', 'is_dir', 'is_symlink', 'stat') for n in ast.walk(branch))
assert inspector.replace(right, left) == old_files['inspect_children.py'].decode()
assert driver.replace('459', '458').replace(config_sha, old_handoff['config_sha256']).replace(inspector_sha, old_handoff['inspector_sha256']) == old_files['executed_driver.py'].decode()
def owned(text):
    return text.split("    with (out/'tests.log').open('xb')as log:\n", 1)[1].split('    current,after=inventory(source),inventory(execution)', 1)[0]
assert owned(driver) == owned(old_files['executed_driver.py'].decode())
assert inventory(execution) == inventory(source) == expected
runtime_map = {k[len('runtime/'):]:v for k,v in expected.items() if k.startswith('runtime/')}
assert runtime_map == config['runtime_source_expectations'] and len(runtime_map) == 403
with tarfile.open(new / 'sources.tar.gz') as archive:
    members = archive.getmembers()
    assert len(members) == len({m.name for m in members}) == 459
    assert all(m.isfile() for m in members)
    assert {m.name:sha(archive.extractfile(m).read()) for m in members} == expected
assert (new / 'worker_driver.py').read_bytes() == old_files['worker_driver.py']
assert config['selected_nodes'] == old_config['selected_nodes'] and len(config['selected_nodes']) == 59
assert sha((execution / 'selected_tests.json').read_bytes()) == config['selection_sha256']
assert all((old / name).read_bytes() == data for name, data in old_files.items())
pins = {p.relative_to(new).as_posix():sha(p.read_bytes()) for p in sorted(new.rglob('*'))
        if p.is_file() and 'execution' not in p.relative_to(new).parts}
handoff = dict(schema='full28-focused-unrun-harness-handoff-v3',
    source_root=str(source), proof_root=str(new), execution_root=str(execution),
    source_count=459, runtime_count=403, static_cases=59,
    source_manifest_sha256=sha(raw_map), source_archive_sha256=handoff_source['source_archive_sha256'],
    driver_sha256=sha(driver.encode()), worker_sha256=sha(old_files['worker_driver.py']),
    inspector_sha256=inspector_sha, config_sha256=config_sha,
    proposal_sha256=sha((new / 'launch-proposal.json').read_bytes()),
    predecessor_handoff_sha256=sha(old_files['handoff.json']),
    correction='Exact459 source and lexical absolute-or-basename sibling aliases; no success/report relaxation.',
    external_source_review=external['root459_source'], pins=pins,
    imports_collection_runs=0, actual_execution_authorized=False)
(new / 'handoff.json').write_bytes(pretty(handoff))
print(json.dumps({k:handoff[k] for k in ('source_count','driver_sha256','worker_sha256','inspector_sha256','config_sha256','proposal_sha256')}, indent=2))
print('handoff', sha((new / 'handoff.json').read_bytes()))
print('Exact459/current/execution/archive and403/runtime plus59 selection; parent ownership text identical; source-only preparation, no collection/import/tests.')
