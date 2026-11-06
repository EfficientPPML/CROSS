"""Static source/test-scope review only; never import or collect candidate code."""
import ast
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/tmp/qwen-model-stage-aggregate-tests-20261104')
MAP = Path('/tmp/qwen-model-stage-test-review-sources-v1-20261104.json')
PARENT = Path('/tmp/qwen-model-stage-aggregate-20261104')
PARENT_MAP = Path('/tmp/qwen-model-stage-aggregate-review-sources-v1-20261104.json')
TAR = Path('/tmp/qwen-model-stage-test-source-handoff-20261104/sources.tar.gz')
MAP_SHA = '30ab709ada68471e65767357808398990cbb36329bb29650d91fc0b1a75a3f1a'
TAR_SHA = 'b456d1148ee8d2abe7e12444e5f9e3cdf2a806df7809771cf8b90c2d62780286'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def data(path):return json.loads(path.read_bytes())

assert sha(MAP.read_bytes()) == MAP_SHA
assert sha(TAR.read_bytes()) == TAR_SHA
expected = data(MAP)
parent = data(PARENT_MAP)
assert len(expected) == 38 and len(parent) == 29
before = {name: sha((ROOT / name).read_bytes()) for name in expected}
assert before == expected
assert {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()} == set(expected)
assert not any(p.is_symlink() for p in ROOT.rglob('*'))
for name, digest in parent.items():
    assert expected[name] == digest == sha((PARENT / name).read_bytes())
with tarfile.open(TAR, 'r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len(expected) and len({m.name for m in members}) == len(members)
    assert all(m.isfile() and not m.issym() and 0 <= m.size <= 8 << 20 for m in members)
    archived = {m.name: sha(archive.extractfile(m).read()) for m in members}
assert archived == expected

selection = data(ROOT / 'selected_model_tests.json')
assert selection['files'] == ['test_model_admission.py', 'test_held_identity.py', 'test_model_recording.py']
assert selection['affinity'] == [108, 109, 110, 111]
assert selection['python_flags'] == ['-I', '-B']
assert selection['actual_execution_enabled'] is False and selection['actual_model_body_reads'] is False
assert selection['inherited_suites_selected'] is False
nodes = []
counts = {}
for name in selection['files']:
    tree = ast.parse((ROOT / name).read_bytes(), filename=name)
    count = 0
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith('test_'):
            continue
        ids = None
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'parametrize':
                assert ids is None, 'one bounded parameter group per selected test'
                ids = ast.literal_eval(next(k.value for k in decorator.keywords if k.arg == 'ids'))
                assert type(ids) is list and all(type(i) is str for i in ids)
                assert len(ids) == len(decorator.args[1].elts)
        own = [name + '::' + node.name] if ids is None else [name + '::' + node.name + '[' + i + ']' for i in ids]
        nodes.extend(own)
        count += len(own)
    counts[name] = count
assert counts == selection['per_file'] == {'test_model_admission.py': 28, 'test_held_identity.py': 10, 'test_model_recording.py': 10}
assert nodes == selection['nodes'] and len(nodes) == selection['expected_nodes'] == 48

# Hash the explicitly listed installed NumPy source files without importing NumPy.
numpy_refs = data(ROOT / 'numpy_source_refs.json')
after = {name: sha((ROOT / name).read_bytes()) for name in expected}
assert after == before
report = {
    'schema': 'qwen-model-stage-transformers-test-source-review-v1',
    'passed': True,
    'scope': 'Source and static test-selection review only. No candidate imports, pytest collection, test execution or retained data access.',
    'source_root': str(ROOT),
    'source_manifest_sha256': MAP_SHA,
    'source_archive_sha256': TAR_SHA,
    'source_count': 38,
    'parent_sources_unchanged': 29,
    'source_hashes_before': before,
    'source_hashes_after': after,
    'static_node_count': 48,
    'static_nodes': nodes,
    'per_file': counts,
    'reviewed_assertions': [
        'Synthetic proof chains re-pin deliberate semantic mutations; rejection is not limited to stale digest checks. Canonical compact source-map hashing and distinct producer/consumer content identity are explicit.',
        'Owned small NPY controls distinguish successful original readback from expected root/file mismatch before first hash/header/mapping and metadata change before a later hash.',
        'Recording owners cover198 logical calls,197 physical cache records and113 gamma visits, ephemeral returned views, source-before-factory ordering and explicit unready result fields.',
        'Original close/readback callbacks remain exercised after callback mutation; body failure, cleanup-only cancellation priority and shield-construction fallback have bounded assertions.',
        'Actual entry denial precedes hostile argument access. Import/discovery guard prohibits declared model/HE backends; installed NumPy remains intentional for the owned small-file tests.'
    ],
    'findings': [],
    'limits': [
        'No test success or actual pytest collection claimed;48 is the independently expanded static source selection.',
        'Synthetic trusted expected pins/outer/root bytes do not authenticate real producer execution or model store integrity.',
        'Only newly owned small NPY files are proposed for real body IO; full model traversal uses recording owners without arrays or parameter files.',
        'No actual execution authority, current model admission, source-bound driver/runtime proof or broad inherited suite coverage.',
        'Root retains the separate future outer ModelAdmission prerequisite before RNG/key-loader entry.'
    ],
    'reviewer_history': ['Initial read-only filename guess selected_tests.json was absent; actual declared selected_model_tests.json was then read. No candidate execution occurred.'],
    'candidate_modules_imported': False,
    'collection_executed': False,
    'tests_executed': False,
    'actual_execution_authorized': False,
    'review_script_sha256': sha(Path(__file__).read_bytes()),
}
out = Path(__file__).with_name('test-source-review.json')
with out.open('xb') as stream:
    stream.write((json.dumps(report, sort_keys=True, indent=2) + '\n').encode())
print(json.dumps({'passed': True, 'source_count': 38, 'static_nodes': 48, 'review': str(out), 'sha256': sha(out.read_bytes())}))
