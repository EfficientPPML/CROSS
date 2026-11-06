"""Static author capture only; no candidate imports, collection or test calls."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/tmp/qwen-keygen-authority-tests-20261104')
PARENT = Path('/tmp/qwen-keygen-authority-boundary-20261104')
OUT = Path('/tmp/qwen-keygen-authority-test-source-handoff-20261104')
MAP = Path('/tmp/qwen-keygen-authority-test-sources-v1-20261104.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def source_map():
    result = {}
    for path in sorted(ROOT.rglob('*')):
        assert not path.is_symlink(), path
        if path.is_file(): result[path.relative_to(ROOT).as_posix()] = sha(path.read_bytes())
        else: assert path.is_dir(), path
    return result


start = source_map()
assert len(start) == 39
parent = json.loads((ROOT/'parent28_sources.json').read_bytes())
assert len(parent) == 28
for name, expected in parent.items():
    assert start[name] == expected == sha((PARENT/name).read_bytes())
assert sha((ROOT/'synthetic_aggregate_metadata.json').read_bytes()) == '18189e5f2f2de3a4939c6afca19d79822465d917caf06c4f6e36668bd37c1f55'
selection = json.loads((ROOT/'selected_pure_tests.json').read_bytes())
assert selection['files'] == ['test_keygen_launch.py','test_keygen_readiness.py','test_keygen_retention.py']
nodes = []
groups = {}
for name in selection['files']:
    tree = ast.parse((ROOT/name).read_text(), filename=name)
    count = 0
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef) or not node.name.startswith('test_'): continue
        cases = None
        for decorator in node.decorator_list:
            assert isinstance(decorator,ast.Call) and isinstance(decorator.func,ast.Attribute)
            assert decorator.func.attr == 'parametrize' and not decorator.keywords
            assert len(decorator.args) == 2 and ast.literal_eval(decorator.args[0]) == 'case'
            assert cases is None
            cases = ast.literal_eval(decorator.args[1])
            assert type(cases) is list and cases and all(type(item) is str for item in cases)
        found = [name+'::'+node.name] if cases is None else [name+'::'+node.name+'['+case+']' for case in cases]
        nodes.extend(found);count += len(found)
    groups[name] = count
assert len(nodes) == len(set(nodes)) == selection['expected_count'] == 48
assert groups == {'test_keygen_launch.py':17,'test_keygen_readiness.py':25,'test_keygen_retention.py':6}
assert nodes == selection['ordered_nodeids']
assert [dict(file=n.split('::')[0],name=n.split('::')[1]) for n in nodes] == selection['ordered_xml_cases']
parsed = []
for name in start:
    if name.endswith('.py'):
        ast.parse((ROOT/name).read_text(),filename=name);parsed.append(name)
assert len(parsed) == 12
assert source_map() == start
raw = canonical(start)
with MAP.open('xb') as stream: stream.write(raw)
with (OUT/'sources.tar.gz').open('xb') as stream:
    with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for name in start: archive.add(ROOT/name,arcname=name,recursive=False)
with tarfile.open(OUT/'sources.tar.gz','r:gz') as archive:
    rows = archive.getmembers()
    assert len(rows) == len(start) and all(row.isfile() for row in rows)
    assert {row.name:sha(archive.extractfile(row).read()) for row in rows} == start
assert source_map() == start
for name,expected in parent.items(): assert sha((PARENT/name).read_bytes()) == expected
handoff = dict(schema='qwen-keygen-pure-test-source-handoff-v1',source_root=str(ROOT),
    source_count=39,source_manifest=str(MAP),source_manifest_sha256=sha(raw),
    source_manifest_encoding='canonical sorted compact JSON plus newline',
    source_archive=str(OUT/'sources.tar.gz'),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
    parent_root=str(PARENT),parent_source_count=28,parent_source_manifest_sha256=start['parent28_sources.json'],
    all_parent28_unchanged=True,source_map=start,selected_files=selection['files'],
    static_expected_count=48,ordered_nodeids=nodes,ordered_xml_cases=selection['ordered_xml_cases'],
    group_counts=groups,parsed_python_files=parsed,candidate_imports=False,
    collection=False,test_execution=False,backend_or_numpy_imports=False,body_reads=False,
    rng=False,keygen=False,actual_model_admission_used=False,actual_role_grant_issued=False,
    scope='Static source authoring/capture only; synthetic expected proof and335 descriptor metadata; no operational authority',
    harness_owner='security_review',cpu_affinity_proposal=[112,113,114,115],
    root_source_and_harness_review_required_before_execution=True,
    capture_script_sha256=sha(Path(__file__).read_bytes()))
with (OUT/'handoff.json').open('xb') as stream: stream.write(canonical(handoff))
print(json.dumps({key:handoff[key] for key in ('source_count','source_manifest_sha256','source_archive_sha256','static_expected_count')}))
print('handoff_sha256='+sha((OUT/'handoff.json').read_bytes()))
