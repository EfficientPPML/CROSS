"""Root source/AST-only review of the focused38/48 successor."""
import ast,hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/tmp/qwen-model-stage-aggregate-tests-20261104')
PARENT=Path('/tmp/qwen-model-stage-aggregate-20261104')
MAP=Path('/tmp/qwen-model-stage-test-review-sources-v1-20261104.json')
HANDOFF=Path('/tmp/qwen-model-stage-test-source-handoff-20261104')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()}
assert sha(MAP)=='30ab709ada68471e65767357808398990cbb36329bb29650d91fc0b1a75a3f1a'
assert sha(HANDOFF/'handoff.json')=='f24064ee60bea4a398a4cd6ae29f656865e83e8d349a7edd1919ca7f001b4b03'
expected=read(MAP);parent=read(Path('/tmp/qwen-model-stage-aggregate-review-sources-v1-20261104.json'))
assert len(expected)==38 and len(parent)==29
assert inventory(SOURCE)==expected and inventory(PARENT)==parent
assert all(expected[n]==digest for n,digest in parent.items())
assert set(expected)-set(parent)=={'READY_TESTS.json','TEST_SCOPE.json','conftest.py','numpy_source_refs.json','parent29_sources.json','selected_model_tests.json','test_held_identity.py','test_model_admission.py','test_model_recording.py'}
assert read(SOURCE/'parent29_sources.json')==parent
archive_path=HANDOFF/'sources.tar.gz'
assert sha(archive_path)=='b456d1148ee8d2abe7e12444e5f9e3cdf2a806df7809771cf8b90c2d62780286'
with tarfile.open(archive_path)as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==38
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()for m in members}==expected
selection=read(SOURCE/'selected_model_tests.json');nodes=[];counts={}
for filename in selection['files']:
    start=len(nodes)
    for fn in ast.parse((SOURCE/filename).read_bytes()).body:
        if not isinstance(fn,ast.FunctionDef)or not fn.name.startswith('test_'):continue
        params=[d for d in fn.decorator_list if isinstance(d,ast.Call)and isinstance(d.func,ast.Attribute)and d.func.attr=='parametrize']
        assert len(params)<=1
        if params:
            call=params[0];ids=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='ids'))
            assert isinstance(call.args[1],ast.List)and len(call.args[1].elts)==len(ids)
            assert all(type(i)is str for i in ids)
            nodes.extend(filename+'::'+fn.name+'['+i+']'for i in ids)
        else:nodes.append(filename+'::'+fn.name)
    counts[filename]=len(nodes)-start
assert nodes==selection['nodes']and len(nodes)==selection['expected_nodes']==48
assert counts==selection['per_file']=={'test_model_admission.py':28,'test_held_identity.py':10,'test_model_recording.py':10}
refs=read(SOURCE/'numpy_source_refs.json')
runtime_pins={row['path']:row['sha256']for row in refs['expected_loaded_modules'].values()}
runtime_pins[refs['additional_source']['path']]=refs['additional_source']['sha256']
assert len(runtime_pins)==4
for path,digest in runtime_pins.items():assert sha(Path(path))==digest
report=dict(schema='qwen-model-stage-root-test-source-review-v1',passed=True,source_count=38,parent_sources_unchanged=29,
    source_manifest_sha256=sha(MAP),source_archive_sha256=sha(archive_path),handoff_sha256=sha(HANDOFF/'handoff.json'),
    selected_nodes=nodes,per_file=counts,test_sources={n:expected[n]for n in selection['files']},guard_sha256=expected['conftest.py'],
    runtime_source_pins=runtime_pins,expected_loaded_numpy_modules=refs['expected_loaded_modules'],
    script_sha256=sha(Path(__file__)),runtime_source_review_sha256=sha(HERE/'implementation-review.json'),
    tests_executed=0,actual_execution_authorized=False,
    scope='28 synthetic externally pinned proof-byte checks;10 newly owned small-file identity controls;10 recording full198/197+113 traversal/lifetime cases. No retained tensor/key/CT bodies, full physical tensors, backend or actual role.',
    limitations='Positive admission chains are synthetic fixtures; no genuine model integrity, archive admission, body-read grant, scientific/calibration/HE or performance result follows.',
    next='Peer and final harness source review before exactly one focused execution; no separate collection needed.')
(HERE/'test-source-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in('selected_nodes','runtime_source_pins','expected_loaded_numpy_modules')},sort_keys=True))
