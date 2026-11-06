"""Root held-source/AST review; no producer or test imports."""
import ast,hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=Path('/tmp/qwen-key-generation-producer-20261104')
NEW=Path('/tmp/qwen-key-generation-producer-v2-20261104')
MAP=Path('/tmp/qwen-key-producer-review-sources-v2-20261104.json')
ARCHIVE=Path('/tmp/qwen-key-producer-source-readiness-20261104/sources.tar.gz')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}
expected=read(MAP);before=read(Path('/tmp/qwen-key-producer-review-sources-v1-20261104.json'))
assert sha(MAP.read_bytes())=='6b8a842e72694d691ea2f4a2d9831331b14e929ad521d701fe445e85f939d874'
assert len(expected)==71 and len(before)==64
assert inventory(NEW)==expected and inventory(OLD)==before
assert {n for n in before if expected[n]!=before[n]}=={'qwen_key_store.py','DESIGN.md','implementation_origin.json'}
assert expected['qwen_key_store.py']=='2e60908c9773debaa85a5d325df1367013ffb99c6ddead2f587dafca78efb0bd'
assert expected['qwen_key_producer.py']=='f85277dc7fe0d7dfcbb01e14500d745d975bb41017199cef6e948d6247427529'
copies=read(NEW/'selected_sources.json')['files'];assert len(copies)==57
for name,row in copies.items():assert expected[name]==row['sha256']==sha(Path(row['path']).read_bytes())
assert sha(ARCHIVE.read_bytes())=='536fe8696324dcc4d87b635a95b647e311daeef99b06b729955d0c1ffe9e0d97'
with tarfile.open(ARCHIVE)as archive:
    members=archive.getmembers()
    assert len(members)==len({m.name for m in members})==71
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==expected
selection=read(NEW/'selected_tests.json');nodes=[];counts={}
for filename in selection['files']:
    start=len(nodes)
    for fn in ast.parse((NEW/filename).read_bytes()).body:
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
assert nodes==selection['nodes'] and len(nodes)==selection['expected_nodes']==30
assert counts==selection['per_file']=={'test_key_store.py':15,'test_key_producer.py':15}
numpy_pins=read(NEW/'harness_provenance.json')['runtime_source_expectations']
for path,digest in numpy_pins.items():assert sha(Path(path).read_bytes())==digest
report=dict(schema='qwen-key-producer-root-successor-source-review-v1',passed=True,
    source_count=71,unchanged_copies=57,producer_unchanged=True,source_manifest_sha256=sha(MAP.read_bytes()),
    source_archive_sha256=sha(ARCHIVE.read_bytes()),selected_nodes=nodes,per_file=counts,
    selected_tests_sha256=expected['selected_tests.json'],runtime_source_pins=numpy_pins,
    script_sha256=sha(Path(__file__).read_bytes()),tests_executed=0,
    scope='Source/AST only. Small owned NPY and synchronous writer faults; full166/335 recording metadata/order with original secret. No real keygen/backend/model or inherited suites.',
    coverage_limit='No claim for injected invalidation cancellation, malformed header/residue negatives, fsync failures or every malformed rotation-return variant.',
    next='Final driver/source/peer review before one focused synthetic execution.',actual_execution_authorized=False)
(HERE/'successor-source-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('selected_nodes','runtime_source_pins')},sort_keys=True))
