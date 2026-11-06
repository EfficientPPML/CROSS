"""Static source capture only: no candidate/NumPy imports or test collection."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile

ROOT=Path('/tmp/qwen-toy5-checkpoint-tests-20261104')
OUT=Path('/tmp/qwen-toy5-checkpoint-test-source-handoff-20261104')
PARENT=Path('/tmp/qwen-toy5-checkpoint-component-20261104')
PARENT_MAP=Path('/tmp/qwen-toy5-checkpoint-source-handoff-20261104/sources.json')
EXTERNAL=Path('/tmp/qwen-toy5-checkpoint-test-sources-v1-20261104.json')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
    state=path.lstat();assert stat.S_ISREG(state.st_mode)and not path.is_symlink()
    raw=path.read_bytes();assert len(raw)==state.st_size and path.lstat()==state
    return raw
def canonical(obj):return (json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def emit(name,obj):(OUT/name).write_bytes(canonical(obj))

parentraw=read(PARENT_MAP)
assert sha(parentraw)=='71c960c8d82f7d617ccb8167ac5bfbc526d5cdf54c2458b954ebfdfbb84f1ed3'
parent=json.loads(parentraw);assert len(parent)==250
assert read(ROOT/'runtime_sources.json')==parentraw
for name,pin in parent.items():
    assert sha(read(PARENT/name))==sha(read(ROOT/'runtime'/name))==pin
added=['runtime_sources.json','numpy_pins.json','q5_test_fixtures.py','conftest.py','test_q5_mapping.py','test_q5_metadata.py','test_q5_checkpoint_state.py','test_q5_entry.py','FOCUSED_TEST_PLAN.md','selected_tests.json','test_provenance.json']
names=sorted(['runtime/'+n for n in parent]+added)
assert len(names)==261
assert sorted(str(p.relative_to(ROOT))for p in ROOT.rglob('*')if p.is_file())==names
assert not any(p.is_symlink()for p in ROOT.rglob('*'))
mapping={name:sha(read(ROOT/name))for name in names}
selected=json.loads(read(ROOT/'selected_tests.json'));nodes=[];counts={}
for name in selected['test_files']:
    count=0
    for fn in ast.parse(read(ROOT/name),filename=name).body:
        if isinstance(fn,ast.FunctionDef)and fn.name.startswith('test_'):
            parameter=[d for d in fn.decorator_list if isinstance(d,ast.Call)and isinstance(d.func,ast.Attribute)and d.func.attr=='parametrize']
            assert len(parameter)<=1
            ids=['']
            if parameter:
                d=parameter[0];ids=ast.literal_eval(next(k.value for k in d.keywords if k.arg=='ids'))
                assert len(ids)==len(ast.literal_eval(d.args[1]))
            for label in ids:nodes.append(name+'::'+fn.name+('['+label+']'if parameter else''));count+=1
    counts[name]=count
assert nodes==selected['ordered_nodes']and len(nodes)==len(set(nodes))==48
assert counts==selected['counts']
for name in added:
    if name.endswith('.py'):ast.parse(read(ROOT/name),filename=name)
np_pins=json.loads(read(ROOT/'numpy_pins.json'))
assert read(ROOT/'numpy_pins.json')==read(PARENT/'reference/runtime_pins.json')
for name,pin in np_pins['sources'].items():assert sha(read(Path(name)))==pin
raw=canonical(mapping);EXTERNAL.write_bytes(raw);(OUT/'sources.json').write_bytes(raw)
with (OUT/'sources.tar.gz').open('xb')as f:
    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for name in names:
                data=read(ROOT/name);info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
with tarfile.open(OUT/'sources.tar.gz','r:gz')as tar:
    archived={m.name:sha(tar.extractfile(m).read())for m in tar}
assert archived==mapping
for name,pin in mapping.items():assert sha(read(ROOT/name))==pin
for name,pin in parent.items():assert sha(read(PARENT/name))==pin
review=dict(schema='qwen-toy5-checkpoint-focused-static-review-v1',source_count=261,runtime250_exact=True,new_files=11,static_cases=48,counts=counts,ordered_nodes=nodes,static_ast_only=True,candidate_imports=False,numpy_imports=False,collection=False,execution=False,metadata_probes=False,capacity_observations=False,backend_or_context=False,actual_role_authorized=False,source_current_archive_equal=True)
emit('static-review.json',review)
handoff=dict(schema='qwen-toy5-checkpoint-focused-source-handoff-v1',source_root=str(ROOT),source_map=str(EXTERNAL),source_manifest_sha256=sha(raw),source_count=261,source_archive_sha256=sha(read(OUT/'sources.tar.gz')),parent_source_manifest_sha256=sha(parentraw),runtime250_unchanged=True,new_files={n:mapping[n]for n in added},static_review_sha256=sha(read(OUT/'static-review.json')),selection_sha256=mapping['selected_tests.json'],static_case_count=48,counts=counts,ordered_nodes=nodes,numpy_pins_sha256=mapping['numpy_pins.json'],numpy_sources_checked_without_import=len(np_pins['sources']),candidate_imports=False,collection=False,execution=False,weight_generation=False,new47_mapping_proven=False,prior140_PASS_inherited=False,actual_execution_authorized=False,backend_or_context=False,capacity_observations=False,scope='New47 mapping/19 synthetic observed-scale original-graph replay; scalar enclosure and targeted attribution/state rejection. Metadata ciphertext stand-ins; no transport/loader/backend proof.',remaining=['peer/root source review','separate exact focused harness','explicit root authorization before any imports/collection/execution'])
emit('handoff.json',handoff)
print(json.dumps({k:v for k,v in handoff.items()if k not in ('new_files','ordered_nodes')},sort_keys=True))
