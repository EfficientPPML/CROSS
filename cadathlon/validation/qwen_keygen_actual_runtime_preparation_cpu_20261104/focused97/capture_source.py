"""Static test selection/capture only; never imports or collects candidate code."""
import ast,hashlib,itertools,json,tarfile
from pathlib import Path
R=Path('/tmp/qwen-keygen-actual-runtime-tests-20261104');O=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
canonical=lambda x:(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
pretty=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
files=['test_keygen_output.py','test_keygen_policy.py','test_keygen_generation.py','test_keygen_entry.py']
nodes=[];per_file={}
for filename in files:
    tree=ast.parse((R/filename).read_text(),filename=filename);local=[]
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef)or not fn.name.startswith('test_'):continue
        axes=[]
        for d in reversed(fn.decorator_list):
            assert isinstance(d,ast.Call)and isinstance(d.func,ast.Attribute)and d.func.attr=='parametrize'
            assert len(d.args)==2 and not d.keywords
            ast.literal_eval(d.args[0]);axis=ast.literal_eval(d.args[1]);assert type(axis)is list
            axes.append(axis)
        suffixes=['']if not axes else ['['+'-'.join(map(str,row))+']'for row in itertools.product(*axes)]
        local.extend(filename+'::'+fn.name+s for s in suffixes)
    per_file[filename]=len(local);nodes+=local
assert len(nodes)==len(set(nodes))
selection=dict(schema='qwen-keygen-actual-focused-static-selection-v1',selected_files=files,selected_nodes=nodes,
    selected_xml_cases=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in nodes],
    per_file=per_file,static_count=len(nodes),cpu_proposal=[112,113,114,115],collected=0,executed=0,
    genuine_numpy_backend_forbidden=True,temporary_module_name_doubles='Exactly three run-flow recording cases; removed before guard postcheck',
    inherited_tests_selected=False,actual_execution_authorized=False)
(R/'selected_tests.json').write_bytes(pretty(selection))
parent_raw=(R/'parent85_sources.json').read_bytes();parent=json.loads(parent_raw)
assert sha(parent_raw)=='28782d1927a4334c778dc8429d92637776bfe6a97a6a42e0d25f99cc82f35699'
def inventory(path):
    paths=list(path.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return {p.relative_to(path).as_posix():sha(p.read_bytes())for p in sorted(paths)if p.is_file()}
assert inventory(R/'runtime')==parent and len(parent)==85
old=Path('/tmp/qwen-keygen-actual-runtime-20261104')
for name in ('keygen_entry_status.json','keygen_static_policy.json'):
    assert(R/'history'/name).read_bytes()==(old/name).read_bytes()
origin=dict(schema='qwen-keygen-actual-focused-origin-v1',parent85_root='/tmp/qwen-keygen-actual-runtime-v2-20261104',
    parent85_compact_map_sha256=sha(parent_raw),all85_runtime_exact=True,source_only=True,
    tests_or_collection=False,real_backend_or_numpy=False,body_reads=False,capacity_observation=False,rng=False,
    source_admission_only='Root03cbf0/d2c6c075 and peer0536dceb source-only; no actual grant or runtime execution authority.')
(R/'test_origin.json').write_bytes(pretty(origin))
sources=inventory(R)
for name in sources:
    if name.endswith('.py'):ast.parse((R/name).read_text(),filename=name)
(O/'sources.json').write_bytes(canonical(sources));(O/'sources-pretty.json').write_bytes(pretty(sources))
with tarfile.open(O/'sources.tar.gz','w:gz')as archive:
    for name in sources:archive.add(R/name,arcname=name,recursive=False)
with tarfile.open(O/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len(sources)and all(m.isfile()for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==sources
assert inventory(R)==sources and inventory(R/'runtime')==parent
handoff=dict(schema='qwen-keygen-actual-focused-source-handoff-v1',source_root=str(R),source_count=len(sources),
    source_manifest=str(O/'sources.json'),source_manifest_sha256=sha((O/'sources.json').read_bytes()),
    source_manifest_encoding='canonical sorted compact JSON plus newline',source_archive_sha256=sha((O/'sources.tar.gz').read_bytes()),
    parent85_compact_map_sha256=sha(parent_raw),all85_runtime_exact=True,static_count=len(nodes),per_file=per_file,
    selected_files=files,selected_nodes=nodes,selected_xml_cases=selection['selected_xml_cases'],selected_tests_sha256=sha((R/'selected_tests.json').read_bytes()),
    collected=0,executed=0,actual_execution_authorized=False,genuine_numpy_backend_forbidden=True,
    runtime_module_origins={},temporary_module_name_doubles=selection['temporary_module_name_doubles'],
    public_synthetic_only=True,original_body_reads=False,capacity_acquisition=False,rng=False,cpu_proposal=[112,113,114,115])
(O/'handoff.json').write_bytes(pretty(handoff))
print(json.dumps(dict(source_count=len(sources),static_count=len(nodes),per_file=per_file,map=handoff['source_manifest_sha256'],archive=handoff['source_archive_sha256'],handoff=sha((O/'handoff.json').read_bytes())),sort_keys=True))
