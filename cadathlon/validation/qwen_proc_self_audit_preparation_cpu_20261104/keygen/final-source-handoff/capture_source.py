"""Stdlib static capture only; imports no candidate/test module."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/tmp/qwen-keygen-proc-self-tests-v2-20261104')
OUT=Path('/tmp/qwen-keygen-proc-self-test-handoff-v2-20261104')
PARENT=Path('/tmp/qwen-keygen-actual-runtime-v3-20261104')
PARENT_MAP=Path('/tmp/qwen-keygen-actual-runtime-v3-handoff-20261104/sources.json')
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda x:(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def mapping(root):
    result={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink()
        if p.is_file():result[p.relative_to(root).as_posix()]=H(p.read_bytes())
    return result

parent_raw=PARENT_MAP.read_bytes();parent=json.loads(parent_raw)
assert H(parent_raw)=='01e5f8681ff5731d8d19d8f488b692bcf38025b83a0dbb3036b536a3d9f1726e'
assert parent_raw==C(parent) and len(parent)==85
assert mapping(PARENT)==parent==mapping(ROOT/'runtime')
assert (ROOT/'runtime_sources.json').read_bytes()==parent_raw

test='test_proc_self_integration.py'
tree=ast.parse((ROOT/test).read_bytes(),filename=test)
funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name.startswith('test_')]
assert len(funcs)==1
f=funcs[0];assert f.name=='test_real_proc_self_audit_boundary'
assert len(f.decorator_list)==1
d=f.decorator_list[0]
assert isinstance(d,ast.Call)and ast.literal_eval(d.args[0])=='case'
cases=ast.literal_eval(d.args[1])
assert len(cases)==32 and len(set(cases))==32
nodeids=[test+'::'+f.name+'['+case+']'for case in cases]
xml_names=[f.name+'['+case+']'for case in cases]
selection={'schema':'qwen-proc-self-static-selection-v1','selected_test_files':[test],
    'ordered_nodeids':nodeids,'ordered_xml_names':xml_names,'expected_tests':32,
    'collection_performed':False,'execution_performed':False}
(ROOT/'test_selection.json').write_bytes(C(selection))
origin={'schema':'qwen-proc-self-test-source-origin-v1','parent_source_root':str(PARENT),
    'parent_source_manifest_sha256':H(parent_raw),'parent_files_unchanged':85,
    'source_only':True,'candidate_imports':False,'collection':False,'tests':False,
    'proc_metadata_reads':False,'capacity_checks':False,'actual_roles_authorized':False,
    'scope':'32 dedicated-child real audit/wrapper cases; own proc metadata only at later authorized execution',
    'prior61_does_not_cover_this_real_integration':True,
    'source_only_predecessor':'/tmp/qwen-keygen-proc-self-tests-20261104',
    'correction':'Negative operations only open/close, with no content read even on unexpected admission'}
(ROOT/'test_origin.json').write_bytes(C(origin))

observed=mapping(ROOT)
assert len(observed)==92
for name,digest in parent.items():assert observed['runtime/'+name]==digest
for name in('proc_self_probe.py','test_proc_self_integration.py','conftest.py'):
    ast.parse((ROOT/name).read_bytes(),filename=name)
probe_tree=ast.parse((ROOT/'proc_self_probe.py').read_bytes())
imports=[n for n in ast.walk(probe_tree)if isinstance(n,(ast.Import,ast.ImportFrom))]
assert any(isinstance(n,ast.Import)and any(a.name=='keygen_dependencies'for a in n.names)for n in imports)
assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='capacity'for n in ast.walk(probe_tree))
static={'schema':'qwen-proc-self-static-review-v1','passed':True,
    'parent85_exact':True,'source_count':len(observed),'static_cases':32,
    'new_python_sources_ast_parsed':3,'candidate_imported':False,'tests_collected_or_run':False,
    'proc_metadata_read':False,'capacity_probe':False,
    'new_files':sorted(set(observed)-{'runtime/'+x for x in parent}),
    'limitations':['Source-only verification; tests remain unexecuted.',
        'Direct builtin audit attests event string, not original high-level argument type.',
        'No resource sufficiency, backend, generation, model or private-body authority.']}
raw=C(observed);(OUT/'sources.json').write_bytes(raw)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as t:
    for name in observed:t.add(ROOT/name,arcname=name,recursive=False)
with tarfile.open(OUT/'sources.tar.gz','r:gz')as t:
    members=t.getmembers()
    assert [m.name for m in members]==list(observed)and all(m.isfile()for m in members)
    assert {m.name:H(t.extractfile(m).read())for m in members}==observed
assert mapping(ROOT)==observed and mapping(PARENT)==parent
old=Path('/tmp/qwen-keygen-proc-self-tests-20261104')
oldmap=json.loads(Path('/tmp/qwen-keygen-proc-self-test-handoff-20261104/sources.json').read_bytes())
assert mapping(old)==oldmap
assert set(oldmap)==set(observed)
changed=[n for n in observed if observed[n]!=oldmap[n]]
assert changed==['SCOPE.md','proc_self_probe.py','test_origin.json']
assert (old/'test_proc_self_integration.py').read_bytes()==(ROOT/'test_proc_self_integration.py').read_bytes()
assert (old/'test_selection.json').read_bytes()==(ROOT/'test_selection.json').read_bytes()
for n in ast.walk(probe_tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='rejected':
        assert len(n.args)==1 and isinstance(n.args[0],ast.Lambda)
        target=n.args[0].body
        assert isinstance(target,ast.Call) and isinstance(target.func,ast.Name)
        assert target.func.id in ('open_fd_only','open_builtin_only')
for fname in ('open_fd_only','open_builtin_only'):
    fnode=next(n for n in probe_tree.body if isinstance(n,ast.FunctionDef)and n.name==fname)
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ('read','readinto','read1')for n in ast.walk(fnode))
static['source_only_predecessor_changes']=changed
static['negative_operations_never_read']=True
(OUT/'static-review.json').write_bytes(C(static))
handoff={'schema':'qwen-proc-self-focused-test-handoff-v1','source_root':str(ROOT),
    'source_count':len(observed),'source_manifest_sha256':H(raw),
    'source_archive_sha256':H((OUT/'sources.tar.gz').read_bytes()),
    'parent_source_manifest_sha256':H(parent_raw),'parent_unchanged_files':85,
    'dependency_module_sha256':parent['keygen_dependencies.py'],
    'selected_test_files':[test],'ordered_nodeids':nodeids,'ordered_xml_names':xml_names,
    'expected_tests':32,'planned_cores':[112,113,114,115],
    'backend_numpy_forbidden':True,'installed_origins':{},'source_only':True,
    'imports_collection_execution_performed':False,'actual_execution_authorized':False,
    'static_review_sha256':H((OUT/'static-review.json').read_bytes())}
(OUT/'handoff.json').write_bytes(C(handoff))
print(json.dumps({'source_count':len(observed),'static_cases':32,
    'map':H(raw),'tar':handoff['source_archive_sha256'],
    'handoff':H((OUT/'handoff.json').read_bytes()),'source_only':True},sort_keys=True))
