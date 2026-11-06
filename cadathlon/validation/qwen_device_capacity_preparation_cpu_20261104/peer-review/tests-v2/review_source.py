"""Source-only independent review; never import or execute candidate files."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

OUT=Path(__file__).resolve().parent
ROOT=Path('/tmp/qwen-device-component-tests-v2-20261104')
HAND=Path('/tmp/qwen-device-component-test-handoff-v2-20261104')
OLD=Path('/tmp/qwen-device-component-tests-20261104')
OLDHAND=Path('/tmp/qwen-device-component-test-handoff-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_bytes())
def tree(root):
    result={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),p
        if p.is_file():result[p.relative_to(root).as_posix()]=sha(p.read_bytes())
    return result
def archived(path):
    result={}
    with tarfile.open(path,'r:gz')as tar:
        for m in tar.getmembers():
            assert m.isfile()and m.name not in result and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts,m.name
            result[m.name]=sha(tar.extractfile(m).read())
    return result
handoff=read(HAND/'handoff.json');current=read(HAND/'sources.json')
assert sha((HAND/'handoff.json').read_bytes())=='3d5a78248daaf74b9039f3d4d1bd60381327728a18df9ff4e49167b6ccd25ed8'
assert sha((HAND/'sources.json').read_bytes())=='1725249e18cf6298eda8516726955efe879635c0132579b65543ad3717493e74'
assert sha((HAND/'sources.tar.gz').read_bytes())=='1645d2fa604ad76afb41b4fdc9ea4ab88c51bd665eed3ed287ca79147923e5aa'
assert len(current)==50 and current==tree(ROOT)==archived(HAND/'sources.tar.gz')==handoff['source_hashes']
assert Path(handoff['source_manifest']).read_bytes()==(HAND/'sources.json').read_bytes()
old=read(OLDHAND/'sources.json')
assert len(old)==49 and sha((OLDHAND/'sources.json').read_bytes())=='39ad4b3709425d86626473d38d7b30df84814019fc4794b2034c31daceb4715a'
assert sha((OLDHAND/'sources.tar.gz').read_bytes())=='1fad9a6017ed4cc141f0952841c8d59a7673bfaa882d298e9eec9f5a540f1348'
assert old==tree(OLD)==archived(OLDHAND/'sources.tar.gz')
changed=sorted(n for n in old if old[n]!=current[n]);added=sorted(set(current)-set(old))
assert changed==['TEST_SCOPE.md','device_child.py']and added==['successor_history.json']
runtime=read(ROOT/'runtime_sources.json')
assert len(runtime)==34 and runtime==tree(ROOT/'runtime')=={n[8:]:h for n,h in current.items()if n.startswith('runtime/')}
origins=read(ROOT/'copied_sources.json')
for name,row in origins.items():assert current[name]==row['sha256']==sha(Path(row['path']).read_bytes())
device=Path('/tmp/qwen-server-device-admission-source-v2-20261104')
device_map=read(Path('/tmp/qwen-server-device-admission-handoff-v2-20261104/sources.json'))
assert len(device_map)==6 and device_map==tree(device)
for name,digest in device_map.items():
    dest=('runtime/'if name.endswith('.py')else'device_metadata/')+name
    assert current[dest]==digest
case_tree=ast.parse((ROOT/'device_cases.py').read_bytes())
cases=ast.literal_eval(next(n.value for n in case_tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CASES'for t in n.targets)))
nodes=['test_device_component.py::test_device_component['+c+']'for c in cases]
assert len(nodes)==len(set(nodes))==30 and nodes==handoff['static_nodeids']==read(ROOT/'selected_tests.json')['static_nodeids']
test=ast.parse((ROOT/'test_device_component.py').read_bytes())
functions=[n for n in test.body if isinstance(n,ast.FunctionDef)]
assert [n.name for n in functions]==['test_device_component']
dec=functions[0].decorator_list[0]
assert ast.unparse(dec)=="pytest.mark.parametrize('scenario', CASES, ids=CASES)"
child=ast.parse((ROOT/'device_child.py').read_bytes())
retained=next(n for n in ast.walk(child)if isinstance(n,ast.FunctionDef)and n.name=='retained_failed')
retained_source=ast.get_source_segment((ROOT/'device_child.py').read_text(),retained)
for token in ('while restores:restores.pop()()',"state['mode']='benign'","original_permit=row['permit']",'capacity.initialize_once(original_permit)',"capacity.observe_capacity(original_permit,fixtures.STAGES[next_index])",'==before'):
    assert token in retained_source
for assignment in ast.walk(child):
    targets=assignment.targets if isinstance(assignment,ast.Assign)else([assignment.target]if isinstance(assignment,(ast.AnnAssign,ast.AugAssign))else[])
    for target in targets:
        assert not(isinstance(target,ast.Attribute)and isinstance(target.value,ast.Name)and target.value.id=='capacity'and target.attr in('_REGISTRY','_ISSUED'))
parent_text=(ROOT/'test_device_component.py').read_text()
for token in ('finally:\n                try:proc.wait()',"type(data['pgid'])is int and data['pgid']==proc.pid","child_result_read=False"):
    assert token in parent_text
assert parent_text.index("'parent-terminal.json'")<parent_text.index('assert code==0')<parent_text.index('data=json.loads(result.read_text())')
targets=['TEST_SCOPE.md','device_child.py','device_fixtures.py','device_cases.py','test_device_component.py','conftest.py','runtime/qwen_device_capacity.py','runtime/qwen_server_process.py','runtime/recording_jax.py','runtime/qwen_device_policy.py','runtime/qwen_server_resource.py']
receipt=dict(schema='qwen-device-focused-cnn-source-review-v1',passed=True,scope='source/hash/AST review only; no candidate imports, collection, tests or probes',
    source_manifest_sha256=sha((HAND/'sources.json').read_bytes()),source_archive_sha256=sha((HAND/'sources.tar.gz').read_bytes()),handoff_sha256=sha((HAND/'handoff.json').read_bytes()),
    source_files=50,runtime_files=34,device_files=6,parent_files=49,parent_unchanged=47,overlays=changed,additions=added,static_nodes=nodes,
    reviewed_target_hashes={n:current[n]for n in targets},all_source_current_archive_exact=True,parent49_current_archive_preserved=True,all_declared_copy_origins_exact=True,
    findings=[],conclusions=[
      'Two phase validation methods are explicitly recorded and restored; real device/resource validators remain selected. Original resource/logical code objects are counted by a scoped profiler.',
      'Document variants rebuild dependent pins and assert specific target messages; later semantic negatives assert real resource validation was reached.',
      'The temporary JAX-shaped fixture is source-owned and removed; installed backends, NumPy and device/proc/sys access are denied. This is not native/device-capacity evidence.',
      'Failed-state helper restores original policy identity, hook/descriptor/default-device mutations and benign statistics before retrying the original permit; hook, initialization and sample counts must remain unchanged. Reissuance remains separate and registry state is never reset.',
      'Post-initialization cases retry the next valid ordered stage; pre-initialization failures retry initialize_once. Early issuance failures cannot establish successful issuance, and no such claim is inferred.',
      'Parent owns Popen immediately, always attempts direct wait in the abnormal cleanup finally, records retirement before child-success assertions, and checks exact-int child PGID equals original child PID even if the early observation missed.'
    ],limitations=['Unexecuted prospective 30 cases; this review is not collection or runtime proof.','No genuine ModelAdmission/science/host/device/resource reservation, lease or actual server execution.','Test cleanup restoration paths are trusted synthetic seams, not arbitrary Python tamper containment.'],
    reviewer_history={'initial_source_check_exit':1,'tool_chunk':'9ca04b','cause':'Naive assignment substring matched equality assertions. Replaced with AST assignment-target check; no candidate source or test execution changed.','initial_script_sha256':sha((OUT/'review_source_initial.py').read_bytes())},
    imports_executed=False,tests_executed=False,capacity_probes=False,actual_authorization=False,review_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'review.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,source_files=50,static_cases=30,receipt_sha256=sha((OUT/'review.json').read_bytes())),sort_keys=True))
