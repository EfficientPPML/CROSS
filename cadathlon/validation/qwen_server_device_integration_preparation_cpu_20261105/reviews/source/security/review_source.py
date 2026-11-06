"""Stdlib-only review of held shared59 source, never candidate imports."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=Path('/tmp/qwen-full28-device-focused-tests-20261104')
HAND=Path('/tmp/qwen-full28-device-focused-test-handoff-20261104')
PARENT=Path('/tmp/qwen-server-device-integration-runtime-v4-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(path.read_bytes())
def pin(path):return dict(path=str(path),sha256=sha(path.read_bytes()))
def canon(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def tree(name):return ast.parse((ROOT/name).read_bytes(),filename=name)
def function(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)

h=load(HAND/'handoff.json');mapping=load(Path(h['source_map']))
assert h['source_count']==len(mapping)==458
assert canon(mapping)==Path(h['source_map']).read_bytes()
assert sha(canon(mapping))==h['source_manifest_sha256']=='6a86b7351e1ad7d0ea43ce0b84035661e68c8bac18ae30e8d1eed3bf434f3517'
assert sha(Path(h['source_archive']).read_bytes())==h['source_archive_sha256']=='3a88579485a582dcaefbf4ded73c4c29fe33bed05aeaf8b3cebd7cf1cb68a86e'
assert {p.relative_to(ROOT).as_posix()for p in ROOT.rglob('*')if p.is_file()}==set(mapping)
assert not any(p.is_symlink()for p in ROOT.rglob('*'))
blobs={n:(ROOT/n).read_bytes()for n in mapping}
assert {n:sha(b)for n,b in blobs.items()}==mapping
with tarfile.open(h['source_archive'],'r:gz')as archive:
    members=archive.getmembers()
    assert len(members)==458 and all(m.isfile()for m in members)
    assert {m.name for m in members}==set(mapping)
    for m in members:assert archive.extractfile(m).read()==blobs[m.name]
parent=load(ROOT/'runtime_sources.json')
assert len(parent)==403 and canon(parent)==blobs['runtime_sources.json']
assert sha(canon(parent))==h['parent_map_sha256']
assert {n[8:]:digest for n,digest in mapping.items()if n.startswith('runtime/')}==parent
assert all(sha((PARENT/n).read_bytes())==digest for n,digest in parent.items())

selection=load(ROOT/'selected_tests.json');plan=load(ROOT/'accepted_plan/nodes.json')
cases=selection['cases'];nodes=selection['nodes'];groups=load(ROOT/'case_groups.json')
assert len(cases)==len(set(cases))==len(nodes)==59
assert nodes==['test_device_integration.py::test_device_integration['+case+']'for case in cases]==h['static_nodes']
assert [row['case']for row in plan['nodes']]==cases
assert groups=={row['case']:row['group']for row in plan['nodes']}
assert collections.Counter(groups.values())==dict(P=7,F=10,O=18,T=1,L=4,E=4,A=6,R=7,B=2)
parent_ast=tree('test_device_integration.py')
tests=[n for n in parent_ast.body if isinstance(n,ast.FunctionDef)and n.name.startswith('test_')]
assert len(tests)==1 and tests[0].name=='test_device_integration'
assert ast.unparse(tests[0].decorator_list[0])=="pytest.mark.parametrize('case', CASES, ids=CASES)"
with_log=next(n for n in tests[0].body if isinstance(n,ast.With))
assert isinstance(with_log.body[0],ast.Assign)and isinstance(with_log.body[0].value,ast.Call)
assert ast.unparse(with_log.body[0].value.func)=='subprocess.Popen'
assert isinstance(with_log.body[1],ast.Try)
owned_try=with_log.body[1]
assert any(isinstance(n,ast.Call)and ast.unparse(n.func)=='child.wait'for n in ast.walk(owned_try))
exception_handler=owned_try.handlers[0]
cleanup_try=next(n for n in exception_handler.body if isinstance(n,ast.Try))
assert any(isinstance(n,ast.Call)and ast.unparse(n.func)=='child.wait'for top in cleanup_try.finalbody for n in ast.walk(top))
assert not any(k.arg=='timeout'for n in ast.walk(tests[0])if isinstance(n,ast.Call)for k in n.keywords)
assert owned_try.finalbody

artifacts=load(ROOT/'ARTIFACT_SELECTION.json')
assert artifacts['children']==59 and artifacts['expected_regular_record_count']==236
assert set(artifacts['fixture_files'])==set(cases)
assert sum(map(len,artifacts['fixture_files'].values()))==artifacts['fixture_regular_count']==14
assert artifacts['records_per_child']==['parent-terminal.json','child-terminal.json','result.json','child.log']
assert artifacts['per_record_limit_bytes']==2<<20
read=ast.unparse(function(parent_ast,'_read_record'))
for token in ('os.O_NOFOLLOW','os.O_NONBLOCK','os.fstat','path.lstat','os.close'):assert token in read
inspect=ast.unparse(function(parent_ast,'_fixtures'))
assert 'follow_symlinks=False'in inspect and 'stat.S_ISLNK'in inspect
fixture_fn=function(parent_ast,'_fixtures')
fixture_walk=function(fixture_fn,'walk')
assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('open','read_bytes','read_text')for n in ast.walk(fixture_walk))
selection_reads=[n for n in ast.walk(fixture_fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('open','read_bytes','read_text')]
assert len(selection_reads)==1 and ast.unparse(selection_reads[0])=="(HERE / 'ARTIFACT_SELECTION.json').read_bytes()"

provenance=load(ROOT/'entry_fixture_provenance.json');raw=blobs['runtime/qwen_server_entry.py'];entry=ast.parse(raw)
dump=lambda node:ast.dump(node,include_attributes=False)
terminal=ast.parse("if __name__=='__main__':\n    raise SystemExit(main())\n").body[0]
assert [i for i,n in enumerate(entry.body)if dump(n)==dump(terminal)]==[len(entry.body)-1]
assert sha(raw)==provenance['entry_sha256']
assert sha(dump(entry.body[-1]).encode())==provenance['excluded_dispatch_ast_sha256']
assert sha(dump(ast.Module(body=entry.body[:-1],type_ignores=entry.type_ignores)).encode())==provenance['definition_ast_sha256']
helpers=load(ROOT/'helper_selection.json');assert len(helpers)==52
assert all(relative in mapping for relative in helpers.values())
assert not any(relative.startswith(('historical/','runtime/runtime/','runtime/common/','runtime/recording_sources/'))for relative in helpers.values())
numeric=[]
for name,relative in helpers.items():
    tr=ast.parse(blobs[relative])
    roots=[]
    for n in tr.body:
        if isinstance(n,ast.Import):roots += [x.name.split('.')[0]for x in n.names]
        elif isinstance(n,ast.ImportFrom)and n.module:roots.append(n.module.split('.')[0])
    if 'numpy'in roots:numeric.append(name)
    assert not set(roots)&{'jax','jaxlib','jaxite_word','libtpu','jax_plugins'}
assert set(numeric)=={'qwen_file_lifecycle','qwen_key_readers','qwen_layout','qwen_public_parameters','qwen_serialization','reused_rns_payload'}
assert mapping['cases_loader_engine.py']=='722e882c25df34d6e6f4404822073eacaf5619c97733a697cac7bb06d04bfa2c'

subjects=('source_loader.py','fixture_base.py','fixture_documents.py','recording_jax.py','owned_child.py',
          'test_device_integration.py','conftest.py','cases_loader_engine.py','cases_device.py','cases_audit.py',
          'SEAMS.json','TEST_SCOPE.md','helper_selection.json','ARTIFACT_SELECTION.json','entry_fixture_provenance.json')
receipt={
 'schema':'full28-device458-shared59-security-source-review-v1',
 'result':'no_additional_concrete_blocker_in_reviewed_source_scope',
 'scope':'Shared fixture/import selection, descriptor restoration, owned child/parent source, bounded public receipts, and L/E8 integration compatibility. Not an outer-driver review or a runtime result.',
 'handoff':pin(HAND/'handoff.json'),'source_map':pin(Path(h['source_map'])),'archive':pin(Path(h['source_archive'])),
 'source_count':458,'parent_count':403,'static_cases':59,'static_group_counts':dict(collections.Counter(groups.values())),
 'reviewed_subjects':{n:mapping[n]for n in subjects},
 'checks':{
  'current_map_archive_exact':True,'nested_parent403_exact':True,'pre_fixture_order_59_exact':True,
  'entry_only_terminal_dispatch_ast_excluded':True,'entry_not_genuine_cli_bootstrap':True,
  'helper_selection52_source_pinned':True,'numerical_core_common_or_historical_implementation_unselected':True,
  'temporary_jax_source_origin_honest':True,'temporary_numpy_empty_definition_name':True,
  'class_descriptors_and_inherited_class_shadows_restored':True,
  'profile_and_temporary_names_restored_before_normal_receipt':True,
  'parent_immediate_popen_try_and_cleanup_finally_wait':True,'healthy_wait_has_no_timeout':True,
  'pid_pgid_starttick_joins_and_early_parent_receipt':True,'bounded_regular_nofollow_record_readback':True,
  'fixture14_inspection_lstat_only':True,'owned_per_child_aliases_rejected':True,
  'loader_engine_original8_module_unchanged':True,
  'shared_graph_has_required_synthetic_key_parent_root_manifest_and_request_digest':True,
  'plain_function_owner_and_runtime_delegates_before_capture':True,
 },
 'observations':[
  'Exact selected source definitions execute only in a future authorized isolated child. The test finder is explicitly separate from the genuine FrozenImports issuer.',
  'Six selected modules have definition-time NumPy imports; L/E installs only an empty owned NumPy name before those definitions. The source-owned recording_jax module has its actual file/spec, and all temporary numerical names must be removed before normal return.',
  'Two phase checks plus the whole scientific packet/readiness ID methods remain explicit recordings. Device/resource validators, snapshot/runtime controls, owner and selected loader/stage/engine/Finalizer bodies remain real source mechanisms.',
  'The shared key-store parent/root/manifest and request_manifest_sha256 fields now satisfy the declared L/E descriptor interface. Fake-FD/stat operations remain narrow synthetic recordings; no actual keys, model or ciphertext bodies are selected.',
  'Parent owns Popen before any observation or receipt write, waits normally without a deadline, and executes direct wait even if abnormal group cleanup fails. PID/PGID/starttime observations are process metadata, not capacity probes.',
  'All236 child record files are bounded public records. The14 tiny fixture files are inspected by metadata only; writer argument hashes are not misrepresented as inspector body readback.',
  'Failed/partial owner cleanup is conservative and does not claim successful backend retirement. A normal child certificate follows fixture restoration and final source/finder checks; failures cannot manufacture that certificate.',
 ],
 'limitations':[
  'Source-only review cannot establish that59 cases will pass or that all prospective branches are reached.',
  'This reviewer authored L/E8 earlier; this review covers its integration with independently authored held shared fixture/parent, and does not replace the separate CNN L/E source review.',
  'No candidate import, collection, tests, native/backend/device/capacity probe, application/private/model/key/ciphertext body read, grant or queue change.',
  'No genuine scientific/model/key authority, HE numerical result, native capacity or performance claim.',
  'The separately authored outer driver/worker is outside this receipt.'
 ],
 'review_driver':pin(Path(__file__)),
 'reviewer_script_history':pin(HERE/'history/assertion-correction.json'),
 'held_sources_unchanged_at_end':all(sha((ROOT/n).read_bytes())==digest for n,digest in mapping.items()),
}
(HERE/'review.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(review=str(HERE/'review.json'),sha256=sha((HERE/'review.json').read_bytes()),sources=458,parent=403,static_cases=59,candidate_imports=False,tests=False),sort_keys=True))
