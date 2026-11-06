"""Source-text-only correction; never import or execute candidate tests."""
from pathlib import Path
import ast,difflib,hashlib,json,os,stat

OLD=Path('/tmp/qwen-full28-device-focused-tests-v2-20261105/cases_loader_engine.py')
ROOT=Path('/tmp/qwen-full28-device-loader-engine-correction-20261105')
OUT=Path('/tmp/qwen-full28-device-loader-engine-correction-handoff-20261105')
source=OLD.read_text();sha=lambda b:hashlib.sha256(b).hexdigest()
old_sha=sha(source.encode())
assert old_sha=='722e882c25df34d6e6f4404822073eacaf5619c97733a697cac7bb06d04bfa2c'
anchor="    p=env.p;env.catalog=object()\n"
addition="""    # Actual ServerBackend.parameter_files evaluates this property before the
    # recorded constructor. The shared model proof is deliberately unadmitted;
    # retain only an explicit metadata marker, never authenticate model bodies.
    original_stores=NS(raw=fc.canonical({'recording_model_stores':True}))
    def model_stores(value):
        assert value is f.model;return original_stores
    f.patch(f.model_module.ModelAdmission,'stores',property(model_stores),
        'L/E model-store metadata property recording before snapshot; no model admission or body reads')
    env.parameter_owners=[]
"""
assert source.count(anchor)==1
fixed=source.replace(anchor,anchor+addition)
old_constructor="""    def parameter_files(*args,**kwargs):
        params=RecordingFiles(env,'parameters');params.catalog=env.catalog;env.parameters=params;return params
"""
new_constructor="""    def parameter_files(*args,**kwargs):
        if env.recording:assert not args and not kwargs
        else:assert len(args)==2 and args[0]is f.public and args[1]is original_stores and not kwargs
        params=RecordingFiles(env,'parameters');env.parameter_owners.append(params)
        params.catalog=env.catalog;env.parameters=params;return params
"""
assert fixed.count(old_constructor)==1;fixed=fixed.replace(old_constructor,new_constructor)
old_assert="    assert env.upload.closed and env.parameters.closed and env.publisher.closed\n"
new_assert="    parameter_owner,=env.parameter_owners\n    assert env.parameters is parameter_owner\n"+old_assert
assert fixed.count(old_assert)==1;fixed=fixed.replace(old_assert,new_assert)
original_ast=ast.parse(source);fixed_ast=ast.parse(fixed)
def shape(node):return ast.dump(node,include_attributes=False)
old_assertions=[shape(n)for n in ast.walk(original_ast)if isinstance(n,ast.Assert)]
new_assertions=[shape(n)for n in ast.walk(fixed_ast)if isinstance(n,ast.Assert)]
assert all(new_assertions.count(n)>=old_assertions.count(n)for n in old_assertions)
old_nodes={n.name:shape(n)for n in original_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new_nodes={n.name:shape(n)for n in fixed_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
changed=[n for n in old_nodes if old_nodes[n]!=new_nodes[n]]
assert changed==['_setup','_engine_case']
def cases(tree):return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CASES'for t in n.targets)))
assert cases(original_ast)==cases(fixed_ast)and len(cases(fixed_ast))==8
assert not (ROOT/'cases_loader_engine.py').exists()
(ROOT/'cases_loader_engine.py').write_text(fixed)
(OUT/'source.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),fixed.splitlines(True),fromfile='held459/cases_loader_engine.py',tofile='corrected/cases_loader_engine.py')))
def pin(path):
    raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=sha(raw))
def dump(name,value):(OUT/name).write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')

# Read only the three explicitly admitted public record sets. No aliases,
# recursive temp traversal or tiny fixture bodies are touched.
proof=Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v3/pytest-owned')
root_receipt=Path('/tmp/qwen-full28-device-focused-root-harness-review-20261105/failed459-terminal-review.json')
receipt_raw=root_receipt.read_bytes();assert sha(receipt_raw)=='93657ed40c0935dc11bf7b0fc3f0f6a8e30ed247d408c78b949cf92371d8dd7b'
receipt=json.loads(receipt_raw);records=[]
for index,case in enumerate(cases(fixed_ast)[4:7]):
    directory=proof/('test_device_integration_engine'+str(index))
    assert stat.S_ISDIR(directory.lstat().st_mode)
    for name in ('child.log','child-terminal.json','parent-terminal.json'):
        path=directory/name;named=path.lstat()
        assert stat.S_ISREG(named.st_mode)and named.st_size<=16384
        fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            held=os.fstat(fd);assert (held.st_dev,held.st_ino,held.st_size)==(named.st_dev,named.st_ino,named.st_size)
            raw=os.read(fd,16385);assert len(raw)==held.st_size
        finally:os.close(fd)
        expected=receipt['children'][case]['records'][name]
        assert sha(raw)==expected['sha256']and len(raw)==expected['bytes']
        if name=='child.log':assert b"AttributeError: 'NoneType' object has no attribute 'closed'"in raw
        else:assert json.loads(raw)['case']==case
        records.append(dict(case=case,**pin(path)))
runtime=OLD.parent/'runtime'
refs={n:pin(runtime/n)for n in ('qwen_server_backend.py','qwen_stage_composition.py','qwen_backend_composition.py','model_admission.py')}
refs['fixture_base.py']=pin(OLD.parent/'fixture_base.py')
dump('source_refs.json',refs)
dump('sources.json',{'cases_loader_engine.py':sha(fixed.encode())})
dump('review.json',dict(schema='full28-loader-engine-fixture-correction-v1',
 result='SOURCE_ONLY_CORRECTION_NOT_EXECUTED',old_source=pin(OLD),new_source=pin(ROOT/'cases_loader_engine.py'),
 root_failed_terminal=pin(root_receipt),bounded_public_records=records,
 diagnosis='All three actual-factory E cases stop at the post-cleanup env.parameters.closed assertion, but source shows env.parameters is initialized None and assigned only in recorded parameter_files. Actual ServerBackend first evaluates phase.model.stores; its real property invokes model authentication on the shared unadmitted marker bytes before that recording constructor. RecordingFiles.close does not clear env.parameters. Thus the observed AttributeError masks an earlier unrecorded admission-boundary failure; it is not evidence of runtime owner loss during close.',
 proof_limit='Original swallowed error is not in the three child logs; the earlier model property failure is source-derived, not an independently observed traceback. Original run remains43PASS16FAIL and is not retrospectively repaired.',
 correction='Pre-snapshot explicit ModelAdmission.stores metadata marker property; unchanged real backend factory passes that exact marker to recorded parameter constructor. Original parameter object is retained only after constructor completion and checked by identity after stage return. No owner is manufactured before constructor routing.',
 assertions=dict(original_count=len(old_assertions),corrected_count=len(new_assertions),all_original_ast_assertions_preserved=True,cases=cases(fixed_ast),changed_top_level_definitions=changed),
 runtime403_changed=False,shared_fixture_changed=False,held459_changed=False,
 candidate_imports=False,collection=False,tests=False,probes=False,fixture_body_reads=False,
 aliases_followed=False,owner_wait_or_live_observation_inherited=False,actual_model_admission=False))
dump('handoff.json',dict(schema='full28-loader-engine-source-correction-handoff-v1',source_root=str(ROOT),source_count=1,
 source=pin(ROOT/'cases_loader_engine.py'),source_map=pin(OUT/'sources.json'),patch=pin(OUT/'source.patch'),review=pin(OUT/'review.json'),
 static_cases=8,original_assertions_preserved=True,runtime403_unchanged=True,held459_preserved=True,tests=False,imports=False,
 scope='Separate L/E source patch for root-owned complete successor integration; no new complete458/459 map or proof'))
dump('files.json',{p.name:sha(p.read_bytes())for p in sorted(OUT.iterdir())if p.is_file()and p.name!='files.json'})
assert OLD.read_text()==source
print(json.dumps(dict(source=pin(ROOT/'cases_loader_engine.py'),handoff=pin(OUT/'handoff.json'),review=pin(OUT/'review.json'),index=pin(OUT/'files.json'))))
