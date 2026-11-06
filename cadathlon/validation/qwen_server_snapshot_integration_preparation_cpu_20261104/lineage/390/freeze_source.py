"""Close a source-only server snapshot successor, without candidate execution."""
import ast,difflib,gzip,hashlib,io,json,pathlib,stat,tarfile
H=pathlib.Path(__file__).resolve().parent
P=pathlib.Path('/tmp/qwen-server-actual-runtime-20261104')
N=pathlib.Path('/tmp/qwen-server-snapshot-runtime-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(d):return(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n').encode()
def read(p):
    a=p.lstat();assert stat.S_ISREG(a.st_mode)and not p.is_symlink();b=p.read_bytes();z=p.lstat()
    assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
    return b
def inventory(root):
    raw={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink()
        if p.is_file():raw[p.relative_to(root).as_posix()]=read(p)
        else:assert p.is_dir()
    return raw
old=json.loads(read(H/'parent387-sources.json'));parent=inventory(P)
assert len(parent)==387 and{n:sha(b)for n,b in parent.items()}==old
with tarfile.open(H/'parent387-sources.tar.gz','r:gz')as t:
    members=t.getmembers();assert len(members)==387
    for m in members:
        assert m.isfile()and m.name in old;f=t.extractfile(m)
        with f:assert sha(f.read())==old[m.name]
code=['qwen_server_process.py','qwen_server_backend.py','qwen_server_entry.py']
raw=inventory(N)
assert {n for n in old if sha(raw[n])!=old[n]}==set(code)
guard=read(pathlib.Path('/tmp/qwen-server-admission-snapshot-source-v2-20261104/qwen_admission_snapshot.py'))
assert raw['qwen_admission_snapshot.py']==guard and sha(guard)=='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
trees={n:ast.parse(raw[n])for n in code+['qwen_admission_snapshot.py']}
process=trees['qwen_server_process.py']
assignments={n.targets[0].id:ast.literal_eval(n.value)for n in process.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('SNAPSHOT_BINDING_SOURCES','EXPECTED_ID_FIELDS')}
closure=assignments['SNAPSHOT_BINDING_SOURCES'];assert type(closure)is tuple and len(closure)==len(set(closure))==20
assert len(assignments['EXPECTED_ID_FIELDS'])==7
cb={n:sha(raw[n])for n in closure};assert set(cb)<=set(raw)
def top(raw):return{n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
dump=lambda n:ast.dump(n,include_attributes=False)
ast_changes={}
for n in code:
    a,b=top(parent[n]),top(raw[n]);ast_changes[n]={'changed':[k for k in a if dump(a[k])!=dump(b[k])],'added':sorted(set(b)-set(a))}
assert set(ast_changes[code[0]]['changed'])=={'RuntimeAdmission','_runtime','run'}
assert set(ast_changes[code[1]]['changed'])=={'_runtime','ActualServerDependencies','ServerBackend'}
assert ast_changes[code[2]]=={'changed':[],'added':[]}
entry_old=parent[code[2]].decode();entry_new=raw[code[2]].decode()
assert entry_new.replace("ROOT_MODULES=frozenset(('qwen_admission_snapshot', 'qwen_server_entry',","ROOT_MODULES=frozenset(('qwen_server_entry',",1)==entry_old
for n in ['qwen_server_resource.py','qwen_server_entry_status.json','execution_authorization.json','qwen_server_dependencies.py','qwen_key_loader.py','qwen_key_readers.py','qwen_finalizer.py','cleanup_origin.py']:
    assert raw[n]==parent[n]
assert json.loads(raw['qwen_server_entry_status.json'])['actual_entry_ready']is False
assert json.loads(raw['execution_authorization.json'])['actual_execution_enabled']is False
functions=top(raw[code[0]])
def dotted(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return dotted(node.value)+'.'+node.attr
    return ''
capture=functions['_capture_snapshot'];callback=next(n for n in capture.body if isinstance(n,ast.FunctionDef)and n.name=='check_bindings')
callback_calls=sorted(set(dotted(n.func)for n in ast.walk(callback)if isinstance(n,ast.Call)))
assert not any(n in callback_calls for n in['packet.check','phase.check','guard.check','runtime.check','capacity_check','source_check'])
runtime_calls=sorted(set(dotted(n.func)for n in ast.walk(functions['_runtime'])if isinstance(n,ast.Call)))
assert 'packet.check'not in runtime_calls and 'phase.check'not in runtime_calls
backend=top(raw[code[1]])
assert not any(isinstance(n,ast.Call)and dotted(n.func)in('phase.check','self.phase.check','phase.expected_ids')for n in ast.walk(trees[code[1]]))
assert sum(isinstance(n,ast.Call)and dotted(n.func)=='phase.expected_ids'for n in ast.walk(process))==1
assert sum(isinstance(n,ast.Attribute)and dotted(n)=='phase.input_digest'for n in ast.walk(process))==1
plan=pathlib.Path('/tmp/qwen-server-snapshot-integration-plan-20261104/PLAN.md')
peer=pathlib.Path('/tmp/qwen-server-snapshot-integration-security-review-20261104/review.json')
provenance={'schema':'qwen-server-snapshot-integration-source-v1','status':'held_unexecuted_source_successor',
 'parent_root':str(P),'parent_source_count':387,'parent_manifest_sha256':sha(read(H/'parent387-sources.json')),'parent_archive_sha256':sha(read(H/'parent387-sources.tar.gz')),
 'accepted_plan_sha256':sha(read(plan)),'plan_peer_review_sha256':sha(read(peer)),
 'code_overlays':{n:{'before':old[n],'after':sha(raw[n])}for n in code},
 'literal_guard_sha256':sha(guard),'additions':['qwen_admission_snapshot.py','SNAPSHOT_INTEGRATION.md','snapshot_integration.json'],
 'unchanged_parent_files':384,'callback_source_closure':cb,'expected_id_fields':list(assignments['EXPECTED_ID_FIELDS']),
 'actual_readiness_changed':False,'complete_capacity_module_unchanged':True,'core_math_loader_reader_policy_unchanged':True,
 'runtime_control_registry':'issuer-owned live runtime/control references; no global proof hash/result cache; failed owners retained',
 'backend_module_capture':'pure declarations imported before snapshot capture; no backend/context construction',
 'test_scope':'Future explicit synthetic integration proof required; no inherited18 integration PASS',
 'candidate_imports':False,'collection_or_tests':False,'device_or_capacity_probes':False,'body_reads':False,'actual_grants':False}
(N/'snapshot_integration.json').write_bytes(canon(provenance))
raw=inventory(N);mapping={n:sha(b)for n,b in raw.items()};assert len(mapping)==390
assert set(mapping)-set(old)==set(provenance['additions'])
(H/'sources.json').write_bytes(canon(mapping));(H/'callback-source-closure.json').write_bytes(canon(cb))
patch=''.join(''.join(difflib.unified_diff(parent[n].decode().splitlines(True),raw[n].decode().splitlines(True),fromfile='parent387/'+n,tofile='successor390/'+n))for n in code)
(H/'source.patch').write_text(patch)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT)as t:
    for n,b in raw.items():
        m=tarfile.TarInfo(n);m.size=len(b);m.mode=0o644;m.mtime=0;m.uid=m.gid=0;m.uname=m.gname='';t.addfile(m,io.BytesIO(b))
(H/'sources.tar.gz').write_bytes(gzip.compress(buf.getvalue(),mtime=0))
assert {n:sha(b)for n,b in inventory(P).items()}==old
assert {n:sha(b)for n,b in inventory(N).items()}==mapping
review={'schema':'qwen-server-snapshot-integration-static-source-review-v1','passed':True,'candidate_imports':False,'tests':False,'source_count':390,'parent387_exact_current_archive':True,'unchanged384':True,'ast_changes':ast_changes,'callback20':cb,'callback_direct_calls':callback_calls,'runtime_calls':runtime_calls,'literal_guard_exact':True,'entry_only_declared_module_name':True,'capacity_and_readiness_unchanged':True,'script_sha256':sha(read(pathlib.Path(__file__)))}
(H/'static-review.json').write_bytes(canon(review))
handoff={'schema':'qwen-server-snapshot-integration-held-handoff-v1','root':str(N),'source_count':390,'source_map_sha256':sha(read(H/'sources.json')),'source_archive_sha256':sha(read(H/'sources.tar.gz')),'source_archive_bytes':(H/'sources.tar.gz').stat().st_size,'parent_map_sha256':provenance['parent_manifest_sha256'],'parent_archive_sha256':provenance['parent_archive_sha256'],'source_patch_sha256':sha(read(H/'source.patch')),'callback_closure_sha256':sha(read(H/'callback-source-closure.json')),'static_review_sha256':sha(read(H/'static-review.json')),'code_hashes':{n:mapping[n]for n in code+['qwen_admission_snapshot.py']},'unchanged_parent_files':384,'code_overlays':3,'additions':3,'source_only':True,'candidate_imports':False,'collection_or_tests':False,'device_or_body_probes':False,'actual_authority':False,'capacity_denial_preserved':True,'readiness_false':True}
(H/'handoff.json').write_bytes(canon(handoff));print(json.dumps({'passed':True,'handoff_sha256':sha(read(H/'handoff.json')),**handoff},sort_keys=True))
