import ast,hashlib,json,pathlib,tarfile
O=pathlib.Path(__file__).parent;R=pathlib.Path('/tmp/qwen-server-snapshot-runtime-v2-20261104');D=pathlib.Path('/tmp/qwen-server-snapshot-runtime-handoff-v2-20261104');B=pathlib.Path('/tmp/qwen-server-snapshot-runtime-handoff-20261104')
H=lambda x:hashlib.sha256(x).hexdigest()
hraw=(D/'handoff.json').read_bytes();h=json.loads(hraw);assert H(hraw)=='9e76bc518782c3e7ca8d8d88d9cd7e02d09312d29082e8cfae42e7e52bb9f607'
raw=(D/'sources.json').read_bytes();m=json.loads(raw);old=json.loads((B/'sources.json').read_bytes())
assert H(raw)==h['source_map_sha256'] and len(m)==391 and len(old)==390
assert not any(x.is_symlink()for x in R.rglob('*'))
assert {x.relative_to(R).as_posix():H(x.read_bytes())for x in R.rglob('*')if x.is_file()}==m
assert H((D/'sources.tar.gz').read_bytes())==h['source_archive_sha256']
with tarfile.open(D/'sources.tar.gz')as t:
    members=t.getmembers();assert len(members)==391 and all(x.isfile()for x in members)
    assert {x.name:H(t.extractfile(x).read())for x in members}==m
changed=sorted(k for k in old if old[k]!=m[k]);assert changed==h['changed_from390']
assert sorted(set(m)-set(old))==['snapshot_dispatch_successor.json']and sum(m[k]==v for k,v in old.items())==386
for n in('qwen_admission_snapshot.py','qwen_server_entry.py','qwen_server_resource.py','qwen_server_entry_status.json','execution_authorization.json','cleanup_origin.py'):assert m[n]==old[n]
p=ast.parse((R/'qwen_server_process.py').read_text());b=ast.parse((R/'qwen_server_backend.py').read_text())
closure=ast.literal_eval(next(n.value for n in p.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SNAPSHOT_BINDING_SOURCES'for t in n.targets)))
assert len(closure)==21 and 'cleanup_origin.py'in closure
cr=(D/'callback-source-closure.json').read_bytes();assert H(cr)==h['callback_closure_sha256']and json.loads(cr)=={n:m[n]for n in closure}
def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
pf=functions(p);bf=functions(b)
def kw(fn,name,target):
    rows=dict(zip((a.arg for a in fn.args.kwonlyargs),fn.args.kw_defaults));assert isinstance(rows[name],ast.Name)and rows[name].id==target
kw(pf['require_runtime'],'_dispatch','_runtime_action');kw(pf['_runtime'],'_require_runtime','require_runtime');kw(pf['run'],'_require_runtime','require_runtime');kw(bf['_runtime'],'_require','require_runtime')
rc=next(n for n in p.body if isinstance(n,ast.ClassDef)and n.name=='RuntimeAdmission')
for fn in rc.body:
    if isinstance(fn,ast.FunctionDef):
        if fn.name in('check','expected_ids_bytes','input_digest'):kw(fn,'_require','require_runtime')
        else:kw(fn,'_dispatch','_runtime_action')
bc=next(n for n in b.body if isinstance(n,ast.ClassDef)and n.name=='ServerBackend')
for fn in bc.body:
    if isinstance(fn,ast.FunctionDef)and fn.name in('__init__','check'):kw(fn,'_check_runtime','_runtime')
assert next(i for i,n in enumerate(p.body)if isinstance(n,ast.Import)and any(a.name=='qwen_server_backend'for a in n.names))>p.body.index(rc)
assert any(isinstance(n,ast.ImportFrom)and n.module=='qwen_server_process'and {x.name for x in n.names}=={'RuntimeAdmission','require_runtime','_expected_ids_bytes'}for n in b.body)
assert not any(isinstance(n,ast.ImportFrom)and n.module=='qwen_server_process'for n in ast.walk(bf['_runtime']))
private_keywords={'_dispatch','_require','_require_runtime','_check_runtime'}
assert not any(k.arg in private_keywords for tree in(p,b)for n in ast.walk(tree)if isinstance(n,ast.Call)for k in n.keywords)
report=dict(schema='qwen-server-snapshot-dispatch-successor-security-source-review-v1',handoff_sha256=H(hraw),source_manifest_sha256=H(raw),source_archive_sha256=h['source_archive_sha256'],source_count=391,unchanged_predecessor390=386,changed=changed,added=h['added'],code_hashes=h['code_hashes'],callback_sources={n:m[n]for n in closure},disposition='no concrete blocker in narrow fixes; meaningful integration tests still required',conclusions=['Exact cleanup_origin4543 is now bound as source/module/exports/code before read cleanup calls, covering is_cleanup/mark transitive lookup.','Runtime methods/properties and require_runtime retain original dispatcher/checker defaults; backend module imports originals once after process definitions; backend constructor/check retain original backend checker.','Supported original caller routes invoke retained dispatcher -> original registry/control -> retained guard before any changed export is invoked; guard rejection marks runtime failed and preserves original owner graph.','Actual source call sites use ordinary fixed argument shapes; no selected call supplies retention keywords. Explicit override of private defaults/registry/closure internals remains outside stated scope.','Entry, literal guard879071, complete-capacity denial, reader/core arithmetic and readiness metadata remain unchanged. This is source-only correction, not inherited38/18 integration proof.'],candidate_imports=False,collection=False,tests=False,body_or_capacity_probes=False,actual_authority=False)
(O/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(H((O/'review.json').read_bytes()))
