from pathlib import Path
import ast,datetime,hashlib,json,tarfile
root=Path('/tmp/qwen-server-snapshot-integration-tests-20261104')
home=Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
handoff=json.loads((home/'handoff.json').read_bytes());mraw=(home/'sources.json').read_bytes();mapping=json.loads(mraw)
assert sha((home/'handoff.json').read_bytes())=='848a375e24ee9b0244f48d1637d10df2e2ab398ffa22510465bb832ae8a19865'
assert sha(mraw)==handoff['source_manifest_sha256']=='9fa73f2f59e6bef71a2de475a09e0aaccb25d56375f7a06452cac7b75e4afc1c'
assert len(mapping)==404
observed={}
for p in root.rglob('*'):
 assert not p.is_symlink(),p
 if p.is_file():observed[p.relative_to(root).as_posix()]=sha(p.read_bytes())
assert observed==mapping
runtime=json.loads((root/'runtime_sources.json').read_bytes());assert len(runtime)==391
parent=Path('/tmp/qwen-server-snapshot-runtime-v2-20261104')
for n,v in runtime.items():assert mapping['runtime/'+n]==v==sha((parent/n).read_bytes())
archive=home/'sources.tar.gz';assert sha(archive.read_bytes())==handoff['source_archive_sha256']=='bbc4c0bd3ce4bc6a1cd0efa10a7afc799e280887132869b846849fee10b2e713'
with tarfile.open(archive,'r:gz')as t:
 members=t.getmembers();assert all(p.isfile()for p in members)
 assert len(members)==404 and len({p.name for p in members})==404
 assert {p.name:sha(t.extractfile(p).read())for p in members}==mapping
for p in root.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
ct=ast.parse((root/'integration_cases.py').read_bytes());cases=ast.literal_eval(next(n.value for n in ct.body if isinstance(n,ast.Assign)))
assert len(cases)==len(set(cases))==36
nodes=['test_snapshot_integration.py::test_snapshot_integration['+x+']'for x in cases]
assert nodes==handoff['static_nodeids']==json.loads((root/'selected_tests.json').read_bytes())['static_nodeids']
entry=ast.parse((root/'runtime/qwen_server_entry.py').read_bytes());dump=lambda n:ast.dump(n,include_attributes=False)
last=ast.parse("if __name__=='__main__':\n    raise SystemExit(main())\n").body[0]
assert dump(entry.body[-1])==dump(last)
prov=json.loads((root/'entry_fixture_provenance.json').read_bytes())
assert sha(dump(last).encode())==prov['excluded_dispatch_ast_sha256']
assert sha(dump(ast.Module(body=entry.body[:-1],type_ignores=entry.type_ignores)).encode())==prov['definition_ast_sha256']
origin=json.loads((root/'origin.json').read_bytes())
for n,v in origin['references'].items():assert sha(Path(n).read_bytes())==v
old=ast.parse(Path('/tmp/qwen-device-component-tests-v2-20261104/test_device_component.py').read_bytes())
new=ast.parse((root/'test_snapshot_integration.py').read_bytes())
def owned_try(t):
 f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
 w=next(n for n in f.body if isinstance(n,ast.With));return next(n for n in w.body if isinstance(n,ast.Try))
assert dump(owned_try(old))==dump(owned_try(new))
out=Path('/tmp/qwen-server-snapshot-integration-root-test-review-20261104');out.mkdir(exist_ok=False)
r={'schema':'qwen-snapshot-integration-root-test-source-review-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'handoff_sha256':sha((home/'handoff.json').read_bytes()),'source_manifest_sha256':sha(mraw),'source_archive_sha256':sha(archive.read_bytes()),'source_count':404,'runtime391_current_and_nested_exact':True,'archive_exact':True,'six_new_python_files_ast_parsed':True,'static_cases':36,'static_nodeids':nodes,'entry_only_terminal_dispatch_omitted':True,'prior_owned_popen_try_ast_exact':True,'root_reads':['969014','55af8d','4622a7','3755bb','989ee5','54959c','3cfbc0','88dd90','7f54bd','94f0cd'],'root_concrete_findings':[],'scope':'Synthetic retained graph, original runtime/backend/control/Finalizer behavior with explicit seams; definition-only entry. No genuine bootstrap, full science, capacity, body integrity, HE or performance result.','status':'Root source review passed; independent peer review and harness review remain; no execution authority','candidate_imports':False,'collection':False,'tests':False,'backend_or_capacity_probes':False,'body_reads':False}
raw=json.dumps(r,sort_keys=True,indent=2).encode()+b'\n';(out/'review.json').write_bytes(raw)
print(json.dumps({'receipt':str(out/'review.json'),'sha256':sha(raw),'source_count':len(mapping),'cases':len(cases)}))
