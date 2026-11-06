from pathlib import Path
import ast,datetime,hashlib,json,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest();dump=lambda n:ast.dump(n,include_attributes=False)
p=Path('/tmp/qwen-server-snapshot-integration-test-handoff-v2-20261104');r=Path('/tmp/qwen-server-snapshot-integration-tests-v2-20261104');old=Path('/tmp/qwen-server-snapshot-integration-tests-20261104')
h=json.loads((p/'handoff.json').read_bytes());assert sha((p/'handoff.json').read_bytes())=='57498a8ea7a0b03d75f4088d173434ec4bca218094e982ed5531c183c748ffa5'
m=json.loads((p/'sources.json').read_bytes());assert len(m)==405 and sha((p/'sources.json').read_bytes())=='7c4f95e68dabba24ee489bfe05ddd77b5de0ca5e7ba1a3e77139ef57a410bbd4'
def inventory(root):
 assert not any(f.is_symlink()for f in root.rglob('*'))
 return {f.relative_to(root).as_posix():sha(f.read_bytes())for f in root.rglob('*')if f.is_file()}
assert inventory(r)==m
om=json.loads(Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104/sources.json').read_bytes());assert inventory(old)==om
assert set(m)-set(om)=={'fixture_successor_history.json'} and set(om)-set(m)==set()
assert [n for n in om if om[n]!=m[n]]==['integration_fixture.py']
for q in (p,p/'before-root-failure-receipts'):
 qm=json.loads((q/'sources.json').read_bytes());assert len(qm)==405
 with tarfile.open(q/'sources.tar.gz')as t:
  members=t.getmembers();assert len(members)==405==len({n.name for n in members}) and all(n.isfile()for n in members)
  assert {n.name:sha(t.extractfile(n).read())for n in members}==qm
assert sha((p/'sources.tar.gz').read_bytes())==h['source_archive_sha256']=='7814e934402820f57da5084c6cefc4d0a924cc5e68a06c219bdeacd47d03ee59'
t=ast.parse((r/'integration_fixture.py').read_bytes());before=ast.parse((old/'integration_fixture.py').read_bytes());cls=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=='Fixture');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='capture_runtime')
local=[n for n in method.body if isinstance(n,ast.FunctionDef)];expected=ast.parse('def source_check():return self.full_boundary()\ndef capacity_check():return self.capacity()\n').body
assert [dump(n)for n in local]==[dump(n)for n in expected]
method.body=[n for n in method.body if n not in local]
call=next(n for n in ast.walk(method)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='_runtime')
assert dump(call.args[4])==dump(ast.Name(id='source_check',ctx=ast.Load())) and dump(call.args[5])==dump(ast.Name(id='capacity_check',ctx=ast.Load()))
call.args[4]=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='full_boundary',ctx=ast.Load());call.args[5]=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='capacity',ctx=ast.Load())
assert dump(t)==dump(before)
history=json.loads((r/'fixture_successor_history.json').read_bytes())
for name,pin in history['independent_root_failed_proof_receipts'].items():assert sha(Path(name).read_bytes())==pin
pre=history['pre_receipt_source_only_checkpoint'];q=p/'before-root-failure-receipts'
assert sha((q/'sources.tar.gz').read_bytes())==pre['source_archive_sha256'] and sha((q/'sources.json').read_bytes())==pre['source_map_sha256'] and sha((q/'fixture_successor_history.json').read_bytes())==pre['history_sha256']
assert json.loads((r/'selected_tests.json').read_bytes())['static_nodeids']==h['static_nodeids'] and len(h['static_nodeids'])==36
out=Path('/tmp/qwen-server-snapshot-integration-root-test-review-20261104')
review={'schema':'qwen-snapshot-integration-root-405-source-review-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'handoff_sha256':sha((p/'handoff.json').read_bytes()),'source_manifest_sha256':sha((p/'sources.json').read_bytes()),'source_archive_sha256':h['source_archive_sha256'],'source_count':405,'runtime391_exact':True,'old404_preserved':True,'prior405_checkpoint_preserved':True,'only_code_change':'Two local plain callback delegates in capture_runtime; normalized AST exact with404','all36_assertions_unchanged':True,'root_reads':['716327','12feb1','d964de','01d4c1'],'peer_review_sha256':'9ac3c531445c242bd630b7095a793c66ea6f8fa306250fded9dd325f54496ec2','status':'Root source delta accepted; no imports/collection/test execution; fresh harness and grant remain','findings':[],'historical_failure':'Preserved25setupFAIL/11PASS; source-only review missed the fixture callback type mismatch; no earlier outcome relabeled'}
b=(json.dumps(review,sort_keys=True,indent=2)+'\n').encode();(out/'source-v2-review.json').write_bytes(b);print(json.dumps({'receipt':str(out/'source-v2-review.json'),'sha256':sha(b)}))
