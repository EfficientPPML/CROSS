from pathlib import Path
import ast,datetime,hashlib,json,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
root=Path('/tmp/qwen-server-admission-snapshot-root-review-20261104')
old=Path('/tmp/qwen-server-admission-snapshot-tests-20261104');new=Path('/tmp/qwen-server-admission-snapshot-tests-v2-20261104')
handoff_path=Path('/tmp/qwen-server-admission-snapshot-test-handoff-v2-20261104/handoff.json')
assert sha(handoff_path.read_bytes())=='7fa0fc4b0672cbffafad4631ecdb97872e482f327c573a190e6e9037ce6a6d49'
h=json.loads(handoff_path.read_bytes());raw=Path(h['source_manifest']).read_bytes();mapping=json.loads(raw)
assert sha(raw)=='478b3411cfa54d154eb91afbb1bccab5b6ae24520c4b6f17873be9bd0d8d44e0' and raw==compact(mapping) and len(mapping)==37
def inventory(p):
 assert not any(q.is_symlink()for q in p.rglob('*'))
 return {q.relative_to(p).as_posix():sha(q.read_bytes())for q in p.rglob('*')if q.is_file()}
assert inventory(new)==mapping==h['source_hashes']
old_map_raw=Path('/tmp/qwen-server-admission-snapshot-test-sources-v1-20261104.json').read_bytes();old_map=json.loads(old_map_raw)
assert sha(old_map_raw)=='8ecdb98742f8f79dd9988cad12a0b4b79c6b33415315c022741dc65b12ff27b2' and len(old_map)==36 and inventory(old)==old_map
assert set(mapping)-set(old_map)=={'successor_history.json'} and set(old_map)<=set(mapping)
assert [n for n in old_map if old_map[n]!=mapping[n]]==['test_snapshot_guard.py']
addition="    assert type(data['pgid'])is int and data['pgid']==proc.pid\n"
after=(new/'test_snapshot_guard.py').read_text();assert after.count(addition)==1 and after.replace(addition,'')==(old/'test_snapshot_guard.py').read_text()
runtime=json.loads((new/'runtime_sources.json').read_bytes());assert len(runtime)==28 and inventory(new/'runtime')==runtime
packet=inventory(Path('/tmp/qwen-server-public-packet-20261104'));assert len(packet)==27
assert {n:h for n,h in runtime.items()if n!='qwen_admission_snapshot.py'}==packet
assert runtime['qwen_admission_snapshot.py']=='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
archive=handoff_path.parent/'sources.tar.gz';assert sha(archive.read_bytes())==h['source_archive_sha256']=='b77f262c89bdb46f73818a42b6346f86f87c2a7b76b12bcf6d2e97446b9fa807'
with tarfile.open(archive)as tf:
 members=tf.getmembers();assert len(members)==len({m.name for m in members})==37
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==mapping
tree=ast.parse(after);cases=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CASES'for t in n.targets)))
tests=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name.startswith('test_')];assert len(tests)==1
param=tests[0].decorator_list[0];assert ast.literal_eval(param.args[0])=='scenario'and param.args[1].id=='CASES'and len(param.keywords)==1 and param.keywords[0].arg=='ids'and param.keywords[0].value.id=='CASES'
nodes=['test_snapshot_guard.py::test_snapshot_guard['+c+']'for c in cases];assert len(nodes)==18 and nodes==h['static_nodeids']
for n in mapping:
 if n.endswith('.py'):ast.parse((new/n).read_bytes())
peer=Path('/tmp/qwen-snapshot-test-cnn-review-20261104/review.json');assert sha(peer.read_bytes())=='3a27509ba7a8ac0ba45d3fec3dfe4ad48428dd36ebcdd45d4e4ade6247eeab24'
record=dict(schema='snapshot37-root-source-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_review_passed=True,source_count=37,runtime_files=28,static_cases=18,source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive.read_bytes()),handoff_sha256=sha(handoff_path.read_bytes()),parent36_unchanged=True,sole_code_change='Unconditional exact-int reported PGID equals original owned child PID before group absence check.',runtime28_exact=True,old_peer_finding_receipt_sha256=sha(peer.read_bytes()),root_full_reads=['bbd475','05b203','5fafdf','970fd4'],scope='Recording validator seam with exact13 frozen admission graph; tests not yet run. No full scientific validation, admission grant, backend, resource capacity, model or HE claim.',imports=False,collection=False,execution_authorized=False,script_sha256=sha(Path(__file__).read_bytes()))
root.mkdir(exist_ok=False);out=root/'source-review.json';out.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),passed=True,static_cases=18)))
