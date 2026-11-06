from pathlib import Path
import ast,datetime,hashlib,json,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
proof=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1')
root=Path('/tmp/qwen-server-admission-snapshot-root-review-20261104')
pins={'handoff.json':'c9c08053c2957f253cbba570ebde3ac750da08eec8c08520f2b6f8752f26d01a','executed_driver.py':'c436247368deed6d0794040e51f7044ea54de7fc354bb70645cc0116512d37d6','worker_driver.py':'55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be','run_config.json':'02d9dec6cfff9dd65b9c32f89fda52a08c47848e33bbab74237b80395b28e47d','launch-proposal.json':'6c50e65185fe0fead4f8bee119a8293ee6786123b3c40888aae88ae3be0bd145'}
for n,h in pins.items():assert sha((proof/n).read_bytes())==h
handoff=json.loads((proof/'handoff.json').read_bytes())
for n,h in handoff['files'].items():assert sha((proof/n).read_bytes())==h
raw=(proof/'sources-before.json').read_bytes();mapping=json.loads(raw);cfg=json.loads((proof/'run_config.json').read_bytes())
assert raw==compact(mapping) and sha(raw)==cfg['source_manifest_sha256']=='478b3411cfa54d154eb91afbb1bccab5b6ae24520c4b6f17873be9bd0d8d44e0'and len(mapping)==37
def inventory(p):
 assert p.resolve()==p and not any(q.is_symlink()for q in p.rglob('*'))
 return {q.relative_to(p).as_posix():sha(q.read_bytes())for q in p.rglob('*')if q.is_file()}
source=Path(cfg['source_root']);assert inventory(source)==inventory(proof/'execution')==mapping
archive=proof/'sources.tar.gz';assert sha(archive.read_bytes())==cfg['source_archive_sha256']=='b77f262c89bdb46f73818a42b6346f86f87c2a7b76b12bcf6d2e97446b9fa807'
with tarfile.open(archive)as tf:
 members=tf.getmembers();assert len(members)==len({m.name for m in members})==37
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==mapping
runtime=json.loads((source/'runtime_sources.json').read_bytes());assert len(runtime)==28 and inventory(source/'runtime')==runtime
source_review=(root/'source-review.json').read_bytes();assert sha(source_review)=='0bd3c1229e9efb7f6bcca8c97a56ad55296138a26304d6bfa7d736058bbfd670'
assert json.loads(source_review)['source_manifest_sha256']==sha(raw)
tree=ast.parse((source/'test_snapshot_guard.py').read_bytes())
cases=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CASES'for t in n.targets)))
nodes=['test_snapshot_guard.py::test_snapshot_guard['+n+']'for n in cases]
assert len(nodes)==18 and nodes==cfg['selected_nodes'] and cfg['selected_files']==['test_snapshot_guard.py']
xml=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in nodes]
assert xml==cfg['selected_xml_cases'] and sha(pretty(nodes))==cfg['selected_nodes_sha256'] and sha(pretty(xml))==cfg['selected_xml_cases_sha256']
assert cfg['runtime_source_expectations']==cfg['runtime_module_origins']=={} and cfg['cpu_affinity']==[112,113,114,115]
assert cfg['actual_execution_authorized']is False and cfg['synthetic_only']is True and cfg['expected_child_records']==18
assert sha((source/'pytest.ini').read_bytes())==cfg['pytest_ini_sha256']=='b7d8a2b1461f216dcacbe357b07d49bf2e53c15b11b097dd198fd5b71bc7d2e0'
assert Path(cfg['owned_tmpdir'])==proof/'tmp' and Path(cfg['pytest_basetemp'])==proof/'pytest-owned'
for n in mapping:
 if n.endswith('.py'):ast.parse((source/n).read_bytes())
for n in ('executed_driver.py','worker_driver.py'):ast.parse((proof/n).read_bytes())
def ownerblock(p):
 return ast.dump(next(n for n in ast.walk(ast.parse(p.read_bytes()))if isinstance(n,ast.Try)and n.body and isinstance(n.body[0],ast.Assign)and any(isinstance(x,ast.Name)and x.id=='child_pgid'for x in n.body[0].targets)),include_attributes=False)
assert ownerblock(proof/'executed_driver.py')==ownerblock(Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v2/executed_driver.py'))
absent=['tests.log','tests.xml','report.json','worker-audit.json','live-observation.json','sources-after.json','execution-after.json','pytest-owned','tmp','owned-pytest-metadata.json','test-child-terminals.json']
assert not any((proof/n).exists()or(proof/n).is_symlink()for n in absent)
record=dict(schema='snapshot-focused-root-preexecution-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,source_count=37,runtime_files=28,static_cases=18,harness_pins=pins,source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive.read_bytes()),source_review_sha256=sha(source_review),source_current_execution_archive_equal=True,static_nodes=nodes,prior_outputs_absent=absent,source_only=True,imports=False,collection=False,execution_authorized=False,root_harness_reads=['6df8e2','905e00'],owned_Popen_block_exact=True,scope=cfg['scope'],script_sha256=sha(Path(__file__).read_bytes()))
out=root/'preexecution-review.json'
with out.open('xb')as f:f.write(pretty(record))
print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),passed=True,static_cases=18)))
