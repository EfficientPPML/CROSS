from pathlib import Path
import ast,datetime,hashlib,json,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
proof=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v2')
root=Path('/tmp/qwen-server-admission-snapshot-root-review-v2-20261104')
pins={'handoff.json':'40aaae7ac0b50c6218ca234146732aca97bc48730fe4c0ea45304253a3788a06','executed_driver.py':'3d70d7531b9284a35da14c08d0d6cb35ecaa5148318a7c5a70e0a0133862a0aa','worker_driver.py':'55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be','run_config.json':'73f36f9634f726699a8c70ec826f2020b59713dc1fc9960b85859d27bb281ab0','launch-proposal.json':'27df759281f03c9f76a704d3b159e61c27ee413c4f3b1f5a1b3df34318c266b8'}
for n,h in pins.items():assert sha((proof/n).read_bytes())==h
handoff=json.loads((proof/'handoff.json').read_bytes())
for n,h in handoff['files'].items():assert sha((proof/n).read_bytes())==h
raw=(proof/'sources-before.json').read_bytes();mapping=json.loads(raw);cfg=json.loads((proof/'run_config.json').read_bytes())
assert raw==pretty(mapping) and sha(raw)==cfg['source_manifest_sha256']=='963b03f0bdc4ca2f09fd03f4986c175d2a3b41b6fe82cddf42761c3b3a53c93b'and len(mapping)==38
def inventory(p):
 assert p.resolve()==p and not any(q.is_symlink()for q in p.rglob('*'))
 return {q.relative_to(p).as_posix():sha(q.read_bytes())for q in p.rglob('*')if q.is_file()}
source=Path(cfg['source_root']);assert inventory(source)==inventory(proof/'execution')==mapping
archive=proof/'sources.tar.gz';assert sha(archive.read_bytes())==cfg['source_archive_sha256']=='bb1eb2fb5613347f5ff2ecf9a27a1f3cdb5944261e154841ea3b19ab4ed75575'
with tarfile.open(archive)as tf:
 members=tf.getmembers();assert len(members)==len({m.name for m in members})==38
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==mapping
runtime=json.loads((source/'runtime_sources.json').read_bytes());assert len(runtime)==28 and inventory(source/'runtime')==runtime
source_review=(root/'source-review.json').read_bytes();assert sha(source_review)=='e2bbc8d2c3a9a12818c82f5af5be63635604d5acb67b3dbc84c30ae56ada30e4'
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
assert cfg['expected_parent_records']==18
old_proof=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1')
assert (proof/'worker_driver.py').read_bytes()==(old_proof/'worker_driver.py').read_bytes()
old_raw=(old_proof/'sources-before.json').read_bytes();old_map=json.loads(old_raw)
assert inventory(old_proof/'execution')==old_map
old_runtime={n[8:]:h for n,h in old_map.items()if n.startswith('runtime/')}
assert old_runtime==runtime
assert sha((proof/'failed-v1/failed-terminal-review.json').read_bytes())=='fc08e491aa93ae6eb0961d7c0d008efd0b2cf83b00f9b47d5a06ef73b376eefd'
record=dict(schema='snapshot-focused-root-preexecution-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,source_count=38,runtime_files=28,static_cases=18,harness_pins=pins,source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive.read_bytes()),source_review_sha256=sha(source_review),source_current_execution_archive_equal=True,static_nodes=nodes,prior_outputs_absent=absent,source_only=True,imports=False,collection=False,execution_authorized=False,root_harness_reads=['6df8e2','905e00','4b4b0e','fc08a2'],owned_Popen_block_exact=True,scope=cfg['scope'],script_sha256=sha(Path(__file__).read_bytes()))
record.update(early_parent_records_required=18,unchanged_worker=True,failed37_runtime28_preserved=True)
out=root/'preexecution-review.json'
with out.open('xb')as f:f.write(pretty(record))
print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),passed=True,static_cases=18)))
