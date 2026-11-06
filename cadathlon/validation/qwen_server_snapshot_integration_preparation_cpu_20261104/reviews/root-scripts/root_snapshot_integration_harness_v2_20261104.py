from pathlib import Path
import ast,datetime,hashlib,json,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest();base=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104');a=base/'preparation-v1';p=base/'preparation-v2'
h=json.loads((p/'handoff.json').read_bytes());assert sha((p/'handoff.json').read_bytes())=='50ffb8a07c5d63d69ffd60e8e40a789f272075851427f7326eee5bf9b3af6555'
for n,v in h['files'].items():assert sha((p/n).read_bytes())==v
old=json.loads((a/'run_config.json').read_bytes());c=json.loads((p/'run_config.json').read_bytes());changes=sorted(k for k in old.keys()|c.keys()if old.get(k)!=c.get(k))
assert changes==['owned_tmpdir','proof_root','pytest_basetemp','source_archive_sha256','source_count','source_manifest_sha256','source_root']
assert (p/'worker_driver.py').read_bytes()==(a/'worker_driver.py').read_bytes()
driver=(p/'executed_driver.py').read_bytes();assert driver.replace(h['config_sha256'].encode(),sha((a/'run_config.json').read_bytes()).encode())==(a/'executed_driver.py').read_bytes();ast.parse(driver)
assert sha(driver)==h['driver_sha256']=='751aa66a81409304713355d5a3b1936ed42c6ce5e3eb6d063f3fb129ce6bef57'
assert sha((p/'run_config.json').read_bytes())==h['config_sha256']=='ceae86e1e99d15f322ace03b7b4d9247add88d581ed1b23d2bd3a9e20461b48e'
mraw=(p/'sources-before.json').read_bytes();m=json.loads(mraw);assert len(m)==405 and c['source_count']==405
assert sha(mraw)==c['source_manifest_sha256']=='7c4f95e68dabba24ee489bfe05ddd77b5de0ca5e7ba1a3e77139ef57a410bbd4'
assert mraw==(json.dumps(m,sort_keys=True,separators=(',',':'))+'\n').encode()
for root in (Path(c['source_root']),p/'execution'):
 assert not any(f.is_symlink()for f in root.rglob('*'))
 assert {f.relative_to(root).as_posix():sha(f.read_bytes())for f in root.rglob('*')if f.is_file()}==m
with tarfile.open(p/'sources.tar.gz')as t:
 members=t.getmembers();assert len(members)==405==len({n.name for n in members})and all(n.isfile()for n in members)
 assert {n.name:sha(t.extractfile(n).read())for n in members}==m
assert sha((p/'sources.tar.gz').read_bytes())==c['source_archive_sha256']=='7814e934402820f57da5084c6cefc4d0a924cc5e68a06c219bdeacd47d03ee59'
assert c['runtime_source_expectations']==json.loads((p/'execution/runtime_sources.json').read_bytes())and len(c['runtime_source_expectations'])==391
assert c['selected_nodes']==json.loads((p/'execution/selected_tests.json').read_bytes())['static_nodeids'] and len(c['selected_nodes'])==36
preserved=json.loads((p/'preserved-v1-files.json').read_bytes())
for n,v in preserved.items():assert sha((a/n).read_bytes())==v
for n in ('report.json','tests.log','tests.xml','tmp','pytest-owned','worker-audit.json','live-observation.json'):assert not(p/n).exists()
out=Path('/tmp/qwen-server-snapshot-integration-root-test-review-20261104');r={'schema':'qwen-snapshot-integration-root-harness-v2-source-review-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'handoff_sha256':sha((p/'handoff.json').read_bytes()),'driver_sha256':h['driver_sha256'],'worker_sha256':h['worker_sha256'],'config_sha256':h['config_sha256'],'proposal_sha256':sha((p/'launch-proposal.json').read_bytes()),'exact405_current_execution_archive':True,'runtime391_and36nodes_unchanged':True,'driver_only_config_pin_changed':True,'worker_exact':True,'changed_config_fields':changes,'v1_failed_evidence_preserved':True,'no_outputs_present':True,'source_delta_root_review_sha256':'92248e7cd47a54a271ffee364993f87f249c231269e6b40bf93eb28390a6b26a','root_read':'8c0ddc','status':'Root narrow source review passed; independent peer harness review and fresh grant required','launch_authorized':False,'candidate_imports':False,'collection':False,'tests':False}
b=(json.dumps(r,sort_keys=True,indent=2)+'\n').encode();(out/'preexecution-v2-review.json').write_bytes(b);print(json.dumps({'receipt':str(out/'preexecution-v2-review.json'),'sha256':sha(b)}))
