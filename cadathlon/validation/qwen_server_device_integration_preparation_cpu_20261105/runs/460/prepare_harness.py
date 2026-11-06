"""Prepare exact460 unrun harness with stdlib only; no candidate execution."""
import ast,difflib,hashlib,json,shutil,stat,tarfile
from pathlib import Path,PurePosixPath
old=Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v3')
new=old.parent/'preparation-v4'
source_handoff=Path('/tmp/qwen-full28-device-focused-test-handoff-v3-20261105/handoff.json')
review=source_handoff.parent/'root-source-review.json'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pretty=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda value:(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def inventory(root):
    paths=sorted(root.rglob('*'));assert all(not p.is_symlink()for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}
source_raw=source_handoff.read_bytes();assert sha(source_raw)=='5ea7bcde5d73a106ee6d5defaeb1879f89dec50fb76a3f457fc4029953b4fc0a'
sh=json.loads(source_raw);source=Path(sh['source_root']);map_raw=Path(sh['source_map']).read_bytes();expected=json.loads(map_raw)
assert len(expected)==sh['source_count']==460 and map_raw==compact(expected)
assert sha(map_raw)==sh['source_manifest_sha256']=='b5697bc650e3476c45500ef7e7e1c6a68e3432fc1c5db20410bffb4f1b1b64a8'
assert inventory(source)==expected
assert sha(review.read_bytes())==sh['root_review_sha256']=='1e1d02960cb34fc96d3a1dbbdbc3452a8ff88a9bee3ca501ba6b4d8ff1c3ebce'
archive=Path(sh['source_archive']);assert sha(archive.read_bytes())==sh['source_archive_sha256']
prior=json.loads((old/'handoff.json').read_bytes())
old_files={n:(old/n).read_bytes()for n in ('executed_driver.py','worker_driver.py','inspect_children.py','run_config.json',
    'launch-proposal.json','handoff.json','driver_provenance.json','sources-before.json','source-handoff.json','prepare_harness.py')}
for name,raw in old_files.items():
    if name!='handoff.json':assert sha(raw)==prior['pins'][name]
assert not new.exists();new.mkdir();execution=new/'execution';execution.mkdir()
for name,pin in expected.items():
    rel=PurePosixPath(name);assert str(rel)==name and not rel.is_absolute()and '..'not in rel.parts
    original=source/name;assert stat.S_ISREG(original.lstat().st_mode)
    raw=original.read_bytes();assert sha(raw)==pin
    target=execution/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
(new/'sources-before.json').write_bytes(map_raw);(new/'source-handoff.json').write_bytes(source_raw)
shutil.copyfile(archive,new/'sources.tar.gz')
for name in ('worker_driver.py','inspect_children.py'):(new/name).write_bytes(old_files[name])
old_config=json.loads(old_files['run_config.json']);config=dict(old_config)
for name in ('proof_root','owned_tmpdir','pytest_basetemp'):config[name]=config[name].replace(str(old),str(new))
config.update(source_root=str(source),source_count=460,source_manifest_sha256=sha(map_raw),source_archive_sha256=sh['source_archive_sha256'],
    external_source_review=dict(path=str(review),sha256=sha(review.read_bytes())))
(new/'run_config.json').write_bytes(pretty(config));config_sha=sha((new/'run_config.json').read_bytes())
driver=old_files['executed_driver.py'].decode();assert driver.count('459')==4
driver=driver.replace('459','460').replace(prior['config_sha256'],config_sha)
(new/'executed_driver.py').write_text(driver)
proposal=json.loads(old_files['launch-proposal.json']);proposal['argv']=[p.replace(str(old),str(new))for p in proposal['argv']]
proposal.update(driver_sha256=sha(driver.encode()),config_sha256=config_sha,source_manifest_sha256=sha(map_raw),source_archive_sha256=sh['source_archive_sha256'])
(new/'launch-proposal.json').write_bytes(pretty(proposal))
history=new/'v3-failed-predecessor';history.mkdir()
for name,raw in old_files.items():(history/(name+'.txt')).write_bytes(raw)
for name in ('executed_driver.py','run_config.json','launch-proposal.json'):
    (history/(name+'.patch')).write_text(''.join(difflib.unified_diff(old_files[name].decode().splitlines(True),(new/name).read_text().splitlines(True),fromfile='failed-preparation-v3/'+name,tofile='unrun-preparation-v4/'+name)))
external={}
for label,path,pin in (
    ('root460_source',review,'1e1d02960cb34fc96d3a1dbbdbc3452a8ff88a9bee3ca501ba6b4d8ff1c3ebce'),
    ('root_failed459',Path('/tmp/qwen-full28-device-focused-root-harness-review-20261105/failed459-terminal-review.json'),'93657ed40c0935dc11bf7b0fc3f0f6a8e30ed247d408c78b949cf92371d8dd7b'),
    ('cnn_failed459',Path('/tmp/qwen-full28-device459-terminal-cnn-review-20261105/review.json'),'887bbd35dc4c8b56ea780aa5d6638c36683ef3d767e4b5a2cd737dc12ba8452a')):
    raw=path.read_bytes();assert sha(raw)==pin;(history/(label+'.json')).write_bytes(raw)
    external[label]=dict(path=str(path),sha256=pin)
provenance=dict(schema='full28-focused-harness-source460-successor-v4',status='HELD_UNRUN_SOURCE_ONLY',
    predecessor_root=str(old),predecessor_handoff_sha256=sha(old_files['handoff.json']),
    predecessor_status='FAILED_43_PASS_16_FAIL_WORKER_NORMAL1_COLLECTOR70',
    source_handoff_sha256=sha(source_raw),source_count=460,runtime_count=403,selected_cases=59,
    worker_exact=True,strict_inspector_exact=True,owned_popen_wait_finally_exact=True,
    driver_delta='Only pinned config digest and four459-to460 count/doc literals.',
    changed_config_fields=sorted(k for k in config if config[k]!=old_config[k]),
    external_reviews=external,old_harness_files={k:sha(v)for k,v in old_files.items()},
    previous_v1_unrun_preserved=True,failed_v2_v3_preserved=True,missing_result_handling='Unchanged fail-closed70, no success inference.',
    candidate_imports_collection_tests=False,actual_execution_authorized=False)
(new/'driver_provenance.json').write_bytes(pretty(provenance));(new/'prepare_harness.py').write_bytes(Path(__file__).read_bytes())
for name in ('executed_driver.py','worker_driver.py','inspect_children.py'):ast.parse((new/name).read_bytes())
assert driver.replace('460','459').replace(config_sha,prior['config_sha256'])==old_files['executed_driver.py'].decode()
for name in ('worker_driver.py','inspect_children.py'):assert(new/name).read_bytes()==old_files[name]
def owned(text):return text.split("    with (out/'tests.log').open('xb')as log:\n",1)[1].split('    current,after=inventory(source),inventory(execution)',1)[0]
assert owned(driver)==owned(old_files['executed_driver.py'].decode())
assert inventory(source)==inventory(execution)==expected
runtime={n[8:]:pin for n,pin in expected.items()if n.startswith('runtime/')};assert runtime==config['runtime_source_expectations']and len(runtime)==403
assert config['selected_nodes']==old_config['selected_nodes']and len(config['selected_nodes'])==59
assert sha((execution/'selected_tests.json').read_bytes())==config['selection_sha256']==sh['selected_tests_sha256']
with tarfile.open(new/'sources.tar.gz')as saved:
    members=saved.getmembers();assert len(members)==len({m.name for m in members})==460 and all(m.isfile()for m in members)
    assert {m.name:sha(saved.extractfile(m).read())for m in members}==expected
assert all((old/name).read_bytes()==raw for name,raw in old_files.items())
pins={p.relative_to(new).as_posix():sha(p.read_bytes())for p in sorted(new.rglob('*'))if p.is_file()and 'execution'not in p.relative_to(new).parts}
handoff=dict(schema='full28-focused-unrun-harness-handoff-v4',source_root=str(source),proof_root=str(new),execution_root=str(execution),
    source_count=460,runtime_count=403,static_cases=59,source_manifest_sha256=sha(map_raw),source_archive_sha256=sh['source_archive_sha256'],
    driver_sha256=sha(driver.encode()),worker_sha256=prior['worker_sha256'],inspector_sha256=prior['inspector_sha256'],config_sha256=config_sha,
    proposal_sha256=sha((new/'launch-proposal.json').read_bytes()),predecessor_handoff_sha256=sha(old_files['handoff.json']),
    correction='Exact460 paths/pins/count only; original strict inspector and worker unchanged.',external_source_review=external['root460_source'],
    pins=pins,imports_collection_runs=0,actual_execution_authorized=False)
(new/'handoff.json').write_bytes(pretty(handoff))
print(json.dumps({k:handoff[k]for k in ('driver_sha256','worker_sha256','inspector_sha256','config_sha256','proposal_sha256')},indent=2))
print('handoff',sha((new/'handoff.json').read_bytes()));print('source460/execution/tar exact,403/59 unchanged; no candidate import, collection or tests')
