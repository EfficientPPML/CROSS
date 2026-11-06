"""Explicit public-file packaging only; no candidate imports, tests or probes."""
import hashlib
import json
from pathlib import Path
import stat

OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
encode=lambda x:(json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
copied={}
def copy(src,dest,expected=None):
    src=Path(src);dest=Path(dest)
    assert not dest.is_absolute() and '..' not in dest.parts
    assert src.is_file() and not src.is_symlink()
    before=src.lstat();assert stat.S_ISREG(before.st_mode)
    raw=src.read_bytes();after=src.lstat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    digest=sha(raw)
    if expected is not None:assert digest==expected,(str(src),'digest differs')
    target=OUT/dest;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(raw)
    copied[dest.as_posix()]=dict(path=str(src),sha256=digest)
def group(root,dest,names):
    for name in names:copy(Path(root)/name,Path(dest)/name)

groups=[
('/tmp/qwen-server-device-admission-handoff-20261104','history/device-v1',
 ['sources.json','sources.tar.gz','handoff.json','static-review.json','capture_source.py']),
('/tmp/qwen-server-device-admission-handoff-v2-20261104','history/device-v2',
 ['sources.json','sources.tar.gz','handoff.json','static-review.json','capture_source.py','source.patch']),
('/tmp/qwen-device-component-test-handoff-20261104','history/tests-v1',
 ['sources.json','sources.tar.gz','handoff.json','capture_source.py','parent_snapshot38.py','parent_owner_adaptation.patch']),
('/tmp/qwen-device-component-test-handoff-v2-20261104','history/tests-v2',
 ['sources.json','sources.tar.gz','handoff.json','capture_successor.py','source.patch']),
('/tmp/qwen-device-component-test-plan-20261104','history/test-plan',['PLAN.md','source_refs.json','files.json']),
('/tmp/qwen-device-component-transformers-review-20261104','peer-review/device-v1',['review_source.py','review.json']),
('/tmp/qwen-device-identity-cnn-review-20261104','peer-review/device-v2',['review_source.py','review.json']),
('/tmp/qwen-device-tests-security-review-20261104','peer-review/tests-v1',['review_source.py','review.json']),
('/tmp/qwen-device-tests-cnn-review-20261104','peer-review/tests-v2',['review_source_initial.py','review_source.py','review.json']),
('/tmp/qwen-device-harness-security-review-20261104','peer-review/harness',['review_harness.py','review.json']),
('/tmp/qwen-device-terminal-security-review-20261104','peer-review/terminal',['review_public.py','review.json','process-absence.json','root-receipt-pins.json']),
('/tmp/qwen-device-component-root-review-20261104','root-review',
 ['preexecution-review.json','run-grant.json','direct-owner.json','root-process-observation.json','terminal-review.json']),
]
for root,dest,names in groups:group(root,dest,names)
copy('/tmp/qwen-server-device-admission-authoring-history-20261104/capacity.syntax-draft.py.txt','history/device-v1/capacity.syntax-draft.py.txt')
for version,root in [('v1','/tmp/qwen-server-device-admission-source-20261104'),('v2','/tmp/qwen-server-device-admission-source-v2-20261104')]:
    mapping=json.loads((OUT/('history/device-'+version)/'sources.json').read_bytes())
    for name,d in mapping.items():copy(Path(root)/name,Path('history/device-'+version)/'source'/name,d)
group('/tmp/qwen-device-component-tests-20261104','history/tests-v1/readable',['device_child.py','test_device_component.py','TEST_SCOPE.md','selected_tests.json'])
source=Path('/tmp/qwen-device-component-tests-v2-20261104')
source_map=json.loads((OUT/'history/tests-v2/sources.json').read_bytes());assert len(source_map)==50
for name,d in source_map.items():copy(source/name,Path('source')/name,d)

proof=Path('/tmp/qwen-device-component-focused-proof-20261104/preparation-v1')
group(proof,'measured',[
 'README.md','prepare_source.py','before_executed_driver.py','before_worker_driver.py',
 'driver_provenance.json','executed_driver.py','executed_driver.py.patch','worker_driver.py','worker_driver.py.patch',
 'run_config.json','source-handoff.json','handoff.json','launch-proposal.json','sources-before.json','sources-after.json',
 'execution-after.json','sources.tar.gz','live-observation.json','tests.log','tests.xml','worker-audit.json',
 'owned-pytest-metadata.json','test-child-terminals.json','report.json'])
metadata=json.loads((proof/'owned-pytest-metadata.json').read_bytes())
regular={n:v for n,v in metadata.items() if v['kind']=='regular'}
aliases={n:v for n,v in metadata.items() if v['kind']=='owned_pytest_alias'}
assert len(regular)==120 and len(aliases)==21 and len(metadata)==141
for name,row in regular.items():
    assert name.startswith('pytest-owned/') and Path(name).name in ('parent-terminal.json','child-terminal.json','result.json','child.log')
    assert '..' not in Path(name).parts and len((proof/name).read_bytes())==row['bytes']
    copy(proof/name,Path('measured/owned')/name,row['sha256'])
# Alias paths are never read, resolved or copied; only their recorded metadata is retained.
for script in ('/tmp/root_device_preexecution_20261104.py','/tmp/root_device_terminal_20261104.py','/tmp/root_snapshot_terminal_v2_20261104.py'):
    copy(script,Path('root-review/scripts')/Path(script).name)
authored=['README.md','verify.py','build_capsule.py']
assert all((OUT/n).is_file() for n in authored)
origins=dict(schema='device-public-capsule-origins-v1',copied=copied,authored=authored,
    scope='Public source, synthetic recording policy/state results, process receipts and reviews only; no model/key/CT/private proof bodies.',
    aliases='Only recorded alias metadata copied; no alias path followed, resolved or reproduced.',
    producer_sources_modified=False,tests_rerun=False,candidate_imported=False)
(OUT/'origins.json').write_bytes(encode(origins))
names=set(copied)|set(authored)|{'origins.json','whitelist.json','files.json'}
(OUT/'whitelist.json').write_bytes(encode(dict(schema='device-public-capsule-whitelist-v1',files=sorted(names))))
actual={str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()}
assert actual==names-{'files.json'}
assert not any(p.is_symlink() for p in OUT.rglob('*'))
index={n:sha((OUT/n).read_bytes()) for n in sorted(actual)}
(OUT/'files.json').write_bytes(encode(index))
print(json.dumps(dict(capsule=str(OUT),indexed_files=len(index),index_sha256=sha((OUT/'files.json').read_bytes()),
    total_bytes=sum((OUT/n).stat().st_size for n in names),copied_public_files=len(copied),regular_child_files=len(regular),recorded_unfollowed_aliases=len(aliases)),sort_keys=True))
