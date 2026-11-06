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
    assert not dest.is_absolute()and '..'not in dest.parts
    assert src.is_file()and not src.is_symlink()
    before=src.lstat();assert stat.S_ISREG(before.st_mode)
    raw=src.read_bytes();after=src.lstat()
    assert(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    digest=sha(raw)
    if expected is not None:assert digest==expected,(str(src),'digest differs')
    target=OUT/dest;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb')as stream:stream.write(raw)
    copied[dest.as_posix()]=dict(path=str(src),sha256=digest)
def group(root,dest,names):
    for name in names:copy(Path(root)/name,Path(dest)/name)

groups=[
('/tmp/qwen-server-admission-snapshot-handoff-20261104','history/guard-v1',
 ['sources.json','sources.tar.gz','handoff.json','capture_source.py','authoring-history.json','unheld-syntax-draft.py.txt']),
('/tmp/qwen-server-admission-snapshot-handoff-v2-20261104','history/guard-v2',
 ['sources.json','sources.tar.gz','handoff.json','build_source.py','guard.patch']),
('/tmp/qwen-server-admission-snapshot-test-handoff-20261104','history/tests-v1',
 ['sources.json','sources.tar.gz','handoff.json']),
('/tmp/qwen-server-admission-snapshot-test-handoff-v2-20261104','history/tests-v2',
 ['sources.json','sources.tar.gz','handoff.json','parent-pgid.patch','pre-release-sources.json','pre-release-sources.tar.gz','pre-release-handoff.json','pre-release-successor_history.json']),
('/tmp/qwen-server-admission-snapshot-test-handoff-v3-20261104','history/tests-v3',
 ['sources.json','sources.tar.gz','handoff.json','author_source.py','source.patch']),
('/tmp/qwen-admission-snapshot-transformers-review-20261104','peer-review/guard',
 ['review_source.py','review.json','review_successor.py','review-v2.json']),
('/tmp/qwen-snapshot-test-cnn-review-20261104','peer-review/test-v1', ['review_source.py','review.json']),
('/tmp/qwen-snapshot-harness-security-review-20261104','peer-review/harness-v1', ['review_source.py','review.json']),
('/tmp/qwen-snapshot-v3-security-source-review-20261104','peer-review/test-v3', ['review_source.py','review.json']),
('/tmp/qwen-snapshot-v2-harness-security-review-20261104','peer-review/harness-v2', ['review_source.py','review.json']),
('/tmp/qwen-snapshot-pass-terminal-security-review-20261104','peer-review/passed-terminal', ['review_public.py','review.json','process-absence.json']),
('/tmp/qwen-server-admission-snapshot-root-review-20261104','root-review/failed',
 ['source-review.json','preexecution-review.json','run-grant.json','root-process-observation.json','failed-terminal-review.json']),
('/tmp/qwen-server-admission-snapshot-root-review-v2-20261104','root-review/passed',
 ['source-review.json','preexecution-review.json','run-grant.json','direct-owner.json','root-process-observation.json','terminal-review.json']),
]
for root,dest,names in groups:group(root,dest,names)
for label,root,hand in [
    ('guard-v1','/tmp/qwen-server-admission-snapshot-source-20261104','/tmp/qwen-server-admission-snapshot-handoff-20261104'),
    ('guard-v2','/tmp/qwen-server-admission-snapshot-source-v2-20261104','/tmp/qwen-server-admission-snapshot-handoff-v2-20261104')]:
    mapping=json.loads((Path(hand)/'sources.json').read_bytes())
    for name,d in mapping.items():copy(Path(root)/name,Path('history')/label/'source'/name,d)
for version,root in [('v1','/tmp/qwen-server-admission-snapshot-tests-20261104'),('v2','/tmp/qwen-server-admission-snapshot-tests-v2-20261104')]:
    group(root,'history/tests-'+version+'/readable',['snapshot_child.py','test_snapshot_guard.py','TEST_SCOPE.md','selected_tests.json'])
group('/tmp/qwen-server-admission-snapshot-test-plan-20261104','history/test-plan',['PLAN.md'])
source=Path('/tmp/qwen-server-admission-snapshot-tests-v3-20261104')
source_map=json.loads((OUT/'history/tests-v3/sources.json').read_bytes());assert len(source_map)==38
for name,d in source_map.items():copy(source/name,Path('source')/name,d)

failed=Path('/tmp/qwen-snapshot-failed-run-retention-20261104')
failed_map=json.loads((failed/'files.json').read_bytes());assert len(failed_map)==44
copy(failed/'files.json','failed/files.json','cea9cc7683be38a068ad2008aef8fac4c7a113cb95045055b11ba4066586fdbb')
for name,d in failed_map.items():copy(failed/name,Path('failed')/name,d)

proof=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v2')
group(proof,'measured',[
 'README.md','prepare_source.py','before_executed_driver.py','before_worker_driver.py',
 'driver_provenance.json','executed_driver.py','executed_driver.py.patch','worker_driver.py','worker_driver.py.patch',
 'run_config.json','source-handoff.json','handoff.json','launch-proposal.json','sources-before.json','sources-after.json',
 'execution-after.json','sources.tar.gz','live-observation.json','tests.log','tests.xml','worker-audit.json',
 'owned-pytest-metadata.json','test-child-terminals.json','report.json'])
metadata=json.loads((proof/'owned-pytest-metadata.json').read_bytes())
regular={n:v for n,v in metadata.items()if v['kind']=='regular'}
aliases={n:v for n,v in metadata.items()if v['kind']=='owned_pytest_alias'}
assert len(regular)==72 and len(aliases)==17 and len(metadata)==89
for name,row in regular.items():
    assert name.startswith('pytest-owned/')and Path(name).name in('parent-terminal.json','child-terminal.json','result.json','child.log')
    assert '..'not in Path(name).parts and len((proof/name).read_bytes())==row['bytes']
    copy(proof/name,Path('measured/owned')/name,row['sha256'])
# Alias paths are neither read, resolved nor copied; their observed metadata is
# already retained in the measured manifest above.

scripts=[
 '/tmp/root_snapshot_test_source_review_20261104.py',
 '/tmp/root_snapshot_test_source_review_20261104.before_spacing_fix.py',
 '/tmp/root_snapshot_preexecution_20261104.py',
 '/tmp/root_snapshot_failed_terminal_20261104.py',
 '/tmp/root_snapshot_preexecution_v2_20261104.py',
 '/tmp/root_snapshot_terminal_v2_20261104.py',
 '/tmp/build_qwen_admission_snapshot_v2_20261104.py',
]
for script in scripts:copy(script,Path('root-review/scripts')/Path(script).name)
authored=['README.md','verify.py','build_capsule.py']
assert all((OUT/n).is_file()for n in authored)
origins=dict(schema='snapshot-public-capsule-origins-v1',copied=copied,authored=authored,
    scope='Public source, synthetic recording-state results, process receipts and review evidence only; no arrays, model/key/CT/private proof bodies.',
    aliases='Only observed alias metadata copied; no alias path followed, resolved or reproduced.',
    producer_sources_modified=False,tests_rerun=False,candidate_imported=False)
(OUT/'origins.json').write_bytes(encode(origins))
names=set(copied)|set(authored)|{'origins.json','whitelist.json','files.json'}
(OUT/'whitelist.json').write_bytes(encode(dict(schema='snapshot-public-capsule-whitelist-v1',files=sorted(names))))
actual={str(p.relative_to(OUT))for p in OUT.rglob('*')if p.is_file()}
assert actual==names-{'files.json'}
assert not any(p.is_symlink()for p in OUT.rglob('*'))
index={n:sha((OUT/n).read_bytes())for n in sorted(actual)}
(OUT/'files.json').write_bytes(encode(index))
print(json.dumps(dict(capsule=str(OUT),indexed_files=len(index),index_sha256=sha((OUT/'files.json').read_bytes()),
    total_bytes=sum((OUT/n).stat().st_size for n in names),copied_public_files=len(copied),new_regular_child_files=len(regular),recorded_unfollowed_aliases=len(aliases)),sort_keys=True))
