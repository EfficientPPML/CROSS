"""Copy a closed public whitelist; no candidate imports, tests or original body reads."""
from pathlib import Path
import hashlib,json,shutil
OUT=Path('/tmp/qwen-keygen-runtime-public-capsule-20261104')
HERE=Path(__file__).parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
rows=[]
def copy(source,destination):
    source=Path(source);target=OUT/destination
    assert source.is_file() and not source.is_symlink() and not target.exists()
    raw=source.read_bytes();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    assert target.read_bytes()==raw
    rows.append(dict(source=str(source),destination=destination,sha256=sha(raw),bytes=len(raw)))
OUT.mkdir(exist_ok=False)
for version,source in [('v1','/tmp/qwen-keygen-actual-runtime-handoff-20261104'),('v2','/tmp/qwen-keygen-actual-runtime-v2-handoff-20261104')]:
    names=['handoff.json','sources.json','sources.tar.gz']
    names+=['capture_source.py','keygen_entry.py.patch','keygen_output.py.patch'] if version=='v1' else ['correction.json','parent85-sources.json','keygen_entry_status.json.patch','keygen_static_policy.json.patch']
    for n in names:copy(Path(source)/n,'source85-'+version+'/'+n)
    runtime=Path('/tmp/qwen-keygen-actual-runtime'+('' if version=='v1' else '-v2')+'-20261104')
    for n in ['keygen_entry_status.json','keygen_static_policy.json','KEYGEN_RUNTIME.md','keygen_implementation_origin.json','keygen_selected_copies.json']:
        copy(runtime/n,'source85-'+version+'/'+n)
proof=Path('/tmp/qwen-keygen-actual-focused-proof-20261104/preparation-v1')
for n in ['sources.tar.gz','sources-after.json','execution-after.json','run_config.json','executed_driver.py','worker_driver.py',
    'executed_driver.py.patch','worker_driver.py.patch','driver_provenance.json','launch-proposal.json','handoff.json','source-handoff.json',
    'selected-tests.json','report.json','tests.xml','tests.log','worker-audit.json','live-observation.json']:
    copy(proof/n,'focused97/'+n)
copy(proof/'sources-before.json','focused97/sources.json')
copy('/tmp/qwen-keygen-actual-runtime-test-handoff-20261104/capture_source.py','focused97/capture_source.py')
copy('/tmp/qwen-keygen-actual-runtime-tests-20261104/TEST_SCOPE.md','focused97/TEST_SCOPE.md')
root=Path('/tmp/qwen-key-generation-actual-root-review-20261104')
for n in ['keygen85-source-review.json','keygen85-peer-metadata-finding.json','keygen85-v2-encoding-review.json',
    'keygen97-preexecution-review.json','keygen97-focused-run-grant.json','keygen97-terminal-review.json','keygen97-direct-owner-receipt.json']:
    copy(root/n,'root/'+n)
for n in ['root_keygen97_review_20261104.py','root_keygen97_terminal_20261104.py']:
    copy(Path('/tmp')/n,'root/'+n)
for source,target,names in [
    ('/tmp/qwen-keygen-runtime85-transformers-review-20261104','peer/runtime85',['review.json','review-v2.json','review_source.py','review_v2.py']),
    ('/tmp/qwen-keygen-runtime-focused-peer-review-20261104','peer/focused',['review.json','harness-review.json','review_source.py','review_harness.py']),
    ('/tmp/qwen-keygen-actual-focused-harness-build-20261104','author/harness',['build.py','review_source.py','review.json'])]:
    for n in names:copy(Path(source)/n,target+'/'+n)
for n in ['README.md','verify.py']:copy(HERE/n,n)
copy(HERE/'build.py','author/capsule_build.py')
(OUT/'public_whitelist.json').write_bytes(pretty(dict(schema='qwen-keygen-runtime-public-whitelist-v1',
    copied_files=rows,no_original_model_key_private_ciphertext_bodies=True,no_repo_edits=True,
    source_archives_only_code_and_public_metadata=True)))
index={p.relative_to(OUT).as_posix():sha(p.read_bytes())for p in sorted(OUT.rglob('*'))if p.is_file()}
(OUT/'index.json').write_bytes(pretty(index))
print(json.dumps(dict(capsule=str(OUT),copied_files=len(rows),indexed_files=len(index),total_files=len(index)+1,
    index_sha256=sha((OUT/'index.json').read_bytes()),whitelist_sha256=sha((OUT/'public_whitelist.json').read_bytes())),sort_keys=True))
