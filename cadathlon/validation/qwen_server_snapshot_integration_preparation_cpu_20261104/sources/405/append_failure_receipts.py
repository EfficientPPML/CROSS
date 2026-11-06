"""Append independently admitted failure provenance before final source hold."""
import gzip,hashlib,io,json,shutil,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent;N=Path('/tmp/qwen-server-snapshot-integration-tests-v2-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
def canonical(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
before=json.loads((H/'sources.json').read_bytes());handoff=json.loads((H/'handoff.json').read_bytes())
assert len(before)==405 and sha((H/'sources.json').read_bytes())=='701d3911a7c7542c60ec304e8674d34c1da216f8d924de82a9cc955befff4b3e'
for name,digest in before.items():assert sha((N/name).read_bytes())==digest
saved=H/'before-root-failure-receipts';assert not saved.exists();saved.mkdir()
for name in('sources.json','sources.tar.gz','handoff.json','source.patch'):
    shutil.copyfile(H/name,saved/name)
shutil.copyfile(N/'fixture_successor_history.json',saved/'fixture_successor_history.json')
root=Path('/tmp/qwen-server-snapshot-integration-root-test-review-20261104')
receipt=root/'failed-terminal-v1.json';owner=root/'direct-owner-failed-v1.json'
assert sha(receipt.read_bytes())=='5505f782e2850765d27940190332fb0c8596622207c406832abd16df75ff0c49'
history=json.loads((N/'fixture_successor_history.json').read_bytes())
history['independent_root_failed_proof_receipts']={str(p):sha(p.read_bytes())for p in(receipt,owner)}
history['pre_receipt_source_only_checkpoint']=dict(source_map_sha256=sha((saved/'sources.json').read_bytes()),source_archive_sha256=sha((saved/'sources.tar.gz').read_bytes()),history_sha256=sha((saved/'fixture_successor_history.json').read_bytes()),change='Only root failure provenance appended; callback fix/assertions/runtime unchanged; no execution of either405 checkpoint.')
(N/'fixture_successor_history.json').write_bytes(canonical(history))
raw={}
for p in sorted(N.rglob('*')):
    assert not p.is_symlink()
    if p.is_file():raw[p.relative_to(N).as_posix()]=p.read_bytes()
current={n:sha(b)for n,b in raw.items()};assert len(current)==405
assert[n for n in before if before[n]!=current[n]]==['fixture_successor_history.json']
(H/'sources.json').write_bytes(canonical(current))
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT)as t:
    for name,b in raw.items():
        m=tarfile.TarInfo(name);m.size=len(b);m.mode=0o644;m.mtime=0;m.uid=m.gid=0;m.uname=m.gname='';t.addfile(m,io.BytesIO(b))
(H/'sources.tar.gz').write_bytes(gzip.compress(buf.getvalue(),mtime=0))
handoff.update(source_manifest_sha256=sha((H/'sources.json').read_bytes()),source_archive_sha256=sha((H/'sources.tar.gz').read_bytes()),
    source_archive_bytes=(H/'sources.tar.gz').stat().st_size,history_sha256=current['fixture_successor_history.json'],
    root_failed_receipts=history['independent_root_failed_proof_receipts'],pre_receipt_checkpoint_preserved=True,
    receipt_append_script_sha256=sha(Path(__file__).read_bytes()))
(H/'handoff.json').write_bytes(canonical(handoff))
print(json.dumps(dict(handoff_sha256=sha((H/'handoff.json').read_bytes()),source_manifest_sha256=handoff['source_manifest_sha256'],source_archive_sha256=handoff['source_archive_sha256'],fixture_sha256=current['integration_fixture.py'],history_sha256=current['fixture_successor_history.json'],source_count=405,runtime_unchanged=True,tests_unrun=True),sort_keys=True))
