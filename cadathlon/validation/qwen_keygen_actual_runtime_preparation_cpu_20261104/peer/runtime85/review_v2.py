from pathlib import Path
import json,hashlib,io,tarfile
old=Path('/tmp/qwen-keygen-actual-runtime-20261104');new=Path('/tmp/qwen-keygen-actual-runtime-v2-20261104');handoff=Path('/tmp/qwen-keygen-actual-runtime-v2-handoff-20261104');out=Path(__file__).parent
sha=lambda raw:hashlib.sha256(raw).hexdigest();canonical=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
a=json.loads(Path('/tmp/qwen-keygen-actual-runtime-handoff-20261104/sources.json').read_bytes());raw=(handoff/'sources.json').read_bytes();b=json.loads(raw)
assert sha(raw)=='28782d1927a4334c778dc8429d92637776bfe6a97a6a42e0d25f99cc82f35699'and canonical(b)==raw
assert len(a)==len(b)==85 and set(a)==set(b)
changed=[n for n in a if a[n]!=b[n]];assert set(changed)=={'keygen_entry_status.json','keygen_static_policy.json'}
for n in a:
 assert sha((old/n).read_bytes())==a[n]and sha((new/n).read_bytes())==b[n]
for n in changed:
 before=(old/n).read_bytes();after=(new/n).read_bytes();assert after==canonical(json.loads(before))and before!=after
archive=(handoff/'sources.tar.gz').read_bytes()
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
 members=tar.getmembers();assert len(members)==85 and all(m.isfile()for m in members)
 assert {m.name:sha(tar.extractfile(m).read())for m in members}==b
review=dict(schema='qwen-keygen-runtime85-encoding-successor-source-review-v1',passed=True,scope='Source/byte comparison only; no candidate import, parser execution, tests, capacity or bodies.',source_root=str(new),source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive),exact85=True,changes={n:dict(before=a[n],after=b[n],bytes=(new/n).stat().st_size)for n in changed},runtime4_and_other83_exact=True,semantic_JSON_objects_unchanged=True,original85_preserved=True,prior_finding_resolved_for_source_encoding=True,actual_entry_ready=False,prior_receipt_sha256=sha((out/'review.json').read_bytes()),limitations=['No runtime/preflight/import claim; all other pending role/dependency/resource admission obligations remain.'])
(out/'review-v2.json').write_text(json.dumps(review,sort_keys=True,indent=2)+'\n');print(sha((out/'review-v2.json').read_bytes()))
