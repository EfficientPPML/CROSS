"""Offline hash/metadata verification; never loads a model, payload or backend."""
from pathlib import Path
import hashlib,json,tarfile
root=Path(__file__).resolve().parent
inventory=json.loads((root/'artifact_sha256.json').read_text())
for name,sha in inventory.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
r=json.loads((root/'source-bound-v1/report.json').read_text());s=json.loads((root/'summary.json').read_text())
assert r['passed'] and s['passed'] and r['tests']==s['tests']==71 and r['exit_code']==0 and r['child_reaped']
assert r['source_before']==r['source_after'] and len(r['source_before'])==9
assert r['owned_workspace_removed'] and r['owned_handle_closed'] and not r['cleanup_errors']
archive=root/'source-bound-v1/sources.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['source_archive_sha256']==s['source_archive_sha256']
with tarfile.open(archive) as tf:
 members=tf.getmembers();assert len(members)==9==len({m.name for m in members}) and all(m.isfile() for m in members)
 assert {m.name:hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in members}==r['source_before']
for name in s['api_modules']:assert hashlib.sha256((root/'sources'/name).read_bytes()).hexdigest()==s['api_modules'][name]==r['source_after'][name]
assert not s['raw_payloads_archived'] and not r['actual_ciphertexts_tested']
print(json.dumps(dict(passed=True,tests=71,sources=9,scope='Offline serializer evidence consistency only; no HE.')))
