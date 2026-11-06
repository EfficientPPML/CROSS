"""Portable public evidence readback; no model/parameter/HE inputs."""
from pathlib import Path
import hashlib,json,tarfile
from verify_silu_v2 import verify
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
verify(P/'source-bound-owner-v1',current=True)
v=json.loads((P/'verification-v2/report.json').read_bytes())
assert v['passed']is True and v['tests']==3 and v['exit_code']==0 and v['reaped']is True
assert v['source_hashes_before']==v['source_hashes_after']
assert sha(P/'verification-v2/sources.tar.gz')==v['source_archive_sha256']
with tarfile.open(P/'verification-v2/sources.tar.gz')as t:
 members=t.getmembers();assert len(members)==len({m.name for m in members})==len(v['source_hashes_before']) and all(m.isfile() for m in members)
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==v['source_hashes_before']
for name,h in v['source_hashes_before'].items():assert sha(P/name)==h,name
receipt=json.loads((P/'terminal_receipt.json').read_bytes())
assert receipt['exit_code']==0 and receipt['reaped']is True and receipt['all_absent']is True
negative=json.loads((P/'negative-binding-v1/report.json').read_bytes())
assert negative['exit_code']==1 and negative['failures']==4 and negative['source_hashes_before']==negative['source_hashes_after']
assert sha(P/'negative-binding-v1/sources.tar.gz')==negative['source_archive_sha256']
with tarfile.open(P/'negative-binding-v1/sources.tar.gz')as t:
 members=t.getmembers();assert len(members)==len({m.name for m in members})==len(negative['source_hashes_before']) and all(m.isfile() for m in members)
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==negative['source_hashes_before']
artifacts=json.loads((P/'artifact_hashes.json').read_bytes())
assert set(artifacts)=={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
for name,h in artifacts.items():assert sha(P/name)==h,name
print('PASS:181 ideal checks/34 exact measured sources; separate3-check/36-source verifier successor; preserved4-negative capsule; no HE/model claim.')
