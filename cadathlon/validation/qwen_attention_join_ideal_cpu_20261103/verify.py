"""Offline public evidence verification; no models, tensors or crypto backend."""
from pathlib import Path
import hashlib,json
from verify_join import verify
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'source-bound-owner-v1/report.json').read_bytes())
verify(P/'source-bound-owner-v1',current=False)
for name,h in r['source_hashes_before'].items():assert sha(P/name)==h,name
receipt=json.loads((P/'terminal_receipt.json').read_bytes())
assert receipt['exit_code']==0 and receipt['reaped']is True and receipt['all_three_pids_absent']is True
for name,row in receipt['helper_provenance'].items():assert sha(P/name)==row['sha256']
assert (P/'cleanup.py').read_bytes()==(P/'validation_cleanup.py').read_bytes()
artifacts=json.loads((P/'artifact_hashes.json').read_bytes())
assert set(artifacts)=={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
for name,h in artifacts.items():assert sha(P/name)==h,name
print('PASS: portable52-source/current archive, all144checks, seven589824-slot aggregate fixtures and recorded terminal receipts; no tensor bodies/backend.')
