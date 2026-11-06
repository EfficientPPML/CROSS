"""Portable public proof verification, without assets or model execution."""
from pathlib import Path
import hashlib,json
from verify_result import verify
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=verify(P/'result-v1')
receipt=json.loads((P/'result-v1/terminal_readback.json').read_bytes())
assert receipt['session']==47717 and receipt['exit_code']==0 and receipt['reaped'] is True
assert all(receipt[n] is True for n in ('producer_pids_absent','owned_groups_absent','current68_equal_start_end_archive'))
assert receipt['report_sha256']==sha(P/'result-v1/report.json')=='393d04e87681c80fdaefe3b8a138e706e1a3d001b9cfba94cbbdc00fe0e571bd'
for name in ('full_contract.py','plan_full.py'):
    assert sha(P/name)==r['source_hashes_before'][name]
inventory=json.loads((P/'artifact_hashes.json').read_bytes())
assert set(inventory)=={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
for name,h in inventory.items():assert sha(P/name)==h,name
print('Portable full28 public archive verified; wholegraph model/HE execution remains unproved.')
