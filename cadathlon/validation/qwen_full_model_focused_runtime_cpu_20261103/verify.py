from pathlib import Path
import hashlib,json,tarfile
from verify_v2 import verify
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=verify(P/'focused-v2')
assert sha(P/'focused-v2/report.json')=='d0827e28deb4832be28ec2d81298c59b832aab9f127085f86a4e2efa2c5f360c'
t=json.loads((P/'focused-v2/terminal_readback.json').read_bytes())
assert t['session']==58804 and t['exit_code']==0 and t['reaped']is True
assert all(t[n]is True for n in ('producer_pids_absent','owned_groups_absent','current84_equal_start_end_archive'))
with tarfile.open(P/'history.tar.gz') as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})
 actual={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms if m.isfile()}
assert actual==json.loads((P/'history_inventory.json').read_bytes())
files=json.loads((P/'artifact_hashes.json').read_bytes())
assert set(files)=={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
assert all(sha(P/n)==h for n,h in files.items())
print('Focused84/144 preserved; six successor negatives preserved; no fullmodel/HE trajectory claim.')
