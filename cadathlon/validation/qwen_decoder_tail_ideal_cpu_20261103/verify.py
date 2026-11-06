"""Portable public source/result verification; no external assets or backend."""
from pathlib import Path
import hashlib,json,tarfile
from verify_result import verify
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=verify(P/'result-v1')
receipt=json.loads((P/'result-v1/terminal_readback.json').read_bytes())
assert receipt['session']==51690 and receipt['exit_code']==0 and receipt['reaped'] is True
assert all(receipt[n] is True for n in ('producer_pids_absent','owned_groups_absent','current54_equal_start_end_archive'))
assert receipt['report_sha256']==sha(P/'result-v1/report.json')=='6a6da2d33af6454568d6fd7a6a12394ab559c2bb87bf7b61e08b1fdb3e9848f6'
assert sha(P/'encrypted_qwen_decoder_tail.py')==r['source_hashes_before']['checkout/cadathlon/common/encrypted_qwen_decoder_tail.py']=='e6ca1701dd90633476f1b754b707366fa580e7a127263e134670c6746b623d66'
history=json.loads((P/'history_inventory.json').read_bytes())
with tarfile.open(P/'history.tar.gz') as t:
    ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(history) and all(m.isfile() for m in ms)
    assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==history
inventory=json.loads((P/'artifact_hashes.json').read_bytes())
assert set(inventory)=={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
for n,h in inventory.items():assert sha(P/n)==h,n
print('Portable tail archive verified, including exact runtime, terminal receipts and preserved failures; no HE claim.')
