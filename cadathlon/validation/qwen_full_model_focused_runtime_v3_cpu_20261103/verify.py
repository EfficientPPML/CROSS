from pathlib import Path
import hashlib,json,tarfile
from verify_result import verify
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=verify(P/'focused-v3')
assert sha(P/'focused-v3/report.json')=='76502ea6f499263fca0604ce8620b664802980df628f5696c384ca05260e104c'
receipt=json.loads((P/'focused-v3/terminal_readback.json').read_bytes())
assert receipt['session']==9542 and receipt['exit_code']==0 and receipt['reaped']is True
assert all(receipt[n]is True for n in ('child_pids_absent','child_groups_absent','current87_equal_start_end_archive'))
assert r['source_hashes_before']['checkout/cadathlon/common/encrypted_qwen_full_model.py']=='f4e8f93a170cd8c9c7570f23eb1db4159c7d6da315560e34691eb5986520ff17'
negative=json.loads((P/'preserved-negatives/report.json').read_bytes())
assert negative['exit_code']==1 and negative['expected_failures']==6 and negative['reaped']is True and negative['before']==negative['after']
with tarfile.open(P/'preserved-negatives/negative-sources.tar.gz') as t:
 members=t.getmembers();assert len(members)==len({m.name for m in members})==86
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members if m.isfile()}==negative['before']
files=json.loads((P/'artifact_hashes.json').read_bytes())
assert set(files)=={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
assert all(sha(P/n)==h for n,h in files.items())
print('Verified87/150 focused successor and6 preserved negatives; no fullmodel/HE numerical claim.')
