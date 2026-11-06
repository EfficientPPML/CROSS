from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'source-bound-v1/report.json').read_bytes());assert r['passed'] and r['test_exit_code']==r['plan_exit_code']==0
assert r['source_hashes_before']==r['source_hashes_after'] and not r['encrypted_execution'] and not r['backend_imported']
assert sha(P/'source-bound-v1/sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(P/'source-bound-v1/sources.tar.gz') as t:
    ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(r['source_hashes_before']) and all(m.isfile() for m in ms)
    assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']
for n,h in r['source_hashes_before'].items():assert sha(P/n)==h,n
import plan_public
assert plan_public.derive()==json.loads((P/'source-bound-v1/public_plan.json').read_bytes())
print('PASS: exact Qwen head-owner source/ideal tests/public key and scale plan; no HE claim.')
