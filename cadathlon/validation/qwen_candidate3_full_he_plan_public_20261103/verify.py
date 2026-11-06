"""Offline readback; execute only public copied source."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'public-v1/report.json').read_bytes());assert r['passed'] and r['test_exit_code']==r['plan_exit_code']==0
assert r['source_hashes_before']==r['source_hashes_after']
assert sha(P/'public-v1/sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(P/'public-v1/sources.tar.gz') as t:
    ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(r['source_hashes_before']) and all(m.isfile() for m in ms)
    assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']
for n,h in r['source_hashes_before'].items():assert sha(P/n)==h,n
import plan
assert plan.run()==json.loads((P/'public-v1/plan.json').read_bytes())
assert not any(r[k] for k in ('model_executed','encrypted_execution','validation_outcomes_read'))
print('PASS: exact public sources, all113 schedules, conditional bounds and dense-shape resources.')
