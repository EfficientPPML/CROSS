"""Offline source/provenance/public-plan verification, no gamma NPZ required."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'source-bound-v1/report.json').read_bytes())
assert r['passed'] is True and r['cleanup_completed'] is True and r['terminal_scope_completed'] is True
assert r['tests']==dict(tests=295,failures=0,errors=0,skipped=0) and r['gamma_tests']==dict(tests=6,failures=0,errors=0,skipped=0)
assert r['test_exit_code']==r['gamma_test_exit_code']==r['plan_exit_code']==0
assert all(x['process_reaped'] and x['exit_code']==0 for x in r['children'])
assert r['source_hashes_before']==r['source_hashes_after'] and r['source_archive_verified'] is True
assert sha(P/'source-bound-v1/sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(P/'source-bound-v1/sources.tar.gz') as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==len(r['source_hashes_before']) and all(m.isfile() for m in members)
    assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
for n,h in r['source_hashes_before'].items():assert sha(P/n)==h,n
gamma=json.loads((P/'gamma-preparation-v1/report.json').read_bytes())
assert gamma['passed'] and gamma['source_hashes_before']==gamma['source_hashes_after'] and gamma['asset_sha256_before']==gamma['asset_sha256_after']
assert len(gamma['arrays'])==113 and r['external_fixture_sha256_before']==r['external_fixture_sha256_after']==gamma['fixture_npz_sha256']
with tarfile.open(P/'gamma-preparation-v1/sources.tar.gz') as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==len(gamma['source_hashes_before'])
    assert all(m.isfile() for m in members)
    assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==gamma['source_hashes_before']
import plan_public
assert json.loads(json.dumps(plan_public.derive()))==json.loads((P/'source-bound-v1/public_plan.json').read_bytes())
print('PASS: all113 RMSNorm ideal circuits, exact public sources/scales/parameter identities; no HE or model claim.')
