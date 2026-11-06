from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'source-bound-v1/report.json').read_bytes())
assert r['passed'] and r['cleanup_completed'] and r['terminal_scope_completed']
assert r['tests']==dict(tests=93,errors=0,failures=0,skipped=0)
assert r['test_exit_code']==r['plan_exit_code']==0 and all(x['process_reaped'] and x['exit_code']==0 for x in r['children'])
assert r['source_hashes_before']==r['source_hashes_after']
assert sha(P/'source-bound-v1/sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(P/'source-bound-v1/sources.tar.gz') as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==len(r['source_hashes_before']) and all(m.isfile() for m in members)
    assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
for name,h in r['source_hashes_before'].items():assert sha(P/name)==h,name
from certificate import derive
expected=derive((P/'inputs/candidate_manifest.json').read_bytes(),json.loads((P/'inputs/continuation_plan.json').read_bytes()))
assert expected==json.loads((P/'source-bound-v1/certificates.json').read_bytes())
assert expected['initial_inverse_refresh_event']['completed_updates']==6
assert len(expected['rational_silu'])==22 and sum(x['doublings'] for x in expected['rational_silu'])==145
print('PASS: frozen exact-real Qwen B2/B4 and existing larger inverse envelopes; no FP/CKKS/refresh-error claim.')
