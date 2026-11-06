from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'source-bound-owner-v1/report.json').read_bytes())
assert r['passed'] and r['cleanup_completed'] and r['terminal_scope_completed']
assert r['tests']==dict(tests=json.loads((P/'owner_inventory.json').read_bytes())['expected_tests'],errors=0,failures=0,skipped=0)
assert r['test_exit_code']==r['plan_exit_code']==0 and all(x['process_reaped'] and x['exit_code']==0 for x in r['children'])
assert r['source_hashes_before']==r['source_hashes_after']
assert sha(P/'source-bound-owner-v1/sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(P/'source-bound-owner-v1/sources.tar.gz') as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==len(r['source_hashes_before']) and all(m.isfile() for m in members)
    assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
for name,h in r['source_hashes_before'].items():assert sha(P/name)==h,name
from collect_owner import verify_inputs
verify_inputs();assert r['predecessor_verified'] is True
from plan_public import derive
expected=derive()
assert expected==json.loads((P/'source-bound-owner-v1/public_plan.json').read_bytes())
assert len(expected['all_layer_K_plans'])==896
assert expected['workloads'][-1]['total28_refreshes']==11656
print('PASS:exact owner/public source closure and all declared ideal checks; no model/backend/HE execution.')
