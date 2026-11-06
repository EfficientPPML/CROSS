"""Offline source-bound ideal proof verifier; no model/crypto inputs."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(base,*,current=False):
    r=json.loads((base/'report.json').read_bytes())
    assert all(r[n]is True for n in ('passed','cleanup_completed','terminal_scope_completed','termination_scope_completed','source_files_unchanged','public_inputs_unchanged','predecessor_verified'))
    assert r['test_exit_code']==r['plan_exit_code']==0
    assert len(r['children'])==2 and all(row['direct_returncode']==0 and all(row[n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')) for row in r['children'])
    assert r['source_hashes_before']==r['source_hashes_after'] and r['public_inputs_before']==r['public_inputs_after']
    assert sha(base/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(base/'sources.tar.gz')as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==len(r['source_hashes_before']) and all(m.isfile() for m in members)
        raw={m.name:t.extractfile(m).read() for m in members}
        assert {n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}==r['source_hashes_before']
    inventory=json.loads(raw['silu_inventory.json']);assert inventory['expected_tests']==181
    expected=dict(tests=181,errors=0,failures=0,skipped=0);suites=ET.parse(base/'tests.xml').getroot().findall('testsuite')
    assert {n:sum(int(s.get(n,0)) for s in suites) for n in expected}==r['tests']==expected
    assert sha(base/'ideal_results.json')==r['ideal_result_sha256']
    metrics=json.loads((base/'ideal_results.json').read_bytes());names={f'layer{layer}.rows{rows}' for layer in range(28) for rows in (1,16)}|{f'layer{layer}.partial3' for layer in (0,3,26,27)}
    assert set(metrics)==names
    for row in metrics.values():
        assert row['max_abs_error']<=row['gate']==1e-8 and row['physical_slots_checked']==65536 and row['output_level']==6
        assert all(d['domain'][0]<=d['minimum']<=d['maximum']<=d['domain'][1] for d in row['domains'])
    plan=json.loads((base/'public_plan.json').read_bytes())
    assert plan['passed']is True and len(plan['sites'])==28 and plan['polynomial_sites']==6 and plan['rational_sites']==22 and plan['total_refreshes']==145 and plan['minimum_output_level']==6
    assert sum(len(p['events']) for p in plan['sites'])==145
    if current:
        for name,h in r['source_hashes_before'].items():assert sha(P/name)==h,name
        from collect_silu import verify_inputs,public_inputs
        verify_inputs();assert public_inputs()==r['public_inputs_after']
        from plan_public import derive
        assert json.loads(json.dumps(derive(),allow_nan=False))==plan
    print(f"PASS:{len(r['source_hashes_before'])} exactsources/181tests/60fullslot idealfixtures/all28 frozenrules; noHE/model claim.")
if __name__=='__main__':verify(P/'source-bound-owner-v1',current=True)
