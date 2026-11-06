"""Verify complete ideal-source/fixture provenance without decrypting anything."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(base,*,current=False):
    r=json.loads((base/'report.json').read_bytes())
    assert r['passed']is True and r['cleanup_completed']is True and r['terminal_scope_completed']is True and r['termination_scope_completed']is True
    assert r['test_exit_code']==r['plan_exit_code']==0 and r['predecessor_verified']is True
    assert all(row['direct_child_reaped']is True and row['direct_returncode']==0 and row['group_empty_verified']is True and row['cleanup_completed']is True for row in r['children'])
    assert r['source_hashes_before']==r['source_hashes_after'] and r['public_arrays_before']==r['public_arrays_after']
    assert r['source_files_unchanged']is True and r['public_arrays_unchanged']is True
    assert sha(base/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(base/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(r['source_hashes_before']) and all(m.isfile() for m in ms)
        raw={m.name:t.extractfile(m).read() for m in ms}
        assert {n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}==r['source_hashes_before']
    inventory=json.loads(raw['join_inventory.json']);assert inventory['expected_tests']==144
    expected=dict(tests=144,errors=0,failures=0,skipped=0)
    suites=ET.parse(base/'tests.xml').getroot().findall('testsuite')
    assert {n:sum(int(s.get(n,0)) for s in suites) for n in expected}==r['tests']==expected
    manifest=json.loads(raw['qkv-export-v1/parameter_manifest.json'])
    assert len(manifest['matrices'])==84
    assert {row['file']:row['file_sha256'] for row in manifest['matrices'].values()}=={k:v for k,v in r['public_arrays_before'].items() if k!='gamma_npz'}
    plan=json.loads((base/'join_actual_inventory.json').read_bytes())
    assert len(plan['layers'])==28 and plan['centered_domain_implications_unproved']==868
    assert plan['linear_key_union']==[1,64,62528,63552]
    for layer in plan['layers']:
        for kind,row in layer['matrices'].items():
            descriptor=manifest['matrices'][f"model.layers.{layer['layer']}.self_attn.{kind}_proj.weight"]
            assert row['array_sha256']==descriptor['sha256'] and row['shape']==descriptor['shape']
    assert sum(event['stage'].endswith('_reserve') for layer in plan['layers'] for event in layer['outer_events'])==9
    assert sha(base/'ideal_results.json')==r['ideal_result_sha256']
    metrics=json.loads((base/'ideal_results.json').read_bytes())
    assert set(metrics)=={'layer0.S1','layer0.S8','layer0.S16','layer0.S19','layer0.S32','layer5.S1','layer9.S1'}
    for row in metrics.values():
        assert row['max_abs_error']<=row['gate']==1e-8 and all(d['passed']is True for d in row['domains'])
        assert row['physical_slots_checked']==((row['sequence']+15)//16)*65536
    if current:
        for name,h in r['source_hashes_before'].items():assert sha(P/name)==h,name
        from collect_join import verify_inputs,public_arrays
        verify_inputs();assert public_arrays()==r['public_arrays_after']
    print(f"PASS: {len(r['source_hashes_before'])} exact sources, 112 component + 32 lifecycle checks and 84 public matrix bindings; no HE/model claim.")
if __name__=='__main__':verify(P/'source-bound-owner-v1',current=True)
