"""Verify public tail evidence without importing arithmetic/backends/assets."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def verify(root):
    r=json.loads((root/'report.json').read_bytes());inventory=json.loads((root/'tail_inventory.json').read_bytes())
    assert r['passed'] is True and r['encrypted_execution'] is False and r['backend_imported'] is False
    assert all(r[n] is True for n in ('predecessor_verified','source_files_unchanged','public_arrays_unchanged','cleanup_completed','terminal_scope_completed','termination_scope_completed'))
    assert r['source_hashes_before']==r['source_hashes_after'] and set(r['source_hashes_before'])==set(inventory['sources'])
    assert r['public_arrays_before']==r['public_arrays_after'] and len(r['public_arrays_before'])==197
    assert len(r['children'])==2 and r['test_exit_code']==r['plan_exit_code']==0
    assert all(c['direct_returncode']==0 and all(c[n] is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')) for c in r['children'])
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(root/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(root/'sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==len(inventory['sources'])
        assert all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    tests={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','errors','failures','skipped')}
    assert tests==r['tests']==dict(tests=inventory['expected_tests'],errors=0,failures=0,skipped=0)
    metrics=json.loads((root/'ideal_results.json').read_bytes());assert sha(root/'ideal_results.json')==r['ideal_result_sha256']
    expected={'layer0.S1.joined','layer0.S19.joined','layer3.S1.standin','layer3.S19.standin','layer15.S1.standin','layer27.S1.standin'}
    assert set(metrics)==expected and sum(v['physical_slots'] for v in metrics.values())==524288
    for name,v in metrics.items():
        assert math.isfinite(v['max_abs_error']) and 0<=v['max_abs_error']<=v['gate']==1e-8
        assert all(d['passed'] is True and all(math.isfinite(d[n]) for n in ('lo','hi')) and d['domain'][0]<=d['lo']<=d['hi']<=d['domain'][1] for d in v['domains'])
        assert all(x['level']==4 and math.isfinite(x['scale']) and x['scale']>0 for x in v['metadata'])
        tiles=(v['sequence']+15)//16
        base={0:(501,257),3:(312,141),15:(308,129),27:(836,633)}[v['layer']]
        assert v['counts']==dict(weighted=base[0]*tiles,multiply=base[1]*tiles,rotate=508*tiles,add=258*tiles)
        assert len(v['outer_refreshes'])==2*tiles
    public=json.loads((root/'public_plan.json').read_bytes())
    assert public['passed'] is True and len(public['layers'])==28 and len(public['examples'])==168 and public['output_level']==4
    for example in public['examples']:
        stages={s['name']:s for s in example['stages']}
        assert stages['original_residual_mask']['level']==24 and example['output']['level']==4
        assert stages['attention_residual']['scale']==example['output']['scale']==stages['original_residual_mask']['scale']
    print('VERIFIED:193 ideal/lifecycle tests, six learned-weight fixtures, 524288 slots; no HE/model-forward claim.')
    return r
if __name__=='__main__':verify(P/'source-bound-owner-v1')
