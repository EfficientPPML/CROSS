"""Independent offline checks of the public schedule and source archive."""
from pathlib import Path
import hashlib,json,math,tarfile
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'report.json').read_text())
assert r['passed'] and not r['forbidden_calls']
assert r['source_hashes_before']==r['source_hashes_after']
with tarfile.open(ROOT/'sources.tar.gz') as t:
    assert len(t.getmembers())==len(r['source_hashes_before'])==108
    for m in t.getmembers():
        assert m.isfile() and not m.issym() and not m.islnk()
        assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==r['source_hashes_before'][m.name]
assert hashlib.sha256((ROOT/'sources.tar.gz').read_bytes()).hexdigest()==r['source_archive_sha256']
assert len(r['profiles'])==6 and r['profiles'][-1]['accepted'] is False
for p in r['profiles'][:-1]:
    assert p['qp_is_secure'] and p['P_covers_each_partition'] and math.prod(p['q']+p['p']).bit_length()<=3523
    assert p['prospective_chained_resources']['total_graph_bootstrap_keys']==116
    assert p['prospective_chained_resources']['raw_key_bytes']==116*p['resources']['rotation_pair_bytes']
    for row in p['probability_only_schedules']:
        if not row['feasible']:continue
        for rnd,n in zip(row['normalized_rounds'],row['iterations']):
            events=rnd['events'];assert all(e['op']!='refresh_denominator' for e in events)
            inverse=next(e for e in events if e['op']=='static_C32_inverse')
            assert inverse['input_level']-inverse['output_level']==2*n-1
            product=next(e for e in events if e['op']=='correlated_numerator_times_z')
            assert min(product['input_levels'])-1==product['output_level']>=2
            assert abs(rnd['product_relative_canonical_difference'])<=1e-12
            for e in events:
                if e['op'].startswith('refresh'):assert e['input_level']>=2
        assert row['block_output_level']>=2
assert r['profiles'][0]['summary']['feasible_K']==list(range(1,75))
assert all(len(p['summary']['feasible_K'])==128 for p in r['profiles'][1:-1])
assert all(p['summary']['K128']['normalization_refreshes']==4 for p in r['profiles'][1:-1])
assert r['profiles'][1]['summary']['K128']['normalization_output_level']==2
assert all(p['summary']['K128']['normalization_output_level']==3 for p in r['profiles'][2:-1])
print(json.dumps({'passed':True,'public_profiles':6,'accepted_profiles':5,'source_files':108,'schedule_rows':640,'encrypted_execution':False},sort_keys=True))
