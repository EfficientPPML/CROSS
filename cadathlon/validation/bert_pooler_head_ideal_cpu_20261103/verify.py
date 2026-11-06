"""Read-only artifact checks and independent pooler continuation accounting."""
from pathlib import Path
import hashlib,json,tarfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_tanh(plan):
    level=plan['input_level']-6;refreshes=[]
    for i,row in enumerate(plan['rounds'],1):
        if level<13:
            assert row['refresh_before'];refreshes.append(i);level=25
        else:assert not row['refresh_before']
        level-=11;assert row['round']==i and row['output_level']==level>=2
        forward=row['inverse_scale_plan'];assert len(forward['rows'])==5
        assert [v['level'] for v in forward['rows']]==list(range(level+9,level,-2))
    assert level==plan['output_level'] and [x['round'] for x in plan['refresh_events']]==refreshes
    return refreshes

def main():
    for n,h in json.loads((HERE/'artifact_sha256.json').read_text()).items():assert sha(HERE/n)==h,n
    d=HERE/'measured';r=json.loads((d/'report.json').read_text())
    assert r['passed'] and r['returncode']==0 and r['child_reaped'] and r['source_files_unchanged'] and r['snapshot_unchanged']
    assert not r['encrypted_execution'] and r['external_checkpoint_unchanged']
    assert r['external_checkpoint_sha256_before']==r['external_checkpoint_sha256_after']=='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
    assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==125
    assert sha(d/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(d/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==125 and all(m.isfile() for m in ms)
        contents={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(v).hexdigest() for n,v in contents.items()}==r['source_hashes_before']
    for name in ('encrypted_bert_rational_tanh.py','encrypted_bert_pooler_head.py'):assert sha(HERE/'source'/name)==r['source_hashes_before']['checkout/cadathlon/common/'+name]
    assert len(r['tanh_cases'])==12 and len(r['head_cases'])==8
    assert {x['level'] for x in r['tanh_cases']}=={8,11,25,49} and len({x['scale'] for x in r['tanh_cases']})==3
    for c in r['tanh_cases']:
        assert c['slots']==65536 and 0<=c['max_abs']<c['gate']==1e-11
        assert c['counts']['multiply']==68 and c['counts']['initial_affine']==6 and c['counts']['integer_affine']==36
        assert c['refreshes']==len(verify_tanh(c['plan']))
    for c in r['head_cases']:
        assert c['slots']==65536 and 0<=c['max_abs']<c['gate']==1e-10
        plan=c['plan'];assert plan['original_position']==0 and plan['classifier_width']==2
        assert [s['level'] for s in plan['stages']]==[11,3,2] and verify_tanh(plan['tanh'])==[1,3,5]
        assert len(c['positions']) in (1,16) and c['positions'][0]==0
    assert r['physical_slots_checked']==20*65536
    assert r['max_abs_vs_independent_tanh']==max(c['max_abs'] for c in r['tanh_cases'])
    assert r['max_abs_vs_independent_head']==max(c['max_abs'] for c in r['head_cases'])
    terminal=json.loads((HERE/'terminal.json').read_text());assert terminal['session_id']==1258 and terminal['observed_exit_code']==0 and terminal['reaped']
    assert '114 passed in 20.39s' in (d/'tests.log').read_text()
    print('Verified125 sources,114 ideal checks,20 recorded full-slot cases and six-round continuation; no HE/full-model claim.')
if __name__=='__main__':main()
