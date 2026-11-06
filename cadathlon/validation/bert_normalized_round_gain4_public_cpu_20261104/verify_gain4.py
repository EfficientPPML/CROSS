"""Portable source/ideal/encoder evidence check; no backend or payload reads."""
from pathlib import Path
import argparse,hashlib,json,math,tarfile,xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(root,current=None):
    root=Path(root);ev=root/'source-bound-v1';r=json.loads((ev/'report.json').read_bytes())
    for key in ('passed','complete','sources_unchanged','source_archive_verified','all_direct_children_reaped','all_owned_groups_empty','cleanup_completed','terminal_scope_completed','termination_scope_completed','plan_file_unchanged','metrics_file_unchanged'):assert r.get(key)is True,key
    assert not r.get('error')and not r.get('cleanup_errors')and r['he']is False and r['model_forward']is False
    pins=r['source_hashes_before'];assert pins==r['source_hashes_after']==r['execution_source_hashes_after']and len(pins)==r['source_count']
    assert sha(ev/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(ev/'sources.tar.gz')as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==len(pins)
        assert all(m.isfile()and 0<=m.size<=8<<20 for m in members)and sum(m.size for m in members)<=32<<20
        raw={m.name:t.extractfile(m).read()for m in members}
    assert {n:hashlib.sha256(b).hexdigest()for n,b in raw.items()}==pins
    if current:assert {n:sha(Path(current)/n)for n in pins}==pins
    predecessor=json.loads(raw['gain4_source_origin.json'])['predecessor_source_hashes'];assert len(predecessor)==114
    final=json.loads(raw['gain4_preparation_pins.json'])
    expected=predecessor|final['helpers']|final['candidate_files']
    assert all(hashlib.sha256(raw[n]).hexdigest()==h for n,h in expected.items())
    assert sha(ev/'encoding.json')==r['plan_sha256'];encoding=json.loads((ev/'encoding.json').read_bytes())
    assert encoding['passed']is True and len(encoding['nonuniform_records'])==36 and len(encoding['uniform_records'])==24
    assert all(x['exact_integer_headroom_passed']is True and x['nonzero_quantized_coefficients']>0 for x in encoding['nonuniform_records'])
    assert all(x['headroom_passed']is True and x['nonzero']is True for x in encoding['uniform_records'])
    assert sha(ev/'metrics.jsonl')==r['metrics_sha256'];rows=[json.loads(line)for line in (ev/'metrics.jsonl').read_bytes().splitlines()]
    general=[x for x in rows if x['kind']=='all_K_fullslot'];original=[x for x in rows if x['kind']=='original_fixture']
    assert len(rows)==386 and len(general)==384 and len(original)==2
    assert {(x['key_count'],x['valid_rows'])for x in general}=={(K,n)for K in range(1,129)for n in (1,7,16)}
    for x in general:
        assert x['passed']is True and x['fixed_gate']==1e-6 and x['slots']==65536 and math.isfinite(x['max_abs_error'])and 0<=x['max_abs_error']<=1e-6
        n=x['updates'];c=x['primitive_counts'];assert x['output_level']==20-n and c['pt']==3 and c['rotate']==14 and c['multiply']==2*n and c['rescale']==2*n+3
    assert {x['name']for x in original}=={'K128_rows16','K15_rows9'}and sum(x['checkpoint_count']for x in original)==54
    assert all(y['passed']is True for x in original for y in [*x['checkpoints'].values(),*x['derived_checks'].values()])
    suites=list(ET.parse(ev/'tests.xml').getroot().iter('testsuite'));counts={k:sum(int(s.attrib.get(k,0))for s in suites)for k in ('tests','errors','failures','skipped')}
    assert counts==r['tests']==dict(tests=1084,errors=0,failures=0,skipped=0)
    assert len(r['processes'])==2 and {p['name']for p in r['processes']}=={'tests','encoding'}
    for p in r['processes']:
        x=p['record'];assert x['direct_returncode']==0 and all(x.get(k)is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
    return dict(passed=True,sources=len(pins),tests=1084,ideal_cases=386,worst_all_K_abs=max(x['max_abs_error']for x in general),current_checked=bool(current),scope='Ideal/structural and public host-encoder preparation only; no genuine HE precision/accuracy claim')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--current',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.root,args.current),sort_keys=True))
