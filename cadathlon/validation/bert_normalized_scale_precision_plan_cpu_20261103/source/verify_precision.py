"""Portable public archive verification; no model, key, CT or backend imports."""
from pathlib import Path
import argparse,hashlib,json,math,tarfile,xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(root,current=None):
    root=Path(root);evidence=root/'source-bound-v1';report=json.loads((evidence/'report.json').read_bytes())
    for name in ('passed','complete','sources_unchanged','source_archive_verified','all_direct_children_reaped','all_owned_groups_empty','cleanup_completed','terminal_scope_completed','termination_scope_completed','plan_file_unchanged'):
        assert report.get(name)is True,name
    assert not report.get('error')and not report.get('cleanup_errors')and report['he']is False and report['model_forward']is False
    expected=report['source_hashes_before'];assert expected==report['source_hashes_after']==report['execution_source_hashes_after']and len(expected)==report['source_count']
    assert sha(evidence/'sources.tar.gz')==report['source_archive_sha256']
    with tarfile.open(evidence/'sources.tar.gz')as t:
        rows=t.getmembers();assert len(rows)==len({m.name for m in rows})==len(expected)
        assert all(m.isfile()and 0<=m.size<=8<<20 for m in rows)and sum(m.size for m in rows)<=32<<20
        raw={m.name:t.extractfile(m).read()for m in rows}
    assert {n:hashlib.sha256(b).hexdigest()for n,b in raw.items()}==expected
    if current:
        assert {n:sha(Path(current)/n)for n in expected}==expected
    pins=json.loads(raw['predecessor_pins.json']);assert len(pins)==8
    assert all(hashlib.sha256(raw[n]).hexdigest()==h for n,h in pins.items())
    assert sha(evidence/'plan.json')==report['plan_sha256'];plan=json.loads((evidence/'plan.json').read_bytes())
    rows=plan['schedules'];assert plan['passed']is True and len(rows)==512
    by={(r['K'],r['gain']):r for r in rows};assert set(by)=={(K,g)for K in range(1,129)for g in (1,2,4,8)}
    for (K,g),r in by.items():
        base=by[K,1];assert r['real_polynomial_unchanged']is True and r['metadata_relabeling']is False
        assert r['counts']==base['counts']and r['output_level']==base['output_level']and r['output_level']>=2
        assert math.isfinite(r['output_relative_scale_error'])and abs(r['output_relative_scale_error'])<1e-12
        assert len(r['inverse']['rows'])==len(base['inverse']['rows'])
        for actual,original in zip(r['inverse']['rows'],base['inverse']['rows']):
            assert actual['scale']*(g*g)==original['scale']
            if 'residual_scale'in actual:assert actual['residual_scale']==original['residual_scale']
    tests=list(ET.parse(evidence/'tests.xml').getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0))for s in tests)for k in ('tests','errors','failures','skipped')}
    assert counts==report['tests']==dict(tests=520,errors=0,failures=0,skipped=0)
    assert len(report['processes'])==2 and {p['name']for p in report['processes']}=={'tests','plan'}
    for p in report['processes']:
        r=p['record'];assert r['direct_returncode']==0 and all(r.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
    # Check every preserved public history/receipt byte against the final
    # artifact map, without claiming post-terminal snapshots were startup-bound.
    if (root/'artifact_hashes.json').exists():
        artifacts=json.loads((root/'artifact_hashes.json').read_bytes());assert all(sha(root/n)==h for n,h in artifacts.items())
    return dict(passed=True,sources=len(expected),tests=520,schedules=512,current_checked=bool(current),scope='Public canonical-input scale plan only; no repaired encrypted-result or accuracy claim')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--current',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.root,args.current),sort_keys=True))
