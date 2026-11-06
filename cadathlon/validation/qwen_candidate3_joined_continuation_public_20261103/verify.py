"""Offline exact-source/artifact verifier; does not import model/HE backends."""
from pathlib import Path
import hashlib,json,sys,tarfile,xml.etree.ElementTree as ET
def digest(raw):return hashlib.sha256(raw).hexdigest()
def verify(directory):
    p=Path(directory);r=json.loads((p/'report.json').read_bytes())
    assert r['passed'] is True and r['cleanup_completed'] is True
    assert r['HE'] is False and r['model_forward'] is False
    assert r['test_exit_code']==r['plan_exit_code']==0
    assert r['source_hashes_before']==r['source_hashes_after']==r['snapshot_hashes_after']
    assert r['source_files_unchanged'] is True and len(r['source_hashes_before'])==27
    assert len(r['children'])==2 and all(x['process_reaped'] is True and x['exit_code']==0 for x in r['children'])
    raw=(p/'sources.tar.gz').read_bytes();assert digest(raw)==r['source_archive_sha256']
    with tarfile.open(p/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==27 and all(m.isfile() for m in ms)
        files={m.name:t.extractfile(m).read() for m in ms}
    assert {name:digest(raw) for name,raw in files.items()}==r['source_hashes_before']
    suites=ET.parse(p/'tests.xml').getroot().findall('testsuite')
    counts={name:sum(int(s.get(name,0)) for s in suites) for name in ('tests','errors','failures','skipped')}
    assert counts==r['tests']==dict(tests=113,errors=0,failures=0,skipped=0)
    raw=(p/'plan.json').read_bytes();assert digest(raw)==r['plan_sha256'];plan=json.loads(raw)
    assert plan['passed'] is True and len(plan['normalizations'])==113 and len(plan['rope_bounds'])==56
    assert plan['all_centered_domains_proved'] is False
    assert len(plan['incompatibilities']['low_norm_rope_inputs'])==9
    assert plan['workloads'][-1]['total_conditional_refreshes']==12931
    assert plan['candidate_manifest_sha256']==digest(files['inputs/candidate_manifest.json'])
    assert plan['source_input_sha256']=={name:row['sha256'] for name,row in json.loads(files['source_inputs.json']).items()}
    print(json.dumps(dict(verified=True,sources=27,tests=113,norms=113,post_rope_bounds=56,conditional_only=True)))
if __name__=='__main__':verify(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent/'source-bound-v1')
