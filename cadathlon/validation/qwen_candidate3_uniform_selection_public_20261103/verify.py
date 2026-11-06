"""Portable artifact/linkage verifier; exact independent proof is preserved separately."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def archive(path,expected):
    with tarfile.open(path) as t:
        ms=t.getmembers();assert len(ms)==len(expected)==len({m.name for m in ms}) and all(m.isfile() for m in ms)
        raw={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==expected
    return raw

def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in pins.items())
    top=ROOT/'source-bound-v1';r=json.loads((top/'report.json').read_text());assert r['passed'] and r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==12
    assert r['tests']==dict(tests=28,failures=0,errors=0,skipped=0) and sum(int(x.attrib['tests']) for x in ET.parse(top/'tests.xml').getroot())==28
    raw=archive(top/'sources.tar.gz',r['source_hashes_before']);assert hashlib.sha256((top/'sources.tar.gz').read_bytes()).hexdigest()==r['source_archive_sha256']
    child=top/'candidate-v3';s=json.loads((child/'report.json').read_text());m=json.loads((child/'manifest.json').read_text());a=json.loads((child/'assessments.json').read_text())
    assert s['passed'] and s['complete'] and not s['model_evaluated'] and not s['HE_executed'] and not s['heldout_or_fresh56_read']
    assert s['source_hashes_before']==s['source_hashes_after']==r['source_hashes_before']
    archive(child/'sources.tar.gz',s['source_hashes_before'])
    assert hashlib.sha256((child/'sources.tar.gz').read_bytes()).hexdigest()==s['source_archive_sha256']
    assert hashlib.sha256((child/'report.json').read_bytes()).hexdigest()==r['selection_report_sha256']
    assert hashlib.sha256((child/'manifest.json').read_bytes()).hexdigest()==r['manifest_sha256']==s['manifest_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
    assert hashlib.sha256((child/'assessments.json').read_bytes()).hexdigest()==s['assessments_sha256']
    assert len(a)==113 and len(s['sites_without_two_level_reserve_from25'])==93
    assert sum(not attempt['selected'] for row in a.values() for attempt in row['attempts'])==610
    assert all(x['covered'] for x in s['silu_radius_recheck'].values())
    review=json.loads((ROOT/'independent-review/report.json').read_text())
    assert review['passed'] and review['selected_manifest_sha256']==r['manifest_sha256'] and review['parent_source_archive_sha256']==r['source_archive_sha256']
    assert review['source_hashes_before']==review['source_hashes_after']==r['source_hashes_before']
    assert hashlib.sha256((ROOT/'independent-review/review.py').read_bytes()).hexdigest()==review['review_script_sha256']
    assert [review[k] for k in ('polynomial_certificates_independently_recomputed','factor4_domains_checked','unsuccessful_polynomial_attempts_retained','public_downstream_radius_checks')]==[36,113,610,28]
    base=json.loads(raw['inputs/candidate2_manifest.json'])
    for name,rule in base['nonlinear'].items():
        if not name.endswith('.inverse_sqrt'):assert m['nonlinear'][name]==rule
    for name in ('softmax','exp','initial_reciprocal','reciprocal_by_keys'):assert m[name]==base[name]
    assert not list(ROOT.rglob('*.npy')) and not list(ROOT.rglob('*.npz'))
    print(json.dumps(dict(passed=True,sources=12,tests=28,selected_norms=113,independent_polynomial_certificates=36,model_evaluated=False,HE_executed=False)))

if __name__=='__main__':main()
