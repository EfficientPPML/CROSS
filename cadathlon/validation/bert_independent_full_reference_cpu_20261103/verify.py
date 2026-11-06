"""Offline exact-source and recorded full-reference gate verifier."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(sha(ROOT/n)==h for n,h in pins.items())
    r=json.loads((ROOT/'report.json').read_text())
    assert r['passed'] is True and r['test_child_reaped'] is True and not r['cleanup_errors']
    assert r['source_files_unchanged'] and r['asset_files_unchanged'] and r['source_archive_verified']
    assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==7
    assert r['asset_hashes_before']==r['asset_hashes_after'] and len(r['asset_hashes_before'])==3
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='85f0876a28244f7e5770c4af5174a6fd1701276906e2efc5ccf1c43824213514'
    with tarfile.open(ROOT/'sources.tar.gz') as t:
        rows=t.getmembers();assert len(rows)==len({x.name for x in rows})==7 and all(x.isfile() for x in rows)
        assert {x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest() for x in rows}==r['source_hashes_before']
    assert r['tests']==dict(tests=35,errors=0,failures=0,skipped=0)
    assert sum(int(x.attrib['tests']) for x in ET.parse(ROOT/'tests.xml').getroot())==35
    for c,K in zip(r['cases'],(15,128)):
        assert c['K']==K and c['approximate_complete'] is True
        assert len(c['exact_native_layers'])==12
        errors=[x[k] for x in c['exact_native_layers'] for k in ('attention_max_abs','block_max_abs')]+[c['exact_native_pooler_max_abs'],c['exact_native_logit_max_abs']]
        assert all(math.isfinite(x) and 0<=x<=r['native_exact_gate']==1e-12 for x in errors)
        assert c['independent_vs_frozen_logit_max_abs']<=r['independent_vs_frozen_logit_gate']==1e-7
        assert c['original_P2_logit_gate_passed'] is True and math.isfinite(c['approximate_native_logit_max_abs'])
        assert len(c['domain_observations'])==156
        for d in c['domain_observations']:
            assert d['below']==d['above']==d['nonfinite']==0 and d['count']>0
            assert d['domain'][0]<=d['minimum']<=d['maximum']<=d['domain'][1]
    assert len(r['reference_arrays'])==52 and r['reference_npz_sha256']=='97ce65202b9cee62ff0de1e2831db36b25a4fdb02d11a92658e8f72cc31aaf23'
    assert r['raw_arrays_archived'] is False and r['labels_loaded'] is False and not list(ROOT.rglob('*.npz'))
    print(json.dumps(dict(passed=True,sources=7,tests=35,cases=2,domain_checks=312,scope='offline provenance/recorded gates; no numerical rerun')))
if __name__=='__main__':main()
