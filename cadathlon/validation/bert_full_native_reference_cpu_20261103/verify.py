"""Offline provenance and recorded-gate verifier; no numerical rerun."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(sha(ROOT/n)==h for n,h in pins.items())
    r=json.loads((ROOT/'report.json').read_text())
    assert r['passed'] is True and r['test_child_reaped'] is True and not r['cleanup_errors']
    assert r['source_files_unchanged'] and r['asset_files_unchanged'] and r['source_archive_verified']
    assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==4
    assert r['asset_hashes_before']==r['asset_hashes_after'] and len(r['asset_hashes_before'])==6
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='7c7d25c2a8ff19420807902427ab472c089381b47c021db57d6e3f9a02cf1e4c'
    with tarfile.open(ROOT/'sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len({x.name for x in members})==4 and all(x.isfile() for x in members)
        assert {x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest() for x in members}==r['source_hashes_before']
    assert r['tests']==dict(tests=15,failures=0,errors=0,skipped=0)
    assert sum(int(s.attrib['tests']) for s in ET.parse(ROOT/'tests.xml').getroot())==15
    expected={}
    for K in (15,128):
        expected[f'K{K}_embeddings']=([128,768],'<f8')
        expected[f'K{K}_positions']=([K],'<i8')
        expected[f'K{K}_pooler_native']=([768],'<f8')
        expected[f'K{K}_logits_native']=([2],'<f8')
        expected.update({f'layer{i}_K{K}_native':([K,768],'<f8') for i in range(12)})
    assert set(r['arrays'])==set(expected)
    for n,(shape,dtype) in expected.items():assert r['arrays'][n]['shape']==shape and r['arrays'][n]['dtype']==dtype and len(r['arrays'][n]['sha256'])==64
    for c,K in zip(r['cases'],(15,128)):
        assert c['K']==K and c['initial_embedding_max_abs']==0.0 and len(c['layers'])==12
        errors=[x['full128_vs_compact_max_abs'] for x in c['layers']]+[c['pooler_compaction_max_abs'],c['logit_compaction_max_abs']]
        assert all(math.isfinite(e) and 0<=e<=r['fixed_native_compaction_gate']==1e-12 for e in errors)
        assert c['original_positions'][0]==0 and len(c['original_positions'])==K
    assert r['fixture_npz_sha256']=='758ff5d71fa3c836f74a974dee97c4a744e17ea283047e2abe15b4675edc202a'
    assert r['allowed_joined_inputs']==['K15_embeddings','K128_embeddings'] and r['later_native_arrays_are_references_only'] is True
    assert r['raw_arrays_archived'] is False and r['labels_loaded'] is False and not list(ROOT.rglob('*.npz'))
    print(json.dumps(dict(passed=True,sources=4,private_array_descriptors=32,native_cases=2,tests=15,scope='offline recorded evidence; no numerical rerun')))
if __name__=='__main__':main()
