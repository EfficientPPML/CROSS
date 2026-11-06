"""Offline archive/provenance verifier; no checkpoint, client arrays, or backend."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    artifacts=json.loads((ROOT/'artifact_sha256.json').read_text())
    assert all(sha(ROOT/name)==digest for name,digest in artifacts.items())
    r=json.loads((ROOT/'report.json').read_text())
    assert r['passed'] is True and r['child_reaped'] is True and not r['cleanup_errors']
    assert r['source_files_unchanged'] is True and r['asset_files_unchanged'] is True and r['source_archive_verified'] is True
    assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==3
    assert r['asset_hashes_before']==r['asset_hashes_after'] and len(r['asset_hashes_before'])==6
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='2b675675fb04babb7d84e5bf53419a01784c8ac3595ac89b8db702fae418d2cb'
    with tarfile.open(ROOT/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len(members)==3 and len({m.name for m in members})==3
        assert {m.name for m in members}=={'client_inputs.py','test_client_inputs.py','collect_client_inputs.py'}
        assert all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
    assert r['tests']==dict(tests=45,failures=0,errors=0,skipped=0)
    xml=ET.parse(ROOT/'tests.xml').getroot()
    assert sum(int(s.attrib['tests']) for s in xml)==45
    for index,k in enumerate((15,128,11)):
        c=json.loads((ROOT/f'case-reports/case{index}.json').read_text())
        assert c==r['cases'][index] and c['case']==index
        assert math.isfinite(c['native_numpy_max_abs']) and c['native_numpy_max_abs']<=c['fixed_gate']==1e-12
        if index<2:assert c['prior_native_max_abs']==0.0
        metadata=c['public_metadata'];positions=metadata['original_positions']
        assert metadata['key_count']==k==len(positions) and positions[0]==0 and positions==sorted(set(positions))
        assert len(metadata['original_attention_mask'])==128 and positions==[i for i,b in enumerate(metadata['original_attention_mask']) if b]
        assert metadata['input_tile_count']==8 and metadata['packing']['slots']==65536
        assert len(c['tile_sha256'])==8 and all(len(x)==64 for x in c['tile_sha256'])
        assert not {'target_idx','input_ids','token_type_ids'} & set(metadata)
    assert not list(ROOT.rglob('*.npz')) and r['raw_arrays_archived'] is False and r['private_target_mapping_archived'] is False
    print(json.dumps(dict(passed=True,source_count=3,cases=3,tests=45,scope='offline provenance and recorded gates; no numerical rerun')))
if __name__=='__main__':main()
