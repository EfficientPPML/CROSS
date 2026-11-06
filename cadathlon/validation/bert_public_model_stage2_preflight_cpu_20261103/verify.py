"""Offline verification of v1 and repaired v2 public stage2 preflights."""
from pathlib import Path
import hashlib,io,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def unpack(blob,expected):
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        ms=t.getmembers();assert len(ms)==len(expected)==len({m.name for m in ms}) and all(m.isfile() for m in ms)
        raw={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==expected
    return raw

def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in pins.items())
    for name,tests,count,expected in (('preflight-v1',68,62,'042e3de4e25b8e301b25c3f46b60c34a43290f086fa51369d3c3ac2eff7ba1ab'),('preflight-v2',72,65,'1d580e2241198e854695ee115c0cdc145d264833a4f2cf888b981a0770eac012')):
        directory=ROOT/name;r=json.loads((directory/'report.json').read_text());assert r['passed'] and r['source_files_unchanged'] and r['source_archive_verified']
        assert r['actual_preprocessing'] is False and r['actual_key_generation'] is False and r['model_forward'] is False
        assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==count
        blob=(directory/'sources.tar.gz').read_bytes();assert hashlib.sha256(blob).hexdigest()==expected==r['source_archive_sha256'];raw=unpack(blob,r['source_hashes_before'])
        assert r['tests']==dict(tests=tests,failures=0,errors=0,skipped=0) and sum(int(x.attrib['tests']) for x in ET.parse(directory/'tests.xml').getroot())==tests
        s=r['public_storage'];assert (s['tensor_count'],s['matrix_count'],s['raw_model_bytes'],s['required_with_margin'])==(196,74,342586376,1416428552)
        origin=json.loads(raw['source_origin.json'])
        for n,h in origin['files'].items():
            if 'checkout/'+n in raw:assert hashlib.sha256(raw['checkout/'+n]).hexdigest()==h
    assert hashlib.sha256(raw['key_source_manifest.json']).hexdigest()=='b98e852d6111a63bbe2f25c5e811c8b56469100e85c3a3af7a26f86688adf4c7'
    inherited=json.loads(raw['key_source_manifest.json']);assert len(inherited)==58
    original=unpack(raw['key_source_archive_fixture.tar.gz'],inherited)
    assert len(original['stage/server_inputs/twelve_layer_plan.json'])==6461050
    assert max(map(len,original.values()))<=8<<20 and sum(map(len,original.values()))<=16<<20
    assert not list(ROOT.rglob('*.npy')) and not list(ROOT.rglob('*.npz'))
    print(json.dumps(dict(passed=True,current_sources=65,current_tests=72,historical_sources=62,actual_preprocessing=False)))

if __name__=='__main__':main()
