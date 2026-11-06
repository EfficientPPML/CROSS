"""Verify recorded public/mocked loader evidence without importing crypto."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in pins.items())
    report=json.loads((ROOT/'report.json').read_text());assert report['passed'] and report['source_files_unchanged'] and report['source_archive_verified'] and report['test_child_reaped'] and not report['cleanup_errors']
    assert report['source_hashes_before']==report['source_hashes_after'] and len(report['source_hashes_before'])==49
    with tarfile.open(ROOT/'sources.tar.gz') as tar:
        rows=tar.getmembers();assert len(rows)==len({m.name for m in rows})==49 and all(m.isfile() for m in rows)
        actual={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in rows}
    assert actual==report['source_hashes_before']
    assert hashlib.sha256((ROOT/'sources.tar.gz').read_bytes()).hexdigest()==report['source_archive_sha256']=='bad8a9341621aefbbb0acf52b144e7bec1aeca83730c6098a326d49946a00cdf'
    assert report['tests']==dict(tests=104,failures=0,errors=0,skipped=0)
    assert sum(int(x.attrib['tests']) for x in ET.parse(ROOT/'tests.xml').getroot())==104
    assert report['actual_key_generation'] is False and report['encrypted_execution'] is False
    assert not list(ROOT.rglob('*.npy')) and not list(ROOT.rglob('*.npz'))
    old=json.loads((ROOT/'superseded-source-bound-v1/report.json').read_text());assert old['tests']['tests']==100
    review=json.loads((ROOT/'independent-lazy-transfer-review/observation.json').read_text());assert review['lazy_key_barrier_called'] is False and review['mapped_backings_closed']>0
    print(json.dumps(dict(passed=True,sources=49,tests=104,scope='recorded public/mocked lifecycle evidence; no actual key or encrypted execution')))
if __name__=='__main__':main()
