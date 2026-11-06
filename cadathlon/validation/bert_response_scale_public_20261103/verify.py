"""Offline archive and exact public enclosure verifier; no backend."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in pins.items())
    report=json.loads((ROOT/'report.json').read_text());assert report['passed'] and report['source_files_unchanged'] and report['source_archive_verified']
    assert report['source_hashes_before']==report['source_hashes_after'] and len(report['source_hashes_before'])==11
    with tarfile.open(ROOT/'sources.tar.gz') as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==11 and all(m.isfile() for m in members)
        raw={m.name:tar.extractfile(m).read() for m in members}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==report['source_hashes_before']
    assert hashlib.sha256((ROOT/'sources.tar.gz').read_bytes()).hexdigest()==report['source_archive_sha256']=='f3c6df871d4f915e874bad0aff38cbb4335216a45956f9bf275bb14b81bd9c89'
    namespace={};exec(compile(raw['response_scale.py'],'response_scale.py','exec'),namespace)
    actual=namespace['scale_certificate'](profile_bytes=raw['server_inputs/p60_profile_and_refresh.json'],manifest_bytes=raw['server_inputs/candidate_manifest.json'])
    assert json.loads(json.dumps(actual))==report['certificate']
    assert report['tests']==dict(tests=13,failures=0,errors=0,skipped=0)
    assert sum(int(x.attrib['tests']) for x in ET.parse(ROOT/'tests.xml').getroot())==13
    print(json.dumps(dict(passed=True,sources=11,tests=13,accepted_scales=actual['accepted_scales'],scope='public metadata enclosure only')))
if __name__=='__main__':main()
