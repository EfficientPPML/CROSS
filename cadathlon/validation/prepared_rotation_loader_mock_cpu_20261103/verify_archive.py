from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
manifest=json.loads((root/'archive_manifest.json').read_text())
for name,digest in manifest['files'].items():assert sha((root/name).read_bytes())==digest,name
for name,count in [('source-bound-v1',108),('source-bound-v2',116)]:
    d=root/name;r=json.loads((d/'report.json').read_text())
    assert r['passed'] and r['child_return_code']==0 and r['child_reaped']
    assert len(r['source_hashes_before'])==14
    assert r['source_hashes_before']==r['source_hashes_after']
    assert sha((d/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
    with tarfile.open(d/'sources.tar.gz') as archive:
        actual={m.name:sha(archive.extractfile(m).read()) for m in archive.getmembers()}
    assert actual==r['source_hashes_before']==r['archive_member_hashes']
    suites=ET.parse(d/'tests.xml').getroot().findall('testsuite')
    assert sum(int(x.attrib['tests']) for x in suites)==count
    assert all(int(x.attrib.get(k,0))==0 for x in suites for k in ('failures','errors'))
for name in ('lookup-negative-v1','replacement-negative-v1'):
    d=root/name;r=json.loads((d/'report.json').read_text())
    assert r['reproduced'] and sha((d/'loader_prepared_rotation.py').read_bytes())==r['source_sha256']
print('PASS:14 exact sources percheckpoint,108 historical/116 final mocked checks; two retained negatives')
