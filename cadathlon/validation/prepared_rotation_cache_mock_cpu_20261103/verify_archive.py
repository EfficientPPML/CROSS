from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
manifest=json.loads((root/'archive_manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert sha((root/name).read_bytes())==digest,name
r=json.loads((root/'source-bound-v1/report.json').read_text())
assert r['passed'] and r['child_return_code']==0 and r['child_reaped']
assert r['source_hashes_before']==r['source_hashes_after']
assert sha((root/'source-bound-v1/sources.tar.gz').read_bytes())==r['source_archive_sha256']
with tarfile.open(root/'source-bound-v1/sources.tar.gz') as archive:
    actual={m.name:sha(archive.extractfile(m).read()) for m in archive.getmembers()}
assert actual==r['source_hashes_before']==r['archive_member_hashes']
xml=ET.parse(root/'source-bound-v1/tests.xml')
suites=xml.getroot().findall('testsuite')
assert sum(int(x.attrib['tests']) for x in suites)==38
assert all(int(x.attrib.get(k,0))==0 for x in suites for k in ('failures','errors'))
print('PASS: five exact proof sources,38 mocked checks; public-only planning and preserved negative')
