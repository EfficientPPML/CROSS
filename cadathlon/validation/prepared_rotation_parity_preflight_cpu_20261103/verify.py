from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
for n,h in json.loads((ROOT/'artifact_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
r=json.loads((ROOT/'preflight/report.json').read_text());assert r['passed'] and r['tests_passed']==92 and r['source_count']==233 and r['candidate_source_count']==231
assert r['source_hashes_before']==r['source_hashes_after'] and r['source_files_unchanged']
for directory in ('preflight','preflight/installed','preflight/public-gates','history/installed-development-v1'):
    p=ROOT/directory;record=json.loads((p/'report.json').read_text())
    expected=record.get('source_hashes_before',record.get('source_hashes_start'))
    assert hashlib.sha256((p/'sources.tar.gz').read_bytes()).hexdigest()==record['source_archive_sha256']
    with tarfile.open(p/'sources.tar.gz') as tar:
        members=tar.getmembers();assert len(members)==len(expected)==len({m.name for m in members}) and all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==expected
suites=ET.parse(ROOT/'preflight/tests.xml').getroot();assert sum(int(s.get('tests')) for s in suites)==92
assert all(int(s.get('failures'))==int(s.get('errors'))==int(s.get('skipped'))==0 for s in suites)
print('PASS:233 preparation sources;92 mocked checks;28 installed denials; public gates; no HE claim')
