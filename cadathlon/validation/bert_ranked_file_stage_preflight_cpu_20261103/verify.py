from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
for n,h in json.loads((p/'artifact_hashes.json').read_text()).items():assert sha(p/n)==h,n
r=json.loads((p/'report.json').read_text());t=json.loads((p/'terminal.json').read_text())
assert r['passed'] and r['test_counts']==dict(tests=141,errors=0,failures=0,skipped=0)
assert len(r['source_hashes_before'])==241 and r['source_hashes_before']==r['source_hashes_after']
assert r['source_files_unchanged'] and r['source_archive_verified'] and r['all_predecessor_copies_and_originals_unchanged']
assert r['backend_audit']==dict(pytest_returncode=0,forbidden_backends_loaded=[])
assert not r['private_retained_bodies_read'] and not r['actual_encrypted_batch'] and not r['ranked_accuracy_or_timing_claim']
assert sha(p/'sources.tar.gz')==r['source_archive_sha256']==t['source_archive_sha256']
with tarfile.open(p/'sources.tar.gz') as z:
 m=z.getmembers();assert len(m)==241==len({x.name for x in m}) and all(x.isfile() for x in m)
 assert {x.name:hashlib.sha256(z.extractfile(x).read()).hexdigest() for x in m}==r['source_hashes_before']
 assert z.extractfile('README.md').read()==(p/'CANDIDATE.md').read_bytes()
for f in (p/'candidate').glob('*.py'):assert sha(f)==r['source_hashes_before'][f.name]
assert t['reaped'] and t['observed_exit_code']==0 and t['session_id']==20777 and t['report_sha256']==sha(p/'report.json')
for report in (p/'history').rglob('report.json'):
 d=json.loads(report.read_text());a=report.parent/'sources.tar.gz'
 if not a.exists():continue
 expected=d.get('source_hashes_before',d.get('source_hashes'))
 with tarfile.open(a) as z:assert {m.name:hashlib.sha256(z.extractfile(m).read()).hexdigest() for m in z.getmembers()}==expected
 if 'source_archive_sha256' in d:assert sha(a)==d['source_archive_sha256']
print('PASS:241 exact sources,141 public/mock checks; no actual ranked worker/HE claim')
