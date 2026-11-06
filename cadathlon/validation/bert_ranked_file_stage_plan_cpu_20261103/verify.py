from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
for n,h in json.loads((p/'artifact_hashes.json').read_text()).items():assert sha(p/n)==h,n
r=json.loads((p/'report.json').read_text());t=json.loads((p/'terminal.json').read_text())
assert r['passed'] and r['test_counts']==dict(tests=38,errors=0,failures=0,skipped=0)
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==40
assert all(r[k] for k in ('source_files_unchanged','source_archive_verified','original_public_sources_unchanged','forbidden_backends_absent'))
assert sha(p/'sources.tar.gz')==r['source_archive_sha256']==t['source_archive_sha256']
with tarfile.open(p/'sources.tar.gz') as z:
 m=z.getmembers();assert len(m)==40==len({x.name for x in m}) and all(x.isfile() for x in m)
 assert {x.name:hashlib.sha256(z.extractfile(x).read()).hexdigest() for x in m}==r['source_hashes_before']
assert t['reaped'] and t['observed_exit_code']==0 and t['session_id']==13544
assert t['report_sha256']==sha(p/'report.json')
a=json.loads((p/'plan.json').read_text());assert len(a['cases'])==9 and a['selected_records']==2124
assert not a['actual_handler_batch_execution'] and not a['ranked_accuracy_or_timing_claim']
assert not r['private_bodies_read'] and not r['model_or_he_execution']
print('PASS:40 exact sources,38 public checks; no actual ranked file-stage execution')
