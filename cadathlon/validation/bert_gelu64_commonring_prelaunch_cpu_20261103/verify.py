from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent/'prelaunch-v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((P/'report.json').read_text());a=json.loads((P/'actual-ring-preflight/report.json').read_text())
assert r['passed'] and len(r['source_hashes_before'])==128
assert r['source_hashes_before']==r['source_hashes_after']==a['source_hashes_before']==a['source_hashes_after']
for folder,report,field in [(P,r,'source_hashes_before'),(P/'actual-ring-preflight',a,'source_hashes_before')]:
 assert sha((folder/'sources.tar.gz').read_bytes())==report['source_archive_sha256']
 with tarfile.open(folder/'sources.tar.gz') as t:
  members=t.getmembers();assert len(members)==128==len({m.name for m in members})
  assert {m.name:sha(t.extractfile(m).read()) for m in members}==report[field]
assert a['passed'] and a['preflight_only'] and not a['key_generation_started'] and not a['encryption_started']
ring=a['ring'];assert ring['actual_qp_is_secure_checked'] and ring['P_coverage_passed']
assert ring['prestage_scale']==1.506919487555368e18 and ring['expected_checkpoint_count']==9
assert ring['rotation_indices']==[] and ring['plan']['output_level']==10 and ring['fixed_checkpoint_absolute_gate']==1e-6
assert '209 passed' in (P/'tests.log').read_text()
print(json.dumps({'passed':True,'source_files':128,'tests':209,'keygen':False,'encryption':False}))
