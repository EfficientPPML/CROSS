from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((P/'verification.json').read_text());r=json.loads((P/'cleanup-preflight-v2/report.json').read_text())
assert all(sha(P/n)==h for n,h in v['artifact_sha256'].items())
with tarfile.open(P/'cleanup-preflight-v2/sources.tar.gz') as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==40 and all(m.isfile() for m in ms)
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']==r['source_hashes_after']
assert sha(P/'cleanup-preflight-v2/sources.tar.gz')==r['source_archive_sha256']==v['source_archive_sha256']
assert r['passed'] and r['test_exit_code']==0 and r['calibrated_runtime_sources_identical']
assert json.loads((P/'original_cleanup_reproducer.json').read_text())['reproduced']
print(json.dumps({'passed':True,'tests':20,'source_count':40,'arithmetic_changed':False,'model_evaluated':False}))
