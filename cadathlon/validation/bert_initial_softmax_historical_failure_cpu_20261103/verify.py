"""Verify a preserved failure, not a numeric pass; no crypto dependencies."""
from pathlib import Path
import hashlib,json,tarfile
from diagnose import derive
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'run-v1/report.json').read_bytes());receipt=json.loads((P/'terminal_readback.json').read_bytes())
assert r['passed']is False and r['complete']is False and r['preflight_only']is False
assert r['errors']==[dict(type='RuntimeError',message='initial-softmax output scale differs')] and r['checks']=={}
assert r.get('server_complete_before_any_decode')is not True
assert r['cleanup_completed']is True and r['owned_key_directory_removed']is True
assert r['cleanup_errors']==r['collector_cleanup_errors']==[]
assert len(r['codec_guards'])==7 and len(r['key_generation_guards'])==2
assert r['source_files_unchanged']is True and r['source_archive_verified']is True
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==110
assert sha(P/'run-v1/report.json')==receipt['report_sha256']=='de1bf6826f362af109fcf525958d5cc61b78c64a8ae36ec0dce5d944a3eef99c'
assert sha(P/'run-v1/sources.tar.gz')==r['source_archive_sha256']==receipt['source_archive_sha256']
with tarfile.open(P/'run-v1/sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==110 and all(m.isfile() for m in ms)
 raw={m.name:t.extractfile(m).read() for m in ms};assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==r['source_hashes_before']
 assert b'if output._ckks_scale!=arithmetic.scale:raise RuntimeError' in raw['checkout/cadathlon/common/encrypted_bert_initial_softmax.py']
assert receipt['session']==34588 and receipt['exit_code']==1 and receipt['reaped']is True
assert all(receipt[n]is True for n in ('worker_absent','owned_group_absent','owned_key_directory_absent','all110_current_sources_equal_start_end_archive'))
assert derive()==json.loads((P/'metadata_diagnosis.json').read_bytes())
assert 'Exit status: 1' in (P/'run-v1.time.txt').read_text()
assert 'RuntimeError: initial-softmax output scale differs' in (P/'run-v1.log').read_text()
art=json.loads((P/'artifact_hashes.json').read_bytes());assert set(art)=={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
for n,h in art.items():assert sha(P/n)==h,n
print('VERIFIED PRESERVED FAILURE:110 exact sources, exit1/reaped, no client numeric gates; one-ULP metadata assertion and cleanup receipts.')
