"""Public source/hash report recovery only, never private outcomes or candidate execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,tarfile,os
B=Path('/tmp/qwen-expanded-candidate-checker-actual-v1-20261104')
O=Path('/home/jianming_gatech/CROSS/cadathlon/validation/readiness_review_20261106')
reads={}
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
 p=Path(p);b=p.read_bytes();assert reads.setdefault(str(p),b)==b;return b
raw=read(B/'worker/report.json');r=json.loads(raw)
mb=read(B/'source_manifest.json');assert sha(mb)=='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a';m=json.loads(mb);assert len(m)==139
assert r['source_hashes_start']==r['source_hashes_end']==m and r['source_manifest_sha256']==sha(mb) and r['source_files_unchanged']
assert {p.relative_to(B/'execution_source').as_posix() for p in (B/'execution_source').rglob('*') if p.is_file()}==set(m)
for n,h in m.items():assert sha(read(B/'execution_source'/n))==h,n
archive=read(B/'sources.tar.gz');assert sha(archive)=='25d3b775825ee5cdece32d2b828fd2018675aa94cc5fb22605f5ab1f1f5e820e'
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
 members=t.getmembers();assert len(members)==139 and {x.name for x in members}==set(m)
 for x in members:assert x.isfile() and not x.issym() and not x.islnk() and sha(t.extractfile(x).read())==m[x.name]
assert r['complete'] is True and r['passed'] is True and r['cleanup_completed'] is True
assert r['acceptance_requires_independent_exit0_reap_and_exact_report_hash'] is True
assert r['encrypted_execution'] is False and r['validation_inputs_read'] is False and r['coefficient_selection_performed'] is False and r['contains_tokens_logits_or_failure_text'] is False
assert r['expected_entries']==r['retained_entries']==r['candidate_complete']==r['semantic_passes']==3200
assert r['phase_counts']=={'candidate':3200,'exact':3200,'native':3200} and r['complete_free_sequences']==r['complete_free_denominator']==16
assert all(r['gates'].values()) and r['audit_denials']==[] and r['complete_measured_context_coverage'] and r['trajectories_complete']
assert r['pid']==2859668 and not Path('/proc/2859668').exists()
record={'utc':datetime.now(timezone.utc).isoformat(),'public_report_sha256':sha(raw),'source_map_sha256':sha(mb),'source_archive_sha256':sha(archive),'source_files_verified':139,'reported_passed':True,'reported_complete':True,'reported_entries':3200,'reported_semantic_passes':3200,'reported_phase_counts':r['phase_counts'],'reported_complete_free_sequences':16,'reported_elapsed_seconds':r['elapsed_s'],'original_worker_absent':True,'original_tool_exit0_reap_receipt_available':False,'terminal_or_numerical_admission':False,'validation_inputs_read':False,'private_outcome_bodies_read':False,'encrypted_execution':False,'full_endpoint_proved':False,'limits':['Own fixed calibration workload only; no validation/row0/coefficient-selection or HE result.','Successful report and absent worker do not reconstruct independent exit0/reap evidence required by the original protocol.','No private outcome bodies, tokens, logits, model arrays or native binaries read; no subject/verifier execution.']}
for p,b in reads.items():assert Path(p).read_bytes()==b,p
for n,b in [('qwen-calibration-public-report.json',raw),('qwen-calibration-source-manifest.json',mb),('qwen-calibration-sources.tar.gz',archive),('qwen-calibration-readback.py',Path(__file__).read_bytes())]:(O/n).write_bytes(b)
(O/'qwen-calibration-public-readback.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({'path':str(O/'qwen-calibration-public-readback.json'),'sha256':sha((O/'qwen-calibration-public-readback.json').read_bytes()),'sources':139,'reported_semantic_passes':3200,'terminal_or_numerical_admission':False}))
