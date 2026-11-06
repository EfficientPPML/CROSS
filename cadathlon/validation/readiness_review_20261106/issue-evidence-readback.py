"""Read archived public evidence, never execute candidate or archived verifiers."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
b=Path('/home/jianming_gatech/CROSS/cadathlon/validation');o=b/'readiness_review_20261106';pins={}
def read(p):
 q=Path(p);raw=q.read_bytes();pins[str(q)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
p1=read(b/'p1_student_post_promotion_20261103/report.json');c1=read(b/'p1_student_post_promotion_20261103/collection.json');p3=read(b/'p3_compact_post_promotion_20261103/process.json');s3=read(b/'p3_original_six_tpu_20261105/final/summary.json');idx=read(b/'p3_original_six_tpu_20261105/final/artifact_hashes.json')
for n,h in idx.items():
 q=b/'p3_original_six_tpu_20261105/final'/n;raw=q.read_bytes();actual=hashlib.sha256(raw).hexdigest();assert actual==h['sha256'] and len(raw)==h['bytes'],n;pins[str(q)]=actual
tr=read(b/'optimized_transformer_tpu_sweep_20261103/final/root-terminal-admission.json');ti=read(b/'optimized_transformer_tpu_sweep_20261103/final/sha256.json')
for n,h in ti.items():
 q=b/'optimized_transformer_tpu_sweep_20261103/final'/n;actual=hashlib.sha256(q.read_bytes()).hexdigest();assert actual==h,n;pins[str(q)]=actual
assert p1['passed'] and p1['cleanup_passed'] and p1['within_budget'] and p1['source_files_unchanged'] and set(p1['cases'])=={'C3','C4'}
assert c1['passed'] and c1['returncode']==0 and c1['external_evaluation_elapsed_s']<300
assert p3['passed'] and p3['returncode']==0 and p3['whole_process_within_300s'] and p3['source_files_unchanged']
assert s3['complete'] and s3['passed_cases']==6 and s3['failed_cases']==0 and len(s3['cases'])==6 and all(x['passed'] for x in s3['cases'])
assert tr['completed']==12 and tr['passed']==7 and tr['failed']==5 and tr['exit_code']==1 and not tr['full_model_or_ranked_pass']
for n in ['p1_student_post_promotion_20261103','p3_compact_post_promotion_20261103']: read(b/n/'verification.json')
result={'utc':datetime.now(timezone.utc).isoformat(),'scope':'Readback of archived reported results and hashes; no rerun, no broader current-runtime or full-model claim','issue1':'Shared-key fix and P1 student schedule previously validated; archived result remains passing','issue2':'Unresolved: original toy transformer sweep7/12; no successful full BERT/Qwen encrypted endpoint or scoring calibration','issue3':'Actual-ring checks implemented; prior CPU/TPU tests are scoped evidence, not arbitrary Python isolation','issue4':{'original_six_pass':True,'elapsed_s':s3['parent_elapsed_s'],'verified_indexed_artifacts':len(idx)},'issue5':{'p1_student_v2_elapsed_s':c1['external_evaluation_elapsed_s'],'p3_compact_v1_elapsed_s':p3['evaluator_process_elapsed_s'],'scope':'These reduced CNN schedules only; excludes original six-case cold schedule and full pretrained transformer workloads'},'full_pretrained_BERT12_five_file_stages_and_ranked100_200_408_seeds42_43_44_complete':False,'full_pretrained_Qwen28_HE_TPU_complete':False,'goal_complete':False,'source_and_result_pins':pins}
for s,h in pins.items():assert hashlib.sha256(Path(s).read_bytes()).hexdigest()==h,s
(o/'issue-evidence-readback.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(o/'issue-evidence-readback.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'path':str(o/'issue-evidence-readback.json'),'sha256':hashlib.sha256((o/'issue-evidence-readback.json').read_bytes()).hexdigest(),'pins':len(pins),'goal_complete':False}))
