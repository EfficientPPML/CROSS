from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
HERE=Path(__file__).resolve().parent;OUT=HERE.parent
H=lambda b:hashlib.sha256(b).hexdigest()
def inventory():return {p.name:H(p.read_bytes())for p in sorted(HERE.iterdir())if p.is_file()}
before=inventory()
with (OUT/'tests.log').open('wb')as log:
 child=subprocess.Popen([sys.executable,'-I','-B',str(HERE/'probe.py'),str(OUT/'test_counts.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 code=child.wait()
def absent(pid,group=False):
 try:(os.killpg if group else os.kill)(pid,0)
 except ProcessLookupError:return True
 return False
after=inventory();assert after==before
with tarfile.open(OUT/'sources.tar.gz','w:gz')as tar:
 for name in sorted(before):tar.add(HERE/name,arcname=name)
counts=json.loads((OUT/'test_counts.json').read_bytes())
review={'schema':'qwen-checker-shield-construction-review-v1','scope':'Exact production supervise/accumulate AST with synthetic resources; no full test rerun, model/backend, private/token/calibration/validation or HE execution.','expected_negative_confirmed':code==1 and counts['tests']==4 and counts['failures']==3 and counts['errors']==0,'child_pid':child.pid,'child_exit_code':code,'child_reaped':True,'child_pid_absent':absent(child.pid),'child_group_absent':absent(child.pid,True),'source_hashes_before':before,'source_hashes_after':after,'source_archive_sha256':H((OUT/'sources.tar.gz').read_bytes()),'test_counts':counts,'finding':'cleanup_ran is set before signals(shield=True) construction outside try. Construction failure skips all resource/source/owned-output cleanup; outer fallback sees cleanup_ran true. Genuine body remains primary.'}
(OUT/'review.json').write_text(json.dumps(review,sort_keys=True,indent=2)+'\n')
print(json.dumps(review))
assert review['expected_negative_confirmed']and review['child_pid_absent']and review['child_group_absent']
