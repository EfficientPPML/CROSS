from pathlib import Path
import hashlib,importlib.util,json,os,sys,unittest
P=Path(__file__).resolve().parent
DENIED=[]
PROTECTED=('/tmp/qwen-expanded-prefix-observer-actual-v2-20261104','/tmp/qwen-expanded-prefix-observer-private-v2-20261104','/tmp/qwen-expanded-prefix-token-inputs-20261104-v1','/home/jianming_gatech/cross_validation_assets','/tmp/qwen-native-calibration-20261103','/tmp/qwen-candidate3-calibration-20261103','/tmp/qwen-candidate3-fresh56-validation-20261103','/tmp/qwen-candidate3-published-regression-20261103')
def audit(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=os.path.abspath(os.fsdecode(args[0]))
  if any(path==r or path.startswith(r+'/')for r in PROTECTED):
   DENIED.append(event);raise PermissionError('actual body forbidden in synthetic root verifier check')
 if event.startswith('socket.')or event in ('subprocess.Popen','os.fork','os.posix_spawn','os.exec','os.system'):
  DENIED.append(event);raise PermissionError('network/process forbidden in synthetic root verifier check')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('test_review',P/'test_root_qwen_initial3200_terminal_review_20261104.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(m))
record=dict(passed=result.wasSuccessful()and result.testsRun==17 and not DENIED,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),audit_denials=DENIED,pid=os.getpid(),pgid=os.getpgrp(),actual_verifier_main_executed=False,actual_model_or_private_body_read=False,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in P.iterdir()if p.is_file()})
(P.parent/'report.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
raise SystemExit(0 if record['passed']else 1)
