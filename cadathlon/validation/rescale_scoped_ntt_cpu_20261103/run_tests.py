"""Scoped rescale control regressions; CPU component evidence, not a model proof."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parent/'checkout';OUT=ROOT.parent
os.chdir(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(ROOT/'cadathlon/tests')]
import pytest
paths=[*ROOT.rglob('*.py'),ROOT/'pytest.ini',Path(__file__)];paths=[p for p in paths if '__pycache__' not in p.parts]
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
before=hashes();started=time.perf_counter()
args=['-q','cadathlon/tests/test_scoped_rescale_ntt.py','cadathlon/tests/test_scoped_ntt.py','cadathlon/tests/test_scoped_mul_ntt.py','cadathlon/tests/test_dispatch_cache_policy.py','cadathlon/tests/test_encrypted_refresh.py','jaxite_word/rescale_test.py','jaxite_word/rescale_double_rescaling_test.py','jaxite_word/hemul_test.py','jaxite_word/hemul_double_rescaling_test.py','--junitxml='+str(OUT/'tests.xml')]
rc=pytest.main(args);after=hashes()
records={name:getattr(sys.modules.get(name),'RECORDS',[]) for name in ['test_scoped_rescale_ntt','test_encrypted_tensor_only']}
r={'scope':__doc__,'returncode':int(rc),'passed':rc==0 and before==after,'elapsed_s':time.perf_counter()-started,'source_hashes_start':before,'source_hashes_end':after,'source_files_unchanged':before==after,'records':records,'command':args}
(OUT/'tests.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':r['passed'],'elapsed_s':r['elapsed_s'],'sources':len(before),'records':records},indent=2),flush=True)
raise SystemExit(0 if r['passed'] else 1)
