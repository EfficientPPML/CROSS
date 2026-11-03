from pathlib import Path
import hashlib,json,sys,time
import pytest
LAB=Path(__file__).resolve().parent;ROOT=LAB/'checkout'
files=[*[p for p in ROOT.rglob('*') if p.is_file() and not {'__pycache__','.pytest_cache'}.intersection(p.parts)],Path(__file__)]
def hashes():return {str(p.relative_to(LAB)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes();started=time.perf_counter()
rc=pytest.main(['-q',str(ROOT/'cadathlon/tests'),'--junitxml='+str(LAB/'tests.xml')]);after=hashes()
r={'scope':'Isolated reusable streaming-linear package/P2 staging integration, including portable ideal/backend-free tests; no encrypted rerun or new calibration.','passed':rc==0 and before==after,'elapsed_s':time.perf_counter()-started,'returncode':int(rc),'source_files_unchanged':before==after,'source_hashes_start':before,'source_hashes_end':after}
(LAB/'report.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if r['passed'] else 1)
