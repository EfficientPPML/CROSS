from pathlib import Path
import hashlib,json,os,sys,time
LAB=Path(__file__).resolve().parent;sys.path.insert(0,str(LAB/'prototype'))
import collect,pytest
os.environ['CROSS_STREAMING_ROOT']=str(LAB/'checkout')
paths=[*sorted((LAB/'promotion/cadathlon/tests').glob('*.py')),Path(__file__)]
def hashes():
 values=collect.hashes(collect.sources())
 values.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 return values
before=hashes();start=time.perf_counter()
rc=pytest.main(['-q',*[str(p) for p in paths[:-1]],'--junitxml='+str(LAB/'promotion_tests.xml')]);after=hashes()
r={'scope':'Portable proposed repository tests against the exact module/runtime that completed the separate encrypted QKV proof; no encryption/key generation in this test run.','passed':rc==0 and before==after,'returncode':int(rc),'elapsed_s':time.perf_counter()-start,'source_files_unchanged':before==after,'source_hashes_start':before,'source_hashes_end':after}
(LAB/'promotion_tests.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if r['passed'] else 1)
