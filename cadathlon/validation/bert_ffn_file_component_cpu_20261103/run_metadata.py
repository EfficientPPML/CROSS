from pathlib import Path
import hashlib,json,sys,time
LAB=Path(__file__).resolve().parent;sys.path.insert(0,str(LAB/'prototype'))
import collect,pytest
before=collect.hashes(collect.sources());before[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();started=time.perf_counter()
rc=pytest.main(['-q',str(LAB/'prototype/test_component.py'),'--junitxml='+str(LAB/'metadata_tests.xml')])
after=collect.hashes(collect.sources());after[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
r={'scope':'Fixed FFN coefficient/depth/profile, strict per-checkpoint transport, distinct-row BSGS public algebra and collector lifecycle. No key generation or encryption.','passed':rc==0 and before==after,'returncode':int(rc),'elapsed_s':time.perf_counter()-started,'source_hashes_start':before,'source_hashes_end':after,'source_files_unchanged':before==after}
(LAB/'metadata_tests.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if r['passed'] else 1)
