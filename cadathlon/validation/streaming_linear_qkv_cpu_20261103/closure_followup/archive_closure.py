from pathlib import Path
import difflib,hashlib,io,json,shutil,subprocess,tarfile
LAB=Path(__file__).resolve().parent;ROOT=Path('/home/jianming_gatech/CROSS');BASE=ROOT/'cadathlon/validation/streaming_linear_qkv_cpu_20261103';OUT=BASE/'closure_followup';OUT.mkdir()
r=json.loads((LAB/'report.json').read_text());assert r['passed'] and r['source_files_unchanged']
with tarfile.open(OUT/'sources.tar.gz','w:gz') as t:
 for rel,wanted in r['source_hashes_start'].items():
  p=LAB/rel;data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
  info=tarfile.TarInfo(rel);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
for name in ('report.json','tests.log','tests.xml','run_tests.py','check_calibration_sources.py','calibration_source_check.json'):shutil.copyfile(LAB/name,OUT/name)
for name in ('initial_setup_failure','missing_checkout_marker_failure','missing_demo_fixture_failure'):shutil.copytree(LAB/name,OUT/name)
patch=[]
for rel in ('cadathlon/tests/test_package.py','cadathlon/tests/test_bert_staging.py'):
 before=(ROOT/rel).read_text();after=(LAB/'checkout'/rel).read_text()
 patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel,n=1))
 shutil.copyfile(LAB/'checkout'/rel,OUT/(Path(rel).name+'.txt'))
(OUT/'package_staging_tests.patch').write_text(''.join(patch))
combined=BASE/'combined_streaming_linear.patch';combined.write_bytes((BASE/'streaming_linear.patch').read_bytes()+(OUT/'package_staging_tests.patch').read_bytes());subprocess.run(['git','apply','--check',str(combined)],cwd=ROOT,check=True)
calibration=json.loads((LAB/'calibration_source_check.json').read_text());assert calibration['passed']
s={'scope':'Packaging/staging test-only follow-up after the four-file reusable candidate; no change to the measured CKKS component/runtime.','passed':True,'tests':75,'elapsed_s':r['elapsed_s'],'source_count':len(r['source_hashes_start']),'source_files_unchanged':True,'followup_patch_sha256':hashlib.sha256((OUT/'package_staging_tests.patch').read_bytes()).hexdigest(),'combined_six_file_patch_sha256':hashlib.sha256(combined.read_bytes()).hexdigest(),'source_archive_sha256':hashlib.sha256((OUT/'sources.tar.gz').read_bytes()).hexdigest(),'calibration_source_identity_passed':True,'p1_measured_files':103,'p3_measured_files':119,'changed_path_overlap':[],'original_numerical_source_archive_unchanged':True,'new_encryption_or_calibration_claim':False}
(OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n');shutil.copyfile(Path(__file__),OUT/'archive_closure.py');print(json.dumps(s,indent=2))
