"""Run exact native fixture preprocessing alone, with owned-process evidence."""
from pathlib import Path
import hashlib,io,json,os,shutil,sys,tarfile,tempfile,time
LAB=Path(__file__).resolve().parent;sys.path.insert(0,str(LAB/'prototype'))
import collect
process=collect.process
out=LAB/(sys.argv[1] if len(sys.argv)>1 else 'assets-preflight-v1');out.mkdir(exist_ok=False);source=collect.sources();source[str(Path(__file__))]=Path(__file__).read_bytes();before=collect.hashes(source)
with tarfile.open(out/'sources.tar.gz','w:gz') as t:
 for name,data in source.items():
  path=Path(name);member=str(path.relative_to(LAB));entry=tarfile.TarInfo(member);entry.size=len(data);t.addfile(entry,io.BytesIO(data))
report={'scope':'Exact trusted native FFN fixture preprocessing only; no keys or encryption.','passed':False,'source_hashes_start':before,'source_archive_sha256':process.sha256(out/'sources.tar.gz')};process.write_json(out/'report.json',report)
owned=Path(tempfile.mkdtemp(prefix='cross-ffn-native-preflight-'));fd,identity=process.open_owned_directory(owned);record=None;caught=None;started=time.perf_counter();(out/'process').mkdir()
process.child_environment_overrides=lambda:{'CROSS_ROOT':str(LAB/'checkout'),'JAX_PLATFORMS':'cpu','PYTHONUNBUFFERED':'1'}
try:
 command=['/tmp/cross-pretrained-validation-env/bin/python',str(LAB/'prototype/assets.py'),'--snapshot','/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c','--private',str(owned/'client_only'),'--model',str(owned/'model')]
 record=process.run_case(command,out/'process',timeout=None,cleanup_grace_s=30.);report['process']=record
 if not isinstance(record,dict) or record.get('returncode')!=0 or record.get('process_reaped') is not True or record.get('result',{}).get('passed') is not True:raise RuntimeError('native preprocessing failed')
 report['preprocessing']=collect.copy_json(owned/'client_only/preprocessing.json',out/'preprocessing.json');collect.copy_json(owned/'model/model_manifest.json',out/'model_manifest.json')
 report['preprocessing_passed']=True
except BaseException as error:caught=error;report['error']=f'{type(error).__name__}: {error}'
finally:
 if record is None:
  try:record=json.loads((out/'process/process.json').read_text())
  except BaseException:record={}
 report['process_reaped']=isinstance(record,dict) and record.get('process_reaped') is True
 report['private_references_removed']=False
 try:
  if not report['process_reaped']:raise RuntimeError('unknown process cleanup; retaining native files')
  report['private_references_removed']=process.remove_owned_directory(owned,fd,identity)
 except BaseException as error:
  report['cleanup_error']=f'{type(error).__name__}: {error}'
  if caught is None:caught=error
  else:caught.add_note('Native cleanup also failed: '+str(error))
 finally:os.close(fd)
 after=collect.hashes(collect.sources());after[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();report['source_hashes_end']=after;report['source_files_unchanged']=before==after;report['elapsed_s']=time.perf_counter()-started
 report['passed']=caught is None and report.get('preprocessing_passed') is True and report['process_reaped'] and report['private_references_removed'] and report['source_files_unchanged']
 process.write_json(out/'report.json',report)
if caught is not None and not isinstance(caught,Exception):raise caught
print(json.dumps({k:report.get(k) for k in ('passed','error','elapsed_s','process_reaped','private_references_removed','source_files_unchanged')}));raise SystemExit(0 if report['passed'] else 1)
