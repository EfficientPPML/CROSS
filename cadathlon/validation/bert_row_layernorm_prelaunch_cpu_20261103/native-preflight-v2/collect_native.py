from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,sys,tarfile,tempfile,time
HERE=Path(__file__).resolve().parent;PROTOTYPE=HERE.parent/'prototype'
paths=[PROTOTYPE/name for name in ('assets.py','test_norm_assets.py','spec.py','contracts.py','transport.py')]+[Path(__file__)]
source={str(p):p.read_bytes() for p in paths};before={name:hashlib.sha256(value).hexdigest() for name,value in source.items()}
with tarfile.open(HERE/'sources.tar.gz','w:gz') as archive:
 for name,data in source.items():
  member=tarfile.TarInfo(Path(name).name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
report={'passed':False,'scope':'Pinned native LayerNorm fixtures only; no encryption/keygen; only hashes and aggregate diagnostics retained.','source_hashes_start':before,'cpu_affinity':sorted(os.sched_getaffinity(0))}
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='cross-norm-native-preflight-') as owned:
 root=Path(owned);env=dict(os.environ,CROSS_ROOT=str(HERE.parent/'checkout'))
 command=['/tmp/cross-pretrained-validation-env/bin/python',str(PROTOTYPE/'assets.py'),'--snapshot','/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c','--private',str(root/'client'),'--model',str(root/'model')]
 with (HERE/'stdout.json').open('w') as out,(HERE/'stderr.log').open('w') as err:
  completed=subprocess.run(command,env=env,stdout=out,stderr=err,check=False)
 report['returncode']=completed.returncode;report['elapsed_s']=time.perf_counter()-start
 for name,path in [('preprocessing.json',root/'client/preprocessing.json'),('model_manifest.json',root/'model/model_manifest.json')]:
  if path.exists():shutil.copyfile(path,HERE/name)
 report['passed']=completed.returncode==0
report['private_references_removed']=not root.exists()
report['source_hashes_end']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report['source_files_unchanged']=before==report['source_hashes_end'];report['passed']=report['passed'] and report['private_references_removed'] and report['source_files_unchanged']
report['source_archive_sha256']=hashlib.sha256((HERE/'sources.tar.gz').read_bytes()).hexdigest()
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
