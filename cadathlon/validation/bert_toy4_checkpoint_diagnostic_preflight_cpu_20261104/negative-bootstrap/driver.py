from pathlib import Path
import tarfile,hashlib,json,subprocess,os,datetime
root=Path('/tmp/bert-toy4-checkpoint-component-20261104');old=root/'development-source-v2';out=Path('/tmp/bert-toy4-bootstrap-security-review-20261104/negative-v1');out.mkdir(parents=True)
r=json.loads((old/'report.json').read_bytes());expected=r['source_hashes_start'];assert len(expected)==246
blob=(old/'sources.tar.gz').read_bytes();assert hashlib.sha256(blob).hexdigest()==r['source_archive_sha256']=='ed5e2b7e21ea6997a29d7c629b29281b8529b263d0e293715d3751b2c78a550c'
with tarfile.open(old/'sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==246 and all(m.isfile()for m in ms);files={m.name:t.extractfile(m).read()for m in ms}
assert {n:hashlib.sha256(b).hexdigest()for n,b in files.items()}==expected
assert {n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in expected}==expected
(out/'predecessor_sources.tar.gz').write_bytes(blob);(out/'expected_sources.json').write_text(json.dumps(expected,sort_keys=True)+'\n')
cases=[]
for name in ('component_worker.py','collect.py'):
 case=out/name.removesuffix('.py');case.mkdir();exe=case/'execution';exe.mkdir()
 for n,b in files.items():
  p=exe/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 marker=case/'helper-executed'
 prefix=('from pathlib import Path\nPath('+repr(str(marker))+').write_text("harmless public-source canary\\n")\nraise RuntimeError("stop before any original helper import or role")\n').encode()
 (exe/'component_protocol.py').write_bytes(prefix+files['component_protocol.py'])
 before={n:hashlib.sha256((exe/n).read_bytes()).hexdigest()for n in expected}
 args=['/tmp/cross-pretrained-validation-env/bin/python','-B',str(exe/name),'--help']
 if name=='component_worker.py':args.insert(1,'-I')
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='')
 child=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 log,_=child.communicate();(case/'stdout.log').write_bytes(log)
 after={n:hashlib.sha256((exe/n).read_bytes()).hexdigest()for n in expected};assert before==after
 absent=not Path('/proc',str(child.pid)).exists()
 try:os.killpg(child.pid,0)
 except ProcessLookupError:group_absent=True
 else:group_absent=False
 assert child.returncode!=0 and marker.exists() and absent and group_absent
 cases.append({'entry':name,'command':args,'pid':child.pid,'direct_returncode':child.returncode,'direct_wait_completed':True,'pid_absent':absent,'group_absent':group_absent,'canary_executed_before_argument_or_source_gate':marker.exists(),'only_intentional_source_change':['component_protocol.py'],'source_hashes_before':before,'source_hashes_after':after})
 (case/'report.json').write_text(json.dumps(cases[-1],indent=2)+'\n')
(out/'driver.py').write_bytes(Path(__file__).read_bytes())
report={'scope':'Two harmless source-only expected bootstrap failures; no original helper execution, backend, model, input, private key or ciphertext read.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'predecessor_source_count':246,'predecessor_archive_sha256':r['source_archive_sha256'],'expected_failure_count':2,'cases':cases,'historical_sources_unchanged':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in expected}==expected}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
with tarfile.open(out/'capsule.tar.gz','w:gz')as t:
 for p in sorted(out.rglob('*')):
  if p.is_file()and p.name!='capsule.tar.gz':t.add(p,arcname=p.relative_to(out).as_posix())
print(json.dumps({'expected_failures':2,'report':str(out/'report.json'),'capsule_sha256':hashlib.sha256((out/'capsule.tar.gz').read_bytes()).hexdigest(),'pids':[x['pid']for x in cases]}))
