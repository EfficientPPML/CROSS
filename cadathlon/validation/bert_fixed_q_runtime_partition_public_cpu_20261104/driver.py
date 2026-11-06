"""Source-bound public-only wrapper; no CKKS state or secret/payload reads."""
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
HERE=Path('/tmp/bert-fixed-q-runtime-partition-audit-20261104')
sys.path.insert(0,str(HERE))
import audit

def run():
 output=HERE/'source-bound-v1';output.mkdir(exist_ok=False)
 before=audit.sources();before['external/check_fixed_q_runtime_partition_20261104.py']=audit.sha(Path(__file__))
 record={'passed':False,'pid':os.getpid(),'source_hashes_before':before,'phases':[]}
 start=time.monotonic()
 try:
  with tarfile.open(output/'sources.tar.gz','w:gz')as tar:
   for name in before:tar.add(Path(__file__)if name.startswith('external/')else HERE/name,arcname=name)
  record['source_archive_sha256']=audit.sha(output/'sources.tar.gz')
  commands=[('tests',[sys.executable,'-m','pytest','--rootdir=.','--confcutdir=.','-q','test_plan.py','test_abstract.py','--junitxml='+str(output/'tests.xml')]),
   ('public-audit',[sys.executable,'audit.py',str(HERE/'actual-v1')])]
  for label,command in commands:
   with (output/(label+'.log')).open('wb')as log:
    child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    status=child.wait()
   record['phases'].append({'name':label,'pid':child.pid,'returncode':status,'waited':True})
   if status:raise RuntimeError(label+' failed')
  result=json.loads((HERE/'actual-v1/report.json').read_bytes())
  record['audit_report_sha256']=audit.sha(HERE/'actual-v1/report.json')
  record['audit_passed']=result['passed'];assert result['passed']is True
  record['passed']=True
 except BaseException as error:
  record['error']={'type':type(error).__name__,'message':str(error)}
  raise
 finally:
  primary=sys.exception()
  try:
   after=audit.sources();after['external/check_fixed_q_runtime_partition_20261104.py']=audit.sha(Path(__file__))
   record['source_hashes_after']=after;record['source_files_unchanged']=before==after
   with tarfile.open(output/'sources.tar.gz','r:gz')as tar:
    archived={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()for m in tar.getmembers()}
   record['source_archive_verified']=archived==before
   record['elapsed_s']=time.monotonic()-start
   record['passed']=record['passed']and primary is None and record['source_files_unchanged']and record['source_archive_verified']
   audit.write(output/'report.json',record)
  except BaseException as error:
   if primary is None:raise
   primary.add_note('wrapper publication failure: '+type(error).__name__)
 return record

if __name__=='__main__':
 r=run();print(json.dumps({'passed':r['passed'],'sources':len(r['source_hashes_before']),'phases':r['phases']}))
 raise SystemExit(0 if r['passed']else 1)
