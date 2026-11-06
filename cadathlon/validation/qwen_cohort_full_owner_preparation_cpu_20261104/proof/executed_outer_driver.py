"""Outer owner for public-only preparation; exact frozen child owns its audit."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time,tarfile
SOURCE=Path('/tmp/qwen-softmax-cohort-full-owner-integration-v2-20261104')
EVIDENCE=Path('/tmp/qwen-cohort-full-owner-combined-preflight-20261104/preflight-v1')
MANIFEST=EVIDENCE/'sources.json'
PIN='ff8d902f71497517609f3629c3adfe12f9fffa5b720ec22aa0db61b5b5bd5d5e'
PYTHON='/tmp/cross-pretrained-validation-env/bin/python'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(name,data):
 (EVIDENCE/name).write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n')
def gone(pgid):
 try:os.killpg(pgid,0)
 except ProcessLookupError:return True
 return False
raw=MANIFEST.read_bytes();assert sha(raw)==PIN
expected=json.loads(raw);execution=EVIDENCE/'execution';assert len(expected)==208
for root in(SOURCE,execution):
 assert {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}==expected
assert not(EVIDENCE/'worker').exists()
env=dict(PATH='/usr/bin:/bin',HOME='/home/jianming_gatech',LANG='C.UTF-8',TZ='UTC',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
unset=('SOFTMAX_REVIEW_SOURCE','PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONPATH','PYTHONHOME')
assert not any(n in env for n in unset)
command=['taskset','-c','112-115',PYTHON,'-I','-B',str(execution/'prepare_expanded.py'),'--source-manifest',str(MANIFEST),'--manifest-sha256',PIN,'--original-root',str(SOURCE),'--output',str(EVIDENCE/'worker')]
dump('launch.json',dict(command=command,environment=env,explicitly_unset=list(unset),source_manifest_sha256=PIN,outer_driver_sha256=sha(Path(__file__).read_bytes()),actual_mode=False,scope='CPU public/synthetic preparation only'))
(EVIDENCE/'executed_outer_driver.py').write_bytes(Path(__file__).read_bytes())
child=None;observed=None;primary=None;code=None;start=time.monotonic()
with(EVIDENCE/'run.log').open('wb')as log:
 try:
  child=subprocess.Popen(command,env=env,cwd=execution,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  observed=dict(outer_pid=os.getpid(),outer_pgid=os.getpgrp(),wrapper_pid=os.getppid(),wrapper_pgid=os.getpgid(os.getppid()),worker_pid=child.pid,worker_pgid=os.getpgid(child.pid))
  dump('live-observation.json',observed)
  code=child.wait()
 except BaseException as error:
  primary=error
  if child is not None and child.poll()is None:
   try:
    os.killpg(child.pid,signal.SIGTERM)
    try:code=child.wait(timeout=30)
    except subprocess.TimeoutExpired:
     os.killpg(child.pid,signal.SIGKILL);code=child.wait()
   except BaseException as cleanup:
    try:primary.add_note('public child retirement failure: '+type(cleanup).__name__)
    except BaseException:pass
parent_after={n:sha((SOURCE/n).read_bytes())for n in expected}
execution_after={n:sha((execution/n).read_bytes())for n in expected}
dump('original-sources-after.json',parent_after);dump('execution-sources-after.json',execution_after)
worker_path=EVIDENCE/'worker/report.json';worker=json.loads(worker_path.read_bytes())if worker_path.exists()else{}
archive=EVIDENCE/'sources.tar.gz'
with tarfile.open(archive)as t:
 members=t.getmembers();archived={m.name:sha(t.extractfile(m).read())for m in members if m.isfile()}
archive_match=len(members)==len(expected)and archived==expected
passed=(primary is None and code==0 and child is not None and gone(child.pid)and parent_after==execution_after==expected and archive_match and worker.get('passed')is True and worker.get('tests')==dict(tests=330,failures=0,errors=0,skipped=0)and worker.get('denied_attempts')==[])
result=dict(schema='qwen-cohort-full-owner-public-outer-v1',passed=passed,source_count=208,tests=worker.get('tests'),child_returncode=code,child_reaped=child is not None and child.returncode is not None,child_group_absent=child is not None and gone(child.pid),source_manifest_sha256=PIN,source_archive_sha256=sha(archive.read_bytes()),archive_matches=archive_match,current_execution_source_exact=parent_after==execution_after==expected,worker_report_sha256=sha(worker_path.read_bytes())if worker_path.exists()else None,outer_driver_sha256=sha(Path(__file__).read_bytes()),elapsed_s=time.monotonic()-start,environment_unset=list(unset),processes=observed if child is not None else None,synthetic_only=True,model_execution=False,encrypted_execution=False,scope='Public component tests and schedule; no learned wholegraph, HE, performance or scientific acceptance claim')
if primary is not None:result['outer_error_type']=type(primary).__name__
dump('outer-report.json',result)
print(json.dumps({k:result[k]for k in('passed','tests','child_returncode','child_group_absent','source_manifest_sha256','worker_report_sha256','elapsed_s')}),flush=True)
raise SystemExit(0 if passed else 1)
