from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET
src=Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104');out=Path('/tmp/qwen-stage-integration-successor-proof-20261104/development-v1');out.mkdir(parents=True,exist_ok=False);exe=out/'execution';exe.mkdir()
sha=lambda x:hashlib.sha256(x).hexdigest()
def inventory(root):return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}
expected_path=Path('/tmp/qwen-stage-integration-v2-sources-20261104.json');expected_raw=expected_path.read_bytes();assert sha(expected_raw)=='225cc51ab57696ad69b6f973f2402d8e2851db27b26fc35e66b5c610032e9f5e';expected=json.loads(expected_raw)
before=inventory(src);assert before==expected;shutil.copyfile(__file__,out/'run.py');(out/'driver-before.json').write_text(json.dumps(dict(sha256=sha(Path(__file__).read_bytes())))+'\n');(out/'sources-before.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
for n,h in before.items():p=exe/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((src/n).read_bytes());assert sha(p.read_bytes())==h
with tarfile.open(out/'sources.tar.gz','w:gz')as tar:
 for n in before:tar.add(exe/n,arcname=n)
cmd=['taskset','-c','176-179','/tmp/cross-pretrained-validation-env/bin/python','-I','-B','-m','pytest','-q','-p','no:cacheprovider','--rootdir='+str(exe),'--confcutdir='+str(exe),'--import-mode=importlib','--junitxml='+str(out/'tests.xml'),str(exe/'test_stage_boundaries.py')]
env=os.environ.copy()
for n in ('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(n,None)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.monotonic()
with (out/'stdout.log').open('wb')as log:
 child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=exe,start_new_session=True)
 (out/'live.json').write_text(json.dumps(dict(wrapper_pid=os.getpid(),wrapper_pgid=os.getpgid(0),worker_pid=child.pid,worker_pgid=os.getpgid(child.pid),command=cmd),indent=2)+'\n')
 code=child.wait()
after=inventory(src);end=inventory(exe)
for n,v in [('sources-after.json',after),('execution-after.json',end)]: (out/n).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
try:os.killpg(child.pid,0);gone=False
except ProcessLookupError:gone=True
xml=ET.parse(out/'tests.xml').getroot();tests=len(xml.findall('.//testcase'));fail=sum(len(xml.findall('.//'+x))for x in ('failure','error','skipped'))
report=dict(scope='Identical12 joined recording boundaries; no actual role/backend/context/body execution',passed=code==0 and tests==12 and fail==0 and before==after==end and gone,source_count=len(before),source_unchanged=before==after==end,test_count=tests,test_failures_errors_skips=fail,exit_code=code,child_reaped=True,child_group_absent=gone,elapsed_seconds=time.monotonic()-start,source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()))
(out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps(report));sys.exit(0 if report['passed']else 1)
