from pathlib import Path
import json,hashlib,tarfile,subprocess,os,time,sys,xml.etree.ElementTree as ET
source=Path('/tmp/qwen-original-embedding-vocabulary-layout-20261104')
out=Path('/tmp/qwen-original-embedding-vocabulary-layout-proof-20261104/development-v1');out.mkdir(parents=True,exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(n,v):(out/n).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
expected={str(p.relative_to(source)):sha(p.read_bytes())for p in sorted(source.rglob('*'))if p.is_file()}
assert len(expected)==9 and not any(p.is_symlink()for p in source.rglob('*'))
dump('sources-before.json',expected);execution=out/'execution';execution.mkdir()
for n in expected:p=execution/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/n).read_bytes())
with tarfile.open(out/'sources.tar.gz','w:gz')as t:
 for n in expected:t.add(execution/n,arcname=n,recursive=False)
runner='''import sys,json,pytest
from pathlib import Path
source=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(source))
denied=[]
def audit(event,args):
 if event=='import'and args and str(args[0]).split('.')[0]in ('torch','transformers','jax','jaxlib','jaxite_word'):
  denied.append(str(args[0]));raise PermissionError('backend import disabled in synthetic layout proof')
sys.addaudithook(audit)
code=pytest.main(['-q','-p','no:cacheprovider','--rootdir='+str(source),str(source/'test_qwen_layout.py'),'--junitxml='+str(out/'tests.xml')])
(out/'worker-audit.json').write_text(json.dumps(dict(denied=denied,loaded_backend=[n for n in sys.modules if n.split('.')[0]in ('torch','transformers','jax','jaxlib','jaxite_word')],exit_code=int(code)),sort_keys=True)+'\\n')
raise SystemExit(int(code)if not denied else 70)
'''
(out/'worker_driver.py').write_text(runner);(out/'executed_driver.py').write_bytes(Path(__file__).read_bytes())
env=dict(PATH='/usr/bin:/bin',HOME='/home/jianming_gatech',LANG='C.UTF-8',TZ='UTC',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
command=['taskset','-c','108-111','/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(out/'worker_driver.py'),str(execution),str(out)]
start=time.monotonic()
with(out/'tests.log').open('wb')as log:
 child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 dump('live-observation.json',dict(outer_pid=os.getpid(),outer_pgid=os.getpgrp(),worker_pid=child.pid,worker_pgid=os.getpgid(child.pid),command=command,environment=env))
 code=child.wait()
try:os.killpg(child.pid,0)
except ProcessLookupError:gone=True
else:gone=False
current={str(p.relative_to(source)):sha(p.read_bytes())for p in sorted(source.rglob('*'))if p.is_file()};after={str(p.relative_to(execution)):sha(p.read_bytes())for p in sorted(execution.rglob('*'))if p.is_file()}
dump('sources-after.json',current);dump('execution-after.json',after)
with tarfile.open(out/'sources.tar.gz')as t:archived={x.name:sha(t.extractfile(x).read())for x in t.getmembers()if x.isfile()}
xml=ET.parse(out/'tests.xml').getroot();counts={n:len(list(xml.iter(tag)))for n,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]};audit=json.loads((out/'worker-audit.json').read_bytes())
r=dict(passed=code==0 and gone and current==after==archived==expected and not any(counts[k]for k in ('failures','errors','skipped'))and audit['denied']==audit['loaded_backend']==[],tests=counts,source_count=len(expected),source_manifest_sha256=sha((out/'sources-before.json').read_bytes()),source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),sources_unchanged=current==after==archived==expected,child_returncode=code,child_reaped=True,child_group_absent=gone,elapsed_s=time.monotonic()-start,synthetic_only=True,model_execution=False,encrypted_execution=False,scope='Public source plus synthetic NumPy layout tests; no actual model/input/key/ciphertext body reads.')
dump('report.json',r);print(json.dumps(r,sort_keys=True));raise SystemExit(0 if r['passed']else 1)
