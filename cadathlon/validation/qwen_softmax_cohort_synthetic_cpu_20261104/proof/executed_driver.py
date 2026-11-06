from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile,time,xml.etree.ElementTree as ET
LAB=Path('/tmp/qwen-causal-softmax-cohort-successor-20261104')
OUT=Path('/tmp/qwen-causal-softmax-cohort-public-proof-20261104/development-v1')
PYTHON='/tmp/cross-pretrained-validation-env/bin/python'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def group_absent(pid):
 try:os.killpg(pid,0)
 except ProcessLookupError:return True
 return False
parent=json.loads((LAB/'cohort_parent_sources.json').read_bytes())
extras=('cohort_parent_sources.json','COHORT_DESIGN.md','checkout/cadathlon/common/encrypted_qwen_causal_softmax_batched.py','test_softmax_cohorts.py','test_cohort_binding_review.py','test_binding_call_order.py')
assert len(parent)==191
manifest={n:sha((LAB/n).read_bytes())for n in (*parent,*extras)}
assert all(manifest[n]==h for n,h in parent.items())and len(manifest)==197
OUT.mkdir(parents=True);src=OUT/'execution_source';src.mkdir()
for n,h in manifest.items():
 target=src/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(LAB/n,target);assert sha(target.read_bytes())==h
shutil.copyfile(__file__,OUT/'executed_driver.py')
dump(OUT/'sources-before.json',manifest)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as archive:
 for n in sorted(manifest):archive.add(src/n,arcname=n,recursive=False)
command=['taskset','-c','108-111',PYTHON,'-I','-B','-m','pytest','--rootdir='+str(src),'--confcutdir='+str(src),'-p','no:cacheprovider','-q','test_softmax_cohorts.py','test_cohort_binding_review.py','test_binding_call_order.py','--junitxml='+str(OUT/'tests.xml')]
env=dict(PATH='/usr/bin:/bin',HOME='/home/jianming_gatech',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
start=time.monotonic()
with (OUT/'stdout.log').open('wb')as stdout,(OUT/'stderr.log').open('wb')as stderr:
 child=subprocess.Popen(command,cwd=src,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 observed=dict(collector_pid=os.getpid(),collector_pgid=os.getpgid(0),wrapper_pid=os.getppid(),wrapper_pgid=os.getpgid(os.getppid()),child_pid=child.pid,child_pgid=os.getpgid(child.pid),command=command)
 dump(OUT/'process-observed.json',observed);code=child.wait()
after={n:sha((LAB/n).read_bytes())for n in manifest};execution={n:sha((src/n).read_bytes())for n in manifest}
dump(OUT/'sources-after.json',after);dump(OUT/'execution-sources-after.json',execution)
with tarfile.open(OUT/'sources.tar.gz')as archive:
 members=archive.getmembers();archived={m.name:sha(archive.extractfile(m).read())for m in members if m.isfile()};archive_matches=len(members)==len(manifest)and archived==manifest
suite=ET.parse(OUT/'tests.xml').getroot();cases=suite.findall('.//testcase');bad=[n for n in cases if any(n.find(tag)is not None for tag in ('error','failure','skipped'))]
passed=code==0 and len(cases)==136 and not bad and after==manifest==execution and archive_matches and group_absent(child.pid)
report=dict(schema='qwen-softmax-cohort-public-proof-v1',passed=passed,synthetic_only=True,model_forward=False,encrypted_execution=False,source_count=len(manifest),parent191_unchanged=all(after[n]==h for n,h in parent.items()),sources_before=manifest,sources_after=after,execution_sources_after=execution,source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),archive_matches=archive_matches,driver_sha256=sha(Path(__file__).read_bytes()),tests=len(cases),bad_tests=len(bad),child_returncode=code,child_reaped=True,child_group_absent=group_absent(child.pid),elapsed_seconds=time.monotonic()-start,processes=observed,scope='fullslot synthetic arithmetic/counts/callback ownership; no real backend, private/model/input/key/CT bodies; no measured acceleration')
dump(OUT/'report.json',report);print(json.dumps({k:report[k]for k in('passed','source_count','tests','bad_tests','child_returncode','child_group_absent','source_archive_sha256','elapsed_seconds')}),flush=True)
raise SystemExit(0 if passed else 1)
