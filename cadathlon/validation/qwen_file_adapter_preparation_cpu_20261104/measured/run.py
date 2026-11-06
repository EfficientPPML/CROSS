from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
import xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
SOURCE=Path('/tmp/qwen-organizer-file-adapter-v4-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def inventory(root):
    rows={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('symlink in captured development sources')
        if path.is_file():rows[path.relative_to(root).as_posix()]=sha(path.read_bytes())
    return rows
def write(name,value):(P/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
before=inventory(SOURCE)
expected_raw=Path('/tmp/qwen-organizer-adapter-development-20261104/collection-v4/sources-before.json').read_bytes()
assert sha(expected_raw)=='d3d75edbcd970ae9fc40e30911d115d27771f0fb784faf7c9a4930b11344f64c'
assert before==json.loads(expected_raw)
execution=P/'execution';execution.mkdir(exist_ok=False)
with tarfile.open(P/'sources.tar.gz','w:gz')as archive:
    for name,pin in before.items():
        raw=(SOURCE/name).read_bytes();assert sha(raw)==pin
        target=execution/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
write('sources-before.json',before)
env=os.environ.copy()
for name in ('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(name,None)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
args=['taskset','-c','176-179','/tmp/cross-pretrained-validation-env/bin/python','-I','-B','-m','pytest','-q','-p','no:cacheprovider','-c','/dev/null',
    '--rootdir='+str(execution),'--confcutdir='+str(execution),'--basetemp='+str(P/'tmp'),'-o','log_file='+str(P/'pytest.log'),
    '--junitxml='+str(P/'tests.xml'),*[str(execution/n)for n in ('test_file_contract.py','test_public_parameters.py','test_file_lifecycle.py','test_request_files.py','test_layout_integration.py','layout_source/test_qwen_layout.py')]]
started=time.monotonic()
with (P/'stdout.log').open('wb')as stream:
    child=subprocess.Popen(args,cwd=execution,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    pid=child.pid;pgid=os.getpgid(pid);write('live.json',dict(pid=pid,pgid=pgid,wrapper_pid=os.getpid(),wrapper_pgid=os.getpgrp(),args=args))
    print(json.dumps(dict(child_pid=pid,child_pgid=pgid)),flush=True)
    code=child.wait()
after=inventory(SOURCE);run_after=inventory(execution)
write('sources-after.json',after);write('execution-after.json',run_after)
try:os.killpg(pgid,0);absent=False
except ProcessLookupError:absent=True
xml=ET.parse(P/'tests.xml').getroot()
cases=xml.findall('.//testcase');bad=sum(len(xml.findall('.//'+kind))for kind in ('failure','error','skipped'))
report=dict(scope='source-captured focused public/synthetic component preparation; no backend/model/actual session execution',test_count=len(cases),expected_tests=191,test_failures_errors_skips=bad,source_count=len(before),exit_code=code,
    child_pid=pid,child_pgid=pgid,child_reaped=True,child_group_absent=absent,source_unchanged=before==after==run_after,
    elapsed_s=time.monotonic()-started,source_archive_sha256=sha((P/'sources.tar.gz').read_bytes()),driver_sha256=sha(Path(__file__).read_bytes()))
report['passed']=code==0 and len(cases)==191 and bad==0 and report['source_unchanged']and absent
write('report.json',report);print(json.dumps(report),flush=True)
raise SystemExit(0 if report['passed']else 1)
