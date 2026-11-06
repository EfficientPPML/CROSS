"""Snapshot ideal owner/public checks without importing a crypto backend."""
from pathlib import Path
import hashlib,io,json,os,signal,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET
from validation_cleanup import cleanup_signals,termination_signals
from validation_terminal import terminal_scope
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_paths():return [P/n for n in json.loads((P/'owner_inventory.json').read_bytes())['sources']]
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def collect():
    out=P/'source-bound-owner-v1';out.mkdir(exist_ok=False);start=time.monotonic();paths=[];primary=None;cleanup_error=None;active=None
    report=dict(passed=False,cleanup_completed=False,encrypted_execution=False,backend_imported=False,children=[],
        scope='Full-slot ideal execution of literal causal-softmax/PV-bypass owner loops with primitive and identity-refresh doubles; no model/backend/HE execution.')
    def child(command,filename):
        nonlocal active
        with (out/filename).open('w') as log:
            active=subprocess.Popen(command,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row=dict(pid=active.pid,process_reaped=False);report['children'].append(row)
            result=active.wait();row.update(process_reaped=True,exit_code=result);active=None;return result
    try:
        write(out/'report.json',report)
        import importlib.metadata,platform
        report['software']=dict(python=sys.version,numpy=importlib.metadata.version('numpy'),platform=platform.platform())
        report['cpu_affinity']=sorted(os.sched_getaffinity(0))
        report['environment']={name:os.environ.get(name) for name in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS')}
        paths=source_paths()
        before={str(p.relative_to(P)):sha(p) for p in paths};report['source_hashes_before']=before
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for p in paths:
                b=p.read_bytes();i=tarfile.TarInfo(str(p.relative_to(P)));i.size=len(b);t.addfile(i,io.BytesIO(b))
        report['test_exit_code']=child([sys.executable,'-m','pytest','-q','test_plan.py','test_causal_softmax_guards.py','test_causal_softmax_ideal.py','--rootdir=.',f'--junitxml={out}/tests.xml'],'tests.log')
        report['plan_exit_code']=child([sys.executable,'plan_public.py'],'plan.log')
        with tarfile.open(out/'sources.tar.gz') as t:
            ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(before)
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms if m.isfile()}==before
        suites=ET.parse(out/'tests.xml').getroot().findall('testsuite');report['tests']={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','errors','failures','skipped')}
        if report['tests']!=dict(tests=json.loads((P/'owner_inventory.json').read_bytes())['expected_tests'],errors=0,failures=0,skipped=0):raise ValueError('complete declared owner/plan checks required')
        report['source_archive_sha256']=sha(out/'sources.tar.gz')
        import shutil
        shutil.copy2(P/'public_plan.json',out/'public_plan.json')
    except BaseException as error:primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,out/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            try:
                if active is not None:
                    if active.poll() is None:
                        os.killpg(active.pid,signal.SIGTERM)
                        try:active.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                    report['children'][-1].update(process_reaped=True,exit_code=active.returncode)
                report['cleanup_completed']=True
            except BaseException as error:cleanup_error=error
            try:
                report['source_hashes_after']={str(p.relative_to(P)):sha(p) for p in paths}
                report['source_files_unchanged']=report.get('source_hashes_before')==report['source_hashes_after']
            except BaseException as error:
                if cleanup_error is None or (not isinstance(error,Exception) and isinstance(cleanup_error,Exception)):cleanup_error=error
                else:cleanup_error.add_note(repr(error))
            report['elapsed_s']=time.monotonic()-start
            report['passed']=(primary is None and cleanup_error is None and report.get('test_exit_code')==report.get('plan_exit_code')==0
                and report.get('source_files_unchanged') is True and all(r['process_reaped'] for r in report['children']))
            write(out/'report.json',report)
    if primary is not None:
        if cleanup_error is not None:primary.add_note(repr(cleanup_error))
        raise primary
    if cleanup_error is not None:raise cleanup_error
    print(json.dumps(dict(passed=report['passed'],tests=report['tests'],sources=len(report['source_hashes_before']),archive=report['source_archive_sha256'])))
    return report
if __name__=='__main__':
    result=None
    try:
        with termination_signals():result=collect()
    except BaseException as error:
        path=P/'source-bound-owner-v1/report.json'
        try:
            r=json.loads(path.read_bytes());r['passed']=False;r['termination_scope_failed']=dict(type=type(error).__name__,message=str(error))
            with cleanup_signals():write(path,r)
        except BaseException as publication:error.add_note(repr(publication))
        raise
    raise SystemExit(0 if result['passed'] else 1)
