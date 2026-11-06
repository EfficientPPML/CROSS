"""Source-bound ideal/public tests, with no numerical HE or model execution."""
from pathlib import Path
import hashlib,io,json,os,signal,subprocess,sys,tarfile,time
import xml.etree.ElementTree as ET
from validation_cleanup import cleanup_signals,termination_signals
from validation_terminal import terminal_scope
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def paths():
    names=['README.md','DESIGN.md','design.json','constants.json','source_inventory.json','site_metadata.json',
        'prepare_gamma_fixtures.py','test_gamma_fixtures.py','ideal_primitives.py','ideal_refresh.py',
        'test_rmsnorm.py','test_rms_guards.py','test_negative_accessors.py','plan_public.py','collect_public.py','verify.py',
        'validation_cleanup.py','validation_terminal.py','gamma-preparation-v1/report.json','gamma-preparation-v1/sources.tar.gz']
    for directory in ('inputs','reference_sources','checkout/cadathlon/common'):
        names.extend(str(x.relative_to(P)) for x in (P/directory).iterdir() if x.is_file())
    return {n:P/n for n in sorted(set(names))}
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def archive_readback(path,expected):
    with tarfile.open(path) as t:
        members=t.getmembers()
        if len(members)!=len({m.name for m in members}) or len(members)!=len(expected) or any(not m.isfile() for m in members):raise ValueError('exact source archive inventory required')
        if {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}!=expected:raise ValueError('archive source bytes differ')
    return True
def collect():
    out=P/'source-bound-v1';out.mkdir(exist_ok=False);start=time.monotonic();files=paths();primary=cleanup_error=None;active=None
    report=dict(passed=False,cleanup_completed=False,encrypted_execution=False,model_forward=False,children=[],
        scope='Actual frozen RMSNorm loops with NumPy primitive/security/type and identity-refresh doubles; real authenticated public gammas only.',
        cpu_affinity=sorted(os.sched_getaffinity(0)),wrapper_pid=os.getpid())
    def child(command,name):
        nonlocal active
        with (out/(name+'.log')).open('w') as stream:
            active=subprocess.Popen(command,cwd=P,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            row=dict(name=name,pid=active.pid,process_reaped=False);report['children'].append(row);write(out/'report.json',report)
            code=active.wait();row.update(process_reaped=True,exit_code=code);active=None;return code
    try:
        write(out/'report.json',report);before={n:sha(p) for n,p in files.items()};report['source_hashes_before']=before
        report['external_fixture_sha256_before']=sha(P/'client_gamma_fixtures.npz')
        gamma=json.loads((P/'gamma-preparation-v1/report.json').read_bytes())
        if gamma['passed'] is not True or report['external_fixture_sha256_before']!=gamma['fixture_npz_sha256']:raise ValueError('authenticated public gamma fixture required')
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for name,path in files.items():
                raw=path.read_bytes();item=tarfile.TarInfo(name);item.size=len(raw);t.addfile(item,io.BytesIO(raw))
        archive_readback(out/'sources.tar.gz',before)
        main=[sys.executable,'-m','pytest','-q','test_rmsnorm.py','test_rms_guards.py','test_negative_accessors.py','--rootdir=.',f'--junitxml={out}/tests.xml']
        report['test_exit_code']=child(main,'tests')
        report['gamma_test_exit_code']=child([sys.executable,'-m','pytest','-q','test_gamma_fixtures.py','--rootdir=.',f'--junitxml={out}/gamma_tests.xml'],'gamma_tests')
        report['plan_exit_code']=child([sys.executable,'plan_public.py'],'plan')
        for name,expected in [('tests',295),('gamma_tests',6)]:
            suites=ET.parse(out/(name+'.xml')).getroot().findall('testsuite');counts={k:sum(int(x.get(k,0)) for x in suites) for k in ('tests','failures','errors','skipped')}
            report[name]=counts
            if counts!=dict(tests=expected,failures=0,errors=0,skipped=0):raise ValueError('complete fixed test inventory required: '+name)
        import shutil
        shutil.copy2(P/'public_plan.json',out/'public_plan.json');report['operation_complete']=True
    except BaseException as error:primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        def attempt(label,fn):
            nonlocal cleanup_error
            try:return fn()
            except BaseException as error:
                report.setdefault('cleanup_errors',[]).append(dict(operation=label,type=type(error).__name__))
                if primary is not None:primary.add_note(label+': '+repr(error))
                elif cleanup_error is None:cleanup_error=error
                elif not isinstance(error,Exception) and isinstance(cleanup_error,Exception):error.add_note(repr(cleanup_error));cleanup_error=error
                else:cleanup_error.add_note(label+': '+repr(error))
        with terminal_scope(report,out/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            def stop():
                if active is not None:
                    if active.poll() is None:
                        os.killpg(active.pid,signal.SIGTERM)
                        try:active.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                    report['children'][-1].update(process_reaped=True,exit_code=active.returncode)
            attempt('child cleanup',stop)
            report['source_hashes_after']=attempt('source final bytes',lambda:{n:sha(p) for n,p in files.items()})
            report['source_files_unchanged']=bool(report.get('source_hashes_before')) and report.get('source_hashes_before')==report['source_hashes_after']
            report['source_archive_verified']=attempt('archive final readback',lambda:archive_readback(out/'sources.tar.gz',report['source_hashes_before'])) is True
            report['source_archive_sha256']=attempt('archive digest',lambda:sha(out/'sources.tar.gz'))
            report['external_fixture_sha256_after']=attempt('fixture final digest',lambda:sha(P/'client_gamma_fixtures.npz'))
            report['external_fixture_unchanged']=report.get('external_fixture_sha256_before')==report['external_fixture_sha256_after'] and report['external_fixture_sha256_after'] is not None
            report['cleanup_completed']=cleanup_error is None and all(x['process_reaped'] for x in report['children'])
            report['elapsed_s']=time.monotonic()-start
            report['passed']=bool(primary is None and cleanup_error is None and report.get('operation_complete') and
                report.get('test_exit_code')==report.get('gamma_test_exit_code')==report.get('plan_exit_code')==0 and
                report['source_files_unchanged'] and report['source_archive_verified'] and report['external_fixture_unchanged'] and report['cleanup_completed'])
            write(out/'report.json',report)
    if primary is not None:raise primary
    if cleanup_error is not None:raise cleanup_error
    print(json.dumps(dict(passed=report['passed'],tests=report['tests'],gamma_tests=report['gamma_tests'],sources=len(report['source_hashes_before']),archive=report['source_archive_sha256'])))
    return report
if __name__=='__main__':
    try:
        with termination_signals():result=collect()
    except BaseException as error:
        path=P/'source-bound-v1/report.json'
        try:
            r=json.loads(path.read_bytes());r['passed']=False;r['termination_scope_failed']=dict(type=type(error).__name__)
            with cleanup_signals():write(path,r)
        except BaseException as publication:error.add_note(repr(publication))
        raise
    raise SystemExit(0 if result['passed'] else 1)
