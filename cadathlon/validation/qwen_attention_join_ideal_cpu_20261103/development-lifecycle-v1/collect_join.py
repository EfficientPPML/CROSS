"""Snapshot ideal owner/public checks without importing a crypto backend."""
from pathlib import Path
import hashlib,io,json,os,signal,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET
from validation_cleanup import cleanup_signals,termination_signals
from validation_terminal import terminal_scope
from component_processes import OwnedChild
from finish import finish
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_paths():return [P/n for n in json.loads((P/'join_inventory.json').read_bytes())['sources']]
def verify_inputs():
    pins=json.loads((P/'source_inputs.json').read_bytes())
    for name,row in pins.items():
        if sha(P/name)!=row['sha256']:raise ValueError('frozen predecessor differs: '+name)
    report=json.loads((P/'qkv-export-v1/report.json').read_bytes())
    if report['passed']is not True or len(report['arrays'])!=84 or report['source_hashes_before']!=report['source_hashes_after']:raise ValueError('public export incomplete')
    if sha(P/'qkv-export-v1/sources.tar.gz')!=report['source_archive_sha256']:raise ValueError('export source archive differs')
    with tarfile.open(P/'qkv-export-v1/sources.tar.gz') as t:
        members=t.getmembers()
        if len(members)!=len({m.name for m in members}) or not all(m.isfile() for m in members):raise ValueError('exact export members required')
        if {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}!=report['source_hashes_before']:raise ValueError('export archive member differs')
    if sha(P/'qkv-export-v1/parameter_manifest.json')!='c707068e1a247f4ef14fda3493a3e2edf9dea2f11ec19e94531a64d80dbcf48b':raise ValueError('pinned public parameter manifest differs')
def public_arrays():
    manifest=json.loads((P/'qkv-export-v1/parameter_manifest.json').read_bytes())
    values={row['file']:sha(P/'public_qkv_parameters'/row['file']) for row in manifest['matrices'].values()}
    if any(values[row['file']]!=row['file_sha256'] for row in manifest['matrices'].values()):raise ValueError('public QKV file differs')
    gamma=Path('/tmp/qwen-rmsnorm-owner-20261103/client_gamma_fixtures.npz')
    values['gamma_npz']=sha(gamma)
    if values['gamma_npz']!='5f40efb5fac32037999ac8aff5f01bae7f937dd0f1d308b2a694883d01b4ef30':raise ValueError('public gamma fixture differs')
    return values
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def collect():
    out=P/'source-bound-owner-v1';out.mkdir(exist_ok=False);start=time.monotonic();paths=[];primary=None;cleanup_error=None;active=None
    report=dict(passed=False,cleanup_completed=False,termination_scope_completed=False,encrypted_execution=False,backend_imported=False,children=[],
        scope='Public genuine Q/K/V plus gamma fixture, full-slot ideal joined owner loops and identity refresh. No model forward/backend/HE or perturbed refresh-domain guarantee.')
    def child(command,filename):
        nonlocal active
        with (out/filename).open('w') as log:
            env=dict(os.environ);env['QWEN_JOIN_METRICS_PATH']=str(out/'ideal_results.json')
            active=OwnedChild(command,cwd=P,stdout=log,env=env)
            report['children'].append(active.record)
            result=active.wait()
            if any(active.record.get(n)is not True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')):
                raise RuntimeError('direct child wait and owned group absence required')
            active=None;return result
    try:
        write(out/'report.json',report)
        import importlib.metadata,platform
        report['software']=dict(python=sys.version,numpy=importlib.metadata.version('numpy'),platform=platform.platform())
        report['cpu_affinity']=sorted(os.sched_getaffinity(0))
        report['environment']={name:os.environ.get(name) for name in ('PYTHONDONTWRITEBYTECODE','PYTHONOPTIMIZE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS')}
        paths=source_paths()
        verify_inputs();report['predecessor_verified']=True;report['public_arrays_before']=public_arrays()
        before={str(p.relative_to(P)):sha(p) for p in paths};report['source_hashes_before']=before
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for p in paths:
                b=p.read_bytes();i=tarfile.TarInfo(str(p.relative_to(P)));i.size=len(b);t.addfile(i,io.BytesIO(b))
        report['test_exit_code']=child([sys.executable,'-m','pytest','-q','test_join.py','test_join_guards.py','test_join_full_tiles.py','test_export_qkv.py','test_export_fd.py','--rootdir=.',f'--junitxml={out}/tests.xml'],'tests.log')
        report['plan_exit_code']=child([sys.executable,'check_join_plan.py'],'plan.log')
        with tarfile.open(out/'sources.tar.gz') as t:
            ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(before)
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms if m.isfile()}==before
        suites=ET.parse(out/'tests.xml').getroot().findall('testsuite');report['tests']={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','errors','failures','skipped')}
        if report['tests']!=dict(tests=json.loads((P/'join_inventory.json').read_bytes())['expected_tests'],errors=0,failures=0,skipped=0):raise ValueError('complete declared owner/plan checks required')
        metrics=json.loads((out/'ideal_results.json').read_bytes())
        if set(metrics)!={'layer0.S1','layer0.S8','layer0.S16','layer0.S19','layer0.S32','layer5.S1','layer9.S1'}:raise ValueError('all seven genuine-weight fixture results required')
        report['ideal_result_sha256']=sha(out/'ideal_results.json')
        report['source_archive_sha256']=sha(out/'sources.tar.gz')
        import shutil
        shutil.copy2(P/'join_actual_inventory.json',out/'join_actual_inventory.json')
    except BaseException as error:primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,out/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            try:
                if active is not None:active.close(primary=primary)
                report['cleanup_completed']=all(all(row.get(n)is True for n in
                    ('direct_child_reaped','group_empty_verified','cleanup_completed')) for row in report['children'])
            except BaseException as error:cleanup_error=error
            try:
                report['source_hashes_after']={str(p.relative_to(P)):sha(p) for p in paths}
                report['source_files_unchanged']=report.get('source_hashes_before')==report['source_hashes_after']
                report['public_arrays_after']=public_arrays()
                report['public_arrays_unchanged']=report.get('public_arrays_before')==report['public_arrays_after']
            except BaseException as error:
                if cleanup_error is None or (not isinstance(error,Exception) and isinstance(cleanup_error,Exception)):cleanup_error=error
                else:cleanup_error.add_note(repr(error))
            report['elapsed_s']=time.monotonic()-start
            report['passed']=(primary is None and cleanup_error is None and report.get('test_exit_code')==report.get('plan_exit_code')==0
                and report.get('source_files_unchanged') is True and report.get('public_arrays_unchanged')is True
                and report['cleanup_completed']is True and len(report['children'])==2
                and all(all(row.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')) for row in report['children']))
            write(out/'report.json',report)
    if primary is not None:
        if cleanup_error is not None:primary.add_note(repr(cleanup_error))
        raise primary
    if cleanup_error is not None:raise cleanup_error
    return report
def main():
    result=finish(P/'source-bound-owner-v1/report.json',collect,write,lambda path:path.read_bytes(),
        termination=termination_signals,shield=cleanup_signals)
    print(json.dumps(dict(passed=result['passed'],tests=result['tests'],sources=len(result['source_hashes_before']),archive=result['source_archive_sha256'])))
    return 0 if result['passed']else 1
if __name__=='__main__':raise SystemExit(main())
