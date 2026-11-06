"""Snapshot ideal owner/public checks without importing a crypto backend."""
from pathlib import Path
import hashlib,io,json,os,signal,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET
from validation_cleanup import cleanup_signals,termination_signals
from validation_terminal import terminal_scope
from component_processes import OwnedChild
from finish import finish
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_paths():return [P/n for n in json.loads((P/'silu_inventory.json').read_bytes())['sources']]
def verify_inputs():
    pins=json.loads((P/'predecessors.json').read_bytes())
    for name,row in pins.items():
        if sha(P/name)!=row['sha256']:raise ValueError('frozen predecessor differs: '+name)
def public_inputs():
    return {str(p.relative_to(P)):sha(p) for p in sorted((P/'inputs').glob('*')) if p.is_file()}
def expected_metrics():
    return {f'layer{layer}.rows{rows}' for layer in range(28) for rows in (1,16)}|{f'layer{layer}.partial3' for layer in (0,3,26,27)}
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def collect():
    out=P/'source-bound-owner-v1';out.mkdir(exist_ok=False);start=time.monotonic();paths=[];primary=None;cleanup_error=None;active=None
    report=dict(passed=False,cleanup_completed=False,termination_scope_completed=False,encrypted_execution=False,backend_imported=False,children=[],
        scope='All28 frozen candidate3 SiLU rules, full-slot ideal payloads/identity refresh, no parameters/model/backend/HE or perturbed refresh-domain guarantee.')
    def child(command,filename):
        nonlocal active
        with (out/filename).open('w') as log:
            env=dict(os.environ);env['QWEN_SILU_METRICS_PATH']=str(out/'ideal_results.json')
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
        verify_inputs();report['predecessor_verified']=True;report['public_inputs_before']=public_inputs()
        before={str(p.relative_to(P)):sha(p) for p in paths};report['source_hashes_before']=before
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for p in paths:
                b=p.read_bytes();i=tarfile.TarInfo(str(p.relative_to(P)));i.size=len(b);t.addfile(i,io.BytesIO(b))
        report['test_exit_code']=child([sys.executable,'-m','pytest','-q','test_silu.py','test_binding_guard.py','test_component_processes.py','test_collector.py','--rootdir=.',f'--junitxml={out}/tests.xml'],'tests.log')
        report['plan_exit_code']=child([sys.executable,'plan_public.py'],'plan.log')
        with tarfile.open(out/'sources.tar.gz') as t:
            ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(before)
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms if m.isfile()}==before
        suites=ET.parse(out/'tests.xml').getroot().findall('testsuite');report['tests']={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','errors','failures','skipped')}
        if report['tests']!=dict(tests=json.loads((P/'silu_inventory.json').read_bytes())['expected_tests'],errors=0,failures=0,skipped=0):raise ValueError('complete declared owner/plan checks required')
        metrics=json.loads((out/'ideal_results.json').read_bytes())
        if set(metrics)!=expected_metrics():raise ValueError('all60 declared fullslot fixture results required')
        if any(row['max_abs_error']>row['gate'] or row['gate']!=1e-8 or row['output_level']!=6 or row['physical_slots_checked']!=65536 for row in metrics.values()):raise ValueError('all-slot fixed metrics failed')
        report['ideal_result_sha256']=sha(out/'ideal_results.json')
        report['source_archive_sha256']=sha(out/'sources.tar.gz')
        import shutil
        shutil.copy2(P/'public_plan.json',out/'public_plan.json')
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
                report['public_inputs_after']=public_inputs()
                report['public_inputs_unchanged']=report.get('public_inputs_before')==report['public_inputs_after']
            except BaseException as error:
                if cleanup_error is None or (not isinstance(error,Exception) and isinstance(cleanup_error,Exception)):cleanup_error=error
                else:cleanup_error.add_note(repr(error))
            report['elapsed_s']=time.monotonic()-start
            report['passed']=(primary is None and cleanup_error is None and report.get('test_exit_code')==report.get('plan_exit_code')==0
                and report.get('source_files_unchanged') is True and report.get('public_inputs_unchanged')is True
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
