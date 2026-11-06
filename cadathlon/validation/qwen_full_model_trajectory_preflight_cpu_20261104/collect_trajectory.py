"""Immutable whole28 fixture collector; explicit execute required for trajectories."""
from pathlib import Path
import argparse,hashlib,io,json,os,sys,tarfile,time,xml.etree.ElementTree as ET
from validation_cleanup import cleanup_signals,termination_signals
from validation_terminal import terminal_scope
from component_processes import OwnedChild
from finish import finish
from trajectory_assets import requirement,runtime_gate,sha
P=Path(__file__).resolve().parent

def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def source_hashes(root=None):
    root=P if root is None else root
    return {n:sha(root/n) for n in json.loads((root/'trajectory_inventory.json').read_bytes())['sources']}
def archive(out,before):
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for n,h in before.items():
            raw=(P/n).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source changed during snapshot')
            m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
def verify_archive(out,before):
    with tarfile.open(out/'sources.tar.gz') as t:
        ms=t.getmembers()
        if len(ms)!=len(before) or len({m.name for m in ms})!=len(ms) or any(not m.isfile() for m in ms):raise ValueError('exact source archive required')
        if {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}!=before:raise ValueError('source archive bytes differ')
def execution_tree(out,before):
    root=out/'execution';root.mkdir(exist_ok=False)
    with tarfile.open(out/'sources.tar.gz') as t:
        for member in t.getmembers():
            name=Path(member.name)
            if name.is_absolute() or '..' in name.parts or name.suffix=='.pyc' or not member.isfile():raise ValueError('plain relative source inventory required')
            raw=t.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest()!=before[member.name]:raise ValueError('execution source hash mismatch')
            destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    if source_hashes(root)!=before:raise ValueError('execution tree differs from archive')
    return root

def collect(out,*,execute=False):
    # main acquired this fresh output before entering republication scope.
    start=time.monotonic();primary=None;cleanup_error=None;active=None;before={};execution=None
    report=dict(passed=False,complete=False,cleanup_completed=False,termination_scope_completed=False,children=[],mode='actual_ideal_trajectory' if execute else 'preflight',
        HE_executed=False,model_forward=False,scope='Fixed full28 packed ideal owner loops; identity refresh, no CKKS/native benchmark claim.')
    def child(command,name):
        nonlocal active
        with (out/(name+'.log')).open('w') as log:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(execution));env.pop('PYTHONOPTIMIZE',None)
            active=OwnedChild(command,cwd=execution,stdout=log,env=env);report['children'].append(active.record)
            result=active.wait()
            if not all(active.record.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')):raise RuntimeError('owned worker group remains')
            active=None;return result
    try:
        write(out/'report.json',report);req=requirement();runtime_gate(req)
        report['requirement_sha256']=sha(P/'trajectory_requirement.json');report['cpu_affinity']=sorted(os.sched_getaffinity(0));report['environment']={n:os.environ.get(n) for n in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS')}
        import importlib.metadata,platform
        report['software']=dict(python=sys.version,numpy=importlib.metadata.version('numpy'),platform=platform.platform())
        before=source_hashes();report['source_hashes_before']=before;archive(out,before);verify_archive(out,before)
        execution=execution_tree(out,before);report['execution_source_hashes_before']=source_hashes(execution)
        if report['execution_source_hashes_before']!=before:raise ValueError('execution source drift before child launch')
        if execute:
            report['fixtures']=[]
            for index,name in enumerate(('S1','S8')):
                code=child([sys.executable,'-B','trajectory_worker.py','--execute','--fixture',str(index),'--output',str(out/name)],name)
                record=json.loads((out/name/'report.json').read_bytes())
                if record.get('source_hashes_before')!=before or record.get('source_hashes_after')!=before:raise ValueError('worker source linkage failed')
                if record.get('fixture_index')!=index:raise ValueError('worker fixture order differs')
                report['fixtures'].append(dict(index=index,report_sha256=sha(out/name/'report.json'),exit_code=code,passed=record.get('passed')is True,
                    complete=record.get('complete')is True,cleanup_completed=record.get('cleanup_completed')is True,termination_scope_completed=record.get('termination_scope_completed')is True,
                    coverage=record.get('coverage'),failure_detail=record.get('failure_detail')))
                write(out/'report.json',report)
            report['complete']=len(report['fixtures'])==2
        else:
            tests=json.loads((P/'trajectory_inventory.json').read_bytes())['tests']
            report['test_exit_code']=child([sys.executable,'-B','audit_public.py',str(out/'audit.json'),'pytest','-q','-p','no:cacheprovider',*tests,'--rootdir=.',f'--junitxml={out}/tests.xml'],'tests')
            suites=ET.parse(out/'tests.xml').getroot().findall('testsuite');report['tests']={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','errors','failures','skipped')}
            expected=json.loads((P/'trajectory_inventory.json').read_bytes())['expected_tests']
            if report['tests']!=dict(tests=expected,errors=0,failures=0,skipped=0):raise ValueError('complete preflight suite required')
            audit=json.loads((out/'audit.json').read_bytes())
            if not audit['passed'] or audit['denied_attempts'] or audit['backend_modules']:raise ValueError('preflight must not read actual bodies/import backend')
            report.update(complete=True,zero_audited_body_reads=True)
    except BaseException as error:primary=error;report['failure_detail']=dict(type=type(error).__name__,message=str(error),notes=getattr(error,'__notes__',[]))
    finally:
        with terminal_scope(report,out/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            def attempt(fn):
                nonlocal cleanup_error
                try:fn()
                except BaseException as e:
                    if cleanup_error is None:cleanup_error=e
                    elif isinstance(cleanup_error,Exception) and not isinstance(e,Exception):e.add_note(repr(cleanup_error));cleanup_error=e
                    else:cleanup_error.add_note(repr(e))
            if active is not None:attempt(lambda:active.close(primary=primary))
            attempt(lambda:verify_archive(out,before));attempt(lambda:report.update(source_hashes_after=source_hashes()))
            if execution is not None:attempt(lambda:report.update(execution_source_hashes_after=source_hashes(execution)))
            report['execution_files_unchanged']=bool(before) and before==report.get('execution_source_hashes_before')==report.get('execution_source_hashes_after')
            report['source_files_unchanged']=bool(before) and before==report.get('source_hashes_after')
            report['cleanup_completed']=cleanup_error is None and all(all(r.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')) for r in report['children'])
            attempt(lambda:report.update(source_archive_sha256=sha(out/'sources.tar.gz')))
            report['passed']=primary is None and cleanup_error is None and report['complete'] and report['source_files_unchanged'] and report['execution_files_unchanged'] and report['cleanup_completed'] and (
                all(all(r[n] for n in ('passed','complete','cleanup_completed','termination_scope_completed')) and r['exit_code']==0 for r in report['fixtures']) if execute else report.get('test_exit_code')==0)
            report['elapsed_s']=time.monotonic()-start
            if cleanup_error:report['cleanup_error']=dict(type=type(cleanup_error).__name__,message=str(cleanup_error))
            write(out/'report.json',report)
    if primary is not None:
        if cleanup_error:primary.add_note(repr(cleanup_error))
        raise primary
    if cleanup_error:raise cleanup_error
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    report=finish(args.output/'report.json',lambda:collect(args.output,execute=args.execute),write,lambda p:p.read_bytes(),termination=termination_signals,shield=cleanup_signals)
    print(json.dumps(dict(passed=report['passed'],mode=report['mode'],sources=len(report.get('source_hashes_before',{})))))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
