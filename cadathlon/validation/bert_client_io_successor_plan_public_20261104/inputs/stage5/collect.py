"""Source-bound client stage5 collector; private outputs retained, never archived."""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,os,stat,sys,tarfile,tempfile,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'runtime')]
import check_transformer_suite as process
import key_gate,reference_gate,dependencies
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
spec=importlib.util.spec_from_file_location('stage5_worker_runtime',HERE/'worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

def sources():
    paths={p.relative_to(HERE).as_posix():p for p in (HERE/'runtime').rglob('*')
        if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
    paths.update({p.name:p for p in HERE.glob('*.py')})
    paths.update({n:HERE/n for n in ('source_origin.json','dependencies.json','key_source_manifest.json','stage3_source_manifest.json','stage4_source_manifest.json')})
    return {n:p.read_bytes() for n,p in paths.items()}
def hashes(raw):return {n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}
def write(path,value):process.write_json(path,value)
def archive(path,raw):
    with tarfile.open(path,'w:gz') as tar:
        for name,data in raw.items():
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
def check_archive(path,expected):
    with tarfile.open(path) as tar:
        rows=tar.getmembers()
        return (len(rows)==len(expected)==len({m.name for m in rows}) and all(m.isfile() for m in rows)
            and {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in rows}==expected)
def outside(path,protected):
    path=Path(path).absolute()
    for base in protected:
        base=Path(base).absolute()
        if path==base or base in path.parents or path.resolve()==base.resolve() or base.resolve() in path.resolve().parents:
            raise ValueError('evidence must be outside all retained private/public prerequisites')
    return path
def fixture_private(evidence,expected_sha):
    raw=key_gate.read(Path(evidence)/'report.json')
    if key_gate.digest(raw)!=key_gate.sha_field(expected_sha):raise ValueError('external fixture report differs before output creation')
    name=key_gate.parse(raw).get('private_workspace')
    if type(name)is not str or not Path(name).is_absolute():raise ValueError('public fixture owner path required')
    return Path(name)

class BorrowedDirectory:
    def __init__(self,path,mode=None):
        self.path=Path(path);self.mode=mode;self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        s=os.fstat(self.fd);self.identity=(s.st_dev,s.st_ino)
        try:self.check()
        except BaseException as primary:
            try:self.close()
            except BaseException as e:primary.add_note('borrowed descriptor close: '+repr(e))
            raise
    def check(self):
        s=self.path.lstat()
        if not stat.S_ISDIR(s.st_mode) or (s.st_dev,s.st_ino)!=self.identity:raise ValueError('borrowed directory replaced')
        if self.mode is not None and stat.S_IMODE(s.st_mode)!=self.mode:raise ValueError('private directory permissions changed')
    def close(self):
        if self.fd is not None:fd=self.fd;self.fd=None;os.close(fd)

def run(output,*,key_evidence,key_report_sha256,key_workspace,model_evidence,model_report_sha256,model_workspace,
        stage3_evidence,stage3_report_sha256,stage4_evidence,stage4_report_sha256,io_workspace,
        fixture_evidence,fixture_report_sha256,reference_evidence,reference_report_sha256,execute=False,
        worker_python=Path('/tmp/cross-pretrained-validation-env/bin/python')):
    # Only authenticated public metadata is read to protect unknown private paths.
    ref=reference_gate.public_metadata(reference_evidence,reference_report_sha256)
    fixture_root=fixture_private(fixture_evidence,fixture_report_sha256)
    protected=(key_workspace,model_workspace,io_workspace,key_evidence,model_evidence,stage3_evidence,stage4_evidence,
        fixture_evidence,reference_evidence,fixture_root,ref['private_workspace'])
    output=outside(output,protected);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();primary=None;cleanup_error=None;record=None;active=False;before={};execution=None;owners=[];result_workspace=None
    report=dict(passed=False,execute=execute,actual_client_decode=False,actual_execution_attempted=False,
        all_processes_reaped=True,external_workspaces_never_removed=True,private_outputs_never_archived=True,
        scope='One size0 client decode and original P2 numeric/prediction gate, not ranked accuracy.',
        wrapper_pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    values=locals().copy()
    names=('key_evidence','key_report_sha256','key_workspace','model_evidence','model_report_sha256','model_workspace',
        'stage3_evidence','stage3_report_sha256','stage4_evidence','stage4_report_sha256','io_workspace',
        'fixture_evidence','fixture_report_sha256','reference_evidence','reference_report_sha256')
    gate_args={n:values[n] for n in names}
    def attempt(label,fn):
        nonlocal cleanup_error
        try:return fn()
        except BaseException as e:
            report.setdefault('cleanup_errors',[]).append(dict(operation=label,type=type(e).__name__,message=str(e)))
            if primary is not None:primary.add_note(label+': '+repr(e))
            elif cleanup_error is None:cleanup_error=e
            elif not isinstance(e,Exception) and isinstance(cleanup_error,Exception):e.add_note('earlier cleanup: '+repr(cleanup_error));cleanup_error=e
            else:cleanup_error.add_note(label+': '+repr(e))
    try:
        write(output/'report.json',report)
        if type(execute)is not bool or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit bool and CPU environment required')
        raw=sources();before=hashes(raw);archive(output/'sources.tar.gz',raw)
        if not check_archive(output/'sources.tar.gz',before):raise ValueError('startup source archive differs')
        execution=output/'execution_source';execution.mkdir()
        for name,data in raw.items():p=execution/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        if {n:hashlib.sha256((execution/n).read_bytes()).hexdigest() for n in before}!=before:raise ValueError('execution source differs')
        write(output/'source_manifest.json',before)
        report['reviewed_dependencies']=dependencies.check()
        profile=raw['runtime/stage/server_inputs/p60_profile_and_refresh.json'];plan=raw['runtime/stage/server_inputs/twelve_layer_plan.json']
        authorized=worker.authenticate(**gate_args,profile_bytes=profile,plan_bytes=plan)
        report['prerequisite']=worker.prerequisite_summary(*authorized)
        for path in (io_workspace,key_workspace,model_workspace,fixture_root,ref['private_workspace']):owners.append(BorrowedDirectory(path))
        report.update(source_hashes_start=before,source_archive_sha256=process.sha256(output/'sources.tar.gz'),
            software=process.software_versions(),worker_python=str(worker_python),
            environment={n:os.environ.get(n) for n in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
        if execute:
            result_workspace=Path(tempfile.mkdtemp(prefix='bert-stage5-private-',dir='/tmp'));os.chmod(result_workspace,0o700)
            owner=BorrowedDirectory(result_workspace,mode=0o700);owners.append(owner)
            report.update(private_result_workspace=str(result_workspace),private_result_workspace_identity=list(owner.identity))
            paths=worker.make_paths(authorized,key_workspace,io_workspace,result_workspace);worker.fresh_paths(paths,result_workspace)
            directory=output/'worker';directory.mkdir();active=True;report['actual_execution_attempted']=True;write(output/'report.json',report)
            args=gate_args|dict(result_workspace=result_workspace,output=directory/'worker_report.json',source_manifest=output/'source_manifest.json')
            command=[str(worker_python),str(execution/'worker.py')]
            for name,value in args.items():command+=['--'+name.replace('_','-'),str(value)]
            previous=process.child_environment_overrides;process.child_environment_overrides=lambda:{'JAX_PLATFORMS':'cpu','PYTHONUNBUFFERED':'1'}
            try:record=process.run_case(command,directory,timeout=None,cleanup_grace_s=120.)
            finally:process.child_environment_overrides=previous
            active=False;report['worker_process']=record
            if (type(record)is not dict or type(record.get('returncode'))is not int or record['returncode'] not in (0,1)
                    or record.get('error') not in (None,('runner exited 1; see stderr.log' if record.get('returncode')==1 else None))
                    or any(record.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))):
                raise ValueError('client worker failed or cleanup incomplete')
            result=record.get('result');proof=key_gate.parse(key_gate.read(directory/'worker_report.json'))
            flags=('operation_complete','actual_client_decode_started','actual_client_decode_completed','readback_complete',
                'source_files_unchanged','keygen_encrypt_native_server_forbidden','terminal_scope_completed','external_workspaces_never_removed')
            if (type(result)is not dict or type(proof)is not dict or any(proof.get(n)is not True for n in flags)
                    or proof.get('error') or proof.get('forbidden_attempts') or any(proof.get(n)is not False for n in ('keygen','encryption','server_compute','model_forward'))
                    or result.get('worker_report_sha256')!=process.sha256(directory/'worker_report.json')
                    or proof.get('source_hashes_start')!=before or proof.get('source_hashes_end')!=before
                    or proof.get('prerequisite')!=report['prerequisite'] or proof.get('reviewed_dependencies')!=report['reviewed_dependencies']):
                raise ValueError('complete client source/phase/terminal proof required')
            readback=proof['readback'];metrics=readback['metrics'];passed=metrics.get('passed')
            if (type(passed)is not bool or proof.get('passed')is not passed or result.get('passed')is not passed
                    or result.get('operation_complete')is not True or record['returncode']!=(0 if passed else 1)
                    or type(readback.get('sample_count'))is not int or readback['sample_count']!=1
                    or any(readback.get(n)is not True for n in ('private_receipt_validated','result_permissions_0600'))
                    or readback.get('public_batch_sha256')!=authorized[2]['public_batch_sha256']):raise ValueError('private same-input result/receipt evidence differs')
            report.update(actual_client_decode=True,numeric_gates_passed=passed,readback=readback,worker_elapsed_s=proof['elapsed_s'])
        after=worker.authenticate(**gate_args,profile_bytes=profile,plan_bytes=plan)
        if worker.prerequisite_summary(*after)!=report['prerequisite']:raise ValueError('completed prerequisite changed')
        for owner in owners:owner.check()
        report['operation_complete']=True
    except BaseException as e:primary=e;report['error']=dict(type=type(e).__name__,message=str(e))
    finally:
        with terminal_scope(report,output/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            if active:
                try:
                    record=key_gate.parse(key_gate.read(output/'worker/process.json'))
                    if type(record)is not dict or any(record.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')):raise ValueError('active client cleanup unknown')
                    report['worker_process']=record
                except BaseException as e:
                    record=dict(process_reaped=False,error='active client cleanup unknown');report['worker_process']=record
                    if primary is None:primary=e
                    else:primary.add_note('active process evidence: '+repr(e))
            report['all_processes_reaped']=not active if record is None else type(record)is dict and all(record.get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
            after=attempt('source readback',lambda:hashes(sources()))
            copied=attempt('execution source readback',lambda:{n:hashlib.sha256((execution/n).read_bytes()).hexdigest() for n in before}) if before and execution is not None else None
            verified=attempt('archive readback',lambda:check_archive(output/'sources.tar.gz',before)) if before else False
            report.update(source_hashes_end=after,execution_source_hashes_end=copied,source_files_unchanged=bool(before) and before==after==copied,source_archive_verified=verified is True)
            retained=True
            for owner in reversed(owners):
                retained=attempt('retained directory identity',lambda:(owner.check(),True)[1]) is True and retained
                attempt('close retained directory descriptor',owner.close)
            report['private_result_workspace_retained']=result_workspace is not None and retained
            report['cleanup_completed']=not report.get('cleanup_errors') and report['all_processes_reaped']
            report['passed']=bool(report.get('operation_complete') and primary is None and cleanup_error is None
                and report['source_files_unchanged'] and verified and report['cleanup_completed'] and retained
                and (not execute or report.get('numeric_gates_passed')is True))
            report['elapsed_s']=time.monotonic()-start
            try:write(output/'report.json',report)
            except BaseException as e:
                prior=primary if primary is not None else cleanup_error
                if prior is None:raise
                prior.add_note('stage5 terminal publication: '+repr(e))
            if primary is not None:raise primary
            if cleanup_error is not None:raise cleanup_error
    print(json.dumps({n:report[n] for n in ('passed','execute','actual_client_decode','elapsed_s')}));return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','key-evidence','key-workspace','model-evidence','model-workspace','stage3-evidence','stage4-evidence',
        'io-workspace','fixture-evidence','reference-evidence'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('key-report-sha256','model-report-sha256','stage3-report-sha256','stage4-report-sha256','fixture-report-sha256','reference-report-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker-python',type=Path,default=Path('/tmp/cross-pretrained-validation-env/bin/python'))
    with termination_signals():report=run(**vars(p.parse_args()))
    raise SystemExit(0 if report['passed'] else 1)
