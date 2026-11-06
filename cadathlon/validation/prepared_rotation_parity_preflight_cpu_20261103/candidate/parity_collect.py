"""Source-bound prepared-rotation parity collector; borrowed upload only."""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,os,shutil,stat,sys,tarfile,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'joined'),
              str(HERE/'joined/checkout/cadathlon/common'),str(HERE/'runtime')]
import check_transformer_suite as process
import key_gate,parity_protocol
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
spec=importlib.util.spec_from_file_location('prepared_parity_worker',HERE/'parity_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

def sources():return parity_protocol.sources()
def hashes(raw):return {name:hashlib.sha256(data).hexdigest() for name,data in raw.items()}
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
        base=Path(base).absolute();resolved=path.resolve();resolved_base=base.resolve()
        if path==base or base in path.parents or resolved==resolved_base or resolved_base in resolved.parents:
            raise ValueError('evidence output must be outside all retained prerequisites')
    return path

_BORROWED_QUARANTINE=[]

class BorrowedDirectory:
    """Keep an existing external directory open; never owns or deletes it."""
    def __init__(self,path):
        self.path=Path(path);self._close_failure=None;self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        info=os.fstat(self.fd);self.identity=(info.st_dev,info.st_ino)
        try:self.check()
        except BaseException as primary:
            try:self.close()
            except BaseException as error:primary.add_note('borrowed descriptor cleanup: '+repr(error))
            raise
    def check(self):
        info=self.path.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_ino)!=self.identity:
            raise ValueError('retained request workspace was replaced')
    def close(self):
        if self._close_failure is not None:raise self._close_failure
        if self.fd is not None:
            try:os.close(self.fd)
            except BaseException as error:
                # A failed close does not establish descriptor retirement; do
                #not erase ownership or blindly retry an uncertain descriptor.
                self._close_failure=error;_BORROWED_QUARANTINE.append(self);raise
            self.fd=None


def validate_worker(record,proof,before,report,keys,model,inputs,path):
    result=record.get('result')
    flags=('passed','actual_rotations_started','all_rotations_completed','request_readback_complete',
        'upload_payloads_unchanged','source_files_unchanged','compiled_owner_cleanup_completed','loader_cleanup_completed',
        'keygen_codecs_native_private_model_forbidden','terminal_scope_completed')
    if (type(result)is not dict or result.get('passed')is not True or result.get('worker_report_sha256')!=process.sha256(path)
        or type(proof)is not dict or any(proof.get(n)is not True for n in flags) or proof.get('forbidden_attempts')
        or proof.get('source_hashes_start')!=before or proof.get('source_hashes_end')!=before
        or proof.get('prerequisite')!=report['prerequisite'] or proof.get('reviewed_sources')!=report['reviewed_sources']
        or proof.get('public_plan')!=report['public_plan']):
        raise ValueError('strict parity worker/source/phase proof differs')
    request=proof['request_readback'];arith=proof.get('arithmetic',{})
    expected=[(level,index,sequence) for level in parity_protocol.LEVELS for sequence,index in enumerate(parity_protocol.INDICES)]
    comparisons=arith.get('comparisons',[])
    if (request.get('sample_count')!=1 or request.get('tile_count')!=8 or request.get('tiles')!=inputs['tiles']
        or request.get('input_digest')!=inputs['input_digest'] or request.get('public_batch_sha256')!=inputs['public_batch_sha256']
        or result.get('context_id')!=keys['context_id'] or result.get('model_id')!=model['model_id']
        or result.get('public_batch_sha256')!=inputs['public_batch_sha256']
        or arith.get('passed')is not True or arith.get('actual_rotation_calls')!=24 or arith.get('input_unchanged')is not True
        or arith.get('level_reduction',{}).get('passed')is not True
        or [(r.get('level'),r.get('index'),r.get('sequence')) for r in comparisons]!=expected
        or any(r.get('passed')is not True or r.get('metadata_equal')is not True or r.get('mismatched_words')!=0 for r in comparisons)):
        raise ValueError('complete fixed rotation parity/binding differs')
    for row in comparisons:
        q=(row['level']+1)*2
        if (type(row.get('compared_words'))is not int or row['compared_words']!=2*131072*q
            or type(row.get('prepared_entries'))is not int or not 1<=row['prepared_entries']<=2
            or type(row.get('prepared_bytes'))is not int or not 0<row['prepared_bytes']<=parity_protocol.MAX_BYTES
            or row.get('left')!=row.get('right')):raise ValueError('all-word/metadata/cache evidence differs')
    levels=arith.get('levels',[])
    if [r.get('level') for r in levels]!=list(parity_protocol.LEVELS):raise ValueError('two level reports required')
    for row,planned in zip(levels,report['public_plan']['rows']):
        if (row.get('retained_entries')!=2 or row.get('retained_bytes')!=planned['max_two_entry_bytes']
            or row.get('stats_delta',{}).get('hits')!=4 or row.get('stats_delta',{}).get('published')!=2):
            raise ValueError('exact prepared reuse/cap evidence differs')

def run(output,*,key_evidence,key_report_sha256,key_workspace,model_evidence,model_report_sha256,model_workspace,
        stage3_evidence,stage3_report_sha256,io_workspace,execute=False,
        worker_python=Path('/tmp/cross-pretrained-validation-env/bin/python')):
    protected=(key_workspace,model_workspace,io_workspace,key_evidence,model_evidence,stage3_evidence)
    output=outside(output,protected);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();primary=None;cleanup_error=None;borrowed=None;paths=None;active=False;record=None;before={};execution=None
    report=dict(passed=False,execute=execute,actual_rotation_parity=False,actual_execution_attempted=False,
        all_processes_reaped=True,external_workspaces_never_removed=True,
        scope='Authenticated same-input encrypted rotation representation parity only; no model, decode, accuracy or speed claim.',
        wrapper_pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    gate_args=dict(key_evidence=key_evidence,key_report_sha256=key_report_sha256,key_workspace=key_workspace,
        model_evidence=model_evidence,model_report_sha256=model_report_sha256,model_workspace=model_workspace,
        stage3_evidence=stage3_evidence,stage3_report_sha256=stage3_report_sha256,io_workspace=io_workspace)
    def attempt(label,fn):
        nonlocal cleanup_error
        try:return fn()
        except BaseException as error:
            report.setdefault('cleanup_errors',[]).append(dict(operation=label,type=type(error).__name__,message=str(error)))
            if primary is not None:primary.add_note(label+': '+repr(error))
            elif cleanup_error is None:cleanup_error=error
            elif not isinstance(error,Exception) and isinstance(cleanup_error,Exception):
                error.add_note('earlier cleanup: '+repr(cleanup_error));cleanup_error=error
            else:cleanup_error.add_note(label+': '+repr(error))
    try:
        write(output/'report.json',report)
        if type(execute)is not bool:raise ValueError('explicit boolean execution policy required')
        parity_protocol.environment()
        raw=sources();before=hashes(raw);archive(output/'sources.tar.gz',raw)
        if not check_archive(output/'sources.tar.gz',before):raise ValueError('startup archive differs')
        execution=output/'execution_source';execution.mkdir()
        for name,data in raw.items():p=execution/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        if {n:hashlib.sha256((execution/n).read_bytes()).hexdigest() for n in before}!=before:raise ValueError('execution source copy differs')
        write(output/'source_manifest.json',before)
        report['reviewed_sources']=parity_protocol.check_sources()
        profile=raw['runtime/stage/server_inputs/p60_profile_and_refresh.json'];plan=raw['runtime/stage/server_inputs/twelve_layer_plan.json']
        authorized=worker.authenticate(**gate_args,profile_bytes=profile,plan_bytes=plan)
        keys,model,inputs=authorized;report['prerequisite']=worker.prerequisite_summary(*authorized)
        paths=worker.make_paths(*authorized,io_workspace)
        borrowed=BorrowedDirectory(paths['upload'])
        public_plan=parity_protocol.plan(profile,plan);report['public_plan']=public_plan
        usage=shutil.disk_usage(output);needed=64<<20
        if usage.free<needed:raise OSError('insufficient public evidence storage')
        report.update(io_workspace=str(io_workspace),upload_identity=list(borrowed.identity),
            storage=dict(required_with_margin=needed,free_bytes=usage.free,exclusions='all borrowed external files, host/device arrays, compiler and conversion temporaries; no key/ciphertext output written'),
            source_hashes_start=before,source_archive_sha256=process.sha256(output/'sources.tar.gz'),
            software=process.software_versions(),worker_python=str(worker_python),
            environment={n:os.environ.get(n) for n in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
        write(output/'report.json',report)
        if execute:
            directory=output/'worker';directory.mkdir();active=True;report['actual_execution_attempted']=True
            write(output/'report.json',report)
            args=gate_args|dict(output=directory/'worker_report.json',source_manifest=output/'source_manifest.json')
            command=[str(worker_python),str(execution/'parity_worker.py')]
            for name,value in args.items():command+=['--'+name.replace('_','-'),str(value)]
            previous=process.child_environment_overrides
            process.child_environment_overrides=lambda:{'JAX_PLATFORMS':'cpu','PYTHONUNBUFFERED':'1'}
            try:record=process.run_case(command,directory,timeout=None,cleanup_grace_s=120.)
            finally:process.child_environment_overrides=previous
            active=False;report['worker_process']=record
            if (type(record)is not dict or type(record.get('returncode'))is not int or record['returncode']!=0 or record.get('error')
                    or any(record.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))):
                raise ValueError('server worker failed or cleanup incomplete')
            result=record.get('result');proof=key_gate.parse(key_gate.read(directory/'worker_report.json'))
            validate_worker(record,proof,before,report,keys,model,inputs,directory/'worker_report.json')
            report.update(actual_rotation_parity=True,context_id=keys['context_id'],model_id=model['model_id'],
                request_readback=proof['request_readback'],arithmetic=proof['arithmetic'],worker_elapsed_s=proof['elapsed_s'])
        after=worker.authenticate(**gate_args,profile_bytes=profile,plan_bytes=plan)
        if worker.prerequisite_summary(*after)!=report['prerequisite']:raise ValueError('completed prerequisite changed')
        borrowed.check();report['operation_complete']=True
    except BaseException as error:primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,output/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            if active:
                try:
                    record=key_gate.parse(key_gate.read(output/'worker/process.json'))
                    if type(record)is not dict or any(record.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')):
                        raise ValueError('active parity cleanup unknown')
                    report['worker_process']=record
                except BaseException as error:
                    report['worker_process']=dict(process_reaped=False,error='active parity cleanup unknown',evidence_error=repr(error))
                    if primary is None:primary=error
                    else:primary.add_note('active process evidence: '+repr(error))
            report['all_processes_reaped']=not active if record is None else type(record)is dict and all(record.get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
            after=attempt('source readback',lambda:hashes(sources()))
            copied=attempt('execution source readback',lambda:{n:hashlib.sha256((execution/n).read_bytes()).hexdigest() for n in before}) if before and execution is not None else None
            verified=attempt('archive readback',lambda:check_archive(output/'sources.tar.gz',before)) if before else False
            report.update(source_hashes_end=after,execution_source_hashes_end=copied,source_files_unchanged=bool(before) and before==after==copied,source_archive_verified=verified is True)
            if borrowed is not None:
                report['io_workspace_retained']=attempt('retained I/O identity',lambda:(borrowed.check(),True)[1]) is True
                # Sibling download/server_report may legitimately appear during root's stage4.
                # Only the borrowed upload subtree is authenticated; no external deletion.
                attempt('close borrowed I/O descriptor',borrowed.close)
            report['cleanup_completed']=not report.get('cleanup_errors') and report['all_processes_reaped']
            report['passed']=bool(report.get('operation_complete') and primary is None and cleanup_error is None
                and report['source_files_unchanged'] and verified and report['cleanup_completed'] and report.get('io_workspace_retained'))
            report['elapsed_s']=time.monotonic()-start
            try:write(output/'report.json',report)
            except BaseException as error:
                prior=primary if primary is not None else cleanup_error
                if prior is None:raise
                prior.add_note('parity terminal publication: '+repr(error))
            if primary is not None:raise primary
            if cleanup_error is not None:raise cleanup_error
    print(json.dumps({n:report[n] for n in ('passed','execute','actual_rotation_parity','io_workspace_retained','elapsed_s')}));return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','key-evidence','key-workspace','model-evidence','model-workspace','stage3-evidence','io-workspace'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('key-report-sha256','model-report-sha256','stage3-report-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--worker-python',type=Path,default=Path('/tmp/cross-pretrained-validation-env/bin/python'))
    p.add_argument('--execute',action='store_true')
    with termination_signals():report=run(**vars(p.parse_args()))
    raise SystemExit(0 if report['passed'] else 1)
