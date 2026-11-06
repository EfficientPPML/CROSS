"""Immutable-execution collector for fresh ranked functional stage proofs.

No healthy timeout, no deletion/retirement, no implicit key or model setup.
An execute=False preparation never calls a handler. Official timing remains the
original harness's five online subprocesses; these durations are diagnostic.
"""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,sys,tarfile,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'joined'),str(HERE/'joined/checkout/cadathlon/common'),str(HERE/'runtime')]
import key_gate,ranked_worker as worker,ranked_execution as execution
import check_transformer_suite as process
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope


def sources():
    # Inherited parent inventory is immutable; new code is explicit by location.
    parent=key_gate.parse((HERE/'parent_source_manifest.json').read_bytes())
    paths={n:HERE/n for n in parent}
    for pattern in ('*.py','*.json'):
        paths.update({p.name:p for p in HERE.glob(pattern)})
    for directory in ('upstream_adapter','timing_upstream'):
        if (HERE/directory).exists():paths.update({p.relative_to(HERE).as_posix():p for p in (HERE/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.md')})
    result={n:p.read_bytes() for n,p in paths.items()}
    if {n:key_gate.digest(result[n]) for n in parent}!=parent:raise ValueError('frozen inherited ranked/runtime source changed')
    return result

def hashes(raw):return {n:key_gate.digest(b) for n,b in raw.items()}
def write(path,value):worker.write(path,value)
def archive(path,raw):
    with tarfile.open(path,'w:gz') as t:
        for name,data in sorted(raw.items()):
            m=tarfile.TarInfo(name);m.size=len(data);t.addfile(m,io.BytesIO(data))
def verify_archive(path,expected):
    with tarfile.open(path) as t:
        rows=t.getmembers()
        return len(rows)==len(expected)==len({m.name for m in rows}) and all(m.isfile() for m in rows) and {m.name:key_gate.digest(t.extractfile(m).read()) for m in rows}==expected

def _external(request,output):
    # Authenticate public reports before discovering protected private paths.
    # Never stat/open private fixtures or references for containment checks.
    authorized=worker.cases.authenticate(request['fixture_evidence'],request['fixture_report_sha256'],request['reference_evidence'],request['reference_report_sha256'])
    case=authorized.case(request['size'],request['seed'])
    bases=execution.protected(case)+(request['staged_root'],request['fixture_evidence'],request['reference_evidence'],request['key_evidence'],request['model_evidence'])
    bases=tuple(x for x in bases if x is not None)
    bases+=tuple(row['evidence'] for row in request['predecessors'])
    return execution.outside(output,bases)


def completed_worker(directory,record,expected,bindings,stage,*,case,previous):
    if (type(record)is not dict or type(record.get('returncode'))is not int or record['returncode']!=0 or record.get('error')
            or any(record.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))):
        raise ValueError('ranked worker exit/reaping/owned process cleanup incomplete')
    result=record.get('result');raw=key_gate.read(directory/'worker_report.json',16<<20)
    if type(result)is not dict or result.get('passed')is not True or result.get('worker_report_sha256')!=key_gate.digest(raw):raise ValueError('ranked worker strict report linkage differs')
    proof=key_gate.parse(raw)
    flags=('passed','actual_stage_started','actual_stage_completed','readback_complete','source_files_unchanged','cleanup_completed','terminal_scope_completed','termination_scope_completed')
    if (any(proof.get(n)is not True for n in flags) or proof.get('error') or proof.get('cleanup_errors') or proof.get('forbidden_attempts')
            or proof.get('bindings')!=bindings or proof.get('stage')!=stage
            or any(proof.get(n)!=expected for n in ('source_hashes_start','source_hashes_end'))):raise ValueError('complete ranked worker source/phase/terminal proof required')
    phases=__import__('ranked_setup').phases(stage) if stage<3 else dict(encryption=stage==3,decryption=stage==5,server_compute=stage==4,model_forward=False,key_generation=False)
    if key_gate.canonical(proof.get('phases'))!=key_gate.canonical(phases):raise ValueError('ranked worker phase separation differs')
    if stage>=3:
        flags={3:('all_ordered_input_bindings_validated',),4:('full_model_graph_completed','all_ordered_response_bindings_validated'),5:('all_ordered_private_outputs_validated',)}[stage]
        if any(proof.get(n)is not True for n in flags):raise ValueError('ranked worker full-count phase evidence missing')
        worker.stage_gate._summaries(stage,proof.get('readback'),count=case.count,previous=previous)
        if stage==5 and proof['readback']['metrics']['official_correct']!=case.official_correct:raise ValueError('ranked official correctness binding differs')
    else:
        data=proof.get('readback')
        if type(data)is not dict:raise ValueError('fresh setup readback required')
        if stage==1:
            if any(data.get(n)is not True for n in ('private_binding_verified','private_permissions_verified','actual_security_preflight_passed')) or type(data.get('private_file_count_at_generation'))is not int or data['private_file_count_at_generation']!=2:
                raise ValueError('fresh key validation incomplete')
        elif type(data.get('tensor_count'))is not int or data['tensor_count']!=196 or type(data.get('matrix_count'))is not int or data['matrix_count']!=74 or data.get('all_readback_fp64_immutable')is not True:
            raise ValueError('fresh model validation incomplete')
    return proof


def run(output,request_path,request_sha256,*,execute=False,worker_python=Path('/tmp/cross-pretrained-validation-env/bin/python')):
    raw_request=key_gate.read(request_path,4<<20)
    if key_gate.digest(raw_request)!=key_gate.sha_field(request_sha256):raise ValueError('externally pinned ranked request differs')
    request=key_gate.parse(raw_request)
    if type(request)is not dict or set(request)!=worker.REQUEST_FIELDS:raise ValueError('ranked request schema differs')
    output=_external(request,output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();primary=None;cleanup_error=None;held=None;before={};execution_source=None;active=False;record=None;bindings=None
    report=dict(schema='cross-bert-ranked-setup-proof-v1' if request['stage']<3 else 'cross-bert-ranked-stage-proof-v1',stage=request['stage'],passed=False,execute=execute,
        actual_execution_attempted=False,operation_complete=False,all_processes_reaped=True,owned_artifacts_retained=False,
        cleanup_completed=False,official_harness_timing=False,external_prerequisites_never_removed=True,
        wrapper_pid=os.getpid(),request_sha256=request_sha256,private_error_text_withheld=True)
    def attempt(label,fn):
        nonlocal cleanup_error
        try:return fn()
        except BaseException as error:
            report.setdefault('cleanup_errors',[]).append({'operation':label,'type':type(error).__name__})
            if primary is not None:primary.add_note(label+': '+type(error).__name__)
            elif cleanup_error is None:cleanup_error=error
            elif not isinstance(error,Exception) and isinstance(cleanup_error,Exception):cleanup_error=error
    try:
        write(output/'report.json',report)
        if type(execute)is not bool or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('exact execution bool and explicit CPU environment required')
        raw=sources();before=hashes(raw);archive(output/'sources.tar.gz',raw)
        if not verify_archive(output/'sources.tar.gz',before):raise ValueError('initial source archive readback differs')
        execution_source=output/'execution_source';execution_source.mkdir()
        for n,data in raw.items():p=execution_source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        if {n:key_gate.digest((execution_source/n).read_bytes()) for n in before}!=before:raise ValueError('immutable execution copy differs')
        write(output/'source_manifest.json',before);(output/'request.json').write_bytes(raw_request)
        a=worker.authenticate(request,before);authorized,case,paths,keys,model,previous,bindings=a
        report.update(bindings=bindings,source_hashes_start=before,source_archive_sha256=key_gate.digest((output/'sources.tar.gz').read_bytes()))
        worker.execution.require_fresh(paths,request['stage']);held=execution.HeldDirectory(request['staged_root'])
        # One fresh official attempt is retained in full; no predecessor or
        #successful sibling is silently deleted to make room for this run.
        per={1:0,2:0,3:8*((100<<20)+65536),4:(6<<20)+65536,5:0}[request['stage']]
        usage=shutil.disk_usage(request['staged_root']);needed=case.count*per+(2<<30)
        if request['stage']==1:
            import client_keys
            p,r=client_keys.public_contract(raw['runtime/stage/server_inputs/p60_profile_and_refresh.json'],raw['runtime/stage/server_inputs/twelve_layer_plan.json'])
            needed+=sum(__import__('math').prod(shape)*4+512 for shape,moduli in client_keys.inventory(p,r).values())
        elif request['stage']==2:needed+=1<<30
        if usage.free<needed:raise OSError('insufficient storage for complete ranked stage and reserve')
        report.update(staged_root=str(request['staged_root']),staged_root_identity=list(held.identity),
            storage={'required_new_bytes_with_margin':needed,'free_bytes':usage.free,'keys_model_existing_io_and_ram_excluded':True})
        write(output/'report.json',report)
        if execute:
            directory=output/'worker';directory.mkdir();active=True;report['actual_execution_attempted']=True;write(output/'report.json',report)
            command=[str(worker_python),str(execution_source/'ranked_worker.py'),'--request-path',str(output/'request.json'),
                '--request-sha256',request_sha256,'--output',str(directory/'worker_report.json'),'--source-manifest',str(output/'source_manifest.json')]
            previous_environment=process.child_environment_overrides
            process.child_environment_overrides=lambda:{'JAX_PLATFORMS':'cpu','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            try:record=process.run_case(command,directory,timeout=None,cleanup_grace_s=120.)
            finally:process.child_environment_overrides=previous_environment
            active=False;report['worker_process']=record
            proof=completed_worker(directory,record,before,bindings,request['stage'],case=case,previous=previous)
            report.update(operation_complete=True,readback=proof['readback'],handler_durations=proof.get('handler_durations',[]))
        held.check()
        if worker.authenticate(request,before)[-1]!=bindings:raise ValueError('ranked prerequisites changed during collection')
        report['passed']=True
    except BaseException as error:
        primary=error;report.update(passed=False,error={'type':type(error).__name__,'message':'ranked collection failed; private exception text withheld'})
    finally:
        with terminal_scope(report,output/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            if active:
                # run_case owns subprocess cancellation/reaping. If it raised,
                # only its strict terminal record can establish cleanup; never
                # infer success from missing metadata or kill an unrelated PID.
                def active_record():
                    value=key_gate.parse(key_gate.read(output/'worker/process.json'))
                    if type(value)is not dict or any(value.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')):
                        raise ValueError('active ranked process cleanup remains unknown')
                    return value
                record=attempt('active process evidence',active_record)
                report['worker_process']=record or {'process_reaped':False,'error':'active cleanup unknown'}
                report['all_processes_reaped']=record is not None
            elif record is not None:
                report['all_processes_reaped']=all(record.get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
            if held is not None:
                attempt('retained attempt identity',held.check);attempt('borrowed attempt handle',held.close)
                report['owned_artifacts_retained']=not bool(report.get('cleanup_errors'))
            current=attempt('current source readback',lambda:hashes(sources())) if before else {}
            report.update(source_hashes_end=current or {},source_files_unchanged=bool(before) and current==before,
                execution_source_hashes_end=attempt('execution source readback',lambda:{n:key_gate.digest((execution_source/n).read_bytes()) for n in before}) if execution_source else {},
                source_archive_verified=bool(before) and bool(attempt('archive source readback',lambda:verify_archive(output/'sources.tar.gz',before))),
                cleanup_completed=cleanup_error is None and not report.get('cleanup_errors') and report['all_processes_reaped'],elapsed_s=time.monotonic()-start)
            report['passed']=bool(report['passed'] and report['source_files_unchanged'] and report['source_archive_verified'] and report['execution_source_hashes_end']==before and report['cleanup_completed'] and report['owned_artifacts_retained'])
            attempt('terminal report',lambda:write(output/'report.json',report))
            if primary is not None:raise primary
            if cleanup_error is not None:raise cleanup_error
    # A proof with execute=False is explicitly not acceptable to stage_gate.
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','request-path'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--execute',action='store_true')
    from ranked_terminal import finish
    args=vars(p.parse_args())
    try:
        result=finish(args['output']/'report.json',lambda:run(**args),write)
    except BaseException as error:
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'private_error_text_withheld':True}));raise SystemExit(130 if not isinstance(error,Exception) else 1)
    print(json.dumps({'passed':result['passed'],'execute':result['execute'],'operation_complete':result['operation_complete']}))
    raise SystemExit(0 if result['passed'] else 1)
