"""Fixed authenticated public ciphertext rotation parity. Never a model runner."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'joined'),str(HERE/'joined/checkout/cadathlon/common'),str(HERE/'runtime')]
from contextlib import redirect_stdout
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
import worker as stage4_worker
import key_gate,server_readback,group_protocol as protocol,parity_boundaries
from group_arithmetic import exercise,OwnerCleanup
write=stage4_worker.write
authenticate=stage4_worker.authenticate
prerequisite_summary=stage4_worker.prerequisite_summary
make_paths=stage4_worker.make_paths

def actual(request,paths,profile,plan,report,output,expected):
    from transport import load_ciphertext
    import key_contexts
    from encrypted_packed_qwen import CompiledArithmetic
    from encrypted_rotation import CompiledRotation
    from grouped_rotation import GroupedCompiledRotation
    from support import require_secure_execution
    report['loaded_local_modules_before_context']=protocol.imported_sources(expected)
    # Reads only the already authenticated public upload. No client/native
    #artifact path is provided to the transport or arithmetic.
    row=request['batch']['requests'][0]
    value,manifest=load_ciphertext(Path(paths['upload'])/row['contract']['request_id']/'tile0',
        contract=row['contract'],role='request',tile_index=0)
    from transport import manifest_digest
    if manifest_digest(manifest)!=request['tiles'][0]['manifest_sha256']:raise ValueError('loaded tile0 changed after request verification')
    report['input_tile_loaded']=True;write(output,report)
    with key_contexts.load_server_grouped(paths['public_keys'],profile_bytes=profile,plan_bytes=plan,dedicated_no_write=True) as handle:
        report['loader_entered']=True;write(output,report)
        baseline=prepared=None;primary=None;retirement=OwnerCleanup()
        try:
            baseline=CompiledArithmetic(handle.ctx,low_memory=True,preserve_dispatch_cache=True)
            retirement.register(baseline)
            prepared=handle.grouped_arithmetic(max_results=protocol.MAX_RESULTS,max_result_bytes=protocol.MAX_RESULT_BYTES,max_digit_bytes=protocol.MAX_DIGIT_BYTES,
                                               low_memory=True,preserve_dispatch_cache=True)
            retirement.register(prepared)
            if type(baseline)is not CompiledArithmetic or type(baseline.rotation)is not CompiledRotation or not isinstance(prepared,CompiledArithmetic) or type(prepared.rotation)is not GroupedCompiledRotation:
                raise TypeError('actual reviewed arithmetic/rotation implementations required')
            report['loaded_local_modules_before_arithmetic']=protocol.imported_sources(expected)
            report['actual_rotations_started']=True;write(output,report)
            def progress(row):
                report.setdefault('completed_comparisons',[]).append(row)
                write(output,report)
                print(json.dumps(dict(level=row['level'],index=row['index'],sequence=row['sequence'],passed=row['passed'],group=row['group'],normalized_index=row['normalized_index'])),flush=True)
            report['arithmetic']=exercise(handle.ctx,value,baseline,prepared,progress=progress,security=require_secure_execution)
            report['all_rotations_completed']=True
        except BaseException as error:primary=error;raise
        finally:
            with cleanup_signals():
                retirement.close(primary)
                report['compiled_owner_cleanup_completed']=retirement.completed
    report['loader_cleanup_completed']=True
    del value

def run(*,key_evidence,key_report_sha256,key_workspace,model_evidence,model_report_sha256,model_workspace,
        stage3_evidence,stage3_report_sha256,io_workspace,output,source_manifest):
    start=time.monotonic();primary=None;expected={};calls=[]
    report=dict(passed=False,pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),actual_rotations_started=False,
        all_rotations_completed=False,input_tile_loaded=False,loader_entered=False,loader_cleanup_completed=False,
        compiled_owner_cleanup_completed=False,source_files_unchanged=False,
        scope='Same-input exact encrypted rotation representation parity; no decrypt, native model, full-model accuracy or performance claim.')
    args=dict(key_evidence=key_evidence,key_report_sha256=key_report_sha256,key_workspace=key_workspace,
        model_evidence=model_evidence,model_report_sha256=model_report_sha256,model_workspace=model_workspace,
        stage3_evidence=stage3_evidence,stage3_report_sha256=stage3_report_sha256,io_workspace=io_workspace)
    try:
        write(output,report);report['environment']=protocol.environment()
        expected=key_gate.parse(key_gate.read(source_manifest))
        report['source_hashes_start']=protocol.hashes(protocol.sources())
        if report['source_hashes_start']!=expected:raise ValueError('exact actual worker source closure differs')
        report['reviewed_sources']=protocol.check_sources()
        profile=(HERE/'runtime/stage/server_inputs/p60_profile_and_refresh.json').read_bytes()
        plan=(HERE/'runtime/stage/server_inputs/twelve_layer_plan.json').read_bytes()
        report['public_plan']=protocol.plan(profile,plan)
        keys,model,inputs=authenticate(**args,profile_bytes=profile,plan_bytes=plan)
        report['prerequisite']=prerequisite_summary(keys,model,inputs)
        paths=make_paths(keys,model,inputs,io_workspace)
        with parity_boundaries.rotation_only(calls,private_paths=(Path(key_workspace)/'client_private',)),redirect_stdout(sys.stderr):
            report['loaded_local_modules_start']=protocol.imported_sources(expected)
            request=server_readback.requests(paths,keys=keys,model=model,stage3=inputs)
            report['request_readback_complete']=True
            report['request_readback']={n:v for n,v in request.items() if n!='batch'}
            write(output,report)
            actual(request,paths,profile,plan,report,output,expected)
            if server_readback.requests(paths,keys=keys,model=model,stage3=inputs)!=request:
                raise ValueError('authenticated upload changed during parity')
            report['upload_payloads_unchanged']=True
        if calls:raise ValueError('forbidden parity action attempted')
        again=authenticate(**args,profile_bytes=profile,plan_bytes=plan)
        if prerequisite_summary(*again)!=report['prerequisite']:raise ValueError('public prerequisite identity changed')
        report.update(passed=True,keygen_codecs_native_private_model_forbidden=True,
                      context_id=keys['context_id'],model_id=model['model_id'])
    except BaseException as error:primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        with terminal_scope(report,output,write,cleanup_signals,lambda:(primary,None)):
            try:
                after=protocol.hashes(protocol.sources());report.update(source_hashes_end=after,
                    source_files_unchanged=bool(expected) and after==expected,forbidden_attempts=calls,elapsed_s=time.monotonic()-start)
                if expected:report['loaded_local_modules_end']=protocol.imported_sources(expected)
                report['passed']=bool(report['passed'] and report['source_files_unchanged']);write(output,report)
            except BaseException as error:
                if primary is None:raise
                primary.add_note('parity final evidence: '+repr(error))
            if primary is not None:raise primary
    print(json.dumps(dict(passed=report['passed'],context_id=report['context_id'],model_id=report['model_id'],
        public_batch_sha256=report['prerequisite']['public_batch_sha256'],
        worker_report_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),elapsed_s=report['elapsed_s'])))
    return 0 if report['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('key-evidence','key-workspace','model-evidence','model-workspace','stage3-evidence','io-workspace','output','source-manifest'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('key-report-sha256','model-report-sha256','stage3-report-sha256'):p.add_argument('--'+n,required=True)
    with termination_signals():exit_code=run(**vars(p.parse_args()))
    raise SystemExit(exit_code)
