"""Client-only size0 decoding after authenticated completed server evidence."""
from pathlib import Path
from contextlib import redirect_stdout
import argparse,hashlib,json,os,stat,sys,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'runtime')]
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
import key_gate,model_gate,input_gate,stage3_gate,stage4_gate,reference_gate,dependencies,boundaries,client_readback

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def hashes(expected):
    result={}
    for name in expected:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('relative execution source required')
        result[name]=hashlib.sha256((HERE/p).read_bytes()).hexdigest()
    return result

def local_modules(expected):
    result={}
    for name,module in tuple(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if HERE in path.parents and path.suffix=='.py':
            relative=path.relative_to(HERE).as_posix()
            if relative not in expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected[relative]:
                raise ValueError('loaded local module outside exact source closure: '+name)
            result[name]=relative
    for name in ('file_stages','key_contexts','transport'):
        module=sys.modules.get(name)
        if module is not None and Path(getattr(module,'__file__','')).resolve()!=HERE/'runtime/stage'/f'{name}.py':
            raise ValueError('mixed client runtime module: '+name)
    return result

def authenticate(*,key_evidence,key_report_sha256,key_workspace,model_evidence,model_report_sha256,model_workspace,
        stage3_evidence,stage3_report_sha256,stage4_evidence,stage4_report_sha256,io_workspace,
        fixture_evidence,fixture_report_sha256,reference_evidence,reference_report_sha256,profile_bytes,plan_bytes):
    # The completed server gate precedes every private input/reference read.
    keys=key_gate.authenticate(key_evidence,key_report_sha256,key_workspace,profile_bytes=profile_bytes,plan_bytes=plan_bytes)
    model=model_gate.authenticate(model_evidence,model_report_sha256,model_workspace,keys=keys)
    requests=stage3_gate.authenticate(stage3_evidence,stage3_report_sha256,io_workspace,keys=keys,model=model)
    server=stage4_gate.authenticate(stage4_evidence,stage4_report_sha256,io_workspace,keys=keys,model=model,requests=requests)
    ref=reference_gate.public_metadata(reference_evidence,reference_report_sha256)
    inputs=input_gate.authenticate(fixture_evidence,fixture_report_sha256)
    if (inputs['input_sha256']!=ref['input_sha256'] or inputs['fixture_report_sha256']!=ref['fixture_report_sha256']
            or type(inputs['valid_token_count'])is not int or inputs['valid_token_count']!=32):
        raise ValueError('same size0 fixture/reference binding required')
    return keys,model,requests,server,inputs,ref

def prerequisite_summary(keys,model,requests,server,inputs,ref):
    return {n:keys[n] for n in ('context_id','key_report_sha256','key_source_archive_sha256')}|{
        n:model[n] for n in ('model_id','model_report_sha256','model_manifest_sha256')}|{
        n:requests[n] for n in ('stage3_report_sha256','public_batch_sha256','input_digest')}|{
        'stage4_report_sha256':server['stage4_report_sha256'],
        'input_sha256':inputs['input_sha256'],'fixture_report_sha256':inputs['fixture_report_sha256'],
        'reference_report_sha256':ref['report_sha256'],'reference_source_archive_sha256':ref['source_archive_sha256']}

def make_paths(authorized,key_workspace,io_workspace,result_workspace):
    import file_stages
    keys,model,requests,server,inputs,ref=authorized
    return dict(size=0,batch_size=1,model_id=file_stages.MODEL_ID,model_revision=file_stages.REVISION,
        public_keys=keys['public_keys'],client_private=Path(key_workspace)/'client_private',
        preprocessed_input=inputs['preprocessed_input'],
        **{n:Path(io_workspace)/n for n in ('upload','download','server_report')},
        decrypted_results=Path(result_workspace)/'decrypted_results')

def fresh_paths(paths,result_workspace):
    root=Path(result_workspace);s=root.lstat()
    if not stat.S_ISDIR(s.st_mode) or stat.S_IMODE(s.st_mode)!=0o700 or list(root.iterdir()):
        raise ValueError('empty real private0700 result workspace required')
    for p in (Path(paths['decrypted_results']),Path(paths['client_private'])/'completed.json'):
        if p.exists() or p.is_symlink():raise FileExistsError('fresh result/completion receipt required')

def load_private_inputs(paths,ref,expected_batch_sha256):
    import file_stages,file_stage_contract
    from file_stage_io import read_manifest
    from transport_contract import digest
    records,input_sha=file_stages._records(paths,1)
    if len(records)!=1 or input_sha!=ref['input_sha256']:raise ValueError('exact original size0 input required')
    batch=read_manifest(Path(paths['upload'])/'batch.json')
    if digest(batch)!=expected_batch_sha256:raise ValueError('completed server request batch differs')
    private=read_manifest(Path(paths['client_private'])/'requests.json')
    file_stage_contract.validate_private_requests(private,batch,[records[0].target_idx],input_sha)
    return batch,reference_gate.load_private(ref,records[0])

def run(*,key_evidence,key_report_sha256,key_workspace,model_evidence,model_report_sha256,model_workspace,
        stage3_evidence,stage3_report_sha256,stage4_evidence,stage4_report_sha256,io_workspace,
        fixture_evidence,fixture_report_sha256,reference_evidence,reference_report_sha256,result_workspace,output,source_manifest):
    start=time.monotonic();primary=None;expected={};calls=[]
    report=dict(passed=False,actual_client_decode_started=False,actual_client_decode_completed=False,readback_complete=False,
        keygen=False,encryption=False,server_compute=False,model_forward=False,external_workspaces_never_removed=True,
        scope='One size0 client decode with original P2 numeric/prediction gates; no ranked accuracy claim.',
        pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    args=locals().copy()
    names=('key_evidence','key_report_sha256','key_workspace','model_evidence','model_report_sha256','model_workspace',
        'stage3_evidence','stage3_report_sha256','stage4_evidence','stage4_report_sha256','io_workspace',
        'fixture_evidence','fixture_report_sha256','reference_evidence','reference_report_sha256')
    gate_args={n:args[n] for n in names}
    try:
        write(output,report)
        if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU decode process required')
        expected=key_gate.parse(key_gate.read(source_manifest));report['source_hashes_start']=hashes(expected)
        if not expected or report['source_hashes_start']!=expected:raise ValueError('execution source startup differs')
        report['reviewed_dependencies']=dependencies.check()
        profile=(HERE/'runtime/stage/server_inputs/p60_profile_and_refresh.json').read_bytes()
        plan=(HERE/'runtime/stage/server_inputs/twelve_layer_plan.json').read_bytes()
        authorized=authenticate(**gate_args,profile_bytes=profile,plan_bytes=plan)
        report['prerequisite']=prerequisite_summary(*authorized)
        paths=make_paths(authorized,key_workspace,io_workspace,result_workspace);fresh_paths(paths,result_workspace)
        batch,reference=load_private_inputs(paths,authorized[-1],authorized[2]['public_batch_sha256'])
        import file_stages
        report['loaded_local_modules_start']=local_modules(expected)
        with boundaries.client_only(calls),redirect_stdout(sys.stderr):
            report['actual_client_decode_started']=True;write(output,report)
            result=file_stages.client_decrypt_decode(paths)
            report['actual_client_decode_completed']=True
            readback=client_readback.validate(paths,request_batch=batch,reference=reference)
            if (type(result)is not dict or type(result.get('sample_count'))is not int or result['sample_count']!=1
                    or result.get('public_batch_sha256')!=authorized[2]['public_batch_sha256']
                    or readback['public_batch_sha256']!=result['public_batch_sha256']):raise ValueError('same size0 handler/result batch required')
        if calls:raise ValueError('client decode attempted forbidden operation')
        after=authenticate(**gate_args,profile_bytes=profile,plan_bytes=plan)
        if prerequisite_summary(*after)!=report['prerequisite']:raise ValueError('prerequisite changed during decode')
        # Re-read the immutable private source, without publishing any values.
        batch2,reference2=load_private_inputs(paths,after[-1],after[2]['public_batch_sha256'])
        readback2=client_readback.validate(paths,request_batch=batch2,reference=reference2)
        if readback2!=readback:raise ValueError('retained private result/receipt changed')
        report.update(readback_complete=True,readback=readback,operation_complete=True,
            passed=readback['metrics']['passed'] is True,context_id=authorized[0]['context_id'],
            model_id=authorized[1]['model_id'],keygen_encrypt_native_server_forbidden=True)
    except BaseException as error:primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        with terminal_scope(report,output,write,cleanup_signals,lambda:(primary,None)):
            try:
                after=hashes(expected) if expected else {}
                report.update(source_hashes_end=after,source_files_unchanged=bool(expected) and after==expected,
                    forbidden_attempts=calls,elapsed_s=time.monotonic()-start)
                if expected:report['loaded_local_modules_end']=local_modules(expected)
                report['passed']=bool(report['passed'] and report['source_files_unchanged']);write(output,report)
            except BaseException as error:
                if primary is None:raise
                primary.add_note('stage5 terminal source/publication: '+repr(error))
            if primary is not None:raise primary
    print(json.dumps(dict(passed=report['passed'],operation_complete=report.get('operation_complete',False),
        worker_report_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
    return 0 if report['passed'] else 1

def parser():
    p=argparse.ArgumentParser()
    for name in ('key-evidence','key-workspace','model-evidence','model-workspace','stage3-evidence','stage4-evidence',
        'io-workspace','fixture-evidence','reference-evidence','result-workspace','output','source-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('key-report-sha256','model-report-sha256','stage3-report-sha256','stage4-report-sha256','fixture-report-sha256','reference-report-sha256'):
        p.add_argument('--'+name,required=True)
    return p
if __name__=='__main__':
    with termination_signals():exit_code=run(**vars(parser().parse_args()))
    raise SystemExit(exit_code)
