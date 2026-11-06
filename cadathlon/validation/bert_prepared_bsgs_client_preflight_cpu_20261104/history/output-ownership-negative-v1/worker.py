"""Fresh source-gated size0 client; original keys/requests remain read-only."""
from pathlib import Path
import argparse,json,os,stat,sys,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'runtime')]
import client_io as io
import client_source_guard as sources

GATE_NAMES=('key_evidence','key_report_sha256','key_workspace','model_evidence','model_report_sha256','model_workspace',
    'stage3_evidence','stage3_report_sha256','stage4_evidence','stage4_report_sha256','io_workspace','response_root',
    'fixture_evidence','fixture_report_sha256','reference_evidence','reference_report_sha256')

def services():
    from types import SimpleNamespace
    import key_gate,model_gate,stage3_gate,combined_server_gate,reference_gate,input_gate,boundaries,separate_public_readback,separate_client_readback
    from cleanup import cleanup_signals,termination_signals
    return SimpleNamespace(**locals())

def bootstrap(path,sha256):
    io.environment();raw=io.read(path)
    if io.digest(raw)!=sha256:raise ValueError('externally bound client source manifest differs')
    expected=io.parse(raw);return expected,sources.check_sources(expected,HERE,fresh=True)

def public_gates(s,args,profile,plan):
    keys=s.key_gate.authenticate(args['key_evidence'],args['key_report_sha256'],args['key_workspace'],profile_bytes=profile,plan_bytes=plan)
    model=s.model_gate.authenticate(args['model_evidence'],args['model_report_sha256'],args['model_workspace'],keys=keys)
    requests=s.stage3_gate.authenticate(args['stage3_evidence'],args['stage3_report_sha256'],args['io_workspace'],keys=keys,model=model)
    server=s.combined_server_gate.authenticate(args['stage4_evidence'],args['stage4_report_sha256'],args['io_workspace'],args['response_root'],keys=keys,model=model,requests=requests)
    reference=s.reference_gate.public_metadata(args['reference_evidence'],args['reference_report_sha256'])
    return keys,model,requests,server,reference

def summary(authorized):
    keys,model,requests,server,ref=authorized
    return {n:keys[n]for n in ('context_id','key_report_sha256','key_source_archive_sha256')}|{n:model[n]for n in ('model_id','model_report_sha256','model_manifest_sha256')}|{n:requests[n]for n in ('stage3_report_sha256','public_batch_sha256','input_digest')}|{
        'stage4_report_sha256':server['stage4_report_sha256'],'response_batch_sha256':server['response_batch_sha256'],
        'response_manifest_sha256':server['response_manifest_sha256'],'response_root':server['response_root'],
        'fixture_report_sha256':ref['fixture_report_sha256'],'input_sha256':ref['input_sha256'],
        'reference_report_sha256':ref['report_sha256'],'reference_source_archive_sha256':ref['source_archive_sha256']}

def public_paths(args,authorized):
    keys,model,requests,server,ref=authorized
    if Path(requests['upload'])!=Path(args['io_workspace'])/'upload'or Path(server['response_root'])!=Path(args['response_root']):raise ValueError('authenticated separate public roots differ')
    return dict(upload=Path(requests['upload']),download=Path(server['response_root'])/'download',server_report=Path(server['response_root'])/'server_report')

def fixture_root_public(args):
    # Bounded externally authenticated PUBLIC report only; no private stat/read.
    raw=io.read(Path(args['fixture_evidence'])/'report.json')
    if io.digest(raw)!=args['fixture_report_sha256']:raise ValueError('external fixture report differs')
    report=io.parse(raw);root=Path(report['private_workspace'])
    if not root.is_absolute():raise ValueError('absolute authenticated fixture workspace required')
    return root

def protect_result(result_root,args,ref,fixture_root):
    protected=tuple(args[n]for n in ('key_workspace','model_workspace','io_workspace','response_root','key_evidence','model_evidence','stage3_evidence','stage4_evidence','fixture_evidence','reference_evidence'))
    return io.outside(result_root,(*protected,ref['private_workspace'],fixture_root,HERE))

def private_inputs(s,args,authorized):
    keys,model,requests,server,ref=authorized
    inputs=s.input_gate.authenticate(args['fixture_evidence'],args['fixture_report_sha256'])
    if inputs['input_sha256']!=ref['input_sha256']or inputs['fixture_report_sha256']!=ref['fixture_report_sha256']or type(inputs['valid_token_count'])is not int or inputs['valid_token_count']!=32:
        raise ValueError('same original size0 input/reference required')
    return inputs

def make_paths(args,authorized,inputs,result_root):
    import file_stages
    return public_paths(args,authorized)|dict(size=0,batch_size=1,model_id=file_stages.MODEL_ID,model_revision=file_stages.REVISION,
        public_keys=authorized[0]['public_keys'],client_private=Path(args['key_workspace'])/'client_private',
        preprocessed_input=inputs['preprocessed_input'],decrypted_results=Path(result_root)/'decrypted_results')

def load_private(paths,ref,expected_batch,s):
    import file_stages,file_stage_contract
    from file_stage_io import read_manifest
    from transport_contract import digest
    records,input_sha=file_stages._records(paths,1)
    if len(records)!=1 or input_sha!=ref['input_sha256']:raise ValueError('same original private size0 input required')
    batch=read_manifest(Path(paths['upload'])/'batch.json')
    if digest(batch)!=expected_batch:raise ValueError('completed request batch changed before private binding')
    private=read_manifest(Path(paths['client_private'])/'requests.json')
    file_stage_contract.validate_private_requests(private,batch,[records[0].target_idx],input_sha)
    return batch,s.reference_gate.load_private(ref,records[0])

def private_state(root):
    """Metadata only, kept in memory; never publishes secret descriptors."""
    root=Path(root);state=root.lstat()
    if not stat.S_ISDIR(state.st_mode)or stat.S_IMODE(state.st_mode)!=0o700:raise ValueError('ordinary private0700 original key directory required')
    rows=[]
    for p in sorted(root.iterdir()):
        info=p.lstat()
        if not stat.S_ISREG(info.st_mode)or stat.S_IMODE(info.st_mode)!=0o600:raise ValueError('ordinary private0600 original key/request files required')
        rows.append((p.name,io.state(info)))
    return (state.st_dev,state.st_ino,tuple(rows))

def body(report,*,expected,args,result_root,result_identity,output,s):
    started=time.monotonic();primary=None;owners=[];calls=[];before_private=None;original_private=None
    try:
        profile=io.read(HERE/'runtime/stage/server_inputs/p60_profile_and_refresh.json');plan=io.read(HERE/'runtime/stage/server_inputs/twelve_layer_plan.json')
        authorized=public_gates(s,args,profile,plan);report['prerequisite']=summary(authorized)
        fixture_root=fixture_root_public(args);protect_result(result_root,args,authorized[-1],fixture_root)
        root=io.HeldDirectory(result_root);owners.append(('result',root))
        if list(root.identity)!=result_identity or stat.S_IMODE(Path(result_root).lstat().st_mode)!=0o700 or list(Path(result_root).iterdir()):raise ValueError('exact empty owned private0700 result root required')
        report.update(result_workspace=str(result_root),result_workspace_identity=list(root.identity),completion_receipt_separate=True)
        for name,path in (('upload',Path(args['io_workspace'])/'upload'),('response',Path(args['response_root'])),('download',Path(args['response_root'])/'download')):
            owners.append((name,io.HeldDirectory(path)))
        import file_stages
        report['source_guard_before_payload']=sources.check_sources(expected,HERE)
        with s.boundaries.client_only(calls):
            public=s.separate_public_readback.verify(public_paths(args,authorized),server=authorized[3],requests=authorized[2])
            report.update(public_payload_verified_before_private=True,public_response_readback=public)
            # Any public replacement between metadata gate and payload check
            #must still fail before the first input/reference/key access.
            repeated=public_gates(s,args,profile,plan)
            if repeated!=authorized:raise ValueError('public producer identities changed before private access')
            report['private_access_started']=True;io.write(output,report)
            inputs=private_inputs(s,args,authorized);paths=make_paths(args,authorized,inputs,result_root)
            original_private=Path(paths['client_private']);private_owner=io.HeldDirectory(original_private);owners.append(('original_private',private_owner))
            before_private=private_state(original_private)
            batch,reference=load_private(paths,authorized[-1],authorized[2]['public_batch_sha256'],s)
            root.check();report['actual_client_decode_started']=True;io.write(output,report)
            result=file_stages.client_decrypt_decode(paths,completion_directory=Path(result_root))
            report['actual_client_decode_completed']=True;root.check()
            readback=s.separate_client_readback.validate(paths,request_batch=batch,reference=reference,completion_directory=Path(result_root))
            if type(result)is not dict or type(result.get('sample_count'))is not int or result['sample_count']!=1 or result.get('public_batch_sha256')!=authorized[2]['public_batch_sha256']or readback['public_batch_sha256']!=result['public_batch_sha256']:
                raise ValueError('same size0 decoder/result/receipt required')
            after=public_gates(s,args,profile,plan)
            if after!=authorized:raise ValueError('public prerequisites changed during decode')
            inputs2=private_inputs(s,args,after)
            if inputs2!=inputs:raise ValueError('original input binding changed')
            batch2,ref2=load_private(paths,after[-1],after[2]['public_batch_sha256'],s)
            root.check()
            if s.separate_client_readback.validate(paths,request_batch=batch2,reference=ref2,completion_directory=Path(result_root))!=readback:
                raise ValueError('private isolated result/receipt changed')
            if private_state(original_private)!=before_private:raise ValueError('original private keys/request ledger changed')
        if calls:raise ValueError('forbidden client operation attempted')
        report.update(readback_complete=True,readback=readback,operation_complete=True,numeric_gates_passed=readback['metrics']['passed']is True,
            body_complete=readback['metrics']['passed']is True,original_private_unchanged=True,keygen_encrypt_native_server_forbidden=True)
    except BaseException as error:primary=error
    finally:
        final=io.Finalizer(report,primary)
        def cleanup():
            if before_private is not None:report['original_private_unchanged']=final.attempt('original private metadata unchanged',lambda:private_state(original_private)==before_private)is True
            for name,owner in reversed(owners):
                report[name+'_retained']=final.attempt('check '+name,owner.check)is True;final.attempt('close '+name,owner.close)
            report['source_hashes_end']=final.attempt('client source readback',lambda:io.hashes(sources.raw_sources(HERE)))
            report['source_guard_end']=final.attempt('loaded client origins',lambda:sources.check_sources(expected,HERE))
        final.shielded(s.cleanup_signals,cleanup)
        report.update(forbidden_attempts=calls,elapsed_s=time.monotonic()-started,cleanup_completed=not report.get('cleanup_errors'),
            source_files_unchanged=report.get('source_hashes_end')==expected,external_workspaces_never_removed=True,
            private_result_policy='retain owned result and isolated receipt after every actual attempt; never retire borrowed artifacts')
        report['body_complete']=bool(report.get('body_complete')and report['cleanup_completed']and report['source_files_unchanged']and report.get('original_private_unchanged')and report.get('result_retained'))
        final.attempt('client body report',lambda:io.write(output,report));final.raise_if_failed()

def run(*,source_manifest,source_manifest_sha256,result_workspace,result_identity,output,**args):
    output=Path(output)
    if output.exists()or output.is_symlink():raise FileExistsError('fresh client worker report required')
    protected=tuple(args[n]for n in ('key_workspace','model_workspace','io_workspace','response_root','key_evidence','model_evidence','stage3_evidence','stage4_evidence','fixture_evidence','reference_evidence'))
    io.outside(output,protected);io.outside(result_workspace,(*protected,output.parent,HERE))
    if type(result_identity)is not list or len(result_identity)!=2 or any(type(x)is not int for x in result_identity):raise ValueError('typed owned result inode required')
    report=dict(passed=False,body_complete=False,cleanup_completed=False,termination_scope_completed=False,source_files_unchanged=False,
        private_access_started=False,public_payload_verified_before_private=False,actual_client_decode_started=False,actual_client_decode_completed=False,
        readback_complete=False,operation_complete=False,keygen=False,encryption=False,server_compute=False,model_forward=False,retirement=False,
        pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),scope='One separately bound size0 client decode; original P2 numeric/prediction gates, no ranked accuracy claim.')
    try:
        expected,guard=bootstrap(source_manifest,source_manifest_sha256);report.update(source_hashes_start=expected,source_guard_start=guard,software=io.software_versions())
        s=services()
    except BaseException as primary:
        report['error']=dict(type=type(primary).__name__,message=str(primary))
        try:io.write(output,report)
        except BaseException as error:primary.add_note('client startup failure publication: '+repr(error))
        raise
    return io.finish(output,report,lambda r:body(r,expected=expected,args=args,result_root=Path(result_workspace),result_identity=result_identity,output=output,s=s),s.termination_signals,s.cleanup_signals)

def main():
    p=argparse.ArgumentParser()
    for n in GATE_NAMES:
        p.add_argument('--'+n.replace('_','-'),required=True,type=str if n.endswith('sha256')else Path)
    for n in ('source-manifest','result-workspace','output'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--result-identity',required=True,type=json.loads)
    args=vars(p.parse_args());r=run(**args);print(json.dumps(dict(passed=r['passed'],operation_complete=r['operation_complete'],worker_report_sha256=io.digest(io.read(args['output'])))))
    return 0 if r['passed']else 1
if __name__=='__main__':raise SystemExit(main())
