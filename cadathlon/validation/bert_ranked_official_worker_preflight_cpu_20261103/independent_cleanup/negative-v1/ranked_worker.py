"""Fresh-key ranked stages3/4/5 with unchanged all-N handlers.

The combined proof worker's call durations are diagnostics, not original
subprocess timing. Public gates precede every private/key/ciphertext payload.
"""
from pathlib import Path
from contextlib import redirect_stdout
import argparse,hashlib,importlib.util,json,os,sys,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'joined'),str(HERE/'joined/checkout/cadathlon/common'),str(HERE/'runtime')]
import key_gate,ranked_case_gate as cases,ranked_stage_gate as stage_gate,ranked_execution as execution
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope


REQUEST_FIELDS={'stage','size','seed','staged_root','attempt_sha256','fixture_evidence','fixture_report_sha256',
    'reference_evidence','reference_report_sha256','key_evidence','key_report_sha256','model_evidence','model_report_sha256','predecessors'}


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');temp.replace(path)

def hashes(expected):
    out={}
    if type(expected)is not dict or not expected:raise ValueError('nonempty reviewed source inventory required')
    for name,h in expected.items():
        p=Path(name)
        if type(name)is not str or p.is_absolute() or '..' in p.parts:raise ValueError('relative source path required')
        key_gate.sha_field(h);out[name]=key_gate.digest((HERE/p).read_bytes())
    return out

def local_modules(expected):
    out={}
    for name,module in tuple(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        p=Path(file).resolve()
        if HERE in p.parents and p.suffix=='.py':
            rel=p.relative_to(HERE).as_posix()
            if expected.get(rel)!=key_gate.digest(p.read_bytes()):raise ValueError('loaded module outside reviewed closure')
            out[name]=rel
    for name in ('file_stages','key_contexts','full_model_engine'):
        module=sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve()!=HERE/'runtime/stage'/f'{name}.py':raise ValueError('mixed file-stage runtime')
    return out


def public_case(request):
    if type(request)is not dict or set(request)!=REQUEST_FIELDS:raise ValueError('exact ranked worker request schema required')
    if type(request['stage'])is not int or request['stage'] not in (1,2,3,4,5):raise ValueError('ranked setup/online stage1..5 required')
    authorized=cases.authenticate(request['fixture_evidence'],request['fixture_report_sha256'],request['reference_evidence'],request['reference_report_sha256'])
    case=authorized.case(request['size'],request['seed'])
    execution.authenticate_attempt(request['staged_root'],request['attempt_sha256'],case)
    return authorized,case


def authenticate(request,expected):
    authorized,case=public_case(request);paths=execution.paths(request['staged_root'],case)
    profile=(HERE/'runtime/stage/server_inputs/p60_profile_and_refresh.json').read_bytes()
    plan=(HERE/'runtime/stage/server_inputs/twelve_layer_plan.json').read_bytes()
    import ranked_setup as setup
    stage=request['stage']
    if stage==1:
        if any(request[n]is not None for n in ('key_evidence','key_report_sha256','model_evidence','model_report_sha256')) or request['predecessors']!=[]:
            raise ValueError('fresh key setup accepts no retained producer fallback')
        return authorized,case,paths,None,None,None,setup.binding(request,case)
    keys=setup.authenticate_keys(request['key_evidence'],request['key_report_sha256'],request=request,case=case,paths=paths,
        expected_sources=expected,profile_bytes=profile,plan_bytes=plan)
    if stage==2:
        if request['model_evidence']is not None or request['model_report_sha256']is not None or request['predecessors']!=[]:
            raise ValueError('fresh model setup accepts no retained model fallback')
        return authorized,case,paths,keys,None,None,setup.binding(request,case)
    model=setup.authenticate_model(request['model_evidence'],request['model_report_sha256'],request=request,case=case,paths=paths,keys=keys,expected_sources=expected)
    previous=None;rows=request['predecessors']
    if type(rows)is not list or len(rows)!=request['stage']-3:raise ValueError('complete ordered producer predecessor chain required')
    for stage,row in zip(range(3,request['stage']),rows):
        if type(row)is not dict or set(row)!={'stage','evidence','report_sha256'} or type(row['stage'])is not int or row['stage']!=stage:
            raise ValueError('ranked predecessor chain schema/order differs')
        previous=stage_gate.authenticate(row['evidence'],row['report_sha256'],stage=stage,case=case,expected_sources=expected,
            context_id=keys['context_id'],model_id=model['model_id'],key_report_sha256=request['key_report_sha256'],
            model_report_sha256=request['model_report_sha256'],previous=previous)
    bindings=dict(case_binding=stage_gate.case_binding(case),context_id=keys['context_id'],model_id=model['model_id'],
        key_report_sha256=request['key_report_sha256'],model_report_sha256=request['model_report_sha256'],
        previous_stage_report_sha256=None if previous is None else previous['report_sha256'])
    return authorized,case,paths,keys,model,previous,bindings


def private_bytes(path):
    from ranked_readback import _private_read
    return _private_read(path,16<<20)


def input_binding(paths,case,*,tokenized):
    raw=private_bytes(paths['preprocessed_input'] if tokenized else paths['test_input'])
    rel=f'io/{execution.NAMES[case.size]}/intermediate/client_preprocessed_input' if tokenized else f'datasets/{execution.NAMES[case.size]}/intermediate/test_input.txt'
    if key_gate.digest(raw)!=dict(case.input_files)[rel]:raise ValueError('fresh official input bytes differ from authenticated ranked selection')
    if len(raw.decode('utf8').splitlines())!=case.count:raise ValueError('fresh ranked input count differs')
    return key_gate.digest(raw)


def guard(stage,calls,paths):
    if stage==3:
        from ranked_encode_boundaries import encryption_only
        return encryption_only(calls)
    if stage==4:
        from ranked_server_boundaries import server_only
        return server_only(calls,private_paths=(paths['client_private'],paths['test_input'].parent,paths['preprocessed_input'].parent))
    from ranked_decode_boundaries import client_only
    return client_only(calls)


def execute(request,authorized,case,paths,keys,model,previous,report,calls):
    import file_stages,ranked_readback as readback,file_stage_io,transport_contract as tc
    stage=request['stage'];execution.require_fresh(paths,stage)
    if stage in (1,2):
        from ranked_setup import execute_setup
        return execute_setup(request,case,paths,keys,report,calls)
    duration=[]
    def timed(label,fn):
        start=time.perf_counter()
        result=fn();duration.append(dict(stage=label,seconds=time.perf_counter()-start,timer_kind='diagnostic-handler-call'))
        return result
    with guard(stage,calls,paths),redirect_stdout(sys.stderr):
        if stage==3:
            input_binding(paths,case,tokenized=False)
            report['actual_preprocessing_started']=True
            timed('Input preprocessing',lambda:execution.adapter().preprocess(paths))
            input_binding(paths,case,tokenized=True);report['actual_preprocessing_completed']=True
            report['actual_stage_started']=True
            result=timed('Input encryption',lambda:file_stages.client_encode_encrypt_input(paths))
            batch=file_stage_io.read_manifest(paths['upload']/'batch.json')
            summary,batch=readback.requests(paths,public=batch['public'],count=case.count,expected_batch_sha256=result['public_batch_sha256'])
            if batch['public']['context_id']!=keys['context_id'] or batch['public']['model_id']!=model['model_id']:raise ValueError('fresh encrypted batch setup identity differs')
            input_binding(paths,case,tokenized=True)
            import file_stage_contract as batches,client_inputs
            records=client_inputs.read_preprocessed_jsonl(paths['preprocessed_input'])
            private=key_gate.parse(private_bytes(paths['client_private']/'requests.json'))
            batches.validate_private_requests(private,batch,[r.target_idx for r in records],case.input_sha256)
            report['all_ordered_input_bindings_validated']=True
        else:
            # Metadata is public; authenticated predecessor is required before
            #the complete payload readback or any loader/context operation.
            old=previous['readback'];batch=file_stage_io.read_manifest(paths['upload']/'batch.json')
            if batch['public']['context_id']!=keys['context_id'] or batch['public']['model_id']!=model['model_id']:raise ValueError('predecessor public setup identity differs')
            summary,batch=readback.requests(paths,public=batch['public'],count=case.count,expected_batch_sha256=old['public_batch_sha256'],
                expected_summary=old if stage==4 else None)
            if stage==4:
                report['actual_stage_started']=True
                result=timed('Encrypted computation',lambda:file_stages.server_encrypted_compute({k:v for k,v in paths.items() if k in execution.SERVER_FIELDS}))
                summary=readback.responses(paths,batch=batch)
                readback.requests(paths,public=batch['public'],count=case.count,expected_batch_sha256=old['public_batch_sha256'],expected_summary=old)
                report.update(full_model_graph_completed=True,all_ordered_response_bindings_validated=True)
            else:
                response=readback.responses(paths,batch=batch,expected_response_sha256=old['response_batch_sha256'])
                if tc.canonical(response)!=tc.canonical(old):raise ValueError('complete response changed since producer')
                input_binding(paths,case,tokenized=True)
                from ranked_private_reference import load
                refs=load(authorized,size=case.size,seed=case.seed)
                report['actual_stage_started']=True
                result=timed('Result decryption',lambda:file_stages.client_decrypt_decode(paths))
                timed('Result postprocessing',lambda:execution.adapter().postprocess(paths))
                raw=private_bytes(paths['predictions'])
                try:predictions=[int(s) for s in raw.decode().splitlines()]
                except Exception:raise ValueError('ranked prediction file is invalid') from None
                summary=readback.client_results(paths,batch=batch,references=refs.records(),count=case.count,predictions=predictions)
                report['all_ordered_private_outputs_validated']=True
        if type(result)is not dict or type(result.get('sample_count'))is not int or result['sample_count']!=case.count or result.get('public_batch_sha256')!=summary['public_batch_sha256']:
            raise ValueError('unchanged all-record handler completion differs')
    if calls:raise ValueError('forbidden ranked phase action attempted')
    report.update(actual_stage_completed=True,readback_complete=True,readback=summary,handler_durations=duration,
        official_harness_timing=False)
    if stage==5 and summary['metrics']['final_ranked_acceptance']is not True:raise ValueError('official ranked integer accuracy gates failed')
    return summary


def run(request_path,request_sha256,output,source_manifest):
    start=time.monotonic();primary=None;expected={};calls=[];held=None;cleanup_error=None;old_umask=os.umask(0o077)
    report=dict(passed=False,actual_stage_started=False,actual_stage_completed=False,readback_complete=False,
        source_files_unchanged=False,cleanup_completed=False,forbidden_attempts=calls,pid=os.getpid(),
        public_error_redacted=True,official_harness_timing=False)
    try:
        write(output,report)
        if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU preparation/worker required')
        expected=key_gate.parse(key_gate.read(source_manifest,4<<20));before=hashes(expected)
        if before!=expected:raise ValueError('worker startup source differs')
        report['source_hashes_start']=before
        raw=key_gate.read(request_path,4<<20)
        if key_gate.digest(raw)!=key_gate.sha_field(request_sha256):raise ValueError('collector request identity differs')
        request=key_gate.parse(raw);a=authenticate(request,expected);authorized,case,paths,keys,model,previous,bindings=a
        stage=request['stage'];report.update(stage=stage,bindings=bindings,phases=__import__('ranked_setup').phases(stage) if stage<3 else dict(encryption=stage==3,decryption=stage==5,server_compute=stage==4,model_forward=False,key_generation=False))
        held=execution.HeldDirectory(request['staged_root']);report['loaded_local_modules_start']=local_modules(expected)
        execute(request,*a[:-1],report,calls)
        held.check()
        if authenticate(request,expected)[-1]!=bindings:raise ValueError('producer prerequisites changed during worker')
        report['passed']=True
    except BaseException as error:
        primary=error;report.update(passed=False,error={'type':type(error).__name__,'message':'ranked worker failed; private exception text withheld'})
    finally:
        with terminal_scope(report,output,write,cleanup_signals,lambda:(primary,cleanup_error)):
            if held is not None:
                try:held.close()
                except BaseException as error:
                    cleanup_error=error;report.setdefault('cleanup_errors',[]).append({'type':type(error).__name__})
                    if primary is not None:primary.add_note('ranked directory close: '+type(error).__name__)
            try:
                after=hashes(expected) if expected else {};report.update(source_hashes_end=after,source_files_unchanged=bool(expected) and after==expected,
                    cleanup_completed=cleanup_error is None,elapsed_s=time.monotonic()-start)
                if expected:report['loaded_local_modules_end']=local_modules(expected)
                report['passed']=bool(report['passed'] and report['source_files_unchanged'] and report['cleanup_completed']);write(output,report)
            except BaseException as error:
                if primary is None:primary=error
                else:primary.add_note('worker finalization: '+type(error).__name__)
            finally:os.umask(old_umask)
            if primary is not None:raise primary
            if cleanup_error is not None:raise cleanup_error
    print(json.dumps({'passed':report['passed'],'worker_report_sha256':key_gate.digest(Path(output).read_bytes())}));return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('request-path','output','source-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--request-sha256',required=True)
    try:
        with termination_signals():result=run(**vars(p.parse_args()))
    except BaseException as error:
        # No private parser/native/codec error messages reach public logs.
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'private_error_text_withheld':True}));raise SystemExit(130 if not isinstance(error,Exception) else 1)
    raise SystemExit(0 if result['passed'] else 1)
