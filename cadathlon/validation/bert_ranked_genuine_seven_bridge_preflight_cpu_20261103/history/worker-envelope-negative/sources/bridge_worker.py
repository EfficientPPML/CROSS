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
import ranked_worker as parent, bridge_protocol as protocol
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
        if not file or not Path(file).is_absolute():continue
        p=Path(file).resolve()
        if HERE in p.parents and p.suffix=='.py':
            rel=p.relative_to(HERE).as_posix()
            if expected.get(rel)!=key_gate.digest(p.read_bytes()):raise ValueError('loaded module outside reviewed closure')
            out[name]=rel
    for name in ('file_stages','key_contexts','full_model_engine'):
        module=sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve()!=HERE/'runtime/stage'/f'{name}.py':raise ValueError('mixed file-stage runtime')
    return out


def public_case(envelope):
    protocol.validate_envelope(envelope)
    return parent.public_case(envelope['request'])


def authenticate(envelope,expected):
    protocol.validate_envelope(envelope)
    authorized,case=parent.public_case(envelope['request'])
    protocol.authorize(envelope,expected,case)
    result=parent.authenticate(envelope['request'],expected)
    protocol.authenticate_partial(envelope,expected,result[-1])
    return result


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


def execute(envelope,authorized,case,paths,keys,model,previous,report,calls):
    from bridge_phases import execute_phase
    return execute_phase(envelope,authorized,case,paths,keys,model,previous,report,calls)


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
        stage=request['request']['stage'];report.update(bridge_phase=request['phase'],bridge_binding=protocol.binding(request,expected,bindings),stage=stage,bindings=bindings,phases=protocol.declared_phases(request['phase']))
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
            finally:
                try:os.umask(old_umask)
                except BaseException as error:
                    report.setdefault('cleanup_errors',[]).append({'operation':'restore umask','type':type(error).__name__})
                    if primary is not None:primary.add_note('worker umask restoration: '+type(error).__name__)
                    elif cleanup_error is None or not isinstance(error,Exception) and isinstance(cleanup_error,Exception):cleanup_error=error
            if primary is not None:raise primary
            if cleanup_error is not None:raise cleanup_error
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('request-path','output','source-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--request-sha256',required=True)
    from bridge_terminal import finish
    args=vars(p.parse_args())
    try:
        result=finish(args['output'],lambda:run(**args),write)
    except BaseException as error:
        # No private parser/native/codec error messages reach public logs.
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'private_error_text_withheld':True}));raise SystemExit(130 if not isinstance(error,Exception) else 1)
    print(json.dumps({'passed':result['passed'],'worker_report_sha256':key_gate.digest(Path(args['output']).read_bytes())}))
    raise SystemExit(0 if result['passed'] else 1)
