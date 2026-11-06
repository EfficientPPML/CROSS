"""One dedicated role of the fixed-gain4 numerical proof; actual execution is gated."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import os
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'common'),str(HERE/'runtime')]
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
from finish import finish
import component_protocol as protocol
import component_contract as contract
import component_key_gate as key_gate
import component_transport as transport


def public_oracle():
    spec=importlib.util.spec_from_file_location('gain4_public_oracle',HERE/'gain4_public_plan.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.build_plan()!=contract.numerical_plan():raise ValueError('independent public oracle plan differs')
    return module


def numeric_errors(actual,expected):
    """Preserve the strict slot gate while keeping failed reports valid JSON.

    Scaling here is only for computing an overflow-safe diagnostic norm. The
    acceptance test compares the original absolute error with the fixed gate.
    Unrepresentable diagnostics are None, never clipped or treated as passes.
    """
    import numpy as np
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=(65536,) or actual.dtype.kind!='f' or not np.isfinite(actual).all():
        raise ValueError('finite real full-slot decode required')
    if expected.shape!=(65536,) or expected.dtype.kind!='f' or not np.isfinite(expected).all():
        raise ValueError('finite real full-slot coordinate oracle required')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        difference=actual.astype(np.longdouble)-expected.astype(np.longdouble)
        maximum=np.max(np.abs(difference))
        if maximum==0:relative=np.longdouble(0)
        elif not np.isfinite(maximum):relative=np.longdouble(np.inf)
        else:
            error_rms=maximum*np.sqrt(np.mean((difference/maximum)**2))
            denominator=max(float(np.sqrt(np.mean(expected*expected))),.01)
            relative=error_rms/denominator
    def finite_float(value):
        return float(value) if np.isfinite(value) and abs(value)<=np.finfo(np.float64).max else None
    absolute,relative=finite_float(maximum),finite_float(relative)
    return dict(max_abs_error=absolute,nrmse=relative,numeric_metrics_finite=absolute is not None and relative is not None,
        absolute_gate=1e-6,passed=bool(np.isfinite(maximum) and maximum<=1e-6))


def validate_config(value):
    names={'stage','workspace','workspace_identity','request_contract','key_evidence','key_workspace',
        'fixture_evidence','fixture_report_sha256','server_evidence','server_report_sha256'}
    if type(value)is not dict or set(value)!=names or value['stage']not in protocol.STAGES:
        raise ValueError('exact single-phase worker configuration required')
    contract.validate_contract(value['request_contract'])
    for n in ('workspace','key_evidence','key_workspace'):
        if type(value[n])is not str or not Path(value[n]).is_absolute():raise ValueError('absolute configured workspace/evidence path required')
    wanted=Path('/tmp')/('cross-normalized-gain4-'+value['request_contract']['nonce'])
    if value['workspace']!=str(wanted):raise ValueError('dedicated nonce-bound proof IO path required')
    for stage in ('fixture','server'):
        needed=value['stage'] in (('server','validate') if stage=='fixture' else ('validate',))
        path,sha=value[stage+'_evidence'],value[stage+'_report_sha256']
        if needed:
            if type(path)is not str or not Path(path).is_absolute():raise ValueError('absolute completed prior evidence required')
            key_gate.sha_field(sha)
        elif path is not None or sha is not None:raise ValueError('unused prior phase must be absent')
    protocol.check_identity(value['workspace'],value['workspace_identity'])
    return value


def prior_gates(config,expected):
    receipts={}
    for stage in ('fixture','server'):
        if config[stage+'_evidence'] is not None:
            receipts[stage]=protocol.terminal_gate(config[stage+'_evidence'],config[stage+'_report_sha256'],stage=stage,
                expected_sources=expected,workspace=config['workspace'],workspace_identity=config['workspace_identity'],request=config['request_contract'])
    if 'server' in receipts and receipts['server']['readback']['input_digest']!=receipts['fixture']['readback']['input_digest']:
        raise ValueError('completed server used different encrypted fixtures')
    return receipts


def fixture(config,keys,profile,plan,report,publish):
    import key_contexts
    from support import require_secure_execution
    request=config['request_contract'];workspace=Path(config['workspace']);folder=workspace/'inputs'
    folder.mkdir(mode=0o700,exist_ok=False);oracle=public_oracle();values=oracle.fixture_arrays();manifests=[]
    if [protocol.sha(a.tobytes()) for a in values]!=request['input_patterns']:
        raise ValueError('original public fixture bytes differ')
    with key_contexts.load_client(keys['public_keys'],Path(config['key_workspace'])/'client_private',profile_bytes=profile,plan_bytes=plan) as client:
        if client.public_manifest['context_id']!=request['context_id']:raise ValueError('loaded client context differs')
        report['loader_entered']=True;publish()
        for i,slots in enumerate(values):
            value=client.ctx.encrypt_slots(slots)
            if not hasattr(value,'_ckks_nsd'):value._ckks_nsd=1
            require_secure_execution(client.ctx,value);value.polynomial.block_until_ready()
            manifests.append(transport.save_ciphertext(value,folder/f'tile{i}',contract=request,role='request',tile_index=i))
            report['completed_ciphertexts']=i+1;publish();del value
    report['loader_cleanup_completed']=True
    digest=transport.request_digest(manifests,request)
    manifest=dict(schema=protocol.MANIFEST_SCHEMA,contract=request,role='request',input_digest=digest,
        tiles=[dict(index=i,manifest_sha256=transport.manifest_digest(m)) for i,m in enumerate(manifests)])
    protocol.write(folder/'manifest.json',manifest,replace=False)
    report['readback']=protocol.readback(workspace,request,role='request')


def server(config,keys,profile,plan,report,publish,receipts):
    import key_contexts
    from gain4_arithmetic import exercise
    workspace=Path(config['workspace']);request=config['request_contract'];prior=receipts['fixture']['readback']
    actual=protocol.readback(workspace,request,role='request')
    if actual!=prior:raise ValueError('fixture artifacts changed after terminal evidence')
    report['all_inputs_authenticated_before_server']=True;publish()
    folder=workspace/'outputs';folder.mkdir(mode=0o700,exist_ok=False);manifests=[];values=[]
    for i,row in enumerate(prior['manifest']['tiles']):
        value,m=transport.load_ciphertext(workspace/'inputs'/f'tile{i}',contract=request,role='request',tile_index=i)
        if transport.manifest_digest(m)!=row['manifest_sha256']:raise ValueError('loaded fixture changed')
        values.append(value)
    proof=contract.numerical_plan()
    with key_contexts.load_server(keys['public_keys'],profile_bytes=profile,plan_bytes=plan) as handle:
        if handle.public_manifest['context_id']!=request['context_id']:raise ValueError('loaded server context differs')
        report['loader_entered']=True;publish()
        def checkpoint(index,value,row):
            if type(index)is not int or index!=len(manifests) or index>=54:raise ValueError('strict checkpoint completion order required')
            expected=proof['checkpoints'][index]
            # Arithmetic independently checks the entire plan row; transport
            # binds actual Polynomial metadata and payload before publication.
            if contract.canonical(row)!=contract.canonical(expected):raise ValueError('checkpoint plan differs')
            value.polynomial.block_until_ready()
            manifests.append(transport.save_ciphertext(value,folder/f'tile{index}',contract=request,role='response',tile_index=index,
                expected_scale=expected['scale'],input_digest=prior['input_digest']))
            report['completed_ciphertexts']=index+1;publish()
        report['arithmetic']=exercise(handle.ctx,*values,checkpoint=checkpoint)
        if len(manifests)!=54:raise ValueError('all54 checkpoints must complete')
    report['loader_cleanup_completed']=True;del values,value
    manifest=dict(schema=protocol.MANIFEST_SCHEMA,contract=request,role='response',input_digest=prior['input_digest'],
        tiles=[dict(index=i,manifest_sha256=transport.manifest_digest(m)) for i,m in enumerate(manifests)])
    protocol.write(folder/'manifest.json',manifest,replace=False)
    report['readback']=protocol.readback(workspace,request,role='response',input_digest=prior['input_digest'])
    if protocol.readback(workspace,request,role='request')!=prior:raise ValueError('fixtures changed during routing')


def validate(config,keys,profile,plan,report,publish,receipts):
    import numpy as np
    import key_contexts
    from support import require_secure_execution
    from gain4_fixtures import definitions,client_oracles,checkpoint_error,derived_checks,diagnostics
    from encrypted_bert_normalization_round import NormalizationRoundContract
    from encrypted_score_reduction import ScoreReductionLayout
    workspace=Path(config['workspace']);request=config['request_contract'];prior=receipts['server']['readback']
    if protocol.readback(workspace,request,role='request')!=receipts['fixture']['readback']:raise ValueError('original encrypted fixtures changed')
    before=protocol.readback(workspace,request,role='response',input_digest=prior['input_digest'])
    if before!=prior:raise ValueError('completed server ciphertexts changed')
    report['all54_authenticated_before_private_read']=True;publish()
    oracle=public_oracle();proof=contract.numerical_plan();expected=oracle.oracle_arrays()
    if len(expected)!=54:raise ValueError('all54 independent original oracle arrays required')
    for row,answer in zip(proof['checkpoints'],expected):
        if protocol.sha(answer.astype('<f8',copy=False).tobytes())!=row['expected_array_sha256']:raise ValueError('original fixed oracle changed')
    errors=[];decoded={c['name']:{}for c in proof['cases']}
    with key_contexts.load_client(keys['public_keys'],Path(config['key_workspace'])/'client_private',profile_bytes=profile,plan_bytes=plan) as client:
        if client.public_manifest['context_id']!=request['context_id']:raise ValueError('loaded validator context differs')
        report['loader_entered']=True;publish()
        for i,answer in enumerate(expected):
            row=proof['checkpoints'][i]
            value,m=transport.load_ciphertext(workspace/'outputs'/f'tile{i}',contract=request,role='response',tile_index=i,expected_scale=row['scale'],input_digest=prior['input_digest'])
            if transport.manifest_digest(m)!=prior['manifest']['tiles'][i]['manifest_sha256']:raise ValueError('ciphertext changed before decode')
            require_secure_execution(client.ctx,value);actual=np.asarray(client.ctx.decrypt_slots(value))
            result=dict(index=i,case=row['case'],label=row['label'],slots=65536,level=row['level'],scale=row['scale'],absolute_gate=1e-6,**checkpoint_error(actual,answer))
            errors.append(result);decoded[row['case']][row['label']]=actual.copy();report['numeric_checkpoints']=errors;publish();del value,actual
    report['loader_cleanup_completed']=True;report['numeric_validation_complete']=len(errors)==54
    report['derived_checks']={};report['final_diagnostics']={}
    raw=(HERE/'gain4_candidate_manifest.json').read_bytes()
    for item,original in zip(definitions(),oracle.fixture_arrays()):
        rule=NormalizationRoundContract(raw,item['key_count']);refs,_=client_oracles(original,item,rule)
        values=decoded[item['name']]
        report['derived_checks'][item['name']]=derived_checks(values,refs,rule.updates)
        layout=ScoreReductionLayout(item['key_mask'],valid_rows=item['valid_rows'])
        report['final_diagnostics'][item['name']]=diagnostics(values['probabilities'],item,layout.input_mask)
    decoded.clear()
    if protocol.readback(workspace,request,role='response',input_digest=prior['input_digest'])!=before:raise ValueError('server outputs changed during validation')
    report['readback']=before
    passed=(len(errors)==54 and all(r['passed']for r in errors)and all(row['passed']for case in report['derived_checks'].values()for row in case.values()))
    maxima=[r['max_abs_error']for r in errors]
    report['numeric_summary']=dict(checkpoints=54,physical_slots=54*65536,maximum_absolute_error=max(maxima)if all(x is not None for x in maxima)else None,absolute_gate=1e-6,passed=passed)
    publish()
    if not passed:raise ValueError('original strict all-slot/derived numerical gate failed')


def run(config,expected,output):
    start=time.monotonic();primary=None;calls=[];stage=config.get('stage');output=Path(output)
    report=dict(passed=False,phase_complete=False,stage=stage,pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        loader_entered=False,loader_cleanup_completed=False,source_files_unchanged=False,termination_scope_completed=False,forbidden_attempts=calls,
        scope='Isolated gain4 normalized round; original failed component retained; never full BERT, ranked accuracy or calibrated timing.')
    def publish():protocol.write(output,report)
    try:
        publish()
        if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU process required')
        profile,plan=protocol.check_sources(expected);report['source_hashes_start']=protocol.hashes(protocol.sources())
        validate_config(config);report.update(workspace_identity=config['workspace_identity'],request_contract=config['request_contract'])
        receipts=prior_gates(config,expected)
        keys=protocol.authenticate_keys(config['key_evidence'],config['key_workspace'],profile=profile,plan=plan)
        report['key_report_sha256']=protocol.KEY_REPORT;report['context_id']=keys['context_id']
        from component_boundaries import role_scope
        with role_scope(stage,protected_paths=(Path(config['key_workspace'])/'client_private',) if stage=='server' else (),calls=calls):
            report['loaded_local_modules_start']=protocol.imported_sources(expected)
            if stage=='fixture':fixture(config,keys,profile,plan,report,publish)
            elif stage=='server':server(config,keys,profile,plan,report,publish,receipts)
            else:validate(config,keys,profile,plan,report,publish,receipts)
        if calls:raise ValueError('forbidden component action attempted')
        protocol.check_identity(config['workspace'],config['workspace_identity'])
        again=protocol.authenticate_keys(config['key_evidence'],config['key_workspace'],profile=profile,plan=plan)
        if again!=keys:raise ValueError('external public key evidence changed')
        report.update(phase_complete=True,passed=True)
    except BaseException as error:
        primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        with terminal_scope(report,output,protocol.write,cleanup_signals,lambda:(primary,None)):
            try:
                after=protocol.hashes(protocol.sources());report.update(source_hashes_end=after,source_files_unchanged=after==expected,
                    forbidden_attempts=calls,elapsed_s=time.monotonic()-start)
                report['loaded_local_modules_end']=protocol.imported_sources(expected)
                report['passed']=bool(report['passed'] and report['source_files_unchanged'] and report['loader_cleanup_completed']);publish()
            except BaseException as error:
                if primary is None:raise
                primary.add_note('component terminal publication: '+repr(error))
            if primary is not None:raise primary
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','source-manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise ValueError('explicit actual phase execution required')
    protocol.require_ready()
    raw=key_gate.read(args.config)
    if protocol.sha(raw)!=key_gate.sha_field(args.config_sha256):raise ValueError('external worker configuration differs')
    if args.output.exists():raise FileExistsError('worker report must be new; existing evidence remains untouched')
    def operation():
        run(key_gate.parse(raw),key_gate.parse(key_gate.read(args.source_manifest)),args.output)
        return key_gate.parse(key_gate.read(args.output))
    report=finish(args.output,operation,protocol.write,key_gate.read,termination=termination_signals)
    return 0 if report['passed']else 1



if __name__=='__main__':raise SystemExit(main())
