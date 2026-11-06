"""One gated dedicated toy4 role: client setup, keyless graph, private readback."""
from pathlib import Path
import argparse,hashlib,json,os,sys,tarfile,time
HERE=Path(__file__).resolve().parent
# Before the first local import: authenticate and load the stdlib-only guard.
_BOOTSTRAP=None
if __name__=='__main__':
    import hashlib,importlib.util,json,stat
    parser0=argparse.ArgumentParser(add_help=False)
    parser0.add_argument('--source-manifest',type=Path,required=True)
    parser0.add_argument('--source-manifest-sha256',required=True)
    args0,_=parser0.parse_known_args()
    def _initial_read(path,limit):
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);primary=None
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode)or not 0<info.st_size<=limit:raise ValueError('bounded bootstrap source required')
            data=b''
            while len(data)<=limit:
                part=os.read(fd,min(1<<20,limit+1-len(data)))
                if not part:break
                data+=part
            if len(data)!=info.st_size or len(data)>limit:raise ValueError('bootstrap source size differs')
            return data
        except BaseException as error:primary=error;raise
        finally:
            try:os.close(fd)
            except BaseException as error:
                if error is not primary and getattr(error,'_cross_toy4_body_origin',False)is not True:error._cross_toy4_cleanup_origin=True
                if primary is None:raise
                primary.add_note('bootstrap descriptor close: '+repr(error))
    manifest0=_initial_read(args0.source_manifest,1<<20)
    if hashlib.sha256(manifest0).hexdigest()!=args0.source_manifest_sha256:raise ValueError('external manifest digest differs before helper imports')
    expected0=json.loads(manifest0)
    guard_path=HERE/'source_guard.py';guard_raw=_initial_read(guard_path,8<<20)
    if type(expected0)is not dict or expected0.get('source_guard.py')!=hashlib.sha256(guard_raw).hexdigest():raise ValueError('source guard differs before import')
    if 'source_guard'in sys.modules:raise ValueError('source guard preloaded')
    spec0=importlib.util.spec_from_file_location('source_guard',guard_path)
    guard0=importlib.util.module_from_spec(spec0);sys.modules['source_guard']=guard0
    # Execute the already authenticated bytes; do not reopen a replaced path.
    exec(compile(guard_raw,str(guard_path),'exec'),guard0.__dict__)
    _BOOTSTRAP=guard0.bootstrap(args0.source_manifest,args0.source_manifest_sha256,root=HERE,entry=__file__)

sys.path[:0]=[str(HERE),str(HERE/'stage'),str(HERE/'runtime/cadathlon/common'),str(HERE/'runtime'),str(HERE/'runtime/jaxite_word')]
import component_protocol as protocol
import component_contract as contract
import component_transport as transport
from cleanup import cleanup_signals
from terminal import terminal_scope
from finish import finish
from publication import OwnedOutput,separate
from cleanup_origin import select,protect_body
_DECODE_FAILURE_QUARANTINE=[]

def fixed_files():return(HERE/'profile.json').read_bytes(),(HERE/'public_plan/plan.json').read_bytes()

def config_check(config):
    fields={'stage','workspace','workspace_identity','nonce','setup_receipt','setup_sha256','server_receipt','server_sha256','server_root_review','server_root_review_sha256'}
    if type(config)is not dict or set(config)!=fields or config['stage']not in protocol.STAGES:raise ValueError('exact single-role configuration required')
    if type(config['workspace'])is not str or not Path(config['workspace']).is_absolute():raise ValueError('absolute owned workspace required')
    if type(config['nonce'])is not str or not contract.NONCE.fullmatch(config['nonce']):raise ValueError('fixed opaque request nonce required')
    protocol.check_identity(config['workspace'],config['workspace_identity'])
    for stage in('setup','server'):
        required=config['stage']in(('server','validate')if stage=='setup'else('validate',))
        path,pin=config[stage+'_receipt'],config[stage+'_sha256']
        if required:
            if type(path)is not str or not Path(path).is_absolute()or type(pin)is not str or not contract.SHA.fullmatch(pin):raise ValueError('externally pinned prior role required')
        elif path is not None or pin is not None:raise ValueError('unused prior role must be absent')
    path,pin=config['server_root_review'],config['server_root_review_sha256']
    if config['stage']=='validate':
        if type(path)is not str or not Path(path).is_absolute()or type(pin)is not str or not contract.SHA.fullmatch(pin):raise ValueError('external completed server root review required')
    elif path is not None or pin is not None:raise ValueError('root review is only a validator prerequisite')
    return config

def priors(config,expected):
    out={}
    for stage in('setup','server'):
        if config[stage+'_receipt']is not None:
            out[stage]=protocol.terminal_gate(config[stage+'_receipt'],config[stage+'_sha256'],stage=stage,
                expected_sources=expected,workspace=config['workspace'],workspace_identity=config['workspace_identity'])
    if 'server'in out:
        for key in('request_contract','input_digest'):
            if contract.canonical(out['server'][key])!=contract.canonical(out['setup'][key]):raise ValueError('prior roles used different request')
    if config['stage']=='validate':
        from response_admission import authenticate_review
        authenticate_review(config,expected,out)
    return out

def device():
    import jax
    if os.environ.get('JAX_PLATFORMS')!='tpu':raise ValueError('explicit original TPU platform required; no CPU fallback')
    jax.config.update('jax_enable_x64',True)
    devices=jax.devices('tpu')
    if not devices or devices[0].platform!='tpu'or devices[0].device_kind!='TPU v6 lite':raise ValueError('original TPU v6 lite device required')
    return jax,devices[0]

def setup(config,report,publish):
    import client_keys,key_contexts
    from fixture_assets import prepare_client_reference,export_public_weights
    from support import require_secure_execution
    workspace=Path(config['workspace']);profile,plan=fixed_files()
    # Security/Pcoverage and measured storage preflight occur inside generate
    #before key_gen imports or the first actual secret/public pair.
    result,refs=prepare_client_reference();del refs
    export_public_weights(workspace/'public_model',result['weights'])
    generated=client_keys.generate(workspace/'public_keys',workspace/'client_private',profile_bytes=profile,plan_bytes=plan)
    public=generated['public_manifest'];request=contract.make_contract(public['context_id'],config['nonce'])
    report.update(request_contract=request,key_context_id=public['context_id'],
        private_permissions_verified=True,private_array_count=1,public_array_count=len(public['arrays']),rotation_count=199,
        key_generation_preflight=generated['preflight'])
    folder=workspace/'inputs';folder.mkdir(mode=0o700,exist_ok=False)
    with key_contexts.load_client(workspace/'public_keys',workspace/'client_private',profile_bytes=profile,plan_bytes=plan)as handle:
        if handle.public_manifest['context_id']!=request['context_id']:raise ValueError('loaded client identity differs')
        value=handle.ctx.encrypt_slots(result['values'][0])
        if not hasattr(value,'_ckks_nsd'):value._ckks_nsd=1
        require_secure_execution(handle.ctx,value);value.polynomial.block_until_ready()
        manifest=transport.save_ciphertext(value,folder/'tile0',contract=request,role='request',tile_index=0)
        del value
    report['loader_cleanup_completed']=True;report['input_manifest_sha256']=transport.manifest_digest(manifest)
    report['input_digest']=transport.request_digest([manifest],request);publish()

def input_readback(workspace,request,prior):
    payload,scale,binding,manifest=transport.validate_ciphertext(workspace/'inputs/tile0',contract=request,role='request',tile_index=0)
    del payload
    if transport.manifest_digest(manifest)!=prior['input_manifest_sha256']or transport.request_digest([manifest],request)!=prior['input_digest']:raise ValueError('original encrypted input changed')
    return manifest

def server(config,report,publish,prior):
    import key_contexts
    import numpy as np
    from fixture_assets import load_public_weights
    from component_arithmetic import exercise
    workspace=Path(config['workspace']);profile,plan=fixed_files();setup=prior['setup'];request=contract.validate_contract(setup['request_contract'])
    input_readback(workspace,request,setup);weights=load_public_weights(workspace/'public_model')
    mask=np.asarray(contract.numerical_plan()['case']['attention_mask'],dtype=bool)
    folder=workspace/'outputs';folder.mkdir(mode=0o700,exist_ok=False);artifacts=[]
    with key_contexts.load_server(workspace/'public_keys',profile_bytes=profile,plan_bytes=plan)as handle:
        if handle.public_manifest['context_id']!=request['context_id']:raise ValueError('loaded keyless context differs')
        value,manifest=transport.load_ciphertext(workspace/'inputs/tile0',contract=request,role='request',tile_index=0)
        if transport.manifest_digest(manifest)!=setup['input_manifest_sha256']:raise ValueError('input changed during loading')
        def checkpoint(index,ciphertext,row):
            if type(index)is not int or index!=len(artifacts):raise ValueError('ordered checkpoint persistence required')
            manifest=transport.save_ciphertext(ciphertext,folder/f'tile{index}',contract=request,role='response',tile_index=index,
                expected_scale=row['scale'],input_digest=setup['input_digest'])
            artifacts.append(dict(index=index,manifest_sha256=transport.manifest_digest(manifest),metadata=dict(row)))
            report.update(completed_checkpoints=len(artifacts),artifacts=artifacts);publish()
        arithmetic=exercise(handle.ctx,value,weights,mask,checkpoint=checkpoint);del value
    report['loader_cleanup_completed']=True
    if len(artifacts)!=96:raise ValueError('all96 opaque checkpoints required')
    # Reauthenticate every saved artifact after graph/owner cleanup and before
    #terminal source publication. No private key or decode exists in this role.
    for i,row in enumerate(artifacts):
        payload,_,_,manifest=transport.validate_ciphertext(folder/f'tile{i}',contract=request,role='response',tile_index=i,
            expected_scale=row['metadata']['scale'],input_digest=setup['input_digest']);del payload
        if transport.manifest_digest(manifest)!=row['manifest_sha256']:raise ValueError('checkpoint changed after server work')
    input_readback(workspace,request,setup)
    report.update(request_contract=request,input_digest=setup['input_digest'],arithmetic=arithmetic,all96_server_outputs_verified=True)

def numeric_errors(actual,expected):
    import numpy as np
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=(65536,)or actual.dtype.kind!='f'or expected.shape!=(65536,)or expected.dtype.kind!='f':raise ValueError('real full65536-slot arrays required')
    if not np.isfinite(actual).all()or not np.isfinite(expected).all():raise ValueError('finite decoded/reference arrays required')
    with np.errstate(over='ignore',invalid='ignore'):
        maximum=np.max(np.abs(actual.astype(np.longdouble)-expected.astype(np.longdouble)))
    finite=bool(np.isfinite(maximum)and maximum<=np.finfo(np.float64).max)
    return dict(max_abs_error=float(maximum)if finite else None,finite=finite,absolute_gate=1e-6,passed=bool(finite and maximum<=1e-6))

def validate(config,report,publish,prior,*,expected_sources):
    import key_contexts,numpy as np
    from fixture_assets import prepare_client_reference
    from scale_replay import client_replay
    workspace=Path(config['workspace']);profile,plan=fixed_files();setup=prior['setup'];server=prior['server']
    request=contract.validate_contract(setup['request_contract']);artifacts=server['artifacts']
    from response_admission import authenticate_all,authenticate_review
    admission=authenticate_all(config,expected_sources,prior)
    replay=admission['replay'];original,references=prepare_client_reference()
    report.update(all_server_work_and_terminal_evidence_before_private=True,independent_scale_replay_verified=True,
        all96_current_responses_authenticated_before_private=True,server_root_review_sha256=admission['review']['sha256'],checks=[])
    with key_contexts.load_client(workspace/'public_keys',workspace/'client_private',profile_bytes=profile,plan_bytes=plan)as handle:
        if handle.public_manifest['context_id']!=request['context_id']:raise ValueError('private readback context differs')
        failures=[]
        for i,row in enumerate(artifacts):
            protocol.check_identity(workspace/'outputs',admission['output_identity'])
            if authenticate_review(config,expected_sources,prior)!=admission['review']:raise ValueError('review changed after private entry')
            record=dict(index=i,kind=row['metadata']['kind'],label=row['metadata']['label'],passed=False,decoder_passed=False)
            value=None
            try:
                value,manifest=transport.load_ciphertext(workspace/'outputs'/f'tile{i}',contract=request,role='response',tile_index=i,
                    expected_scale=replay['trace'][i]['scale'],input_digest=setup['input_digest'])
                if transport.manifest_digest(manifest)!=row['manifest_sha256']or transport.manifest_digest(manifest)!=admission['manifest_sha256'][i]:raise ValueError('server artifact changed')
                # Exactly the original decoder and0.03125 bound. Each failed
                #checkpoint is preserved while later independent readbacks run.
                decoded=handle.ctx.decrypt_slots(value);record['decoder_passed']=True
                record.update(numeric_errors(decoded,references[i]))
                if i==95:
                    logits=np.asarray(decoded)[:8192].reshape(2,8,8,64)[:,0,0,:2]
                    golden=np.asarray(json.loads((HERE/'runtime/cadathlon/agent_pset/p2_bert/test/test4.out').read_bytes())['logits'])
                    original_gate=bool(np.isfinite(logits).all()and np.all(np.abs(logits-golden)<=.02+.01*np.abs(golden))and np.array_equal(logits[:,1]>=logits[:,0],golden[:,1]>=golden[:,0]))
                    report['original_final_toy_gate_passed']=original_gate
                del value,decoded
            except Exception as error:
                # Preserve each independent diagnosis, but never reclaim a
                #decoder's unknown pending values as if successful. Retain all
                #failed wrappers and force loader quarantine after the final
                #readback; this dedicated process then exits unsuccessfully.
                _DECODE_FAILURE_QUARANTINE.append((value,handle,error));failures.append(error)
                record['error']=dict(type=type(error).__name__,message=str(error))
            report['checks'].append(record);publish()
        if failures:
            report['all96_readbacks_attempted']=len(report['checks'])==96
            raise RuntimeError('one or more checkpoint readbacks failed; client owners quarantined')from failures[0]
    report['loader_cleanup_completed']=True
    report['numeric_passed']=bool(len(report['checks'])==96 and all(r['passed']and r['decoder_passed']for r in report['checks'])and report.get('original_final_toy_gate_passed')is True)

def install_server_audit(workspace,attempts):
    forbidden=[(Path(workspace)/'client_private').resolve(),(HERE/'public_plan/public_graph.py').resolve(),(HERE/'public_plan/dense_reference.py').resolve(),(HERE/'runtime/cadathlon/agent_pset/p2_bert/src/reference.py').resolve()]
    def hook(event,args):
        if event!='open'or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if any(path==p or p in path.parents for p in forbidden):attempts.append(str(path));raise RuntimeError('private/client-oracle file read forbidden in server')
    sys.addaudithook(hook)

def run(config,expected,out,*,publication):
    report=dict(stage=config['stage'],passed=False,complete=False,cleanup_completed=False,source_verified=False,
        fresh_source_guard=None if _BOOTSTRAP is None else _BOOTSTRAP[1],loader_cleanup_completed=False,source_hashes_start=expected,workspace=config['workspace'],workspace_identity=config['workspace_identity'],forbidden_attempts=[])
    primary=None;start=time.monotonic();path=out/'report.json'
    def publish():publication.write(path,report)
    try:
        publish();protocol.check_sources(expected);config_check(config);prior=priors(config,expected)
        publication.check()
        with tarfile.open(out/'sources.tar.gz','x:gz')as archive:
            for name in expected:archive.add(HERE/name,arcname=name,recursive=False)
        publication.check();raw=protocol.read(out/'sources.tar.gz');protocol.verify_archive(raw,expected);report['source_archive_sha256']=protocol.sha(raw)
        if config['stage']=='server':install_server_audit(config['workspace'],report['forbidden_attempts'])
        jax,chosen=device();report['device']=dict(platform=chosen.platform,kind=chosen.device_kind,id=chosen.id)
        with jax.default_device(chosen):
            if config['stage']=='setup':setup(config,report,publish)
            elif config['stage']=='server':server(config,report,publish,prior)
            else:validate(config,report,publish,prior,expected_sources=expected)
        report['complete']=True;report['passed']=report.get('numeric_passed',True)
    except BaseException as error:primary=protect_body(error);report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,path,publication.write,cleanup_signals,lambda:(primary,None)):
            try:
                protocol.check_identity(config['workspace'],config['workspace_identity']);after=protocol.hashes(protocol.sources())
                report.update(source_hashes_end=after,source_verified=after==expected,elapsed_s=time.monotonic()-start,
                    imports=protocol.imported_sources(expected),cleanup_completed=report['loader_cleanup_completed'])
                report['passed']=bool(report['passed']and report['source_verified']and report['cleanup_completed']and not report['forbidden_attempts'])
                publish()
            except BaseException as error:
                primary=select(primary,[('terminal source/readback failure',error)])
            if primary is not None:raise primary
    return report

def main():
    if _BOOTSTRAP is None:raise ValueError('fresh authenticated isolated script entry required')
    parser=argparse.ArgumentParser()
    for name in('config','source-manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise ValueError('explicit actual role execution required')
    protocol.require_ready();raw=protocol.read(args.config)
    if protocol.sha(raw)!=args.config_sha256:raise ValueError('external worker configuration hash differs')
    config=protocol.parse(raw);config_check(config);expected=_BOOTSTRAP[0]
    if protocol.sha(protocol.read(args.source_manifest))!=args.source_manifest_sha256:raise ValueError('source manifest changed after admission')
    args.output=args.output.absolute()
    separate(args.output,[HERE,args.config,args.source_manifest,config['workspace'],config['setup_receipt'],config['server_receipt'],config['server_root_review']])
    publication=OwnedOutput(args.output).acquire()
    report=finish(args.output/'report.json',lambda:run(config,expected,args.output,publication=publication),publication.write,publication.read)
    return 0 if report['passed']else 1

if __name__=='__main__':raise SystemExit(main())
