"""Fixed full-slot secure refresh proof; client codecs run only outside evaluation."""
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import signal
import shutil
import resource
import sys
import time
import numpy as np

LAB=Path(__file__).resolve().parents[1];ROOT=LAB/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
DEGREE=131072;SLOTS=65536;DEPTH=49;DNUM=4;BOUND=16.;META_PRECISION=10;GATE=1e-3


def sources():
    paths=[*ROOT.rglob('*.py'),*Path(__file__).parent.glob('*.py'),LAB/'public_plan.json']
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def input_pattern():
    """Independent random real slots plus predeclared boundary/interior probes."""
    values=np.random.default_rng(20261103).uniform(-BOUND,BOUND,SLOTS)
    points=np.array([-16.,-14.2,-8.,-1.,-1e-6,-1e-12,0.,1e-12,1e-6,1.,8.,14.2,16.])
    values[:len(points)]=points
    if len(np.unique(values))!=SLOTS or np.max(np.abs(values))>BOUND:
        raise AssertionError('fixed independent-slot pattern is invalid')
    if any(np.array_equal(values,np.tile(values[:period],SLOTS//period)) for period in (1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768)):
        raise AssertionError('pattern unexpectedly has a shorter period')
    return values


def metrics(actual, expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=(SLOTS,) or expected.shape!=(SLOTS,) or actual.dtype.kind!='f' or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {'passed':False,'finite_and_shape':False,'slots_checked':SLOTS,'absolute_error_gate':GATE}
    diff=np.abs(actual-expected)
    error=float(np.max(diff))
    return {'passed':error<GATE,'finite_and_shape':True,'slots_checked':SLOTS,
        'absolute_error_gate':GATE,'max_abs_error':error,
        'rms_error':float(np.sqrt(np.mean(diff*diff))),
        'max_relative_error_with_1e_12_floor':float(np.max(diff/np.maximum(np.abs(expected),1e-12))),
        'worst_slot_index':int(np.argmax(diff)),'per_4096_slot_block_max_abs_error':diff.reshape(16,4096).max(axis=1).tolist(),
        'boundary_interior_probe_errors':diff[:13].tolist()}


def validate_key_directory(value):
    candidate=Path(value)
    if candidate.is_symlink():raise ValueError('owned key directory cannot be a symlink')
    path=candidate.resolve(strict=True)
    if not path.is_dir() or path==Path('/dev/shm') or not path.is_relative_to('/dev/shm'):
        raise ValueError('an owned child directory under /dev/shm is required')
    return path


def required_key_space(params, rotations):
    q,p=params['q_towers'],params['p_towers']
    alpha=(len(q)+DNUM-1)//DNUM;parts=(len(q)+alpha-1)//alpha
    return len(rotations)*(2*parts*(len(q)+len(p))*DEGREE*4+8192)+64*1024**2


CODEC_METHODS=('encrypt','encrypt_slots','encrypt_slots_batch','decrypt','decrypt_slots','decrypt_slots_batch','decode')
KEYGEN_METHODS=('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen')


def install_generator_guards(registry, restorations):
    names=('key_gen','jaxite_word.key_gen')
    if any(name not in registry for name in names):
        raise AssertionError('both historical and packaged key_gen aliases must be loaded')
    modules={registry[name] for name in names}
    def forbidden(*args,**kwargs):raise AssertionError('runtime key generation forbidden')
    for module in modules:
        for name in KEYGEN_METHODS:
            original=getattr(module,name)
            restorations.append(lambda m=module,n=name,f=original:setattr(m,n,f))
            setattr(module,name,forbidden)


def install_codec_guards(context_class, server_getter, phase_getter, restorations):
    for name in CODEC_METHODS:
        original=getattr(context_class,name)
        def guarded(self,*args,_original=original,_name=name,**kwargs):
            if phase_getter()=='server' or self is server_getter():
                raise AssertionError('codec forbidden in keyless evaluation: '+_name)
            return _original(self,*args,**kwargs)
        restorations.append(lambda n=name,f=original:setattr(context_class,n,f))
        setattr(context_class,name,guarded)


def ciphertext_fingerprint(value):
    array=np.asarray(value.polynomial)
    return dict(payload_sha256=hashlib.sha256(array.tobytes()).hexdigest(),payload_shape=list(array.shape),
        payload_dtype=str(array.dtype),degree=value.degree,batch=value.batch,num_elements=value.num_elements,
        num_moduli=value.num_moduli,moduli=list(value.moduli),degree_layout=list(value.degree_layout),
        precision=value.precision,scale=value._ckks_scale,nsd=value._ckks_nsd,
        ntt_backend=type(value.ntt_ctx).__module__+'.'+type(value.ntt_ctx).__name__,
        field_backend=type(value.ntt_ctx.ff_ctx).__module__+'.'+type(value.ntt_ctx.ff_ctx).__name__)


def release_all(contexts, owner, key_store, active_error, callbacks=None):
    """Attempt all releases, preserving a numerical/cancellation failure."""
    import jax
    from jaxite_word import ckks_ctx
    errors=[]
    callbacks=[] if callbacks is None else callbacks
    while callbacks:
        release=callbacks.pop(0)
        try:release()
        except BaseException as error:errors.append(error)
        release=None
    if owner is not None:
        try:owner.close()
        except BaseException as error:errors.append(error)
    while contexts:
        context=contexts.pop()
        if context is not None:
            try:ckks_ctx.evict_context_caches(context)
            except BaseException as error:errors.append(error)
        context=None
    for release in (jax.clear_caches,gc.collect):
        try:release()
        except BaseException as error:errors.append(error)
    if key_store is not None:
        try:key_store.close()
        except BaseException as error:errors.append(error)
    if errors:
        if active_error is not None:
            for error in errors:active_error.add_note('Cleanup also failed: '+repr(error))
        else:
            primary=next((e for e in errors if not isinstance(e,Exception)),errors[0])
            for error in errors:
                if error is not primary:primary.add_note('Other cleanup failure: '+repr(error))
            raise primary


def run(report):
    import jax
    from jaxite_word import CKKSContext,key_gen,he_params
    from encrypted_refresh import RefreshProfile,refresh_plan,BoundedSecureRefresh,ClientRotationKeyStore
    from encrypted_ntt import enable_scoped_ntt_subsets
    from support import require_secure_execution
    jax.config.update('jax_enable_x64',True)
    devices=jax.devices()
    if jax.default_backend()!='cpu' or not devices or any(device.platform!='cpu' for device in devices):
        raise ValueError('this component is CPU-only; no TPU lease is implied')
    if sorted(os.sched_getaffinity(0))!=[132,133,134,135]:
        raise ValueError('fixed proof requires CPU affinity132-135')
    if os.environ.get('BOOTSTRAP_DISABLE_HOISTED_LT')=='1':
        raise ValueError('full-slot proof requires the planned hoisted transform path')
    report['bootstrap_disable_hoisted_lt']=os.environ.get('BOOTSTRAP_DISABLE_HOISTED_LT','unset-default-enabled')
    key_directory=validate_key_directory(os.environ['CROSS_PUBLIC_KEY_DIRECTORY'])
    params=RefreshProfile(DEGREE,DEPTH,56,61,DNUM).parameters()
    plan=refresh_plan(DEGREE,SLOTS,(4,4),max_level=DEPTH)
    q,p=params['q_towers'],params['p_towers'];params['degree_layout']=(256,512)
    if len(q)!=100 or len(p)!=24 or len(plan['rotations'])!=106 or plan['output_level']!=25:
        raise ValueError('fixed approved public plan changed')
    if not he_params.qp_is_secure(q,p,DEGREE,he_params.SecurityTarget(128,'classical')):
        raise AssertionError('actual QP is insecure before key generation')
    he_params.assert_p_coverage(q,p,DNUM)
    report['actual_security_and_P_coverage_checked_before_keygen']=True
    required=required_key_space(params,plan['rotations'])
    if shutil.disk_usage(key_directory).free<required:
        raise ValueError('owned tmpfs lacks complete public key storage plus64MiB before keygen')
    report['required_public_key_free_bytes']=required
    report.update(profile={'degree':DEGREE,'slots':SLOTS,'depth':DEPTH,'dnum':DNUM,'q':q,'p':p,
        'scale':params['scaling_factor'],'level_budget':[4,4],'rotation_count':106,'period':SLOTS,
        'log2_QP':math.log2(math.prod(q+p))},plan=plan,requested_backend='cpu',
        requested_device_kinds=[d.device_kind for d in devices],key_directory=str(key_directory),
        key_filesystem={'total':os.statvfs(key_directory).f_blocks*os.statvfs(key_directory).f_frsize,
            'free':os.statvfs(key_directory).f_bavail*os.statvfs(key_directory).f_frsize})
    start=time.perf_counter()
    def progress(message):print(f'[{time.perf_counter()-start:.2f}s] {message}',file=sys.stderr,flush=True)
    client=server=refresh=owner=key_store=checkpoint=None
    restorations=[];phase='client-setup'
    try:
        progress('client key generation starting')
        keys=key_gen.gen_pke_pair(q,[],DEGREE)
        evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=DNUM,noise_std=he_params.DEFAULT_SIGMA,noise_scale=1)
        evaluation=[jax.numpy.asarray(evaluation[name],dtype='uint32').transpose(0,2,1) for name in ('a','b')]
        key_store=ClientRotationKeyStore.generate(keys['secret_key'],q,p,degree=DEGREE,dnum=DNUM,
            rotations=plan['rotations'],directory=key_directory)
        report['public_key_store_required_bytes']=key_store.required_bytes
        client=CKKSContext(dict(params,**keys));server=CKKSContext(dict(params,evaluation_key=evaluation))
        install_generator_guards(sys.modules,restorations)
        install_codec_guards(CKKSContext,lambda:server,lambda:phase,restorations)
        phase='server'
        server.program_initialization(list(plan['rotations']),dnum=DNUM,r=256,c=512,batch=1,pregenerated_rotation_keys=key_store.keys)
        del keys,evaluation
        if client is server or server.secret_key is not None or server._param_cache.secret_key is not None or server.parameters.get('secret_key') is not None or server.public_key is not None:
            raise AssertionError('separate keyless evaluator required')
        owner=enable_scoped_ntt_subsets(server)
        values=input_pattern();report['input_pattern_sha256']=hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
        phase='client-encoding'
        encrypted=client.encrypt_slots(values)
        report['input_ring']=require_secure_execution(client,encrypted)
        encrypted=server.he_level_reduce[DEPTH,2].level_reduce(encrypted)
        report['server_input_ring']=require_secure_execution(server,encrypted)
        report['server_secret_absent']=True
        report['server_public_key_absent']=True
        input_before=ciphertext_fingerprint(encrypted)
        phase='server'
        refresh=BoundedSecureRefresh(server,period=SLOTS,level_budget=(4,4),low_memory=True,
            compiled=True,preserve_dispatch_cache=True,public_bound=BOUND,meta_precision=META_PRECISION)
        if refresh.engine._sparse or refresh.engine._gap!=1 or refresh.plan['period']!=SLOTS:
            raise AssertionError('full independent-slot branch required')
        checkpoint=refresh.engine._stage_checkpoint
        stages=[]
        def trace(tag,ct=None):
            checkpoint(tag,ct);stages.append(tag);progress('bootstrap stage '+tag)
        refresh.engine._stage_checkpoint=trace
        report['setup_s']=time.perf_counter()-start
        progress('full-slot keyless refresh starting')
        began=time.perf_counter();refreshed=refresh.refresh(encrypted);jax.block_until_ready(refreshed.polynomial)
        report['refresh_s']=time.perf_counter()-began
        level=server._param_cache.level_for_num_q(refreshed.num_moduli)
        if level!=25 or refreshed._ckks_nsd!=1 or refresh.refresh_count!=1:
            raise AssertionError('unexpected full-slot refresh output metadata/count')
        if not math.isclose(refreshed._ckks_scale,params['scaling_factor'],rel_tol=1e-12,abs_tol=0.):
            raise AssertionError('refresh did not restore canonical scale')
        refreshed_scale=refreshed._ckks_scale
        square_divisor=math.prod(server._param_cache.q_moduli_at_level(25)[-2:])
        expected_square_scale=refreshed_scale*refreshed_scale/square_divisor
        report['refreshed_ring']=require_secure_execution(refresh,refreshed)
        square=refresh.evaluator.multiply(refreshed,refreshed);jax.block_until_ready(square.polynomial)
        if server._param_cache.level_for_num_q(square.num_moduli)!=24 or square._ckks_nsd!=1:
            raise AssertionError('unexpected squared continuation metadata')
        if not math.isclose(square._ckks_scale,expected_square_scale,rel_tol=1e-15,abs_tol=0.):
            raise AssertionError('square scale differs from actual rescale recurrence')
        report['scales']={'input':encrypted._ckks_scale,'canonical':params['scaling_factor'],'refreshed':refreshed_scale,
            'square_divisor':square_divisor,'expected_square':expected_square_scale,'actual_square':square._ckks_scale}
        report['input_fingerprint_before']=input_before
        report['input_fingerprint_after']=ciphertext_fingerprint(encrypted)
        report['input_payload_and_metadata_unchanged']=input_before==report['input_fingerprint_after']
        if not report['input_payload_and_metadata_unchanged']:
            raise AssertionError('refresh mutated the input ciphertext or metadata')
        report['squared_ring']=require_secure_execution(server,square)
        if stages.count('approxmod-re')!=2 or stages.count('approxmod-im')!=2:
            raise AssertionError('meta refresh did not execute both full-slot branches twice')
        report.update(server_completed_before_client_decode=True,full_slot_real_approxmod_passes=2,
            full_slot_imag_approxmod_passes=2,periodic_projection_aliases=1,periodic_projection_rotations=0,
            stage_checkpoints=stages,server_codecs_and_runtime_keygen_forbidden=True)
        phase='client-validation'
        decoded_refresh=np.asarray(client.decrypt_slots(refreshed));decoded_square=np.asarray(client.decrypt_slots(square))
        report['refresh_check']=metrics(decoded_refresh,values);report['square_check']=metrics(decoded_square,values*values)
        report['numerical_passed']=report['refresh_check']['passed'] and report['square_check']['passed']
        report.update(output_level=25,square_level=24,full_slots_checked=65536,unique_input_values=65536,
            component_s_through_validation=time.perf_counter()-start)
    finally:
        active=sys.exc_info()[1]
        callbacks=restorations
        if refresh is not None:
            callbacks.append(refresh.evaluator._clear_working_caches)
            callbacks.extend(helper.clear for helper in (refresh.compiled_primitives or ()))
        contexts=[client,server];refresh=client=server=checkpoint=None
        release_all(contexts,owner,key_store,active,callbacks)
        report['cleanup_completed']=True
        report['peak_host_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['memory_scope']='Linux process ru_maxrss only; not accelerator HBM or whole-machine peak'


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'cadathlon/validation'))
    import check_transformer_suite as collector
    report={'passed':False,'numerical_passed':False,'scope':'One full independent-slot bounded refresh and square only; no full-model or calibration claim.',
        'public_bound':BOUND,'meta_precision':META_PRECISION,'absolute_error_gate':GATE,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'input_pattern':'default_rng(20261103) uniform[-16,16], with13 fixed signed boundary/interior probes;65536uniquevalues,no short period'}
    caught=None
    try:
        collector.write_json(args.report,report)
        report['source_hashes_start']=sources()
        collector.write_json(args.report,report)
        with collector.termination_signals():run(report)
    except BaseException as error:
        caught=error;report['error']=type(error).__name__+': '+str(error);report['notes']=list(getattr(error,'__notes__',()))
    finally:
        try:
            report['source_hashes_end']=sources();report['source_files_unchanged']=bool(report.get('source_hashes_start')) and report['source_hashes_end']==report.get('source_hashes_start')
        except BaseException as error:
            report['source_files_unchanged']=False
            if caught is None:caught=error
            else:caught.add_note('Source verification also failed: '+repr(error))
        report['passed']=caught is None and report.get('numerical_passed') is True and report.get('cleanup_completed') is True and report.get('source_files_unchanged') is True
        try:collector.write_json(args.report,report)
        except BaseException as error:
            if caught is None:raise
            caught.add_note('Final proof report also failed: '+repr(error))
    print(json.dumps(report,allow_nan=False),flush=True)
    if caught is not None and not isinstance(caught,Exception):raise caught
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
