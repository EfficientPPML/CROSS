"""Actual CPU remaining K128/s4 inverse rounds; no refresh or full softmax."""
import argparse,gc,hashlib,importlib.metadata,io,json,math,os,resource,signal,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('explicit CPU backend required')
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(HERE)]
import jax,numpy as np
from jaxite_word import CKKSContext,ckks_ctx,he_params,key_gen,util
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_packed_qwen import CompiledArithmetic
from encrypted_ntt import enable_scoped_ntt_subsets
from public_schedule import PublicSchedule
from encrypted_newton import scaled_newton_inverse

def source_bytes():
    paths={Path(__file__),HERE/'public_schedule.py',HERE/'baseline_manifest.json',HERE/'test_newton.py'}
    for folder in ('jaxite_word','cadathlon/common'):paths.update((ROOT/folder).glob('*.py'))
    return {str(p.relative_to(HERE)):p.read_bytes() for p in sorted(paths)}

def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def parameters():
    degree,depth,dnum=131072,41,3
    q=composite_prime_gen(2,2*(depth+1),61,56,2*degree,31)
    p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
    target=he_params.SecurityTarget(128,'classical')
    if not he_params.qp_is_secure(q,p,degree,target):raise ValueError('actual full Q/P ring is insecure')
    he_params.assert_p_coverage(q,p,dnum)
    alpha=(len(q)+dnum-1)//dnum;parts=(len(q)+alpha-1)//alpha
    return dict(degree=degree,depth=depth,dnum=dnum,q_towers=q,p_towers=p,
        slots=degree//2,degree_layout=[256,512],scale=float(q[-2])*q[-1],
        actual_log2_QP=sum(map(math.log2,q+p)),QP_bit_length=math.prod(q+p).bit_length(),
        security_bits=128,security_bit_length_ceiling=he_params.security_ceiling(degree,target),
        raw_evaluation_key_bytes=2*parts*degree*(len(q)+len(p))*4,
        top_ciphertext_payload_bytes=2*degree*len(q)*4,
        saved_update_payload_bytes=sum(2*degree*(2*(level+1))*4 for level in (40,37,34,31,28,25,22,19,16,13,10,7,4)),
        resource_scope='Raw key/ciphertext payload estimates only; NTT tables, controls, executables and host temporaries add overhead.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    raw=source_bytes();before={n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}
    with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
        for name,data in raw.items():
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    report=dict(scope=__doc__,passed=False,source_hashes_before=before,
        started_utc=datetime.now(timezone.utc).isoformat(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        fixed_absolute_error_gate=1e-6,checks=[],phases=[],
        versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','numpy')},
        environment={n:os.environ.get(n) for n in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        opt_ins={'scoped_NTT_subsets':True,'guarded_dispatch_cache':True,'compiled_arithmetic':True},
        limitations=['Four independent remaining-round inverse components only, selected at declared9/12/13/13 checkpoints; no composed softmax, refresh or model.',
                    'Publicly sampled declared-domain inputs; no all-input CKKS error bound.',
                    'Client and keyless server are separate objects in one dedicated process, not an OS sandbox.'])
    write(args.output/'report.json',report)
    started=time.perf_counter();caught=None;patched=[];old_handlers={}
    client=server=arithmetic=provider=keys=evaluation=encrypted_x=z=product=correction=ct=inverse_result=observed=None
    checkpoints=[]
    def interrupted(signum,frame):
        for number in old_handlers:signal.signal(number,signal.SIG_IGN)
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        old_handlers[number]=signal.signal(number,interrupted)
    def phase(name,value=None):
        if value is not None:value.polynomial.block_until_ready()
        row=dict(name=name,elapsed_s=time.perf_counter()-started,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if value is not None:row.update(level=arithmetic._level(value),scale=value._ckks_scale,nsd=value._ckks_nsd)
        report['phases'].append(row);write(args.output/'report.json',report);print(json.dumps(row),flush=True)
    try:
        ring=parameters();report['ring']=ring
        coefficients=json.loads((HERE/'baseline_manifest.json').read_text())['power_coefficients_in_t_2x']
        schedule=PublicSchedule(coefficients)
        contracts=[schedule.stage(128,stage) for stage in (1,2,3,4)]
        assert [item['iterations'] for item in contracts]==[9,12,13,13]
        c=32.;segment=ring['slots']//4
        x_parts=[];initial_parts=[];public_contracts=[]
        for contract in contracts:
            a,b=contract['a'],contract['b']
            points=np.unique(np.r_[np.linspace(a,b,129),np.geomspace(a,b,129),contract['inactive_dummy'],
                                   np.nextafter(a,b),np.nextafter(b,a)])
            initial=2./(c*(a+b))
            x_parts.append(points[np.arange(segment)%len(points)])
            initial_parts.append(np.full(segment,initial))
            public_contracts.append(dict(contract,K=128,s=4.,C=c,initial_z=initial,
                distinct_input_count=len(points),input_points=points.tolist(),segment_slots=segment,
                theoretical_true_inverse_range=[4/(128*b),4/(128*a)]))
        x=np.concatenate(x_parts);initial_by_slot=np.concatenate(initial_parts)
        report['public_contracts']=public_contracts
        report['checkpoint_policy']='Retain all13updates, decode only after server completion; stage1/2/3/4 outputs use their own segments at updates9/12/13/13.'
        assert jax.default_backend()=='cpu' and all(d.platform=='cpu' for d in jax.devices())
        assert ring['raw_evaluation_key_bytes']<512*1024**2
        phase('actual-ring security, public contract and resource preflight passed')
        if args.preflight:
            report['preflight_only']=True;report['passed']=True
        else:
            q,p,N,dnum=ring['q_towers'],ring['p_towers'],ring['degree'],ring['dnum']
            scale=ring['scale']
            keys=key_gen.gen_pke_pair(q,[],N)
            evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
            params=dict(degree=N,num_slots=ring['slots'],scaling_factor=scale,output_scale=scale,
                q_towers=q,p_towers=p,composite_degree=2,degree_layout=(256,512),p=56,
                CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
            client=CKKSContext(dict(params,**keys))
            server=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(evaluation[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
            def forbidden(*args,**kwargs):raise AssertionError('server encryption/decryption/secret-dependent key generation forbidden')
            for module in {key_gen,sys.modules.get('key_gen')}-{None}:
                for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):
                    patched.append((module,name,getattr(module,name)));setattr(module,name,forbidden)
            report['key_generation_guarded_modules']=sorted({module.__name__ for module,_,_ in patched})
            assert set(report['key_generation_guarded_modules'])=={'jaxite_word.key_gen','key_gen'}
            server.program_initialization([],dnum=dnum,r=256,c=512,batch=1,pregenerated_rotation_keys={})
            assert server.secret_key is None and server._param_cache.secret_key is None and server.parameters.get('secret_key') is None
            for name in ('encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch'):setattr(server,name,forbidden)
            client._polynomial_ntt_ctx=server._param_cache.ntt_q_max
            provider=enable_scoped_ntt_subsets(server)
            arithmetic=CompiledArithmetic(server,low_memory=True,preserve_dispatch_cache=True)
            keys=evaluation=None;report['keyless_server']=True
            from support import require_secure_execution
            encrypted_x=client.encrypt_slots(x);encrypted_x._ckks_nsd=1
            require_secure_execution(server,encrypted_x)
            phase('client input encrypted; keyless server ready',encrypted_x)
            class ObservedArithmetic:
                # Records only ciphertext level/scale and public operation count.
                # It never reads x, expected plaintext, or a client codec.
                def __init__(self):
                    self.ctx=server
                    self.operations=0
                def record(self,value):
                    self.operations+=1
                    if (self.operations-1)%3==0:
                        phase(f'Newton update{(self.operations+2)//3} complete',value)
                    return value
                def weighted_sum(self,values,weights,*,bias=0.):
                    return self.record(arithmetic.weighted_sum(values,weights,bias=bias))
                def multiply(self,left,right):
                    return self.record(arithmetic.multiply(left,right))
            observed=ObservedArithmetic()
            inverse_result=scaled_newton_inverse(observed,encrypted_x,normalization_factor=c,
                initial_estimate=initial_by_slot,updates=13,capture_updates=range(1,14))
            z=inverse_result.final
            assert observed.operations==37
            phase('entire thirteen-update server circuit complete; client validation begins',z)
            expected=x*((-c*initial_by_slot)*initial_by_slot)+2.*initial_by_slot
            expected_level,expected_scale=40,scale
            decoded_updates={}
            for update in range(1,14):
                if update>1:
                    drop=math.prod(q[2*(expected_level-2):2*(expected_level-1)])
                    expected_scale=expected_scale*scale/drop
                    expected_level-=3
                    expected=expected*(2.-c*(x*expected))
                ct=inverse_result.checkpoints[update]
                decoded=np.asarray(client.decrypt_slots(ct))
                error=float(np.max(np.abs(decoded-expected)))
                metadata=(arithmetic._level(ct)==expected_level and ct._ckks_nsd==1
                          and math.isclose(ct._ckks_scale,expected_scale,rel_tol=1e-12))
                report['checks'].append(dict(update=update,passed=bool(metadata and error<1e-6),
                    max_abs_error_against_exact_float_circuit=error,metadata_ok=metadata,level=expected_level,
                    scale=ct._ckks_scale,expected_scale=expected_scale,nsd=ct._ckks_nsd,
                    min_decoded=float(decoded.min()),max_decoded=float(decoded.max()),
                    all_slots_checked=ring['slots'],execution_rings=require_secure_execution(server,ct)))
                if update in (9,12,13):decoded_updates[update]=decoded
            assert expected_level==4
            stage_checks=[]
            for index,contract in enumerate(contracts):
                selected=slice(index*segment,(index+1)*segment)
                decoded=decoded_updates[contract['iterations']][selected]
                reference=4./(128*x[selected])
                error=float(np.max(np.abs(decoded-reference)))
                stage_checks.append(dict(stage=contract['stage'],update=contract['iterations'],
                    passed=bool(error<1e-6),max_abs_error_against_true_inverse=error,
                    segment_start=index*segment,all_segment_slots_checked=segment,
                    min_decoded=float(decoded.min()),max_decoded=float(decoded.max()),
                    ideal_upper=8.,excursion_above_ideal_upper=max(0.,float(decoded.max())-8.)))
            report['declared_stage_outputs']=stage_checks
            report['provider_stats']=dict(provider.stats)
            report['passed']=len(report['checks'])==13 and all(c['passed'] for c in report['checks']) and all(c['passed'] for c in stage_checks)
            phase('client validation finished',z)
    except BaseException as error:
        caught=error;report['passed']=False;report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        cleanup=[];cleanup_errors=[]
        for number in old_handlers:signal.signal(number,signal.SIG_IGN)
        def attempt(label,operation):
            try:return operation()
            except BaseException as error:cleanup.append(f'{label}: {type(error).__name__}: {error}');cleanup_errors.append(error)
        if provider is not None:attempt('public NTT owner close',provider.close)
        if arithmetic is not None:
            attempt('rotation compiler clear',lambda:arithmetic.rotation.clear(executables=True))
            for helper in (arithmetic._plaintext_multiply,arithmetic._rescale):attempt('linear compiler clear',lambda:helper.clear(executables=True))
            attempt('multiplication compiler clear',arithmetic._multiplication_compiler.clear_executables)
        for context in (server,client):
            if context is not None:attempt('context eviction',lambda:ckks_ctx.evict_context_caches(context))
        checkpoints.clear();context=helper=client=server=arithmetic=provider=keys=evaluation=encrypted_x=z=product=correction=ct=inverse_result=observed=None
        attempt('JAX cache clear',jax.clear_caches);attempt('GC',gc.collect)
        for module,name,original in patched:attempt('restore key generation guard',lambda:setattr(module,name,original))
        final=attempt('source verification',source_bytes)
        after={n:hashlib.sha256(v).hexdigest() for n,v in final.items()} if final is not None else {}
        report.update(source_hashes_after=after,source_files_unchanged=before==after,cleanup_errors=cleanup,
            elapsed_s=time.perf_counter()-started,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        report['passed']=bool(report['passed'] and before==after and not cleanup)
        cleanup_cancel=next((e for e in cleanup_errors if not isinstance(e,Exception)),None)
        try:
            try:write(args.output/'report.json',report)
            except BaseException as error:
                primary=caught if caught is not None else cleanup_cancel
                if primary is None:raise
                primary.add_note(f'Final evidence write failed: {error}')
        finally:
            for number,handler in old_handlers.items():signal.signal(number,handler)
        if caught is None and cleanup_cancel is not None:raise cleanup_cancel
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
