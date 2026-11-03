"""Secure CPU integer affine and two-update C32 Newton feasibility, no full inverse/refresh."""
import argparse,copy,gc,hashlib,importlib.metadata,io,json,math,os,resource,signal,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('Explicit CPU backend required')
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(HERE)]
import jax,numpy as np
from jaxite_word import CKKSContext,ckks_ctx,he_params,key_gen,util
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_packed_qwen import CompiledArithmetic
from integer_affine import integer_affine,affine_at_scale,backward_scale_plan
from support import require_secure_execution


def sources():
    paths={Path(__file__),HERE/'integer_affine.py',HERE/'test_public_plan.py',HERE/'base_manifest.json'}
    for name in ('jaxite_word','cadathlon/common'):paths.update((ROOT/name).glob('*.py'))
    return {str(p.relative_to(HERE)):p.read_bytes() for p in sorted(paths)}


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    initial=sources();before={n:hashlib.sha256(b).hexdigest() for n,b in initial.items()}
    with tarfile.open(args.output/'sources.tar.gz','w:gz') as tar:
        for name,data in initial.items():
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    report=dict(scope=__doc__,passed=False,fixed_absolute_error_gate=1e-6,
        started_utc=datetime.now(timezone.utc).isoformat(),source_hashes_before=before,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','numpy')},
        checks=[],phases=[],limitations=['Two updates compared to the exact float circuit, not a converged inverse.',
            'No full thirteen-update inverse, normalization chain, softmax, model or refresh.',
            'Client/server are separate objects in a dedicated process, not an OS sandbox.',
            'Sampled inputs do not establish an all-input CKKS noise bound.'])
    write(args.output/'report.json',report)
    started=time.perf_counter();caught=None;patched=[];handlers={}
    client=server=arithmetic=keys=evaluation=encrypted_x=signed_ct=z1=product=correction=z2=ct=None
    affine_outputs=[];checks=[];original_decode=None;client_decoding=False
    def phase(name):
        row=dict(name=name,elapsed_s=time.perf_counter()-started)
        report['phases'].append(row);write(args.output/'report.json',report);print(json.dumps(row),flush=True)
    def interrupted(signum,frame):
        for number in handlers:signal.signal(number,signal.SIG_IGN)
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):handlers[number]=signal.signal(number,interrupted)
    try:
        N,depth,dnum=16384,3,4;slots=N//2
        q=composite_prime_gen(2,8,61,56,2*N,31);p=util.generate_p_towers(q,dnum,N,aux_bits=30)
        target=he_params.SecurityTarget(128,'classical')
        if not he_params.qp_is_secure(q,p,N,target):raise ValueError('Actual Q/P ring is insecure')
        he_params.assert_p_coverage(q,p,dnum)
        scale=float(q[-2])*q[-1]
        report['ring']=dict(degree=N,depth=depth,dnum=dnum,q_towers=q,p_towers=p,scale=scale,slots=slots,
            actual_log2_QP=sum(map(math.log2,q+p)),QP_bit_length=math.prod(q+p).bit_length(),
            security_bit_length_ceiling=he_params.security_ceiling(N,target),security_bits=128)
        assert jax.default_backend()=='cpu' and all(d.platform=='cpu' for d in jax.devices())
        plan=backward_scale_plan(q,input_scale=scale,first_output_level=2,updates=2,final_scale=scale)
        report['scale_plan']=plan
        lower,upper=.5/128,.041910546372478145
        points=np.unique(np.r_[np.linspace(lower,upper,129),np.geomspace(lower,upper,129),
            (lower+upper)/2,np.nextafter(lower,upper),np.nextafter(upper,lower)])
        x=points[np.arange(slots)%len(points)]
        initial_z=2./(32.*(lower+upper));weight=-32.*initial_z*initial_z;bias=2.*initial_z
        first=x*weight+bias;expected_product=x*first;expected_correction=2.-32.*expected_product
        second=first*expected_correction
        signed=np.resize(np.r_[np.linspace(-.75,.75,65),-0.,2.**-20,-2.**-20],slots)
        report['public_inputs']=dict(K=128,s=4,C=32,interval=[lower,upper],initial_estimate=initial_z,
            distinct_denominators=len(points),signed_affine_points=68,all_slots=slots,
            first_affine_weight=weight,first_affine_bias=bias)
        keys=key_gen.gen_pke_pair(q,[],N);evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
        params=dict(degree=N,num_slots=slots,scaling_factor=scale,output_scale=scale,q_towers=q,p_towers=p,
            composite_degree=2,degree_layout=(128,128),p=56,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
        client=CKKSContext(dict(params,**keys))
        server=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(evaluation[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
        def forbidden(*args,**kwargs):raise AssertionError('Server secret/keygen/encryption/decryption forbidden')
        for module in {key_gen,sys.modules.get('key_gen')}-{None}:
            for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):
                patched.append((module,name,getattr(module,name)));setattr(module,name,forbidden)
        report['guarded_keygen_modules']=sorted({m.__name__ for m,_,_ in patched})
        assert report['guarded_keygen_modules']==['jaxite_word.key_gen','key_gen']
        server.program_initialization([],dnum=dnum,r=128,c=128,batch=1,pregenerated_rotation_keys={})
        client._polynomial_ntt_ctx=server._param_cache.ntt_q_max
        for name in ('encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch'):setattr(server,name,forbidden)
        assert server.secret_key is None and server._param_cache.secret_key is None and server.parameters.get('secret_key') is None
        arithmetic=CompiledArithmetic(server,low_memory=True,preserve_dispatch_cache=True)
        encrypted_x=client.encrypt_slots(x);signed_ct=client.encrypt_slots(signed)
        encrypted_x._ckks_nsd=signed_ct._ckks_nsd=1
        original_decode=client.decrypt_slots
        def guarded_decode(*args,**kwargs):
            if not client_decoding:raise AssertionError('Client decode attempted before fixed server circuit ended')
            return original_decode(*args,**kwargs)
        client.decrypt_slots=guarded_decode
        source_payloads=[np.asarray(v.polynomial).copy() for v in (encrypted_x,signed_ct)]
        for value in (encrypted_x,signed_ct):
            value._public_absolute_bound=99.;value._public_bound_provenance='sentinel'
            value._public_row_l2_bound=123.;value._public_head_l2_bounds=(1.,2.)
            value._public_normalization_affine=('not valid for outputs',)
        source_metadata=[dict(vars(v)) for v in (encrypted_x,signed_ct)]
        keys=evaluation=None;phase('actual secure ring and keyless server ready')
        negative_checks=[]
        for bad in (True,1.5,2**64):
            try:integer_affine(arithmetic,signed_ct,bad)
            except ValueError:negative_checks.append('multiplier '+repr(bad))
            else:raise AssertionError('invalid coefficient accepted')
        for bad in (True,float('inf'),float('nan')):
            try:integer_affine(arithmetic,signed_ct,1,bias=bad)
            except ValueError:negative_checks.append('bias '+repr(bad))
            else:raise AssertionError('invalid bias accepted')
        for attr,bad in (('_ckks_nsd',2),('_ckks_scale',0.)):
            bad_ct=copy.copy(signed_ct);setattr(bad_ct,attr,bad)
            try:integer_affine(arithmetic,bad_ct,32)
            except ValueError:negative_checks.append(attr)
            else:raise AssertionError('invalid metadata accepted')
        had_parameter='secret_key' in server.parameters
        original_parameter=server.parameters.get('secret_key')
        server.parameters['secret_key']='sentinel'
        try:
            try:integer_affine(arithmetic,signed_ct,32)
            except ValueError:negative_checks.append('secret-bearing context')
            else:raise AssertionError('secret-bearing context accepted')
        finally:
            if had_parameter:server.parameters['secret_key']=original_parameter
            else:server.parameters.pop('secret_key')
        report['negative_contract_checks']=negative_checks
        # Independent public residue oracle, not an alternate server kernel.
        raw=source_payloads[1].astype(np.uint64);moduli=np.asarray(q,dtype=np.uint64)
        for coefficient,bias_value in ((-32,2.),(-1,0.),(0,2.),(1,0.),(7,-.125),(32,2.)):
            output=integer_affine(arithmetic,signed_ct,coefficient,bias=bias_value)
            expected_raw=(raw*np.asarray([coefficient%m for m in q],dtype=np.uint64))%moduli
            expected_raw[:,0]=(expected_raw[:,0]+np.asarray([ckks_ctx.nearest_int(bias_value*scale)%m for m in q],dtype=np.uint64))%moduli
            exact=np.array_equal(np.asarray(output.polynomial),expected_raw.astype(np.uint32))
            cleared=not any(n.startswith('_public_') for n in vars(output))
            assert exact and cleared and output._ckks_scale==scale and output._ckks_nsd==1 and arithmetic._level(output)==3
            affine_outputs.append((coefficient,bias_value,output,coefficient*signed+bias_value))
            report['checks'].append(dict(name=f'integer affine residues {coefficient},{bias_value}',passed=bool(exact and cleared),exact_residues=exact,semantic_attributes_removed=cleared))
        phase('integer affine residue and negative-contract checks passed')
        z1=affine_at_scale(arithmetic,encrypted_x,weight,bias,output_scale=plan['initial_target_scale'])
        product=arithmetic.multiply(encrypted_x,z1)
        correction=integer_affine(arithmetic,product,-32,bias=2.)
        z2=arithmetic.multiply(z1,correction)
        z2.polynomial.block_until_ready()
        checks=[('update1',z1,first,2,plan['rows'][0]['scale']),
            ('product',product,expected_product,1,plan['rows'][1]['product_scale']),
            ('integer correction',correction,expected_correction,1,plan['rows'][1]['product_scale']),
            ('update2',z2,second,0,plan['rows'][1]['scale'])]
        # The existing unbounded multiplication clones its left wrapper. The
        # injected product sentinels must be removed by integer_affine; this
        # component does not change the existing multiplication contract.
        report['generic_product_inherited_semantic_attributes']=sorted(n for n in vars(product) if n.startswith('_public_'))
        assert all(not any(n.startswith('_public_') for n in vars(v)) for v in (z1,correction,z2))
        phase('entire fixed server circuit complete; client decoding now enabled')
        client_decoding=True
        for coefficient,bias_value,value,expected in affine_outputs:
            checks.append((f'decoded integer affine {coefficient},{bias_value}',value,expected,3,scale))
        for name,ct,expected,level,expected_scale in checks:
            decoded=np.asarray(client.decrypt_slots(ct));error=float(np.max(np.abs(decoded-expected)))
            metadata=arithmetic._level(ct)==level and ct._ckks_nsd==1 and ct._ckks_scale==expected_scale
            report['checks'].append(dict(name=name,passed=bool(metadata and error<1e-6),max_abs_error=error,
                all_slots_checked=slots,level=level,scale=ct._ckks_scale,expected_scale=expected_scale,
                exact_metadata_recurrence=metadata,execution_rings=require_secure_execution(server,ct)))
        immutable=True
        for value,payload,metadata in zip((encrypted_x,signed_ct),source_payloads,source_metadata):
            immutable &= np.array_equal(np.asarray(value.polynomial),payload)
            immutable &= set(vars(value))==set(metadata)
            immutable &= all(vars(value)[k] is v or (isinstance(v,(float,int,str,tuple)) and vars(value)[k]==v) for k,v in metadata.items())
        report['input_immutability']=bool(immutable)
        report['true_inverse_error_diagnostic_only']=float(np.max(np.abs(np.asarray(client.decrypt_slots(z2))-4/(128*x))))
        report['passed']=bool(immutable and all(c['passed'] for c in report['checks']))
        phase('all-slot client validation complete')
    except BaseException as error:
        caught=error;report['passed']=False;report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        cleanup=[];cleanup_exceptions=[]
        for number in handlers:signal.signal(number,signal.SIG_IGN)
        def attempt(label,operation):
            try:return operation()
            except BaseException as error:
                cleanup.append(f'{label}: {type(error).__name__}: {error}');cleanup_exceptions.append(error)
        if arithmetic is not None:attempt('clear arithmetic',arithmetic._clear_working_caches)
        for context in (server,client):
            if context is not None:attempt('evict context',lambda context=context:ckks_ctx.evict_context_caches(context))
        affine_outputs.clear();checks.clear();context=client=server=arithmetic=keys=evaluation=None
        encrypted_x=signed_ct=z1=product=correction=z2=ct=original_decode=None
        source_metadata=source_payloads=[];value=output=bad_ct=metadata=None
        attempt('clear JAX caches',jax.clear_caches);attempt('collect GC',gc.collect)
        for module,name,original in patched:attempt('restore keygen '+name,lambda module=module,name=name,original=original:setattr(module,name,original))
        after_raw=attempt('final source read',sources)
        after={n:hashlib.sha256(b).hexdigest() for n,b in (after_raw or {}).items()}
        report.update(source_hashes_after=after,source_files_unchanged=before==after,cleanup_errors=cleanup,
            elapsed_s=time.perf_counter()-started,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        report['passed']=bool(report['passed'] and before==after and not cleanup)
        primary=caught or next((e for e in cleanup_exceptions if not isinstance(e,Exception)),None)
        try:write(args.output/'report.json',report)
        except BaseException as error:
            if primary is None:raise
            primary.add_note(f'Final report publication failed: {error}')
        finally:
            for number,handler in handlers.items():signal.signal(number,handler)
        if caught is None and primary is not None:raise primary
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
