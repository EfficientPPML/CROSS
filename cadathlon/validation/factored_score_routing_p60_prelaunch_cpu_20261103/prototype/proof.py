"""Two genuine common-profile factored score routes; no attention, refresh or model claim."""
from pathlib import Path
import gc,hashlib,json,math,os,resource,sys,time
import numpy as np
LAB=Path(__file__).resolve().parents[1];ROOT=LAB/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import proof_support as support
import fixtures
CPU_AFFINITY=[152,153,154,155];DEGREE=131072;SLOTS=65536;DEPTH=49;DNUM=8

def sources():
    paths=[*ROOT.rglob('*.py'),*Path(__file__).parent.glob('*.py')]
    paths.extend(LAB/name for name in ('public_plan.json','profile_audit.json','base_provenance.json','routing_preflight_origin.json'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def weighted_scale(input_scale,canonical_scale,divisor):
    # Match weighted_sum/PTCT/rescale operation order; never force the result to delta.
    output_scale_before_rescale=canonical_scale*divisor
    encode_scale=output_scale_before_rescale/input_scale
    return input_scale*encode_scale/divisor

def stage_scale_plan(q,canonical_scale,input_scale):
    first=weighted_scale(input_scale,canonical_scale,math.prod(q[10:12]))
    second=weighted_scale(first,canonical_scale,math.prod(q[8:10]))
    third=weighted_scale(second,canonical_scale,math.prod(q[6:8]))
    fourth=weighted_scale(third,canonical_scale,math.prod(q[4:6]))
    return [first,second,second,second,third,fourth]

def public_parameters():
    from encrypted_refresh import RefreshProfile
    from encrypted_factored_score_routing import FactoredScoreLayout
    from jaxite_word import he_params
    params=RefreshProfile(DEGREE,DEPTH,60,61,DNUM).parameters();params['degree_layout']=(256,512)
    q,p=params['q_towers'],params['p_towers'];rotations=FactoredScoreLayout(7,15).rotation_indices()
    audit=json.loads((LAB/'profile_audit.json').read_text());fixed=next(x for x in audit['profiles'] if x['name']=='depth49_p60_d8')
    if q!=fixed['q'] or p!=fixed['p'] or params['scaling_factor']!=fixed['canonical_scale']:raise ValueError('fixed actual profile differs')
    if len(q)!=100 or len(p)!=14 or len(rotations)!=54:raise ValueError('actual common profile/key plan differs')
    if not he_params.qp_is_secure(q,p,DEGREE,he_params.SecurityTarget(128,'classical')):raise ValueError('insecure actual Q/P')
    he_params.assert_p_coverage(q,p,DNUM)
    alpha=(len(q)+DNUM-1)//DNUM;parts=(len(q)+alpha-1)//alpha;pair=2*parts*(len(q)+len(p))*DEGREE*4
    scale=params['scaling_factor'];divisor=math.prod(q[12:14]);product_scale=scale*scale/divisor
    if math.isclose(product_scale,scale,rel_tol=1e-12):raise ValueError('genuine product would not test noncanonical alignment')
    plan=dict(degree=DEGREE,slots=SLOTS,depth=DEPTH,scale_bits=60,dnum=DNUM,q=q,p=p,canonical_scale=scale,
        actual_log2_QP=math.log2(math.prod(q+p)),actual_QP_bits=math.prod(q+p).bit_length(),security_ceiling=3523,
        actual_128_classical_secure=True,P_coverage=True,rotations=list(rotations),rotation_count=len(rotations),
        key_pair_bytes=pair,raw_rotation_bytes=len(rotations)*pair,raw_rotation_GiB=len(rotations)*pair/2**30,
        relinearization_bytes=pair,public_key_bytes=2*len(q)*DEGREE*4,headers_allowance_bytes=len(rotations)*8192,
        extra_free_margin_bytes=4*1024**3,required_free_bytes=len(rotations)*(pair+8192)+4*1024**3,
        noncanonical_input_level=5,noncanonical_product_operand_level=6,noncanonical_divisor=divisor,
        noncanonical_actual_scale=product_scale,relative_noncanonical_scale_gap=product_scale/scale-1,
        expected_route_rotations=108,expected_route_public_multiplications=112,expected_route_rescales=8,
        separate_input_ciphertext_products=1,separate_input_rescales=1,
        expected_checkpoints=12,expected_stage_levels=list(fixtures.LEVELS),key_tile=7,offset=15,
        exact_stage_scales={name:stage_scale_plan(q,scale,value) for name,value in (('canonical',scale),('noncanonical',product_scale))})
    return params,plan

def run(report,report_path):
    import jax
    from jaxite_word import CKKSContext,key_gen,he_params
    from encrypted_refresh import ClientRotationKeyStore
    from encrypted_ntt import enable_scoped_ntt_subsets
    from encrypted_packed_qwen import CompiledArithmetic
    from encrypted_factored_score_routing import FactoredScoreLayout,FactoredScoreRouting
    from support import require_secure_execution
    from check_transformer_suite import write_json
    jax.config.update('jax_enable_x64',True)
    if sorted(os.sched_getaffinity(0))!=CPU_AFFINITY:raise ValueError('fixed CPU152-155 affinity required')
    if jax.default_backend()!='cpu' or any(d.platform!='cpu' for d in jax.devices()):raise ValueError('CPU-only component')
    directory=support.validate_key_directory(os.environ['CROSS_PUBLIC_KEY_DIRECTORY'])
    params,plan=public_parameters()
    if plan!=json.loads((LAB/'public_plan.json').read_text())['plan']:raise ValueError('approved preflight plan changed')
    free=__import__('shutil').disk_usage(directory).free
    if free<plan['required_free_bytes']:raise ValueError('insufficient tmpfs for public rotation keys plus4GiB')
    report.update(plan=plan,actual_security_and_P_coverage_checked_before_keygen=True,tmpfs_free_bytes_before_keygen=free)
    q,p=params['q_towers'],params['p_towers'];scale=params['scaling_factor'];rotations=plan['rotations'];began=time.perf_counter()
    client=server=ops=provider=key_store=router=None;restorations=[];phase='client-setup';originals={};inputs={};retained=[]
    def progress(label):
        print(f'[{time.perf_counter()-began:.2f}s] {label}',file=sys.stderr,flush=True)
    try:
        progress('client public key generation started')
        keys=key_gen.gen_pke_pair(q,[],DEGREE)
        evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=DNUM,noise_std=he_params.DEFAULT_SIGMA,noise_scale=1)
        evaluation=[jax.numpy.asarray(evaluation[name],dtype='uint32').transpose(0,2,1) for name in ('a','b')]
        key_store=ClientRotationKeyStore.generate(keys['secret_key'],q,p,degree=DEGREE,dnum=DNUM,rotations=rotations,directory=directory)
        client=CKKSContext(dict(params,**keys));server=CKKSContext(dict(params,evaluation_key=evaluation))
        support.install_generator_guards(sys.modules,restorations)
        support.install_codec_guards(CKKSContext,lambda:server,lambda:phase,restorations)
        phase='server';server.program_initialization(rotations,dnum=DNUM,r=256,c=512,batch=1,pregenerated_rotation_keys=key_store.keys)
        del keys,evaluation
        if any(v is not None for v in (server.secret_key,server._param_cache.secret_key,server.parameters.get('secret_key'),server.public_key)):
            raise AssertionError('keyless separate public server required')
        provider=enable_scoped_ntt_subsets(server);ops=CompiledArithmetic(server,low_memory=True,preserve_dispatch_cache=True)
        phase='client-encoding';values=fixtures.values()
        originals['x']=client.encrypt_slots(values);originals['one']=client.encrypt_slots(np.ones(SLOTS))
        for ct in originals.values():ct._ckks_nsd=1 # Fresh codec boundary with explicit noise_scale_degree=1 parameters.
        del ct
        original_fingerprints={name:support.ciphertext_fingerprint(value) for name,value in originals.items()}
        report['client_input_security']={name:require_secure_execution(client,value) for name,value in originals.items()}
        report['input_pattern_sha256']=hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
        phase='server';report['keyless_server']=True
        inputs['canonical']=ops._lower(originals['x'],5)
        left=ops._lower(originals['x'],6);right=ops._lower(originals['one'],6)
        inputs['noncanonical']=ops.multiply(left,right);jax.block_until_ready(inputs['noncanonical'].polynomial);del left,right
        if ops._level(inputs['noncanonical'])!=5 or inputs['noncanonical']._ckks_scale!=plan['noncanonical_actual_scale']:
            raise AssertionError('genuine noncanonical input metadata differs from actual product recurrence')
        if inputs['canonical']._ckks_scale!=scale or type(inputs['noncanonical']._ckks_nsd)is not int or inputs['noncanonical']._ckks_nsd!=1:
            raise AssertionError('input metadata differs')
        input_fingerprints={name:support.ciphertext_fingerprint(value) for name,value in inputs.items()}
        report['route_input_security']={name:require_secure_execution(server,value) for name,value in inputs.items()}
        report['route_input_scales']={name:value._ckks_scale for name,value in inputs.items()}
        report['phases']=[];total_counts={}
        for case,gain in (('canonical',1.),('noncanonical',.125)):
            def capture(label,value,counts,case=case):
                jax.block_until_ready(value.polynomial)
                position=sum(entry[0]==case for entry in retained)
                if position>=6 or label!=fixtures.LABELS[position] or ops._level(value)!=fixtures.LEVELS[position]:
                    raise AssertionError('actual stage order/level differs from fixed route')
                if type(value._ckks_nsd)is not int or value._ckks_nsd!=1 or value._ckks_scale!=plan['exact_stage_scales'][case][position]:
                    raise AssertionError('route stage failed actual scale alignment')
                retained.append((case,label,value,support.ciphertext_fingerprint(value)))
                report['phases'].append(dict(case=case,label=label,level=ops._level(value),scale=value._ckks_scale,nsd=value._ckks_nsd,
                    counts=dict(counts),elapsed_s=time.perf_counter()-began,execution_ring=require_secure_execution(server,value)))
                write_json(report_path,report);progress(case+' '+label)
            router=FactoredScoreRouting(ops,FactoredScoreLayout(7,15),progress=capture)
            forward=router.forward(inputs[case],gain=gain);backward=router.inverse(forward);del forward,backward
            for name,count in router.counts.items():total_counts[name]=total_counts.get(name,0)+count
        if len(retained)!=12 or total_counts.get('rotations')!=108 or total_counts.get('public_multiplications')!=112 or total_counts.get('actual_composite_rescales')!=8:
            raise AssertionError('actual operation inventory differs')
        report['route_counts']=total_counts;report['separate_input_counts']={'ciphertext_products':1,'rescales':1}
        report['input_fingerprints_before']=dict(originals=original_fingerprints,routes=input_fingerprints)
        report['input_fingerprints_after']=dict(originals={n:support.ciphertext_fingerprint(v) for n,v in originals.items()},routes={n:support.ciphertext_fingerprint(v) for n,v in inputs.items()})
        report['inputs_unchanged']=report['input_fingerprints_before']==report['input_fingerprints_after']
        report['retained_outputs_unchanged']=all(support.ciphertext_fingerprint(v)==saved for _,_,v,saved in retained)
        report['output_security_before_decode']={case+':'+label:require_secure_execution(server,v) for case,label,v,_ in retained}
        if not report['inputs_unchanged'] or not report['retained_outputs_unchanged']:raise AssertionError('an original input or retained output changed')
        report.update(all_twelve_outputs_before_client_decode=True,codecs_and_both_keygen_aliases_guarded=True,no_refresh=True)
        phase='client-validation';references={case:fixtures.oracle(values,gain) for case,gain in (('canonical',1.),('noncanonical',.125))}
        report['checks']={}
        for case,label,value,_ in retained:
            decoded=np.asarray(client.decrypt_slots(value));report['checks'][case+':'+label]=support.metrics(decoded,references[case][label])
        del value,decoded
        report['numerical_passed']=len(report['checks'])==12 and all(v['passed'] for v in report['checks'].values())
        report['component_s']=time.perf_counter()-began
    finally:
        primary=sys.exc_info()[1];report['cleanup_completed']=False
        if ops is not None:
            restorations.append(ops._clear_working_caches)
            restorations.append(lambda:ops.rotation.clear(executables=True))
            for helper in (ops._plaintext_multiply,ops._rescale):
                restorations.append(lambda helper=helper:helper.clear(executables=True))
            restorations.append(ops._multiplication_compiler.clear_executables)
        originals.clear();inputs.clear();retained.clear();router=None
        report['cleanup_completed']=support.release_all([client,server],provider,key_store,primary,restorations)
        client=server=ops=provider=key_store=None
        report['peak_host_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'cadathlon/validation'))
    import check_transformer_suite as collector
    report=dict(passed=False,numerical_passed=False,scope=__doc__,fixed_absolute_error_gate=1e-6,
        cpu_affinity=sorted(os.sched_getaffinity(0)),same_process_separate_keyless_objects=True,full_model_or_refresh_claim=False)
    caught=None
    try:
        collector.write_json(args.report,report);report['source_hashes_start']=sources();collector.write_json(args.report,report)
        with collector.termination_signals():run(report,args.report)
    except BaseException as error:caught=error;report['error']=type(error).__name__+': '+str(error);report['notes']=list(getattr(error,'__notes__',()))
    finally:
        try:report['source_hashes_end']=sources();report['source_files_unchanged']=bool(report.get('source_hashes_start')) and report['source_hashes_start']==report['source_hashes_end']
        except BaseException as error:
            report['source_files_unchanged']=False
            if caught is None:caught=error
            else:caught.add_note('Source verification also failed: '+repr(error))
        report['passed']=caught is None and report.get('numerical_passed')is True and report.get('cleanup_completed')is True and report.get('source_files_unchanged')is True
        try:collector.write_json(args.report,report)
        except BaseException as error:
            if caught is None:raise
            caught.add_note('Final report also failed: '+repr(error))
    print(json.dumps(report,allow_nan=False),flush=True)
    if caught is not None and not isinstance(caught,Exception):raise caught
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
