"""Public exact-ring/level/resource integration plan; no keygen, contexts or HE."""
from pathlib import Path
from fractions import Fraction
import datetime, hashlib, importlib.util, io, json, math, os, shutil, sys, tarfile, time
LAB=Path(__file__).resolve().parent
ROOT=LAB/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(LAB/'inputs')]
from public_schedule import PublicSchedule
from encrypted_integer_affine import backward_scale_plan
from encrypted_refresh import RefreshProfile,refresh_plan
from jaxite_word import CKKSContext,he_params,key_gen
from jaxite_word.bootstrapping import Bootstrap
from jaxite_word.bsgs import compute_bsgs_params
import jax

FORBIDDEN=[]
def forbidden(*args,**kwargs):
    FORBIDDEN.append('context/keygen/backend construction attempted')
    raise AssertionError(FORBIDDEN[-1])
CKKSContext.__init__=Bootstrap.__init__=forbidden
for module in {key_gen,sys.modules.get('key_gen')} - {None}:
    for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):
        if hasattr(module,name):setattr(module,name,forbidden)
jax.devices=jax.local_devices=jax.default_backend=forbidden

N=131072;SLOTS=N//2;RESERVE=2
PROFILE_SPECS=[('current49_p56_d4',49,56,4),('depth52_p56_d6',52,56,6),
               ('depth52_p60_d12',52,60,12),('depth53_p60_d14',53,60,14)]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

def dense_linear_keys(rows,cols):
    active={d%SLOTS for d in range(-(rows-1),cols)}
    n1,n2=compute_bsgs_params(SLOTS,len(active),2.)
    babies={d%n1 for d in active}-{0};giants={(d//n1)*n1 for d in active}-{0}
    return babies|giants,dict(shape=[rows,cols],active_diagonal_upper=len(active),n1=n1,n2=n2,
        baby_key_count=len(babies),giant_key_count=len(giants),total_distinct_keys=len(babies|giants))

def key_inventory():
    square,square_info=dense_linear_keys(768,768)
    up,up_info=dense_linear_keys(3072,768);down,down_info=dense_linear_keys(768,3072)
    linears=square|up|down
    heads={1<<bit for bit in range(6)}|{(-(1<<bit))%SLOTS for bit in range(6)}
    token_rows={4096*offset for offset in range(1,16)}
    routing={sign*63*(1<<bit)%SLOTS for sign in (-1,1) for bit in range(4)}
    routing|={sign*16*(1<<bit)%SLOTS for sign in (-1,1) for bit in range(4)}
    routing|={sign*256*tile%SLOTS for sign in (-1,1) for tile in range(1,8)}
    reductions={sign*16*(1<<bit)%SLOTS for sign in (-1,1) for bit in range(7)}
    norm={1<<bit for bit in range(10)}|{(-(1<<bit))%SLOTS for bit in range(12)}
    groups={'dense_QKV_attention_output_FFN':linears,'head_sum_and_broadcast':heads,
        'cyclic_KV_row_rotations':token_rows,'all128_factored_routes_and_inverse':routing,
        'score_reduce_broadcast':reductions,'row_LayerNorm':norm}
    union=set().union(*groups.values())
    return dict(groups={name:sorted(values) for name,values in groups.items()},
        counts={name:len(values) for name,values in groups.items()},graph_rotation_union=sorted(union),
        graph_rotation_count=len(union),dense_matrix_upper_contracts=[square_info,up_info,down_info],
        scope='Exact direct keys for listed public dense-diagonal upper contract. Pinned weights may omit zero diagonals; actual preprocess inventory must still be verified. No composed signed-key substitution assumed.')

def profile(name,depth,pbits,dnum,keys):
    try:params=RefreshProfile(N,depth,pbits,61,dnum).parameters()
    except Exception as exc:return {'name':name,'accepted':False,'error':repr(exc)}
    q,p=params['q_towers'],params['p_towers'];alpha=math.ceil(len(q)/dnum);parts=math.ceil(len(q)/alpha)
    coverage=[math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha)]
    refresh=refresh_plan(N,SLOTS,(4,4),max_level=depth)
    boot=set(refresh['rotations']);graph=set(keys['graph_rotation_union']);union=boot|graph
    pair=2*parts*(len(q)+len(p))*N*4
    raw=len(union)*pair
    scale=params['scaling_factor']
    levels={str(L):2*(2*(L+1))*N*4 for L in range(depth+1)}
    return dict(name=name,accepted=True,degree=N,slots=SLOTS,layout=[256,512],precision=32,
        depth=depth,scale_bits=pbits,first_bits=61,dnum=dnum,actual_partitions=parts,alpha=alpha,
        q=q,p=p,Q_count=len(q),P_count=len(p),log2_QP=math.log2(math.prod(q+p)),
        QP_bit_length=math.prod(q+p).bit_length(),security_ceiling=3523,
        qp_is_secure=he_params.qp_is_secure(q,p,N,he_params.SecurityTarget(128,'classical')),
        P_covers_each_partition=math.prod(p)>=max(coverage),P_bits=math.prod(p).bit_length(),
        maximum_Q_partition_bits=max(coverage).bit_length(),canonical_scale=scale,
        base_Q_over_scale=float(math.prod(q[:2]))/scale,
        maximum_Q_word_bits=max(q).bit_length(),maximum_P_word_bits=max(p).bit_length(),
        refresh={'input_minimum':2,'output_level':refresh['output_level'],'boot_depth':refresh['boot_depth'],
            'level_budget':[4,4],'period':SLOTS,'aliases':1,'projection_rotations':0,
            'meta_precision':10,'genuine_bootstraps_per_refresh':2,'ApproxMod_paths_per_refresh':4,
            'bootstrap_rotation_count':len(boot),'bootstrap_rotations':sorted(boot)},
        resources={'rotation_pair_bytes':pair,'relinearization_pair_bytes':pair,
            'bootstrap_only_key_pairs_bytes':len(boot)*pair,'graph_plus_bootstrap_count':len(union),
            'graph_plus_bootstrap_rotations':sorted(union),'raw_rotation_store_bytes':raw,
            'raw_rotation_store_GiB':raw/2**30,'rotation_file_header_allowance_bytes':len(union)*8192,
            'store_preflight_with_64MiB_bytes':raw+len(union)*8192+64*2**20,
            'public_key_bytes':2*len(q)*N*4,'client_secret_bytes':len(q)*N*4,
            'top_ciphertext_bytes':levels[str(depth)],'ciphertext_bytes_by_level':levels,
            'sixteen_top_row_ciphertexts_bytes':16*levels[str(depth)],
            'thirteen_outputs_at_levels_from_refresh_bytes':sum(levels[str(refresh['output_level']-1-2*i)] for i in range(13)) if refresh['output_level']>=25 else None,
            'scope':'Raw uint32 payload only. Not peak host RSS or TPU HBM; excludes controls/NTT tables, keygen temporaries, JIT/compiler, simultaneous BSGS transforms, frontend copies, prior retained requests and allocator overhead.'})

def attempt_round(profile,L,contract,refresh_probability,refresh_denominator):
    R=profile['refresh']['output_level'];n=contract['iterations'];q=profile['q'];delta=profile['canonical_scale']
    events=[];refreshes=0
    if refresh_probability:
        if L<2:return None
        events.append(dict(op='refresh_probability',input_level=L,output_level=R,ideal_bound=1.,semantic='All refreshed slots assumed to remain positive/zero-padded within error margin; not numerically established.'))
        L=R;refreshes+=1
    if L<3:return None
    masked=L-1;squared=L-2;denominator=L-3
    events.extend([dict(op='input_mask',input_level=L,output_level=masked),
                   dict(op='square_saved_numerator',input_level=masked,output_level=squared),
                   dict(op='anchor_sum_dummy_broadcast',input_level=squared,output_level=denominator,
                        public_dummy=contract['inactive_dummy'],rotations=14)])
    squared_scale=delta*delta/math.prod(q[2*masked:2*masked+2])
    if refresh_denominator:
        if denominator<2:return None
        events.append(dict(op='refresh_denominator',input_level=denominator,output_level=R,ideal_bound=contract['b'],
                           semantic='Perturbs sum-of-same-squares identity; conditional ratio/mass certificate needs explicit numerical margin.'))
        denominator=R;refreshes+=1
    inverse=denominator-(2*n-1)
    product=min(squared,inverse)-1
    if inverse<0 or product<RESERVE:return None
    # Plan final inverse scale to make the actual numerator product canonical,
    # using its retained square scale and the actual next dropped prime pair.
    product_input_level=product+1
    target_inverse=delta*math.prod(q[2*product_input_level:2*product_input_level+2])/squared_scale
    scale_plan=backward_scale_plan(q,input_scale=delta,first_output_level=denominator-1,
                                   updates=n,final_scale=target_inverse)
    actual_inverse=scale_plan['rows'][-1]['scale']
    actual_product=squared_scale*actual_inverse/math.prod(q[2*product_input_level:2*product_input_level+2])
    relative=actual_product/delta-1
    if abs(relative)>1e-12:raise AssertionError(('canonical product tolerance failed',profile['name'],L,n,relative))
    a,b=contract['a'],contract['b'];seed=2./(32*(a+b));weight=(-32*seed)*seed
    encoded_min=abs(weight)*scale_plan['initial_plaintext_scale']
    assert math.isfinite(encoded_min) and encoded_min>=.5
    events.extend([dict(op='static_C32_inverse',input_level=denominator,output_level=inverse,
                        updates=n,levels_consumed=2*n-1,ideal_z_upper=1/(32*a)),
                   dict(op='correlated_numerator_times_z',input_levels=[squared,inverse],output_level=product,ideal_bound=1/32),
                   dict(op='integer32_restore',input_level=product,output_level=product,addition_chain_doublings=5,ideal_bound=1.)])
    return dict(input_level=events[0]['input_level'],output_level=product,refresh_count=refreshes,
                events=events,scale_plan=scale_plan,saved_square_scale=squared_scale,
                inverse_target_scale=target_inverse,actual_product_scale=actual_product,
                product_relative_canonical_difference=relative,
                minimum_nonzero_initial_coefficient_scaled=encoded_min,
                mode=[refresh_probability,refresh_denominator])

def solve(profile,K,schedule,*,allow_denominator_refresh=True):
    # Fresh Q/K projection1 + CTCT1 + forward routing9 (gain1/8 folded
    # into its existing source filter) =11. No extra optional key mask here.
    score_level=profile['depth']-11
    # Score mean2, masked centered t=(scores-mean)/16 PT1, degree8 balanced
    # powers3 + final coefficients/mask PT1, explicit masked exp-mean2,
    # C1 six-update inverse11, base*inverse1, public1/K scale PT1 =22.
    initial=score_level-22
    initial_events=[('QK_score_map',profile['depth'],score_level),('score_mean',score_level,score_level-2),
        ('masked_center_and_t',score_level-2,score_level-3),('degree8_powers_and_masked_coefficients',score_level-3,score_level-7),
        ('masked_mean_exp_dummy',score_level-7,score_level-9),('C1_initial_inverse_6_updates',score_level-9,score_level-20),
        ('base_times_inverse',score_level-20,score_level-21),('public_1_over_K',score_level-21,initial)]
    assert initial>=2
    states={initial:(0,[])}
    unsupported=None
    for j in range(5):
        contract=schedule.stage(K,j);next_states={}
        for L,(count,history) in states.items():
            for rp in (False,True):
                for rd in ((False,True) if allow_denominator_refresh else (False,)):
                    attempt=attempt_round(profile,L,contract,rp,rd)
                    if attempt is None:continue
                    out=attempt['output_level'];score=count+attempt['refresh_count']
                    prior=next_states.get(out)
                    if prior is None or score<prior[0]:next_states[out]=(score,history+[attempt])
        if not next_states:
            unsupported=dict(stage=j,iterations=contract['iterations'],reason='No allowed branch leaves product at level>=2; input/output minimum of bounded refresh is enforced.');break
        states=next_states
    if unsupported:return dict(K=K,feasible=False,initial_probability_level=initial,initial_events=initial_events,
        iterations=[schedule.stage(K,j)['iterations'] for j in range(5)],failure=unsupported)
    # Pareto choice: minimize refresh count, then maximize final level. No
    # claim of optimality over alternate circuits or mid-inverse refresh.
    L,(count,history)=min(states.items(),key=lambda item:(item[1][0],-item[0]))
    R=profile['refresh']['output_level']
    continuation=[];additional=0
    if L<24:
        assert L>=2;continuation.append(dict(op='refresh_final_probability',input_level=L,output_level=R,bound=1.));L=R;additional+=1
    continuation.append(dict(op='inverse_score_map',input_level=L,output_level=L-9));L-=9
    continuation.append(dict(op='probability_times_saved_V_and_sum',input_level=L,output_level=L-1));L-=1
    continuation.append(dict(op='attention_output_linear_and_residual_add',input_level=L,output_level=L-1));L-=1
    assert L>=12
    continuation.append(dict(op='attention_LayerNorm',input_level=L,output_level=L-11));L-=11
    if L<18:
        assert L>=2;continuation.append(dict(op='refresh_attention_norm',input_level=L,output_level=R,
            requested_bound=64.,reason='Observed attention LayerNorm magnitude49.37 exceeds16;64 is not a proven all-input/model contract.'));L=R;additional+=1
    continuation.append(dict(op='FFN_768_3072_768_and_residual_add',input_level=L,output_level=L-16));L-=16
    # RowLayerNorm helper itself allows output1, but inter-block bounded
    # refresh requires2: reserve that additional level before entering LN.
    if L<13:
        assert L>=2;continuation.append(dict(op='refresh_FFN_residual',input_level=L,output_level=R,
            requested_bound='unproved; requires public residual bound'));L=R;additional+=1
    assert L>=13
    continuation.append(dict(op='MLP_LayerNorm',input_level=L,output_level=L-11));L-=11
    assert L>=2
    return dict(K=K,feasible=True,iterations=[schedule.stage(K,j)['iterations'] for j in range(5)],
        initial_probability_level=initial,initial_events=initial_events,normalized_rounds=history,
        normalization_refreshes=count,normalization_output_level=history[-1]['output_level'],
        continuation=continuation,continuation_refreshes=additional,
        total_refreshes_per_query_tile_first_block=count+additional,block_output_level=L,
        next_block_refresh_eligible=L>=2)

def initial_C1_certificate():
    a,b=Fraction(1,2),Fraction(2)
    initial=2./(float(a)+float(b))
    w=Fraction.from_float((-1*initial)*initial);bias=Fraction.from_float(2.*initial)
    residual=lambda x:1-x*(w*x+bias)
    candidates=[a,b];vertex=-bias/(2*w)
    if a<=vertex<=b:candidates.append(vertex)
    values=[residual(x) for x in candidates]
    rmax=max(abs(v) for v in values)
    final=rmax**32
    assert final<Fraction(1,10**12)
    return dict(domain=[.5,2.],updates=6,normalization_factor=1,initial_estimate=initial,
        rounded_weight=float(w),rounded_bias=float(bias),first_residual_minimum=float(min(values)),
        first_residual_maximum=float(max(values)),final_residual_upper_float=float(final),
        exact_final_residual_numerator=str(final.numerator),exact_final_residual_denominator=str(final.denominator),
        exact_remaining_five_updates_square_residual=True,
        reciprocal_mass_contract='Reuse the exact same binary64 public1/K in mean reduction and final probability scaling; it cancels in the ideal common-factor normalization.',
        scope='Exact real arithmetic after the rounded initial public affine. Excludes all CKKS encoding, multiplication, rotation and refresh error.')


def main():
    start=time.perf_counter()
    sources=[Path(__file__),LAB/'test_plan.py',LAB/'source_origin.json',*sorted((LAB/'inputs').glob('*')),
             *sorted((ROOT/'jaxite_word').glob('*.py')),*sorted((ROOT/'cadathlon/common').glob('*.py'))]
    source_bytes={str(path.relative_to(LAB)):path.read_bytes() for path in sources if path.is_file()}
    before={name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()}
    with tarfile.open(LAB/'sources.tar.gz','w:gz') as archive:
        for name,data in source_bytes.items():
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    baseline=json.loads((LAB/'inputs/baseline_manifest.json').read_text())
    schedule=PublicSchedule(baseline['power_coefficients_in_t_2x'])
    keys=key_inventory()
    profiles=[profile(*spec,keys) for spec in PROFILE_SPECS]
    for p in profiles:
        if not p['accepted']:continue
        rows=[solve(p,K,schedule) for K in range(1,129)]
        p['all_K_schedules']=rows
        p['summary']={'feasible_K':[r['K'] for r in rows if r['feasible']],
            'infeasible_K':[r['K'] for r in rows if not r['feasible']],
            'K15':rows[14],'K128':rows[127],
            'normalization_refresh_range':[min(r['normalization_refreshes'] for r in rows if r['feasible']),max(r['normalization_refreshes'] for r in rows if r['feasible'])]}
        pair=p['resources']['rotation_pair_bytes'];one=p['resources']['top_ciphertext_bytes']
        p['resources']['BSGS_FFN_current_preprocess_key_stack_upper_bytes']=(63+60)*pair
        p['resources']['BSGS_63_baby_ciphertexts_upper_bytes']=63*one
        p['resources']['BSGS_64_plaintext_giant_group_upper_bytes']=64*one//2
        p['resources']['warning']='These BSGS stacks alone can exceed an accelerator capacity. Raw disk-key fit does not establish HBM fit. Reviewed chunked linear and explicit primitive implementation differ from legacy BSGS preprocessing; each actual path needs its own live-set proof.'
    after={name:hashlib.sha256((LAB/name).read_bytes()).hexdigest() for name in before}
    result={'scope':__doc__,'passed':before==after and not FORBIDDEN,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,
        'source_archive_sha256':hashlib.sha256((LAB/'sources.tar.gz').read_bytes()).hexdigest(),
        'forbidden_calls':FORBIDDEN,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'key_inventory':keys,'profiles':profiles,'initial_C1_certificate':initial_C1_certificate(),
        'filesystem_snapshot':{path:dict(zip(('total','used','free'),shutil.disk_usage(path))) for path in ('/tmp','/dev/shm')},
        'schedule_scope':['Exact levels for explicitly masked reductions and retained numerator branch. Dynamic program minimizes refresh count only within start-probability/denominator-refresh choices, with unchanged static inverse and at least2 final levels.',
            'Initial six-update C1 inverse reuses the generic integer helper. Its combined softmax execution is not measured. The ideal public seed is0.8 on[0.5,2]; the rounded affine residual certificate must be retained.',
            'The scale examples start from nominal canonical masked/refresh scales. Actual integration must supply actual public ciphertext scales to the same backward planner and recheck each primitive; these examples do not prove exact bootstrap output scale bits. Each inverse targets a public scale computed to make the following retained-square product canonical. This uses genuine initial affine encoding plus exact binary64 recurrence; no metadata relabeling. It is a new public plan, not an encrypted proof.',
            'The denominator refresh breaks exact numerical equality to the retained sum of squares. Conditional positivity, ratio and mass contracts require a fresh CKKS/refresh error budget.',
            'Inactive probabilities are mathematically zero. CKKS residuals after masking/refresh must be controlled separately; no plaintext clipping is permitted.'],
        'bound_and_noise_budget':{
            'conservative_fullslot_component_bound':16.,'attention_norm_required_candidate_bound':64.,
            'bound64_preconditioning_example':49.37/64.,
            'C32_ideal_inverse_bound':8.,'correlated_product_ideal_bound':1/32,
            'denominator_minimum_true_for_K128_unit_mass':1/128,
            'conditional_contract_lower_a_for_K128':1/256,
            'illustrative_denom_error':1e-3,
            'illustrative_relative_denom_error_at_1_over_128':.128,
            'illustrative_probability_mass_with_plus_error':(1/128)/(1/128+1e-3),
            'illustrative_probability_mass_with_minus_error':(1/128)/(1/128-1e-3),
            'scope':'Algebraic sensitivity examples only, not error predictions. A1e-3 component refresh gate cannot imply the1e-12 ideal mass premise,1e-6 encrypted component gate or final model accuracy. Every refreshed denominator needs a correlated numerical error budget, and lower public preconditioning bounds need their own evidence.'},
        'common_ring_constraints':['All Q/K/V, attention, FFN and LayerNorm ciphertexts must be freshly generated/evaluated under the same N131072 exact orderedQ/P, dnum, scale, batch1, layout256x512 and key identity. Existing smaller-ring results cannot be relabeled or directly composed.',
            'RowTile F4096 has16rows/65536slots and8tiles for128tokens. Existing full-width FFN N65536 covers8rows/tile, LayerNorm N32768 covers4rows/tile; common-ring port and all-row tests remain required.',
            'The observed p56 LayerNorm component failed its fixed1e-6 gate. The p60/d13 smaller-ring retry now passed its fixed1e-6 gate, but p60 common-chain candidates do not inherit its numerical outcome.',
            'Attention LayerNorm observed output magnitude49.37 needs at least64 if refreshed without further justified public scaling. Current full-slot refresh proof is bound16 and pending; neither bound64 nor p60/d12/d14 bootstrap has numerical validation.',
            'p60 common profiles have only roughly1bit base-Q/scale headroom. Even after public bound preconditioning to[-1,1], this changes bootstrap numeric conditioning from p56; security and available levels alone do not validate refresh.',
            'After FFN, the residual addition uses the original normalized branch, with actual scales aligned by learned-linear output; preserve that saved ciphertext through the phase. MLP LayerNorm output/residual bounds remain unproved.',
            'Public direct key union is a dense-diagonal upper inventory. Disk-backed raw keys do not eliminate formatted key/control/NTT/FFT/JIT workspace or concurrent ciphertext storage.'],
        'external_evidence':{
            'precision60_layernorm':{'report':'inputs/precision60_layernorm_report.json','passed':json.loads((LAB/'inputs/precision60_layernorm_report.json').read_text())['passed'],'scope':'Separate N32768/Q26/P3/d13 depth12, two fixed rows/phase, all131072decodedslots, fixed1e-6; attention output3.2229643e-8 andMLP2.02292938e-9. Root reported exit0/reaped90252; not a same-ring proof.'},
            'native408':{'report':'inputs/native408_report.json','completed_examples':json.loads((LAB/'inputs/native408_report.json').read_text())['completed_examples'],'all_examples_have_existing_domain_violation':True,'scope':'Native float32 trajectories only; does not measure the approximate or encrypted trajectory. Full-model existing contracts fail coverage, so this level plan is conditional and cannot authorize complete MRPC inference.'}},
        'proof_gaps':['All408 audited MRPC examples violate at least one existing full-model valid-token domain; even layer0 MLP variance has one below-domain input. Existing fixed domains must not be silently reused as a full-dataset certificate.',
            'No encrypted normalization round, multi-round softmax, full-width attention composition or complete BERT block is established here.',
            'Full-slot49/d4 bound16 proof remains frozen. New52/53 chains, p60 bootstrapping, bound64 and stricter component accuracy require separate authorized proofs.',
            'Even a1e-3 refresh component pass is insufficient for the unchanged1e-6 inverse/component gates or model accuracy without a propagated error budget.'],
        'elapsed_s':time.perf_counter()-start}
    (LAB/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'source_count':len(before),'graph_keys':keys['graph_rotation_count'],'profiles':[
        {k:p.get(k) for k in ('name','accepted','Q_count','P_count','log2_QP','actual_partitions','base_Q_over_scale')}|({'refresh_output':p['refresh']['output_level'],'union_keys':p['resources']['graph_plus_bootstrap_count'],'raw_GiB':p['resources']['raw_rotation_store_GiB'],'feasible_K_count':len(p['summary']['feasible_K']),'K128_refreshes':p['summary']['K128'].get('normalization_refreshes'),'K128_output':p['summary']['K128'].get('normalization_output_level')} if p['accepted'] else {}) for p in profiles],'elapsed_s':result['elapsed_s']},indent=2))

if __name__=='__main__':main()
