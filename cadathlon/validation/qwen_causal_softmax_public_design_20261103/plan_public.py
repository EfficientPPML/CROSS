"""Proposed causal-softmax public schedule; no owner, payload or backend run."""
from collections import Counter
from pathlib import Path
import hashlib,json,math,sys
from encrypted_scale_planning import drop_divisor,multiply_scale,weighted_sum_scale

ROOT=Path(__file__).resolve().parent
MANIFEST='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
DELTA=1.1467305003699405e18
def sha(raw):return hashlib.sha256(raw).hexdigest()
def inputs():
    pins=json.loads((ROOT/'input_sources.json').read_bytes())
    for name,row in pins.items():
        if sha((ROOT/name).read_bytes())!=row['sha256']:raise ValueError('changed public source '+name)
    if sha((ROOT/'inputs/candidate_manifest.json').read_bytes())!=MANIFEST:raise ValueError('candidate changed')
    manifest=json.loads((ROOT/'inputs/candidate_manifest.json').read_bytes())
    cert=json.loads((ROOT/'inputs/certificates.json').read_bytes())
    continuation=json.loads((ROOT/'inputs/continuation_plan.json').read_bytes())
    profile=next(p for p in json.loads((ROOT/'inputs/profile_audit.json').read_bytes())['profiles'] if p['name']=='depth49_p60_d8')
    assert cert['passed'] is True and cert['manifest_sha256']==MANIFEST
    assert cert['continuation_plan_sha256']==sha((ROOT/'inputs/continuation_plan.json').read_bytes())
    assert cert['initial_inverse_refresh_event']==dict(phase='after_completed_initial_reciprocal',completed_updates=6,
        input_level=3,output_level=25,public_bound=4,frozen_plan_tail=continuation['static_initial_softmax_level_plan']['tail'])
    assert len(profile['q'])==100 and len(profile['p'])==14 and profile['canonical_scale']==DELTA
    return manifest,cert,continuation,profile

def branch_plan(manifest,cert,profile,layer,k,*,input_scale=DELTA,refresh_mode='canonical'):
    """Replay literal weighted/CT metadata; identity refresh scales are examples."""
    if type(layer)is not int or not 0<=layer<28 or type(k)is not int or not 1<=k<=32:raise ValueError('typed public layer/K required')
    if refresh_mode not in ('canonical','alternating_ulp'):raise ValueError('unknown scale example')
    if k==1:return dict(layer=layer,K=1,nonlinear_bypass=True,refreshes=[],counts={},output_level=None,
        value_source='actual expanded V at original first key; no probability ciphertext required')
    q=profile['q'];events=[];stages=[];counts=Counter();rounds=[]
    def node(label,kind,level,scale):
        if level<2 or not math.isfinite(scale) or scale<=0:raise ValueError('invalid public metadata')
        out=dict(level=level,scale=scale);stages.append(dict(label=label,kind=kind,**out));return out
    def weighted(label,*values):
        level=min(v['level'] for v in values);counts['weighted_sums']+=1
        return node(label,'weighted',level-1,weighted_sum_scale([v['scale'] for v in values],DELTA,drop_divisor(q,level))['output_scale'])
    def mul(label,x,y):
        level=min(x['level'],y['level']);counts['ciphertext_products']+=1
        return node(label,'multiply',level-1,multiply_scale(x['scale'],y['scale'],drop_divisor(q,level)))
    def add(label,x,y):
        # Ordinary add/subtract preserves left scale after level alignment.
        if not math.isclose(x['scale'],y['scale'],rel_tol=1e-12):raise ValueError('ordinary addition scale mismatch')
        counts['ordinary_additions']+=1
        return node(label,'add_or_subtract',min(x['level'],y['level']),x['scale'])
    def reduction(value):
        counts['rotations']+=5;counts['ordinary_additions']+=5;return value
    def broadcast(value):
        counts['rotations']+=5;counts['ordinary_additions']+=5;return value
    def refresh(label,value,bound,**details):
        scale=DELTA if refresh_mode=='canonical' else math.nextafter(DELTA,math.inf if len(events)%2==0 else -math.inf)
        event=dict(label=label,input_level=value['level'],input_scale=value['scale'],output_level=25,
            actual_output_scale_example=scale,public_bound=bound,**details)
        events.append(event);counts['refreshes']+=1
        return node(label,'refresh',25,scale)
    def inverse(label,denominator,rule,event_bound=None):
        y=weighted(label+'.first',denominator);levels=[y['level']]
        for update in range(2,rule['iterations']+1):
            if event_bound is not None and update==7:
                assert y['level']==6
                y=refresh(label+'.refresh_before7',y,event_bound,completed_updates=6)
            product=mul(label+f'.{update}.product',denominator,y)
            correction=weighted(label+f'.{update}.correction',product)
            y=mul(label+f'.{update}.update',y,correction);levels.append(y['level'])
        return y,levels

    s=manifest['softmax'][str(layer)]['squarings'];rule=manifest['reciprocal_by_keys'][str(k)]
    kc=cert['softmax_K'][k-1];assert kc['K']==k and kc['existing_inverse_refresh_bound_certified']
    assert manifest['exp']['strategy']=='power' and manifest['exp']['degree']==8
    assert len(manifest['exp']['evaluation_coefficients'])==9 and all(manifest['exp']['evaluation_coefficients'][1:])
    source=node('scores','input',21,float(input_scale))
    masked=weighted('source_branch_mask',source)
    mean=broadcast(weighted('mean_anchor_alpha',reduction(masked)))
    centered=add('centered',masked,mean)
    normalized=weighted('normalized_t_branch_mask',centered)
    assert normalized['level']==18
    t=refresh('normalized_t.refresh',normalized,2)
    powers={1:t}
    def power(n):
        if n not in powers:powers[n]=mul('power'+str(n),power(n//2),power((n+1)//2))
        return powers[n]
    base=weighted('base_exp',*[power(n) for n in range(1,9)])
    denominator=broadcast(weighted('initial_mean_anchor_dummy1',reduction(weighted('initial_mean_mask_alpha',base))))
    initial_inverse,initial_levels=inverse('initial_inverse',denominator,manifest['initial_reciprocal'])
    assert initial_levels==[18,15,12,9,6,3]
    initial_inverse=refresh('completed_initial_inverse.refresh',initial_inverse,4,completed_updates=6)
    probability=weighted('initial_probability_alpha_mask',mul('initial_quotient',base,initial_inverse))
    assert probability['level']==19
    for iteration in range(s):
        prefix='round'+str(iteration)
        probability=refresh(prefix+'.probability.refresh',probability,2)
        square=mul(prefix+'.square',*(2*[weighted(prefix+'.mask',probability)]))
        denominator=broadcast(weighted(prefix+'.denominator_anchor_dummy1',reduction(square)))
        y,levels=inverse(prefix+'.inverse',denominator,rule,kc['existing_inverse_refresh_bound'])
        probability=mul(prefix+'.quotient',square,y)
        rounds.append(dict(round=iteration,denominator_level=denominator['level'],inverse_levels=levels,
            inverse_output_level=y['level'],probability_output_level=probability['level'],
            probability_output_scale=probability['scale']))
    probability=refresh('final_probability.refresh',probability,2)
    output=weighted('final_branch_mask',probability)
    expected=dict(weighted_sums=14+s*(rule['iterations']+2),ciphertext_products=18+2*s*rule['iterations'],
        ordinary_additions=21+10*s,rotations=20+10*s,refreshes=3+2*s)
    assert dict(counts)==expected and output['level']==24
    return dict(layer=layer,K=k,nonlinear_bypass=False,centered_domain=manifest['softmax'][str(layer)]['centered_domain'],
        mapped_t_factor=2.**(1-s),rounded_alpha_hex=float(1/k).hex(),squarings=s,
        exp_evaluation_coefficients_hex=[float(x).hex() for x in manifest['exp']['evaluation_coefficients']],
        initial_iterations=6,original_round_iterations=rule['iterations'],original_K_domain=rule['domain'],
        initial_first_affine_hex=[float(manifest['initial_reciprocal'][key]).hex() for key in ('rounded_first_weight','rounded_first_bias')],
        round_first_affine_hex=[float(rule[key]).hex() for key in ('rounded_first_weight','rounded_first_bias')],
        initial_inverse_levels=initial_levels,rounds=rounds,refreshes=events,counts=dict(counts),
        input_level=21,input_scale=input_scale,output_level=24,output_scale=output['scale'],stages=stages,
        scale_scope='Actual binary64 primitive replay for the supplied example refresh scales; no actual refresh or CKKS error claim.')

def derive():
    m,c,continuation,p=inputs();records=[];examples=[];hist=Counter()
    for layer in range(28):
        for k in range(1,33):
            row=branch_plan(m,c,p,layer,k)
            if k>1:
                for e in row['refreshes']:hist[str(e['public_bound'])]+=1
                row['refreshes']=[{key:e[key] for key in ('label','input_level','output_level','public_bound')} for e in row['refreshes']]
                row.pop('stages');row.pop('exp_evaluation_coefficients_hex')
                for r in row['rounds']:r.pop('probability_output_scale')
            records.append(row)
    for layer in (0,3):
        for k in (2,8,32):
            for mode in ('canonical','alternating_ulp'):
                examples.append(branch_plan(m,c,p,layer,k,input_scale=multiply_scale(DELTA,DELTA,drop_divisor(p['q'],23)),refresh_mode=mode))
    reduce_keys={shift%65536 for bit in range(5) for shift in (1<<bit,-(1<<bit))}
    audit=json.loads((ROOT/'inputs/profile_audit.json').read_bytes())
    pv=json.loads((ROOT/'inputs/pv_plan.json').read_bytes())
    bootstrap=set(audit['bootstrap_rotation_indices']);pvkeys=set(pv['workloads'][-1]['required_component_rotation_keys'])
    union=sorted(bootstrap|pvkeys|reduce_keys)
    workloads=[]
    for size in (1,8,16,19,32):
        layers=[]
        for layer in range(28):
            subset=[r for r in records if r['layer']==layer and 2<=r['K']<=size]
            total=Counter()
            for row in subset:total.update(row['counts'])
            total['ordinary_additions']+=sum(max(0,min(size,16*(tile+1))-max(2,16*tile+1)) for tile in range((size+15)//16))
            layers.append(dict(layer=layer,nontrivial_query_branches=size-1,counts=dict(total)))
        workloads.append(dict(sequence=size,tile_count=(size+15)//16,K_by_tile=[list(range(16*t+1,min(size,16*(t+1))+1)) for t in range((size+15)//16)],
            K1_direct_actual_V_bypass=True,layers=layers,total28_refreshes=sum(x['counts'].get('refreshes',0) for x in layers)))
    return dict(scope='Reviewable implementation design and public level/metadata plan only. No causal-softmax owner or HE run yet.',
        candidate_manifest_sha256=MANIFEST,certificate_sha256=sha((ROOT/'inputs/certificates.json').read_bytes()),
        layout=dict(slots=65536,row_width=4096,heads=16,key_stride=32,max_sequence=32,score_entry_level=21,probability_output_level=24),
        profile={key:p[key] for key in ('name','degree','slots','depth','dnum','Q_count','P_count','log2_QP','canonical_scale')},
        all_layer_K_plans=records,actual_scale_examples=examples,workloads=workloads,
        refresh_histogram_all28_S32=dict(hist),
        keys=dict(softmax_reduction=sorted(reduce_keys),bootstrap_count=len(bootstrap),predecessor_head_PV_count=len(pvkeys),
            combined_component_indices=union,combined_component_count=len(union),raw_rotation_key_bytes=len(union)*p['rotation_pair_bytes'],
            scope='Head/RoPE/QK/PV + proposed softmax + bootstrap only; not full-model key union or memory peak.'),
        retained_exp_basis_raw_bytes=sum(2*131072*2*(26-math.ceil(math.log2(n)))*4 for n in range(1,9)),
        exp_zero_margin_layers=[3,5,12,15],limitations=[
            'Each row branch refreshes a full independent65536-slot ciphertext; serial SIMD utilization is low.',
            'All B2/B4/B8..B128 envelopes are conditional exact-real certificates; refresh/FP/CKKS perturbations are excluded.',
            'No guarantee that noisy normalized t remains within the original exp domain, especially zero-margin endpoints.',
            'Original K1 bypass requires a new explicit PV path; existing frozen PV bytes are unchanged.',
            'No changed coefficients, domains, iteration counts, tracked/integer-affine replacement or model-data selection.',
            'Component raw key/basis bytes exclude controls, temporary buffers, compiler, device/host duplication and transport.'])

if __name__=='__main__':
    report=derive();(ROOT/'public_plan.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert not any(name in sys.modules for name in ('jax','jaxlib','torch','transformers','safetensors'))
    print(json.dumps(dict(plans=len(report['all_layer_K_plans']),refreshes_S32=report['workloads'][-1]['total28_refreshes'],
        keys=report['keys']['combined_component_count'],basis_bytes=report['retained_exp_basis_raw_bytes'])))
