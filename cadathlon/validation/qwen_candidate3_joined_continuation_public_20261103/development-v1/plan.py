"""Public candidate3 join ledger and conditional stored-coefficient envelopes.

No model/weights/inputs, cryptographic context or arithmetic backend is read.
Refresh metadata examples are supplied assumptions, never scale assignments.
"""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'common'))
from encrypted_qwen_rmsnorm import QwenRMSContract, rms_scale_plan
from encrypted_scale_planning import weighted_sum_scale, multiply_scale, drop_divisor

MANIFEST = '7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
RELATIVE = F(1, 1000000)
U = F(1, 2**53)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def ratio(value): return [str(value.numerator), str(value.denominator)]
def upper(value):
    value = F(value); out = float(value)
    if F.from_float(out) < value: out = math.nextafter(out, math.inf)
    return out
def sqrt_upper(value):
    value = F(value); out = math.sqrt(float(value))
    while F.from_float(out)**2 < value: out = math.nextafter(out, math.inf)
    return out
def guard_bound(value):
    target = F(value)*F(9,8); result = 1
    while result < target: result *= 2
    return result


def authenticated():
    origins = json.loads((ROOT/'source_inputs.json').read_bytes())
    for name, row in origins.items():
        if digest((ROOT/name).read_bytes()) != row['sha256']:
            raise ValueError('fixed public dependency changed: '+name)
    raw = (ROOT/'inputs/candidate_manifest.json').read_bytes()
    if digest(raw) != MANIFEST: raise ValueError('unchanged candidate3 required')
    return origins, raw


def rope_coefficients(config):
    """Same public expression as frozen head owner; every allowed position."""
    theta = config['rope_theta']
    positions = config['max_position_embeddings']
    if theta != 1000000 or positions != 40960: raise ValueError('fixed public RoPE architecture required')
    angles = np.arange(positions)[:,None]*theta**(-np.arange(0,128,2)/128)[None,:]
    c, s = np.cos(angles), np.sin(angles)
    if c.dtype != np.dtype('float64') or s.dtype != np.dtype('float64'):
        raise ValueError('actual binary64 stored coefficient tables required')
    if not (np.isfinite(c).all() and np.isfinite(s).all() and
            np.max(np.abs(c)) <= 1 and np.max(np.abs(s)) <= 1):
        raise ValueError('stored trigonometric coefficient magnitude premise failed')
    summed = c*c+s*s
    maximum = float(summed.max())
    # This does NOT presume correctly rounded libm sin/cos. Treat their actual
    #output bytes as exact coefficients. For |c|,|s|<=1, each rounded square has
    #absolute error <=u; addition of two numbers <=1 has error <=2u. Thus the
    #exact dyadic c²+s² is at most fl(fl(c*c)+fl(s*s))+4u, including zero/tiny
    #products. The only floating premise here is binary64 RN basic arithmetic.
    exact_upper = F.from_float(maximum) + 4*U
    at = tuple(int(v) for v in np.unravel_index(int(summed.argmax()), summed.shape))
    witness = F.from_float(float(c[at]))**2 + F.from_float(float(s[at]))**2
    if witness > exact_upper: raise ValueError('exact stored-coefficient witness exceeds enclosure')
    return dict(positions=positions,frequencies=64,coefficient_pairs=positions*64,
        numpy_version=np.__version__,cos_f64le_sha256=digest(c.astype('<f8').tobytes()),
        sin_f64le_sha256=digest(s.astype('<f8').tobytes()),
        max_rounded_square_sum=maximum,absolute_rounding_allowance=ratio(4*U),
        exact_square_sum_upper=ratio(exact_upper),exact_square_sum_upper_float=upper(exact_upper),
        maximum_rounded_witness_position_frequency=list(at),exact_witness_square_sum=ratio(witness),
        scope='Actual stored public cos/sin bytes on every allowed original position; no assertion that libm is correctly rounded, no HE arithmetic/refresh perturbation covered.'), exact_upper


def paired_bound(gamma, square_sum):
    # Pairwise Cauchy: |c gamma_i x_i +/- s gamma_j x_j|²
    # <= max_gamma²(c²+s²) sum_j x_j². Same-input v>=sum(x²)/128.
    value = F.from_float(sqrt_upper(F(128)*square_sum))*F.from_float(gamma)*(1+RELATIVE)
    return dict(exact_upper=ratio(value),absolute_upper=upper(value),public_refresh_bound=guard_bound(value))


class Replay:
    """Actual ordered binary64 metadata equations; no ciphertext objects."""
    def __init__(self, q, delta, scenario):
        self.q, self.delta, self.scenario = q, delta, scenario
        self.events, self.stages, self.additions = [], [], []
    def node(self, label, level, scale, kind):
        if type(level) is not int or level < 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError('invalid public continuation metadata')
        result = dict(level=level, scale=scale)
        self.stages.append(dict(label=label,kind=kind,**result))
        return result
    def weighted(self, label, *values):
        level=min(v['level'] for v in values)
        scale=weighted_sum_scale([v['scale'] for v in values],self.delta,drop_divisor(self.q,level))['output_scale']
        return self.node(label,level-1,scale,'weighted')
    def multiply(self, label, left, right):
        level=min(left['level'],right['level'])
        scale=multiply_scale(left['scale'],right['scale'],drop_divisor(self.q,level))
        return self.node(label,level-1,scale,'multiply')
    def refresh_scale(self):
        direction = {'lower':-1,'canonical':0,'upper':1}.get(self.scenario)
        if direction is None: direction = (-1,1)[len(self.events)%2]
        return self.delta if direction==0 else math.nextafter(self.delta,math.inf if direction>0 else -math.inf)
    def refresh(self,label,value,bound,*,category):
        if value['level']<2: raise ValueError('existing refresh binding requires input level >=2')
        scale=self.refresh_scale()
        self.events.append(dict(label=label,input_level=value['level'],input_scale=value['scale'],
            output_level=25,assumed_output_scale=scale,public_bound=bound,category=category))
        return self.node(label,25,scale,'assumed_refresh')
    def add(self,label,left,right):
        compatible=math.isclose(left['scale'],right['scale'],rel_tol=1e-12,abs_tol=0.)
        self.additions.append(dict(label=label,left_scale=left['scale'],right_scale=right['scale'],compatible=compatible))
        if not compatible:raise ValueError('ordinary residual addition scale incompatibility')
        return self.node(label,min(left['level'],right['level']),left['scale'],'add_left_metadata')
    def polynomial(self,label,value,rule):
        mapped=self.weighted(label+'.map',value);basis={1:mapped}
        def term(n):
            if n not in basis:
                product=self.multiply(label+f'.basis{n}.product',term(n//2),term((n+1)//2))
                basis[n]=(product if rule['strategy']=='power' else
                    self.weighted(label+f'.basis{n}',product,*(() if n%2==0 else (mapped,))))
            return basis[n]
        indices=[n for n in range(1,len(rule['evaluation_coefficients']))
                 if rule['strategy']=='chebyshev' or rule['evaluation_coefficients'][n]!=0]
        return self.weighted(label+'.sum',*[term(n) for n in indices])
    def reciprocal(self,label,value,rule):
        result=self.weighted(label+'.first',value)
        for i in range(2,rule['iterations']+1):
            product=self.multiply(label+f'.{i}.product',value,result)
            correction=self.weighted(label+f'.{i}.correction',product)
            result=self.multiply(label+f'.{i}.update',result,correction)
        return result


def derive():
    origins,raw=authenticated();manifest=json.loads(raw)
    meta=(ROOT/'inputs/site_metadata.json').read_bytes()
    config=json.loads((ROOT/'inputs/config.json').read_bytes())
    full=json.loads((ROOT/'inputs/full_plan.json').read_bytes())
    rms=json.loads((ROOT/'inputs/rms_plan.json').read_bytes())
    soft=json.loads((ROOT/'inputs/softmax_plan.json').read_bytes())
    learned=json.loads((ROOT/'inputs/learned_norm_bounds.json').read_bytes())
    profile=next(row for row in json.loads((ROOT/'inputs/profile_audit.json').read_bytes())['profiles'] if row['name']=='depth49_p60_d8')
    q,delta=profile['q'],profile['canonical_scale']
    contracts={row['site']:QwenRMSContract(raw,meta,row['site']) for row in rms['sites']}
    bound_sites={row['site']:row for row in full['normalization']['sites']}
    if len(contracts)!=113 or sum(len(c.events) for c in contracts.values())!=260:raise ValueError('exact113-site circuit required')
    rope_table,m=rope_coefficients(config)
    rope_bounds={};norm_records=[]
    for name,c in contracts.items():
        old=bound_sites[name];g=learned['normalizations']['qwen:'+old['learned_weight_tensor']]
        if g['gamma_f64le_sha256']!=c.gamma_sha256:raise ValueError('gamma-bound identity differs')
        norm_records.append(dict(site=name,method=c.method,degree=c.degree,iterations=c.iterations,
            input_level=25,output_level=c.output_level,internal_inverse_refreshes=len(c.events),
            raw_input_bound=old['conditional_input_refresh_bound'],output_bound=old['conditional_norm_output_refresh_bound'],
            gamma_max_abs=g['gamma_max_abs'],gamma_sha256=c.gamma_sha256))
        if '.q.' in name or '.k.' in name:
            rope_bounds[name]=dict(site=name,gamma_max_abs=g['gamma_max_abs'],
                gamma_sha256=c.gamma_sha256,requires_extra_pre_rope_refresh=c.output_level<3,
                norm_output_level=c.output_level,**paired_bound(g['gamma_max_abs'],m))
    scores=[]
    for layer in range(28):
        qg=F.from_float(rope_bounds[f'{layer}.q.inverse_sqrt']['gamma_max_abs'])
        kg=F.from_float(rope_bounds[f'{layer}.k.inverse_sqrt']['gamma_max_abs'])
        # Orthogonal 2x2 pair blocks have operator norm sqrt(c²+s²).
        score=F.from_float(sqrt_upper(128))*qg*kg*(1+RELATIVE)**2*m
        domain=manifest['softmax'][str(layer)]['centered_domain'];rows=[]
        for k in range(2,33):
            alpha=F.from_float(1/k)
            centered=(abs(1-alpha)+(k-1)*abs(alpha))*score
            rows.append(dict(K=k,absolute_upper=upper(centered),upper_fraction=ratio(centered),
                proves_frozen_centered_domain=(centered<=-F(domain[0]) and centered<=F(domain[1]))))
        scores.append(dict(layer=layer,score_abs_upper=upper(score),score_upper_fraction=ratio(score),
            frozen_centered_domain=domain,centered_rows=rows))

    scenario_reports=[]
    for scenario in ('lower','canonical','upper','alternating'):
        replay=Replay(q,delta,scenario);layers=[]
        value=dict(level=49,scale=delta)
        def norm(name,value):
            c=contracts[name];scales=[];event_start=len(replay.events)
            for event in c.events:
                pseudo=dict(level=event[2],scale=delta)
                assumed=replay.refresh(name+'.inverse_event',pseudo,event[1],category='inverse_only')
                scales.append(assumed['scale'])
            result=rms_scale_plan(q,c,input_scale=value['scale'],refresh_scales=scales)
            # Replace event input metadata with the replayed actual schedule.
            actual_events=[s for s in result['stages'] if s['kind']=='refresh']
            if len(actual_events)!=len(c.events):raise ValueError('norm refresh mismatch')
            event_index=event_start
            for i,stage in enumerate(result['stages']):
                if stage['kind']=='refresh':
                    previous=result['stages'][i-1]
                    if previous['level']!=replay.events[event_index]['input_level']:
                        raise ValueError('actual inverse refresh predecessor differs')
                    replay.events[event_index]['input_scale']=previous['scale']
                    event_index+=1
            return replay.node(name,result['output_level'],result['output_scale'],'unchanged_rms'),digest(canonical(result))
        for layer in range(28):
            before=len(replay.events);p=f'{layer}.'
            if layer==0:value=replay.node('embedding_level_reduce',25,value['scale'],'level_reduce')
            else:value=replay.refresh(p+'raw_input',value,bound_sites[p+'input.inverse_sqrt']['conditional_input_refresh_bound'],category='outer')
            residual=value
            normalized,norm_digest=norm(p+'input.inverse_sqrt',value)
            normalized=replay.refresh(p+'normalized_input',normalized,bound_sites[p+'input.inverse_sqrt']['conditional_norm_output_refresh_bound'],category='outer')
            query=replay.weighted(p+'q_linear',normalized);key=replay.weighted(p+'k_linear',normalized);projected=replay.weighted(p+'v_linear',normalized)
            heads={};head_norms={}
            for kind,source in (('q',query),('k',key)):
                site=p+kind+'.inverse_sqrt';contract=contracts[site]
                source=replay.refresh(site+'.raw',source,bound_sites[site]['conditional_input_refresh_bound'],category='outer')
                result,hash_plan=norm(site,source);head_norms[kind]=hash_plan
                if result['level']<3:
                    result=replay.refresh(site+'.pre_rope',result,bound_sites[site]['conditional_norm_output_refresh_bound'],category='pre_rope_reserve')
                rope=replay.weighted(site+'.rope',result,result,result)
                heads[kind]=replay.refresh(site+'.post_rope',rope,rope_bounds[site]['public_refresh_bound'],category='post_rope')
            expanded_key=replay.weighted(p+'key_gqa',*[heads['k']]*9)
            expanded_value=replay.weighted(p+'value_gqa',*[projected]*9)
            key_broadcast=replay.weighted(p+'key_broadcast',*[expanded_key]*16)
            product=replay.multiply(p+'qk',heads['q'],key_broadcast)
            score=replay.weighted(p+'scores',*[product]*16)
            # Existing literal owner verifies every intermediate. Its final
            #actual refresh is L25 followed by the actual branch mask ->L24.
            #Use a supplied return scale here, not a relabelled prior value.
            probability_return=dict(level=25,scale=replay.refresh_scale())
            probability=replay.weighted(p+'softmax_final_mask',probability_return)
            probability_anchor=replay.weighted(p+'probability_anchor',*[probability]*16)
            value_broadcast=replay.weighted(p+'value_broadcast',*[expanded_value]*16)
            pv_product=replay.multiply(p+'pv_product',probability_anchor,value_broadcast)
            attended=replay.weighted(p+'attended_final_mask',pv_product,expanded_value)
            output=replay.weighted(p+'o_linear',attended)
            post_residual=replay.add(p+'attention_residual',residual,output)
            post_input=replay.refresh(p+'post_raw',post_residual,bound_sites[p+'post.inverse_sqrt']['conditional_input_refresh_bound'],category='outer')
            post,post_digest=norm(p+'post.inverse_sqrt',post_input)
            post=replay.refresh(p+'post_normalized',post,bound_sites[p+'post.inverse_sqrt']['conditional_norm_output_refresh_bound'],category='outer')
            gate=replay.weighted(p+'gate_linear',post);up=replay.weighted(p+'up_linear',post)
            rule=manifest['nonlinear'][p+'silu']
            if rule['method']=='polynomial':
                silu=replay.polynomial(p+'silu',gate,rule)
            else:
                y=replay.polynomial(p+'silu_base',gate,rule['base'])
                for i in range(rule['doublings']):
                    y=replay.refresh(p+f'tanh{i}.refresh',y,2,category='silu_tanh')
                    denominator=replay.weighted(p+f'tanh{i}.denominator',replay.multiply(p+f'tanh{i}.square',y,y))
                    inverse=replay.reciprocal(p+f'tanh{i}.inverse',denominator,rule['reciprocal'])
                    y=replay.weighted(p+f'tanh{i}.double',replay.multiply(p+f'tanh{i}.quotient',y,inverse))
                sigmoid=replay.weighted(p+'sigmoid',y)
                silu=replay.multiply(p+'silu',gate,sigmoid)
            gated=replay.multiply(p+'gated',silu,up)
            down=replay.weighted(p+'down',gated)
            value=replay.add(p+'mlp_residual',post_residual,down)
            layers.append(dict(layer=layer,input_norm_plan_sha256=norm_digest,qk_norm_plan_sha256=head_norms,
                post_norm_plan_sha256=post_digest,query=heads['q'],expanded_key=expanded_key,
                expanded_value=expanded_value,scores=score,probabilities=probability,attended=attended,
                attention_output=output,post_residual=post_residual,silu=silu,gated=gated,down=down,
                layer_output=value,nonsoftmax_refresh_events=replay.events[before:]))
        value=replay.refresh('final.raw',value,bound_sites['final.inverse_sqrt']['conditional_input_refresh_bound'],category='outer')
        final,final_digest=norm('final.inverse_sqrt',value)
        logits=replay.weighted('vocabulary_block',final)
        scenario_reports.append(dict(scenario=scenario,layers=layers,final_norm_plan_sha256=final_digest,
            final_norm=final,all38_vocabulary_blocks=logits,
            residual_additions=replay.additions,nonsoftmax_refresh_counts=dict(Counter(e['category'] for e in replay.events)),
            nonsoftmax_event_count=len(replay.events),metadata_trace_sha256=digest(canonical(replay.stages))))

    outer=next(r for r in scenario_reports if r['scenario']=='canonical')['nonsoftmax_refresh_counts']
    if sum(outer.values())!=638 or outer['inverse_only']!=260 or outer['silu_tanh']!=145:
        raise ValueError('fixed joined event accounting mismatch')
    workloads=[]
    for size in (1,8,16,19,32):
        tiles=(size+15)//16
        soft_row=next(row for row in soft['workloads'] if row['sequence']==size)
        soft_events=sum(row['counts'].get('refreshes',0) for row in soft_row['layers'])
        workloads.append(dict(sequence=size,row_tiles=tiles,softmax_refreshes=soft_events,
            inverse_only_refreshes=260*tiles,silu_tanh_refreshes=145*tiles,
            outer_and_pre_rope_refreshes=232*tiles+1,
            total_conditional_refreshes=637*tiles+1+soft_events,
            attended_level=22 if size==1 else 20,attention_output_level=21 if size==1 else 19,
            all_block_output_level=4,final_norm_level=7,all_vocabulary_blocks_level=6,
            K1_scope='Actual V0 direct mask; no invented probability result. The complete28-layer metadata examples above use a nontrivial query; S1 has two extra attention levels and the same MLP endpoint.'))
    preliminary_keys=set(soft['keys']['combined_component_indices'])
    for row in full['linear_maps'].values():preliminary_keys.update(row['required_keys_dense_case'])
    for row in full['vocabulary']['maps']:preliminary_keys.update(row['required_keys_dense_case'])
    for row in rms['sites']:preliminary_keys.update(row['norm_rotation_keys'])
    return dict(schema='qwen-candidate3-joined-continuation-public-v1',passed=True,
        scope='Public level/conditional scale and exact-real range ledger only; no joined owner, encrypted execution, model forward or trajectory/domain coverage claim.',
        candidate_manifest_sha256=MANIFEST,source_input_sha256={n:v['sha256'] for n,v in origins.items()},
        architecture=dict(layers=28,hidden=1024,query_heads=16,kv_heads=8,head_dim=128,mlp=3072,vocabulary=151936),
        rope_table=rope_table,rope_bounds=list(rope_bounds.values()),normalizations=norm_records,
        score_domain_outer_bounds=scores,all_centered_domains_proved=all(r['proves_frozen_centered_domain'] for x in scores for r in x['centered_rows']),
        metadata_examples=scenario_reports,workloads=workloads,
        incompatibilities=dict(refresh_before_rope_only=dict(query_level=24,expanded_key_level=23,score_level=20,required_score_level=21),
            low_norm_rope_inputs=[n for n,c in contracts.items() if ('.q.' in n or '.k.' in n) and c.output_level<3],
            natural_value_path=dict(projection24=24,expansion23=23,attended20=20,output_projection19=19),
            rms_entry=25,raw_post_attention_levels=[19,21],raw_block_level=4),
        resources=dict(preliminary_dense_case_key_union=sorted(preliminary_keys),key_count=len(preliminary_keys),
            raw_rotation_pair_bytes=profile['rotation_pair_bytes'],raw_rotation_keys_bytes=len(preliminary_keys)*profile['rotation_pair_bytes'],
            retained_degree512_basis_raw_bytes=full['normalization']['degree512_retained_basis_raw_bytes'],
            all38_L6_logit_ciphertext_bytes=38*2*131072*14*4,
            scope='Known routing/norm/bootstrap plus dense-shape chained gaps. Actual authenticated public matrix zero-diagonal patterns can change chained gaps; this is not a final full-model key inventory or memory peak.'),
        limitations=[
            'Four supplied refresh-return metadata scenarios are examples, not an exhaustive reachable-scale proof. Runtime must replay every actual returned scale and enforce ordinary addition compatibility.',
            'All norm/score/SiLU declared domains remain assumptions. Input-independent score outer bounds do not in general imply frozen centered domains; no clipping or refitting is proposed.',
            'Same-input variance and norm-relative certificates survive only identity refresh in this exact-real design. Inverse/pre-RoPE/post-RoPE and raw refresh perturbations need separate correlated error/domain analysis.',
            'Stored public RoPE coefficient tables are evaluated on all40960 original positions. Future runtime must bind the same bytes/software or independently certify its own coefficient tables.',
            'Existing inverse-only refresh binding does not authorize new outer/post-RoPE events. A distinct typed outer-event contract and actual B32..32768 range/precision validation are missing.',
            'Qwen SiLU/MLP/residual/final-vocabulary/joined28-layer/file-stage owners and complete model key planning remain unimplemented. Existing generic linears are reusable primitives, not full-model HE evidence.',
            'A future dedicated Qwen engine must enforce synchronous ordinary arithmetic and quarantine all public maps/owners on entered-body failure, including results not returned after a failed barrier.'
        ])


if __name__=='__main__':
    report=derive()
    if any(name in sys.modules for name in ('jax','jaxlib','torch','transformers','safetensors')):
        raise RuntimeError('public planner imported forbidden backend/model package')
    (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(norms=len(report['normalizations']),post_rope=len(report['rope_bounds']),
        extra_pre_rope=len(report['incompatibilities']['low_norm_rope_inputs']),
        refreshes_S32=report['workloads'][-1]['total_conditional_refreshes'],
        keys=report['resources']['key_count'],all_centered_domains_proved=report['all_centered_domains_proved'])))
