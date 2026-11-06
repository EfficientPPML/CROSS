"""Public candidate3 level/resource design; no model, activations or HE import."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from types import SimpleNamespace
import ast, hashlib, json, math
from numbers import Integral, Real

ROOT=Path(__file__).resolve().parent
MANIFEST='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ceil_sqrt(x):
    x=F(x);y=math.sqrt(float(x))
    while F.from_float(y)**2<x:y=math.nextafter(y,math.inf)
    return y
def power2_guard(x):
    x=F(x)*F(9,8);b=1
    while b<x:b*=2
    return b
def upper_float(x):
    x=F(x);y=float(x)
    if F.from_float(y)<x:y=math.nextafter(y,math.inf)
    return y
def ct_bytes(level):return 2*131072*(2*(level+1))*4

def original_update_level(y,v):
    square=y-1; product=min(square,v)-1; correction=product-1
    return min(y,correction)-1

def norm_plan(name,rule):
    """Keep x25/v23 unchanged. Refresh y only, at explicit group boundaries."""
    lo,hi=map(F.from_float,rule['domain']);dim=128 if name.split('.')[1:2] in (['q'],['k']) else 1024
    events=[];steps=[]
    if rule['method']=='newton':
        n=rule['iterations'];assert rule['depth']==4*n-3
        w,b=map(F.from_float,(rule['rounded_first_weight'],rule['rounded_first_bias']))
        first=[w*x+b for x in (lo,hi)];assert min(first)>0
        cert=rule['first_exact_residual_abs_bound'];R=F(int(cert['numerator']),int(cert['denominator']));assert 0<=R<1
        # 0<x*y1^2<2 and y1>0; f(z)=z(3-z^2)/2 maps (0,sqrt(2)) into (0,1].
        # Subsequent positive y <= 1/sqrt(x). First affine is bounded separately.
        y_bound=max(max(first),F.from_float(ceil_sqrt(1/lo)))
        y=22;steps.append(dict(iteration=1,y_level=y,operation='unchanged rounded first affine'))
        for i in range(2,n+1):
            if original_update_level(y,23)<4:
                events.append(dict(before_iteration=i,input_level=y,output_level=25,
                    value='inverse_estimate_y',public_bound=power2_guard(y_bound)))
                y=25
            y=original_update_level(y,23)
            assert y>=4
            steps.append(dict(iteration=i,y_level=y,operation='square, v product, affine correction, y product'))
        inverse=y;output=min(25,y)-2
    else:
        assert rule['strategy']=='chebyshev'
        d=rule['degree'];assert rule['depth']==2*math.ceil(math.log2(d))+2
        inverse=23-rule['depth'];y=inverse
        y_bound=F(1000001,1000000)*F.from_float(ceil_sqrt(1/lo))
        if inverse-2<2:
            events.append(dict(after='inverse_polynomial',input_level=inverse,output_level=25,
                value='inverse_polynomial_output',public_bound=power2_guard(y_bound)))
            y=25
        output=min(25,y)-2
    assert output>=2
    return dict(site=name,method=rule['method'],degree=rule.get('degree'),iterations=rule.get('iterations'),
        domain=rule['domain'],group_width=dim,unrefreshed_rms_depth=rule['depth']+4,
        entry_level=25,variance_level=23,inverse_level_before_terminal_refresh=inverse,
        output_level=output,refresh_events=events,iteration_levels=steps,
        ideal_inverse_abs_upper=upper_float(y_bound),inverse_refresh_bound=power2_guard(y_bound),
        conditional_input_refresh_bound=power2_guard(F.from_float(ceil_sqrt(dim*hi))),
        bound_scope='Ideal same-input variance relation and declared domain only; coefficient certificate/first-affine residual as pinned. No FP/CKKS/refresh error allowance.')

def reciprocal_plan(n,denominator_level,first_level):
    """Unchanged affine -1/+2 static Newton: three levels per later update."""
    y=first_level;levels=[y];events=[]
    def update(y):return min(y,min(y,denominator_level)-2)-1
    for i in range(2,n+1):
        if update(y)<4:events.append(dict(before_iteration=i,input_level=y,output_level=25));y=25
        y=update(y);assert y>=3;levels.append(y)
    return dict(iterations=n,denominator_level=denominator_level,y_levels=levels,refresh_events=events)

def production_planner():
    """Execute only unchanged public AST definitions, avoiding backend imports."""
    ns=dict(math=math,Integral=Integral,Real=Real)
    from typing import Optional,Tuple
    from dataclasses import dataclass
    ns.update(Optional=Optional,Tuple=Tuple,dataclass=dataclass)
    for filename,names in [('bsgs.py',{'compute_bsgs_params'}),('chained_streaming_linear.py',{'validate_rotation_strategy','LinearRotationPlan','linear_rotation_plan'})]:
        tree=ast.parse((ROOT/'inputs'/filename).read_text())
        selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert len(selected)==len(names)
        exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),ns)
    return ns

def dense_linear_plan(rows,cols):
    assert max(rows,cols)<=4096
    ns=production_planner();active=tuple(sorted({x%65536 for x in range(-(rows-1),cols)}))
    n1,n2=ns['compute_bsgs_params'](65536,len(active),2.)
    p=ns['linear_rotation_plan'](SimpleNamespace(active=active,n1=n1,dimension=65536),strategy='chained')
    return dict(output_width=rows,input_width=cols,active_diagonals_dense_upper=len(active),n1=n1,
        logical_rotation_calls=p.nonzero_rotation_calls,required_keys_dense_case=list(p.required_keys),
        weighted_giant_groups=len(p.giant_indices),ciphertext_depth=1,
        scope='Dense-shape plan; actual public zero-diagonal inventory remains to be authenticated.')

def run():
    origins=json.loads((ROOT/'input_sources.json').read_bytes())
    for n,row in origins.items():assert sha(ROOT/'inputs'/n)==row['sha256']
    assert sha(ROOT/'inputs/candidate_manifest.json')==MANIFEST
    m=json.loads((ROOT/'inputs/candidate_manifest.json').read_bytes());c=json.loads((ROOT/'inputs/config.json').read_bytes())
    assert tuple(c[k] for k in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','vocab_size'))==(1024,3072,28,16,8,128,151936)
    profile=next(x for x in json.loads((ROOT/'inputs/profile_audit.json').read_bytes())['profiles'] if x['name']=='depth49_p60_d8')
    assert len(profile['q'])==100 and len(profile['p'])==14 and profile['normalized_refresh_output_level']==25
    norms=[norm_plan(n,r) for n,r in m['nonlinear'].items() if n.endswith('inverse_sqrt')]
    learned=json.loads((ROOT/'inputs/learned_norm_bounds.json').read_bytes())
    assert learned['passed'] and learned['model_assets']['qwen']['config_sha256']==sha(ROOT/'inputs/config.json')
    for site in norms:
        name=site['site']
        if name=='final.inverse_sqrt':tensor='model.norm.weight'
        else:
            layer,kind,_=name.split('.')
            tensor=f'model.layers.{layer}.'+({'input':'input_layernorm.weight','post':'post_attention_layernorm.weight','q':'self_attn.q_norm.weight','k':'self_attn.k_norm.weight'}[kind])
        bound=learned['normalizations']['qwen:'+tensor]
        assert not bound['centered'] and bound['width']==site['group_width']
        native=F(*map(int,bound['upper_fraction']));approximate=native*F(1000001,1000000)
        site.update(learned_weight_tensor=tensor,gamma_f64le_sha256=bound['gamma_f64le_sha256'],
            conditional_norm_output_abs_upper=upper_float(approximate),conditional_norm_output_refresh_bound=power2_guard(approximate))
    assert len(norms)==113
    counts=Counter((r['method'],r['degree']) for r in norms)
    missing=sum(r['unrefreshed_rms_depth']+2>25 for r in norms);assert missing==93
    assert sum(r['method']=='newton' for r in norms)==77
    maps={n:dense_linear_plan(*shape) for n,shape in dict(q=(2048,1024),k=(1024,1024),v=(1024,1024),o=(1024,2048),gate=(3072,1024),up=(3072,1024),down=(1024,3072)).items()}
    vocab=[dense_linear_plan(min(4096,151936-start),1024) for start in range(0,151936,4096)]
    assert len(vocab)==38 and vocab[-1]['output_width']==384
    silu=[]
    for l in range(28):
        r=m['nonlinear'][str(l)+'.silu']
        if r['method']=='polynomial':
            assert r['depth']==18
            silu.append(dict(layer=l,method=r['method'],degree=r['degree'],input24_output6=True,gated_output5_down4=True,refresh_count=0))
        else:
            assert r['reciprocal']['iterations']==5 and r['reciprocal']['depth']==13
            assert r['base']['depth']==6
            # Base24->18; refresh tanh base25, each unchanged doubling costs17->8.
            silu.append(dict(layer=l,method=r['method'],doublings=r['doublings'],base_output_level=18,
                refresh_before_each_doubling=True,refresh_count=r['doublings'],doubling_input25_output8=True,
                sigmoid7_silu6_gated5_down4=True,
                refresh_amplitude='B2 is a proposed checkpoint requiring a separate exact frozen rational envelope and HE perturbation proof.'))
    basis_bytes=sum(ct_bytes(22-2*math.ceil(math.log2(k))) for k in range(1,513))
    rounds={}
    for k in range(2,33):
        n=m['reciprocal_by_keys'][str(k)]['iterations'];p=reciprocal_plan(n,22,21)
        p.update(input_probability_refresh25=True,mask24_square23_denominator22=True,
            output_level=min(23,p['y_levels'][-1])-1,
            conditional_inverse_refresh_bound=power2_guard(F(2*k)*F(1000001,1000000)))
        assert p['output_level']>=2
        rounds[str(k)]=p
    result=dict(scope='Public architecture/manifest level and raw-storage design only; no model, validation outputs, JAX, keys or HE execution.',
        candidate_manifest_sha256=MANIFEST,source_inputs=origins,profile={k:profile[k] for k in ('name','degree','slots','depth','scale_bits','dnum','Q_count','P_count','log2_QP','actual_128_bit_QP_secure','P_covers_each_partition','canonical_scale','rotation_pair_bytes','public_key_bytes','normalized_refresh_output_level')},
        normalization=dict(sites=norms,site_count=113,sites_without_C25_two_level_reserve=missing,
            inverse_refresh_calls_per_full_norm_set=sum(len(x['refresh_events']) for x in norms),
            inverse_refresh_bound_histogram=dict(Counter(str(e['public_bound']) for x in norms for e in x['refresh_events'])),
            input_refresh_bound_histogram=dict(Counter(str(x['conditional_input_refresh_bound']) for x in norms)),
            norm_output_refresh_bound_histogram=dict(Counter(str(x['conditional_norm_output_refresh_bound']) for x in norms)),
            polynomial_degrees=dict(Counter(str(x['degree']) for x in norms if x['method']=='polynomial')),
            degree512_retained_basis_raw_bytes=basis_bytes),
        linear_maps=maps,vocabulary=dict(block_count=38,last_block_outputs=384,maps=vocab,
            logical_rotation_calls=sum(x['logical_rotation_calls'] for x in vocab),
            raw_all38_output_ciphertexts_L6_bytes=38*ct_bytes(6),
            weight_float64_bytes=151936*1024*8,
            scope='38 streamed4096-wide output maps, only last-token row live; optional compaction to3 outputs is not implemented/counted.'),
        per_sequence={str(s):dict(row_tiles=(s+15)//16,input_raw_L49_bytes=((s+15)//16)*ct_bytes(49),
            seven_linears_per_layer_calls=7*((s+15)//16),all28_linear_calls=28*7*((s+15)//16),
            all28_dense_shape_linear_rotation_calls=28*((s+15)//16)*sum(x['logical_rotation_calls'] for x in maps.values()),
            causal_nontrivial_query_branches_per_layer=s-1,
            original_unpartitioned_abstract_depth=m['depth_by_sequence'][str(s)]['final_depth']) for s in (8,16,19,32)},
        silu=silu,static_softmax_round_level_plans=rounds,
        static_initial_softmax_level_plan=dict(prefix='score mean/mask2, public map1, explicit normalized_t B2 refresh->25',
            tail='power exp basis/sum4->21, masked mean2->19, first affine18 plus five3-level updates->3, inverse B4 refresh->25, retained base product20, final1/K mask19',
            scope='Conservative physical reduction plan; exact Qwen exp positivity/rational mass envelope and all scale replay remain to implement.'),
        softmax={str(l):dict(m['softmax'][str(l)],initial_inverse_updates=6,
            per_K_original_updates={k:v['iterations'] for k,v in m['reciprocal_by_keys'].items()}) for l in range(28)},
        limitations=['No final key union until Qwen16-head/D128/GQA/causal routing is implemented; BERT130-key inventory is not inherited.',
            'Every inserted refresh changes the noisy circuit: ideal continuation is not an error certificate.',
            'Public bounds are conditional on original declared domains and same-input variance; no refreshed perturbation allowance.',
            'Scale replay must use actual binary64 ordering and actual returned refresh scale; no scale relabeling.',
            'Raw byte sums exclude controls, temporaries, compiler/backend allocations, host/device duplicates and ciphertext transport.',
            'Full prefill/decode/file handlers and all151936 encrypted logits remain unimplemented/unproved.'])
    return result

if __name__=='__main__':
    result=run();(ROOT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(sites=113,gap_sites=93,inverse_refresh_calls=result['normalization']['inverse_refresh_calls_per_full_norm_set'],
        inverse_bounds=result['normalization']['inverse_refresh_bound_histogram'],raw_basis_bytes=result['normalization']['degree512_retained_basis_raw_bytes'],vocab_rotations=result['vocabulary']['logical_rotation_calls'])))
