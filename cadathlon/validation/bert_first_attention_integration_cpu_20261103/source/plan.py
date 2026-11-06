"""First BERT attention sublayer, exact public metadata/count plan only."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import json,math,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'inputs/common'))
from encrypted_bert_initial_softmax import initial_softmax_plan
from encrypted_bert_normalization_round import normalization_round_plan,NormalizationRoundContract
from encrypted_scale_planning import weighted_sum_scale,multiply_scale,drop_divisor
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_score_reduction import ScoreReductionLayout
from encrypted_row_normalization import RowLayerNorm
from encrypted_streaming_linear import RowTileLayout

PROFILE=json.loads((HERE/'inputs/profile_and_refresh.json').read_text())
P=PROFILE['profile'];Q=P['q'];DELTA=P['canonical_scale']
RAW=(HERE/'inputs/candidate_manifest.json').read_bytes();MANIFEST=json.loads(RAW)

def weighted(scale,level):
    return weighted_sum_scale([scale],DELTA,drop_divisor(Q,level))['output_scale']

def starting_scales():
    projection=weighted(DELTA,49)
    product=multiply_scale(projection,projection,drop_divisor(Q,48))
    head=weighted(product,47);scores=weighted(head,46)
    initial=initial_softmax_plan(Q[:92],input_level=45,canonical_scale=DELTA,input_scale=scores)
    return dict(projection=projection,qk_product=product,head_compaction=head,scores=scores,initial=initial)

def schedule(K,*,refresh_scale=DELTA,tail_reserve=18):
    """Only public K/level controls refresh. The reset scale is an assumption.

    Real callers must pass the returned ciphertext's actual scale into each
    copied planner, not assign refresh_scale to its metadata.
    """
    if type(K)is not int or not 1<=K<=128:raise ValueError('public integer K1..128 required')
    contract=NormalizationRoundContract(RAW,K);level=23;scale=starting_scales()['initial']['final_scale']
    rows=[];refreshes=0
    for index in range(1,6):
        refresh=level<contract.updates+7
        source_level=level;source_scale=scale
        if refresh:
            assert level>=2
            level=25;scale=refresh_scale;refreshes+=1
        item=normalization_round_plan(Q[:2*(level+1)],input_level=level,canonical_scale=DELTA,
            contract=contract,input_scale=scale)
        rows.append(dict(round=index,source_level=source_level,source_scale=source_scale,
            refresh_before=refresh,refresh_bound=2. if refresh else None,plan=item))
        level=item['levels']['probabilities'];scale=item['scales']['probabilities']
    final_refresh=level<tail_reserve
    final_probability=dict(level=level,scale=scale,refresh_for_tail=final_refresh)
    if final_refresh:
        assert level>=2
        level=25;scale=refresh_scale;refreshes+=1
        # Fixed continuation contract: every final refresh has a real public
        # mask afterward. No runtime skip/configuration flag is proposed.
        scale=weighted(scale,level);level-=1
    return dict(K=K,updates=contract.updates,rounds=rows,final_probability=final_probability,
        pre_round_refreshes=sum(r['refresh_before'] for r in rows),total_refreshes_per_query_tile=refreshes,
        tail_entry_level=level,tail_entry_scale=scale,tail_reserve=tail_reserve,post_refresh_mask=final_refresh,
        tail_levels=dict(value_mixing=level-3,output_dense_residual=level-4,attention_norm=level-15),
        refresh_scale_assumption=refresh_scale,actual_refresh_output_scale_must_be_replayed=True)

def tail_scales(probability_scale,*,post_refresh_mask=False):
    entry=25
    if post_refresh_mask:probability_scale=weighted(probability_scale,entry);entry-=1
    inverse_local=weighted(probability_scale,entry);inverse_head=weighted(inverse_local,entry-1)
    value_scale=starting_scales()['projection']
    mixed=multiply_scale(inverse_head,value_scale,drop_divisor(Q,entry-2))
    dense=weighted(mixed,entry-3)
    assert math.isclose(dense,DELTA,rel_tol=1e-12)
    # The public HEAdd facade checks this tolerance and preserves its first
    # operand's actual scale. Its small mismatch remains numerical error.
    mean=weighted(dense,entry-4);centered=dense
    assert math.isclose(mean,centered,rel_tol=1e-12)
    square=multiply_scale(centered,centered,drop_divisor(Q,entry-5));variance=weighted(square,entry-6)
    t=weighted(variance,entry-7);terms={1:(entry-8,t)}
    def power(n):
        if n not in terms:
            left=power(n//2);right=power((n+1)//2);level=min(left[0],right[0])
            terms[n]=(level-1,multiply_scale(left[1],right[1],drop_divisor(Q,level)))
        return terms[n]
    coefficients=np.polynomial.chebyshev.cheb2poly(MANIFEST['norms']['0.attention']['polynomial']['coefficients'])
    ordered=[power(i) for i in range(1,len(coefficients)) if coefficients[i]!=0]
    polynomial_level=min(x[0] for x in ordered)
    inverse=weighted_sum_scale([x[1] for x in ordered],DELTA,drop_divisor(Q,polynomial_level))['output_scale']
    normalized=multiply_scale(centered,inverse,drop_divisor(Q,polynomial_level-1))
    output=weighted(normalized,polynomial_level-2)
    return dict(probability=probability_scale,inverse_local=inverse_local,inverse_head=inverse_head,
        mixed_values=mixed,output_dense=dense,residual_add_first_operand_scale=dense,
        residual_relative_scale_difference=dense/DELTA-1,mean=mean,variance=variance,
        inverse_norm=inverse,normalized=normalized,attention_norm_output=output,
        output_level=polynomial_level-3,post_refresh_public_mask=post_refresh_mask)

def inventory():
    matrices=json.loads((HERE/'matrix_inventory.json').read_text())['tensors']
    dense=set().union(*(set(x['rotation_keys']) for x in matrices.values()))
    groups=dict(attention_qk_pv=set(BertAttentionTile.required_rotation_indices()),
        softmax=set(ScoreReductionLayout.rotation_indices()),row_norm=set(RowLayerNorm.required_rotation_indices(RowTileLayout(65536))),
        dense=dense,bootstrap=set(PROFILE['refresh_plan']['rotations']))
    graph=set().union(*(v for k,v in groups.items() if k!='bootstrap'));all_keys=graph|groups['bootstrap']
    return dict(groups={k:sorted(v) for k,v in groups.items()},graph=sorted(graph),all=sorted(all_keys),
        graph_count=len(graph),all_count=len(all_keys),raw_rotation_bytes=len(all_keys)*PROFILE['resources']['rotation_pair_bytes'])

def operation_counts(K):
    n=NormalizationRoundContract(RAW,K).updates
    tiles=math.ceil(K/16);dense_maps=16+2*tiles
    # Actual primitive calls, not critical-path levels; refresh internals are
    # deliberately excluded. Weighted_sum performs one rescale per group.
    stages={
        'dense_maps':dict(rotations=dense_maps*86,ct_products=0,pt_products=dense_maps*1535,rescales=dense_maps*24),
        'query_tiles_QK_and_PV':dict(rotations=2*tiles*4328,ct_products=2*tiles*128,pt_products=2*tiles*3584,rescales=2*tiles*384),
        'initial_softmaxes':dict(rotations=tiles*28,ct_products=tiles*18,pt_products=tiles*16,rescales=tiles*26),
        'normalization_rounds':dict(rotations=5*tiles*14,ct_products=5*tiles*2*n,pt_products=5*tiles*3,rescales=5*tiles*(2*n+3)),
        'final_refresh_output_masks':dict(rotations=0,ct_products=0,pt_products=tiles,rescales=tiles),
        'attention_layernorms':dict(rotations=tiles*42,ct_products=tiles*13,pt_products=tiles*16,rescales=tiles*18)}
    total=Counter()
    for item in stages.values():total.update(item)
    return dict(public_valid_query_count=K,query_tiles=tiles,key_value_tiles=8,dense_map_count=dense_maps,
        stages=stages,nonbootstrap_total=dict(total),bounded_meta10_refresh_calls=tiles*schedule(K)['total_refreshes_per_query_tile'],
        scope='Public valid-query compaction; still all8 K/V tiles and128 key routes per query tile. Refresh internals, level drops, encoding, additions and I/O excluded.')

def resources(keys):
    def ct(level):return 131072*2*4*2*(level+1)
    return dict(rotation_key_pair_bytes=PROFILE['resources']['rotation_pair_bytes'],
        raw_rotation_GiB=keys['raw_rotation_bytes']/2**30,relinearization_pair_bytes=PROFILE['resources']['relinearization_pair_bytes'],
        public_key_bytes=PROFILE['resources']['public_key_bytes'],secret_key_client_only_bytes=PROFILE['resources']['secret_key_bytes'],
        ciphertext_bytes_by_level={str(L):ct(L) for L in (49,48,45,25,24,23,22,21,20,10,9,7,5)},
        all_X_Q_K_V_payload_bytes=8*ct(49)+24*ct(48),saved_K_V_plus_one_X_Q_bytes=16*ct(48)+ct(49)+ct(48),
        dense_64_baby_payload_at_top_bytes=64*ct(49),all8_final_output_bytes=8*ct(9),
        caution='Raw payload/key accounting only: format copies, compiler/NTT controls, mmap residency, pending groups and bootstrap workspace are additional. No peak-memory or runtime claim.')

def report():
    assert math.prod(Q+P['p']).bit_length()<=3523
    alpha=math.ceil(len(Q)/P['dnum'])
    assert math.prod(P['p'])>=max(math.prod(Q[i:i+alpha]) for i in range(0,len(Q),alpha))
    keys=inventory();schedules=[schedule(K) for K in range(1,129)]
    near=[]
    for sign in (-1,1):
        value=DELTA
        for ulp in range(1,5):
            value=math.nextafter(value,math.inf if sign>0 else -math.inf)
            near.append(dict(input_scale=value,input_hex=value.hex(),ULP=sign*ulp,
                K128=schedule(128,refresh_scale=value),tail=tail_scales(value)))
    base_capacity=math.prod(Q[:2])/DELTA/2
    return dict(passed=True,scope=__doc__,execution='Public NumPy/binary64 metadata only; no JAX, model forward, context, keys, ciphertext or HE.',
        profile=P,profile_refresh=PROFILE['refresh_plan'],initial_path=starting_scales(),K1=schedules[0],K15=schedules[14],K128=schedules[-1],
        all_K=schedules,refresh_adjacent_scale_probes=near,tail_scale_example=tail_scales(DELTA,post_refresh_mask=True),
        key_inventory=keys,operations_K128=operation_counts(128),operations_K15=operation_counts(15),resources=resources(keys),
        mandatory_post_refresh_mask=dict(tail_reserve=18,tail_scale_example=tail_scales(DELTA,post_refresh_mask=True),
            levels={'refreshed_probability':25,'public_mask':24,'value_mixing':21,'output_dense_residual':20,'attention_norm':9},
            counted_pt_products=8,counted_rescales=8,
            reason='Explicitly suppress invalid-key/head/row noise after final refresh; PV itself has no public key mask. Ideal refresh preserves zeros, but no exact noisy-zero guarantee is assumed.'),
        probability_refresh=dict(public_bound=2.,engineering_margin_only=True,fullslot_B2_measured=False,
            input_reserve=2,output_level=25,period=65536,aliases=1,meta_precision=10,
            ideal_max_before_precondition=1.,ideal_max_after_precondition=.5,
            actual_p60_adjustment_divisor=64,ideal_base_max=.5/64,full_bound_base_max=1/64,
            base_signed_capacity=base_capacity,restored_error_multiplier=2.,
            caveat='Actual encoding/input/refresh errors and repeated nonlinear domain coverage remain unproved; B16/B128 evidence does not certify B2.'),
        no_backend=not any(x.split('.')[0] in ('jax','jaxlib','torch','transformers') for x in sys.modules))

if __name__=='__main__':
    result=report();assert result['no_backend']
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','scope','execution','operations_K128','resources','probability_refresh')},indent=2))
