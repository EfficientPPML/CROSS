import math
import pytest
import plan

@pytest.mark.parametrize('K',range(1,129))
def test_public_schedule_preserves_exact_counts_and_reserve(K):
    s=plan.schedule(K);n=plan.MANIFEST['normalization_reciprocal_by_public_K'][str(K)]['iterations']
    assert s['updates']==n and len(s['rounds'])==5
    for row in s['rounds']:
        p=row['plan'];assert p['consumed_levels']==n+5
        assert row['refresh_before']==(row['source_level']<n+7)
        assert p['levels']['probabilities']>=2
        assert abs(p['output_relative_scale_error'])<1e-12
    assert s['total_refreshes_per_query_tile']==(3 if K==1 else 5)
    assert s['tail_levels']==dict(value_mixing=21,output_dense_residual=20,attention_norm=9)
    assert s['post_refresh_mask'] and s['tail_entry_level']==24

def test_actual_noncanonical_initial_and_round_metadata_not_assigned_delta():
    p=plan.starting_scales()['initial'];assert p['final_scale'].hex()=='0x1.fd402a0ff4fffp+59'
    s=plan.schedule(128)
    assert [r['plan']['levels']['probabilities'] for r in s['rounds']]==[5,7,7,7,7]
    assert s['rounds'][0]['plan']['scales']['probabilities'].hex()=='0x1.fd402a0ff6f3ep+59'
    assert s['rounds'][1]['plan']['scales']['probabilities'].hex()=='0x1.fd402a0ff677dp+59'

def test_exact_independent_key_set_union_and_operation_counts():
    k=plan.inventory();groups=k['groups']
    independent=set().union(*map(set,groups.values()))
    assert set(k['all'])==independent and k['all_count']==129 and k['graph_count']==91
    assert set(groups['softmax'])<=set(groups['attention_qk_pv'])
    assert set(k['graph'])-set(groups['attention_qk_pv'])=={63488,64064}
    assert plan.operation_counts(128)['nonbootstrap_total']==dict(rotations=73120,ct_products=3336,pt_products=106848,rescales=8432)
    assert plan.operation_counts(128)['bounded_meta10_refresh_calls']==40

@pytest.mark.parametrize('direction',(-1,1))
def test_refresh_output_adjacent_metadata_is_replayed(direction):
    scale=plan.DELTA
    for _ in range(4):
        scale=math.nextafter(scale,math.inf if direction>0 else -math.inf)
        s=plan.schedule(128,refresh_scale=scale)
        assert s['rounds'][1]['plan']['input_scale']==scale
        assert plan.tail_scales(scale)['output_level']==10

def test_bound2_headroom_and_denominator_error_are_only_conditional():
    capacity=math.prod(plan.Q[:2])/plan.DELTA/2
    assert .5/64<1/64<capacity
    # Triangle/Cauchy bounds do not require refreshed entries to stay positive.
    K=128;epsilon=.001;mass=1.;norm=1.
    lower=(mass-K*epsilon)**2/K;upper=(norm+math.sqrt(K)*epsilon)**2
    assert lower>.5/K and upper<2.
    assert lower<1/K # The domain check alone is not an accuracy guarantee.

def test_explicit_final_mask_is_one_real_level_and_retains_capacity():
    assert plan.tail_scales(plan.DELTA,post_refresh_mask=True)['output_level']==9
    assert plan.tail_scales(plan.DELTA)['output_level']==10

def test_pinned_matrices_and_dense_payload_budget():
    import json
    m=json.loads((plan.HERE/'matrix_inventory.json').read_text())
    for row in m['tensors'].values():
        assert row['active_diagonals']==1535 and row['n1']==64 and row['giant_groups']==24
        assert row['rotation_calls']==86 and row['rotation_keys']==[1,64,64064]
    r=plan.resources(plan.inventory())
    assert r['dense_64_baby_payload_at_top_bytes']==int(6.25*2**30)
    assert r['raw_rotation_GiB']==114.890625
