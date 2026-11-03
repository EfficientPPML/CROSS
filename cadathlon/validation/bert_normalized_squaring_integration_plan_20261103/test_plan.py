"""Public-only checks of scheduling contracts, not encrypted execution."""
import json, math
from pathlib import Path
import plan
LAB=Path(__file__).resolve().parent

def test_public_counts_and_inactive_midpoints():
    baseline=json.loads((LAB/'inputs/baseline_manifest.json').read_text())
    schedule=plan.PublicSchedule(baseline['power_coefficients_in_t_2x'])
    certificate=json.loads((LAB/'inputs/report.json').read_text())
    assert certificate['contracts_checked']==640 and certificate['passed']
    assert [schedule.stage(128,j)['iterations'] for j in range(5)]==[8,9,12,13,13]
    assert [schedule.stage(15,j)['iterations'] for j in range(5)]==[8,9,10,10,10]
    for K in range(1,129):
        for j in range(5):
            c=schedule.stage(K,j)
            assert c['a']<=c['inactive_dummy']<=c['b']

def test_actual_profiles_and_all_branch_reserves():
    report=json.loads((LAB/'report.json').read_text())
    profiles=report['profiles']
    assert [(p['Q_count'],p['P_count'],p['actual_partitions']) for p in profiles]==[(100,24,4),(106,17,6),(106,10,12),(108,9,14)]
    for p in profiles:
        assert p['qp_is_secure'] and p['P_covers_each_partition']
        assert p['QP_bit_length']<=3523
        assert math.prod(p['p'])>=max(math.prod(p['q'][i:i+p['alpha']]) for i in range(0,p['Q_count'],p['alpha']))
        for row in p['all_K_schedules']:
            if not row['feasible']:continue
            for rnd in row['normalized_rounds']:
                assert rnd['output_level']>=2
                assert abs(rnd['product_relative_canonical_difference'])<=1e-12
                assert rnd['minimum_nonzero_initial_coefficient_scaled']>=.5
                for event in rnd['events']:
                    if event['op'].startswith('refresh_'):assert event['input_level']>=2
            for event in row['continuation']:
                if event['op'].startswith('refresh_'):assert event['input_level']>=2
            assert row['block_output_level']>=2
    assert profiles[0]['summary']['feasible_K']==list(range(1,38))
    assert all(len(p['summary']['feasible_K'])==128 for p in profiles[1:])

def test_saved_branch_and_inverse_product_scale():
    report=json.loads((LAB/'report.json').read_text())
    baseline=json.loads((LAB/'inputs/baseline_manifest.json').read_text())
    schedule=plan.PublicSchedule(baseline['power_coefficients_in_t_2x'])
    c=schedule.stage(128,4)
    p=report['profiles'][1]
    assert plan.attempt_round(p,2,c,False,True) is None
    assert plan.attempt_round(p,4,c,False,True) is None
    assert plan.attempt_round(p,5,c,False,True)['output_level']==2
    assert plan.attempt_round(p,2,c,True,True)['output_level']==2
    assert plan.attempt_round(report['profiles'][0],25,c,False,True) is None
    assert plan.attempt_round(report['profiles'][3],29,c,False,True)['output_level']==3

def test_profile29_continuation_and_resource_scope():
    report=json.loads((LAB/'report.json').read_text())
    p28,p29=report['profiles'][2:]
    r28=p28['summary']['K128'];r29=p29['summary']['K128']
    assert r28['normalization_refreshes']==7 and r29['normalization_refreshes']==6
    assert any(e['op']=='refresh_FFN_residual' for e in r28['continuation'])
    assert not any(e['op']=='refresh_FFN_residual' for e in r29['continuation'])
    assert r29['block_output_level']==2
    assert p29['resources']['graph_plus_bootstrap_count']==224
    assert p29['resources']['raw_rotation_store_bytes']==224*2*14*117*131072*4
    assert report['forbidden_calls']==[]


def test_initial_mean_inverse_rounded_affine_certificate():
    from fractions import Fraction
    certificate=plan.initial_C1_certificate()
    upper=Fraction(int(certificate['exact_final_residual_numerator']),int(certificate['exact_final_residual_denominator']))
    assert upper<=Fraction(1,10**12) and certificate['updates']==6
    assert certificate['normalization_factor']==1
