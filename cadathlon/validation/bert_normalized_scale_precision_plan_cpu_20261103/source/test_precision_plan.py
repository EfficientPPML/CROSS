import math
import pytest
import precision_plan as p

@pytest.fixture(scope='module')
def inputs():return p.public_inputs()

@pytest.mark.parametrize('K',range(1,129))
def test_gain1_matches_unchanged_frozen_plan(inputs,K):
    ring,raw,_=inputs;c=p.NormalizationRoundContract(raw,K)
    old=p.normalization_round_plan(ring['q'][:52],input_level=25,canonical_scale=ring['canonical_scale'],contract=c)
    new=p.plan(ring,raw,K,1)
    assert new['output_scale']==old['scales']['probabilities']
    assert new['mask']['output_scale']==old['scales']['masked_input']
    assert new['numerator_scale']==old['scales']['squared_numerator']
    assert new['anchors']['output_scale']==old['scales']['denominator']
    assert new['inverse']=={k:v for k,v in old['tracked_inverse_scale_plan'].items()}
    assert new['counts']==dict(ciphertext_products=old['logical_ciphertext_products'],plaintext_products=old['logical_plaintext_products'],rotations=old['logical_rotations'],composite_rescales=old['actual_composite_rescales'],critical_levels=old['consumed_levels'])

@pytest.mark.parametrize('K',range(1,129))
@pytest.mark.parametrize('gain',(2,4,8))
def test_literal_two_branch_binary64_replay(inputs,K,gain):
    ring,raw,_=inputs;row=p.plan(ring,raw,K,gain);q=ring['q'];delta=ring['canonical_scale']
    def drop(L):return q[2*L]*q[2*L+1]
    for stage in (row['mask'],row['anchors']):
        assert stage['output_scale']==stage['input_scale']*stage['plaintext_scale']/drop(stage['input_level'])
    x=row['anchors']['output_scale'];y=row['inverse']['rows'][0]['scale'];L=21
    e=x*y/drop(L);eL=L-1
    for i,r in enumerate(row['inverse']['rows']):
        assert r['scale']==y
        if 'residual_scale'in r:assert r['residual_scale']==e
        if i+1<len(row['inverse']['rows']):
            y=y*e/drop(eL)
            if i+2<len(row['inverse']['rows']):e=e*e/drop(eL);eL-=1
    assert row['output_scale']==row['numerator_scale']*y/drop(row['inverse_level'])
    assert abs(row['output_relative_scale_error'])<1e-12
    baseline=p.plan(ring,raw,K,1)
    for actual,original in zip(row['inverse']['rows'],baseline['inverse']['rows']):
        assert actual['scale']*(gain*gain)==original['scale']
        if 'residual_scale'in actual:assert actual['residual_scale']==original['residual_scale']
    assert row['minimum_y_scale_over_canonical']*(gain*gain)==baseline['minimum_y_scale_over_canonical']
    assert row['output_level']>=2

@pytest.mark.parametrize('gain',(True,0,-1,3,16,2.0,float('nan')))
def test_unreviewed_gain_rejected(inputs,gain):
    with pytest.raises(ValueError):p.plan(*inputs[:2],128,gain)

def test_complete_public_inventory_and_no_accuracy_claim():
    r=p.derive();assert len(r['schedules'])==512 and len(r['sensitivity'])==16
    assert r['security']['actual_QP_bit_length']<=r['security']['classical128_ceiling']
    assert all('assumption'in x for x in r['sensitivity'])
    assert any('input ciphertext'in x for x in r['limitations'])
