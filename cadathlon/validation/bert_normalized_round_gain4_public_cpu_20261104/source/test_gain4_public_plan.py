"""Actual input-scale continuation, compared to the unchanged predecessor."""
import math
import pytest
from test_gain4_ideal import Q,D,RAW,g,NormalizationRoundContract
from encrypted_bert_normalization_round import normalization_round_plan

@pytest.mark.parametrize('K',range(1,129))
@pytest.mark.parametrize('incoming',(math.nextafter(D,0.),D,math.nextafter(D,math.inf),D*(1-2e-13),D*(1+2e-13)))
def test_actual_scale_replay_preserves_residual_and_real_counts(K,incoming):
    contract=NormalizationRoundContract(RAW,K)
    base=normalization_round_plan(Q[:52],input_level=25,canonical_scale=D,contract=contract,input_scale=incoming)
    new=g.gain4_round_plan(Q[:52],input_level=25,canonical_scale=D,contract=contract,input_scale=incoming)
    for key in ('logical_ciphertext_products','logical_plaintext_products','logical_rotations','actual_composite_rescales','levels'):
        assert new[key]==base[key]
    assert new['scales']['masked_input']==4*base['scales']['masked_input']
    for key in ('squared_numerator','denominator'):assert new['scales'][key]==16*base['scales'][key]
    for a,b in zip(new['tracked_inverse_scale_plan']['rows'],base['tracked_inverse_scale_plan']['rows']):
        assert a['scale']*16==b['scale']
        if 'residual_scale'in a:assert a['residual_scale']==b['residual_scale']
    assert new['scales']['probabilities']==base['scales']['probabilities']
    assert math.isclose(new['scales']['probabilities'],D,rel_tol=1e-12)
