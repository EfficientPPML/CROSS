import copy
from fractions import Fraction
import math
import sys
import numpy as np
import pytest
from q5_checkpoint_contract import numerical_plan,all_checkpoints
from q5_actual_metadata_replay import weighted_scale,product_scale,refresh_enclosure,check_refresh,check_refreshes,client_replay
from q5_test_fixtures import independent_weighted


@pytest.mark.parametrize('level',[2,6,26,49],ids=['L2','L6','L26','L49'])
def test_binary64_recurrences_preserve_original_operation_order(level):
    profile=numerical_plan()['profile'];D=profile['q_towers'][2*level]*profile['q_towers'][2*level+1]
    for scale in (math.nextafter(profile['scale'],0.),profile['scale'],math.nextafter(profile['scale'],math.inf),math.ldexp(1.25,55)):
        assert weighted_scale(scale,level)==independent_weighted(profile,scale,level)
        other=math.nextafter(scale,math.inf)
        assert product_scale(scale,other,level)==(scale*other)/D


def test_final26_to25_enclosure_is_independently_rounded(math_fixture):
    f=math_fixture;profile=f.plan['profile'];D=profile['q_towers'][52]*profile['q_towers'][53]
    T=Fraction(profile['scale']*D);u=Fraction(1,2**53)
    low=T*(1-u)*(1-u);high=T*(1+u)*(1+u)
    # Independent closed enumeration around rounded target, then exact rational
    # filtering; candidate uses a directed walk from its rational lower end.
    target=float(T);values={target};lower=upper=target
    for _ in range(8):
        lower=math.nextafter(lower,0.);upper=math.nextafter(upper,math.inf);values.update((lower,upper))
    wanted=sorted({x/D for x in values if low<=Fraction(x)<=high})
    certificate=refresh_enclosure(26)
    assert certificate['accepted_scales']==wanted
    assert certificate['output_level']==25
    for obs in f.observations:
        assert check_refresh(obs,obs['index'])==obs['after']
        assert obs['after']['scale']in wanted


@pytest.mark.parametrize('level',[25,27,True],ids=['not25','not27','notbool'])
def test_alignment_requires_observed_level26(level):
    with pytest.raises(ValueError,match='observed refresh alignment input26'):
        refresh_enclosure(level)


@pytest.mark.parametrize('case',['index_bool','wrong_bound','missing_call','precondition_weight','restore_weight','rounded_output','late_level27','subnormal_input'],ids=['index_bool','wrong_bound','missing_call','precondition_weight','restore_weight','rounded_output','late_level27','subnormal_input'])
def test_refresh_semantic_corruptions_reach_specific_checks(math_fixture,case):
    f=math_fixture;obs=copy.deepcopy(f.observations[0]);message=''
    if case=='index_bool':obs['index']=False;message='original ordered refresh bound'
    elif case=='wrong_bound':obs['public_bound']=2.;message='original ordered refresh bound'
    elif case=='missing_call':obs['weighted_calls'].pop(0);message='actual refresh input/count'
    elif case=='precondition_weight':obs['weighted_calls'][0]['weight']=1.;message='actual original preconditioning'
    elif case=='restore_weight':obs['weighted_calls'][-1]['weight']=0.5;message='original64-alias'
    elif case=='rounded_output':
        obs['weighted_calls'][-1]['after']['scale']=math.nextafter(obs['after']['scale'],math.inf)
        obs['after']=dict(obs['weighted_calls'][-1]['after']);message='actual weighted metadata recurrence'
    elif case=='late_level27':
        final=obs['weighted_calls'][-1];final['before']['level']=27
        final['after']=dict(level=26,scale=independent_weighted(f.plan['profile'],final['before']['scale'],27),nsd=1)
        obs['after']=dict(final['after']);message='observed refresh alignment input26'
    else:
        obs['weighted_calls'][-1]['before']['scale']=sys.float_info.min/2;message='positive finite binary64'
    with pytest.raises(ValueError,match=message):check_refresh(obs,0)


@pytest.mark.parametrize('case',['partial47','reordered_rows','missing19','export_scale'],ids=['partial47','reordered_rows','missing19','export_scale'])
def test_complete_order_and_export_metadata_are_required(math_fixture,case):
    f=math_fixture;rows=copy.deepcopy(f.rows);observations=copy.deepcopy(f.observations)
    if case=='partial47':rows.pop();message='complete ordered47'
    elif case=='reordered_rows':rows[13],rows[15]=rows[15],rows[13];message='checkpoint contract differs'
    elif case=='missing19':observations.pop();message='complete ordered19'
    else:
        rows[3]['scale']=math.nextafter(rows[3]['scale'],math.inf);message='export/observed refresh metadata differs'
    with pytest.raises(ValueError,match=message):check_refreshes(observations,rows)


def test_later_event_scale_cannot_bypass_actual_scalar_replay(math_fixture):
    f=math_fixture;rows=copy.deepcopy(f.rows)
    # Refresh boundaries remain cross-consistent. Change only a later ordinary
    # event to force the full original-graph scalar replay rejection, not the
    # earlier refresh-export gate.
    rows[-1]['scale']=math.nextafter(rows[-1]['scale'],math.inf)
    check_refreshes(f.observations,rows)
    with pytest.raises(ValueError,match='independent actual metadata replay differs: scale'):
        client_replay(np,f.reference_root,f.weights,f.embeddings,rows,f.observations)
