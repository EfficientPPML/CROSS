"""Independent public algebra, level joins and pinned metadata checks."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import plan


@pytest.fixture(scope='module')
def report(): return plan.derive()


def test_unchanged113_rules_and_all_refresh_counts(report):
    assert report['candidate_manifest_sha256']==plan.MANIFEST
    assert len(report['normalizations'])==113
    source=json.loads((plan.ROOT/'inputs/rms_plan.json').read_bytes())
    prior={r['site']:r for r in source['sites']}
    for row in report['normalizations']:
        original=prior[row['site']]
        assert (row['method'],row['degree'],row['iterations'])==(original['method'],original['degree'],original['iterations'])
        assert row['output_level']==original['scale_example']['output_level']
        assert row['internal_inverse_refreshes']==len(original['events'])
    assert sum(r['internal_inverse_refreshes'] for r in report['normalizations'])==260
    assert [r['total_conditional_refreshes'] for r in report['workloads']]==[638,3270,6278,8043,12931]


@pytest.mark.parametrize('layer',range(28))
def test_every_layer_scale_and_capacity(report,layer):
    for example in report['metadata_examples']:
        row=example['layers'][layer]
        assert [row[key]['level'] for key in ('query','expanded_key','expanded_value','scores',
            'probabilities','attended','attention_output','silu','gated','down','layer_output')]==[25,24,23,21,24,20,19,6,5,4,4]
        assert all(math.isfinite(row[key]['scale']) and row[key]['scale']>0 for key in
                   ('query','expanded_key','expanded_value','scores','probabilities','attended','layer_output'))
        assert all(e['input_level']>=2 and e['output_level']==25 for e in row['nonsoftmax_refresh_events'])
        assert all(r['compatible'] for r in example['residual_additions'])
        assert example['final_norm']['level']==7
        assert example['all38_vocabulary_blocks']['level']==6


def test_literal_prior_refresh_placement_is_one_level_short(report):
    problem=report['incompatibilities']['refresh_before_rope_only']
    q=25-1;k=25-1-1
    score=min(q,k-1)-1-1
    assert (q,k,score)==(24,23,20)
    assert score<problem['required_score_level']==21
    expected={'5.q','7.q','12.q','15.q','22.q','23.q','24.q','25.q','9.k'}
    assert {s.removesuffix('.inverse_sqrt') for s in report['incompatibilities']['low_norm_rope_inputs']}==expected


@pytest.mark.parametrize('index',range(56))
def test_post_rope_bound_public_gamma_identity_and_headroom(report,index):
    row=report['rope_bounds'][index]
    native=json.loads((plan.ROOT/'inputs/learned_norm_bounds.json').read_bytes())['normalizations']
    layer,kind,_=row['site'].split('.')
    prior=native[f'qwen:model.layers.{layer}.self_attn.{kind}_norm.weight']
    assert row['gamma_sha256']==prior['gamma_f64le_sha256']
    assert row['gamma_max_abs']==prior['gamma_max_abs']
    value=F(*map(int,row['exact_upper']))
    assert value*F(9,8)<=row['public_refresh_bound']
    assert 64<=row['public_refresh_bound']<=2048


@pytest.mark.parametrize('c,s',[(0,1),(1,0),(F(3,5),F(4,5)),(-F(3,5),F(4,5)),(F(1,3),F(-2,3))])
@pytest.mark.parametrize('x,y',[(1,2),(-3,4),(0,0),(5,-7)])
def test_original_pair_cauchy_without_coordinate_infinity_loss(c,s,x,y):
    c,s,x,y=map(F,(c,s,x,y));gi,gj=F(-2),F(3)
    lhs=(c*gi*x-s*gj*y)**2
    rhs=max(abs(gi),abs(gj))**2*(c*c+s*s)*(x*x+y*y)
    assert lhs<=rhs


def test_stored_coefficient_square_enclosure_exact_dyadic_spots(report):
    config=json.loads((plan.ROOT/'inputs/config.json').read_bytes())
    positions=np.array([0,1,2,7,31,1024,40959],dtype=np.int64)
    angles=positions[:,None]*config['rope_theta']**(-np.arange(0,128,2)/128)[None,:]
    cs,ss=np.cos(angles),np.sin(angles)
    bound=F(*map(int,report['rope_table']['exact_square_sum_upper']))
    for c,s in zip(cs.ravel(),ss.ravel()):
        assert F.from_float(float(c))**2+F.from_float(float(s))**2<=bound
    assert bound>1 and bound<=1+F(1,2**50)
    assert report['rope_table']['coefficient_pairs']==40960*64


def test_uniform_rounding_allowance_includes_tiny_products():
    for c,s in [(1.,math.nextafter(0.,1.)),(.5,.5),(1.,0.),(np.float64(2**-600),1.),
                (.9999999999999999,math.nextafter(0.,1.))]:
        c,s=float(c),float(s)
        exact=F.from_float(c)**2+F.from_float(s)**2
        evaluated=F.from_float((c*c)+(s*s))
        assert exact<=evaluated+4*plan.U


def test_score_bound_does_not_prove_fixed_exp_domains(report):
    assert report['all_centered_domains_proved'] is False
    rows=[r for layer in report['score_domain_outer_bounds'] for r in layer['centered_rows']]
    assert len(rows)==868 and all(not r['proves_frozen_centered_domain'] for r in rows)
    for layer in report['score_domain_outer_bounds']:
        score=F(*map(int,layer['score_upper_fraction']))
        for row in layer['centered_rows']:
            k=row['K'];alpha=F.from_float(1/k)
            assert F(*map(int,row['upper_fraction']))==(abs(1-alpha)+(k-1)*alpha)*score


def test_residual_compatibility_uses_actual_production_validation(report):
    tree=ast.parse((plan.ROOT/'inputs/he_ops.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_additive_metadata')
    ns={'math':math};exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_he_ops.py','exec'),ns)
    validate=ns['_validate_additive_metadata']
    for example in report['metadata_examples']:
        for row in example['residual_additions']:
            validate(SimpleNamespace(_ckks_scale=row['left_scale'],_ckks_nsd=1),
                     SimpleNamespace(_ckks_scale=row['right_scale'],_ckks_nsd=1),'public plan')
    with pytest.raises(ValueError):
        validate(SimpleNamespace(_ckks_scale=1.,_ckks_nsd=1),SimpleNamespace(_ckks_scale=1.000001,_ckks_nsd=1),'counterexample')


def test_actual_weighted_rounding_is_preserved():
    profile=next(r for r in json.loads((plan.ROOT/'inputs/profile_audit.json').read_bytes())['profiles'] if r['name']=='depth49_p60_d8')
    replay=plan.Replay(profile['q'],profile['canonical_scale'],'canonical')
    for level in (2,3,6,19,20,23,24,25,38):
        for scale in (profile['canonical_scale'],math.nextafter(profile['canonical_scale'],math.inf)):
            actual=replay.weighted('probe',dict(level=level,scale=scale))
            divisor=profile['q'][2*level]*profile['q'][2*level+1]
            assert actual==dict(level=level-1,scale=(scale*((profile['canonical_scale']*divisor)/scale))/divisor)
    with pytest.raises(ValueError,match='level >=2'):
        replay.refresh('insufficient',dict(level=1,scale=profile['canonical_scale']),2,category='test')


def test_no_model_backend_import_in_fresh_process():
    code="import sys;import plan;assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))"
    subprocess.run([sys.executable,'-c',code],cwd=plan.ROOT,check=True,timeout=20)


def test_resources_are_conditional_not_full_inventory(report):
    assert report['resources']['key_count']==166
    assert len(set(report['resources']['preliminary_dense_case_key_union']))==166
    assert report['resources']['all38_L6_logit_ciphertext_bytes']==557842432
    assert len(report['limitations'])==7
