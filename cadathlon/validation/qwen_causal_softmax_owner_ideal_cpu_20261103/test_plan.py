import ast,json,math,sys
from collections import Counter
from pathlib import Path
import pytest
from plan_public import inputs,branch_plan,derive,DELTA,ROOT

@pytest.fixture(scope='module')
def data():return inputs()

@pytest.mark.parametrize('layer',range(28))
def test_every_original_K_rule_and_event(data,layer):
    m,c,old,p=data
    for k in range(1,33):
        x=branch_plan(m,c,p,layer,k,refresh_mode='alternating_ulp')
        if k==1:
            assert x['nonlinear_bypass'] and not x['refreshes'] and x['output_level'] is None
            continue
        n=m['reciprocal_by_keys'][str(k)]['iterations'];s=m['softmax'][str(layer)]['squarings']
        assert x['original_round_iterations']==n and x['squarings']==s
        assert x['initial_inverse_levels']==[18,15,12,9,6,3]
        for row in x['rounds']:
            assert row['inverse_levels']==[21,18,15,12,9,6]+list(range(19,19-3*(n-6),-3))
            assert row['probability_output_level']==old['static_softmax_round_level_plans'][str(k)]['output_level']
        events=x['refreshes'];assert len(events)==3+2*s
        assert events[1]['label']=='completed_initial_inverse.refresh' and events[1]['completed_updates']==6
        assert (events[1]['input_level'],events[1]['public_bound'])==(3,4)
        inverse=[e for e in events if 'before7' in e['label']]
        assert len(inverse)==s and all((e['input_level'],e['completed_updates'],e['public_bound'])==(6,6,c['softmax_K'][k-1]['existing_inverse_refresh_bound']) for e in inverse)
        assert all(e['input_level']>=2 for e in events)
        assert x['output_level']==24 and math.isclose(x['output_scale'],DELTA,rel_tol=1e-12)

def test_literal_frozen_original_inverse_uses_three_stages(data):
    tree=ast.parse((ROOT/'inputs/frozen_arithmetic.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Arithmetic')
    inv=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='inverse')
    loop=next(n for n in inv.body if isinstance(n,ast.For))
    methods=[n.value.func.attr for n in loop.body if isinstance(n,ast.Assign)]
    assert methods==['multiply','affine','multiply']
    qwen=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenArithmetic')
    attention=next(n for n in qwen.body if isinstance(n,ast.FunctionDef) and n.name=='causal_attention')
    rowloop=next(n for n in attention.body if isinstance(n,ast.For))
    bypass=rowloop.body[0]
    assert isinstance(bypass,ast.If) and ast.unparse(bypass.test)=='row == 0'
    assert isinstance(bypass.body[-1],ast.Continue)
    assert 'projected.data' in ast.unparse(bypass)

def test_no_second_exp_map_and_frozen_coefficient_bytes(data):
    m,c,_,p=data;x=branch_plan(m,c,p,0,8)
    names=[s['label'] for s in x['stages']]
    a=names.index('normalized_t.refresh');b=names.index('base_exp')
    assert all(s['kind']=='multiply' for s in x['stages'][a+1:b])
    assert b-a-1==7
    assert x['exp_evaluation_coefficients_hex']==c['exp']['actual_power_hex']
    assert x['rounded_alpha_hex']==float(1/8).hex()

@pytest.mark.parametrize('k',[2,3,5,10,19,32])
@pytest.mark.parametrize('layer',[0,3])
def test_counts_by_independent_literal_accounting(data,k,layer):
    m,c,_,p=data;x=branch_plan(m,c,p,layer,k)
    n=m['reciprocal_by_keys'][str(k)]['iterations'];rounds=m['softmax'][str(layer)]['squarings']
    # Prefix3 + exp sum1 + initial mean2 + inverse6 + probability1 + final mask1.
    assert x['counts']['weighted_sums']==14+rounds*(1+1+n)
    # Seven exp powers, ten initial inverse products, one quotient; square+2(n-1)+quotient each round.
    assert x['counts']['ciphertext_products']==18+rounds*(1+2*(n-1)+1)
    assert x['counts']['rotations']==2*10+rounds*10
    assert x['counts']['ordinary_additions']==2*10+1+rounds*10

def test_heterogeneous_tiles_and_total_cost():
    r=derive();by={x['sequence']:x for x in r['workloads']}
    assert by[19]['K_by_tile']==[list(range(1,17)),[17,18,19]]
    assert by[32]['K_by_tile']==[list(range(1,17)),list(range(17,33))]
    assert by[1]['total28_refreshes']==0
    assert by[32]['total28_refreshes']==31*(22*13+6*15)==11656
    assert sum(r['refresh_histogram_all28_S32'].values())==11656
    assert r['retained_exp_basis_raw_bytes']==382*1024**2
    assert r['keys']['combined_component_count']==162
    assert r['keys']['raw_rotation_key_bytes']==162*912*1024**2
    assert r['exp_zero_margin_layers']==[3,5,12,15]

@pytest.mark.parametrize('layer,k',[(True,2),(1.,2),(0,True),(0,2.),(-1,2),(28,2),(0,0),(0,33)])
def test_typed_public_domain_rejection(data,layer,k):
    m,c,_,p=data
    with pytest.raises(ValueError):branch_plan(m,c,p,layer,k)

def test_no_model_or_HE_backend_imported():
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))
