from fractions import Fraction as F
import importlib.util,json,math
from pathlib import Path
import pytest
import plan

def rules():return json.loads((Path(__file__).parent/'inputs/candidate_manifest.json').read_bytes())['nonlinear']

@pytest.mark.parametrize('iterations',range(15,25))
def test_retained_variance_schedule(iterations):
    r=dict(rules()['3.input.inverse_sqrt'],iterations=iterations,depth=4*iterations-3)
    p=plan.norm_plan('3.input.inverse_sqrt',r)
    # Independently walk four real depth edges with v23/x25 retained.
    y=22;events={e['before_iteration']:e for e in p['refresh_events']}
    seen=[y]
    for i in range(2,iterations+1):
        if i in events:assert events[i]['input_level']==y;y=25
        squared=y-1;product=min(23,squared)-1;correction=product-1;y=min(y,correction)-1
        assert y>=4
        seen.append(y)
    assert seen==[s['y_level'] for s in p['iteration_levels']]
    assert p['output_level']==min(25,y)-2>=2
    assert len(events)==(iterations-1)//5

def test_degree512_terminal_refresh_is_before_original_product():
    p=plan.norm_plan('3.k.inverse_sqrt',rules()['3.k.inverse_sqrt'])
    assert p['inverse_level_before_terminal_refresh']==3 and p['output_level']==23
    assert p['refresh_events']==[dict(after='inverse_polynomial',input_level=3,output_level=25,value='inverse_polynomial_output',public_bound=p['inverse_refresh_bound'])]

def test_exact_newton_magnitude_induction():
    # Exact rational identity: 1-[z(3-z²)/2]² = (1-z²)²(4-z²)/4.
    # For 0<z²<2 the next residual is nonnegative and next iterate positive.
    for z in [F(1,1000),F(1,2),F(1),F(7,5)]:
        assert 0<z*z<2
        out=z*(3-z*z)/2
        assert 1-out*out==(1-z*z)**2*(4-z*z)/4 and 0<out<=1

@pytest.mark.parametrize('name',[n for n in rules() if n.endswith('inverse_sqrt')])
def test_every_bound_and_frozen_iteration(name):
    r=rules()[name];p=plan.norm_plan(name,r)
    assert p['inverse_refresh_bound']>=F.from_float(p['ideal_inverse_abs_upper'])*F(9,8)
    assert p['conditional_input_refresh_bound']**2>=F.from_float(r['domain'][1])*p['group_width']*F(81,64)
    assert p['output_level']>=2
    if r['method']=='newton':assert len(p['iteration_levels'])==r['iterations']

def test_shapes_and_dense_actual_public_planner():
    r=plan.run();assert r['normalization']['sites_without_C25_two_level_reserve']==93
    assert r['linear_maps']['q']['output_width']==2048 and r['linear_maps']['o']['input_width']==2048
    assert r['vocabulary']['block_count']==38 and r['vocabulary']['last_block_outputs']==384
    assert r['normalization']['degree512_retained_basis_raw_bytes']==7511998464
    assert {s:r['per_sequence'][s]['row_tiles'] for s in ('8','16','19','32')}=={'8':1,'16':1,'19':2,'32':2}

@pytest.mark.parametrize('keys',range(2,33))
def test_original_reciprocal_counts_and_physical_depth(keys):
    m=json.loads((Path(__file__).parent/'inputs/candidate_manifest.json').read_bytes())
    r=m['reciprocal_by_keys'][str(keys)];p=plan.run()['static_softmax_round_level_plans'][str(keys)]
    assert len(p['y_levels'])==r['iterations'] and len(p['refresh_events'])==1
    # Explicit public refresh changes no value in this exact-real identity check.
    events={e['before_iteration'] for e in p['refresh_events']}
    for x in (F.from_float(r['domain'][0]),F(1),F.from_float(r['domain'][1])):
        y=F.from_float(r['rounded_first_weight'])*x+F.from_float(r['rounded_first_bias'])
        z=y
        for i in range(2,r['iterations']+1):
            if i in events:y=F(y)
            product=x*y;correction=2-product;y=y*correction
            z=z*(2-x*z)
        assert y==z
    assert p['output_level']>=2

def test_gamma_amplitudes_are_not_a_B128_claim():
    r=plan.run();n=r['normalization']
    assert max(map(int,n['norm_output_refresh_bound_histogram']))==8192
    assert n['input_refresh_bound_histogram']['32768']==50
    assert max(map(int,n['inverse_refresh_bound_histogram']))==64

def test_no_backend_or_model_import():
    import sys
    assert not any(n in sys.modules for n in ('jax','torch','transformers','safetensors'))
