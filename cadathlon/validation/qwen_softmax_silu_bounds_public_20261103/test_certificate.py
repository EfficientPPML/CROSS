from pathlib import Path
from fractions import Fraction as F
import json,math
import pytest
from certificate import derive,reciprocal_certificate,exp_certificate,power_to_chebyshev,horner,dyadic_upper,record,unpack,exact
P=Path(__file__).resolve().parent
RAW=(P/'inputs/candidate_manifest.json').read_bytes();M=json.loads(RAW);PLAN=json.loads((P/'inputs/continuation_plan.json').read_bytes())

@pytest.fixture(scope='module')
def report():return derive(RAW,PLAN)

def clenshaw(c,x):
    b1=b2=F()
    for coefficient in reversed(c[1:]):b0=coefficient+2*x*b1-b2;b2=b1;b1=b0
    return c[0]+x*b1-b2

@pytest.mark.parametrize('n',range(13))
def test_exact_monomial_basis_identity(n):
    coefficients=[F()]*n+[F(1)];converted=power_to_chebyshev(coefficients)
    for x in (F(-1),F(-7,8),F(-1,3),F(),F(1,5),F(3,4),F(1)):
        assert clenshaw(converted,x)==x**n

@pytest.mark.parametrize('key',['initial',*[str(k) for k in range(1,33)]])
def test_exact_rounded_first_affine_and_independent_full_power_residual(key):
    rule=M['initial_reciprocal'] if key=='initial' else M['reciprocal_by_keys'][key]
    result=reciprocal_certificate(rule);a,b=map(exact,rule['domain']);w=exact(rule['rounded_first_weight']);bias=exact(rule['rounded_first_bias'])
    for x in (a,(a+b)/2,b):
        first=1-x*(w*x+bias)
        # Independent single exact power; no iterative directed-rounding code.
        final=first**(2**(rule['iterations']-1))
        assert 0<=final<=unpack(result['final_residual_upper'])<=F(1,10**12)
        assert (1-final)/x>0 and (1-final)/x<=1/x

def test_exp_actual_polynomial_bounds_use_derivative_not_ideal_exp(report):
    coefficients=list(map(exact,M['exp']['evaluation_coefficients']));r=report['exp']
    assert unpack(r['minimum'])==horner(coefficients,F(-1))
    assert unpack(r['maximum'])==horner(coefficients,F(1))
    for i in range(129):
        x=F(i,64)-1
        assert unpack(r['minimum'])<=horner(coefficients,x)<=unpack(r['maximum'])
        derivative=horner([n*coefficients[n] for n in range(1,len(coefficients))],x)
        assert derivative>=unpack(r['derivative_positive_lower'])>0

@pytest.mark.parametrize('K',[1,3,8,32])
def test_actual_shared_alpha_cancellation_in_initial_chain(K,report):
    coefficients=list(map(exact,M['exp']['evaluation_coefficients']));base=[horner(coefficients,F(2*i,K)-1) for i in range(K)]
    alpha=exact(1./K);D=alpha*sum(base);rule=M['initial_reciprocal'];inverse=exact(rule['rounded_first_weight'])*D+exact(rule['rounded_first_bias'])
    residual=1-D*inverse
    for _ in range(rule['iterations']-1):inverse=inverse*(2-D*inverse);residual=residual*residual
    probabilities=[alpha*(x*inverse) for x in base];mass=sum(probabilities)
    assert mass==1-residual and all(p>0 for p in probabilities)
    assert unpack(report['softmax_K'][K-1]['mass_domain'][0])<=mass<=1
    if K==3:assert K*alpha!=1 and mass!=(K*alpha)*(1-residual)

def test_all_fixed_domains_and_existing_amplitudes_certified(report):
    assert len(report['softmax_layers'])==28 and len(report['softmax_K'])==32
    assert all(x['refresh_bound_certified'] for x in report['softmax_layers'])
    assert all(x['existing_inverse_refresh_bound_certified'] for x in report['softmax_K'])
    assert unpack(report['initial_inverse_guarded_upper'])<=4
    assert report['initial_inverse_refresh_event']['completed_updates']==M['initial_reciprocal']['iterations']==6
    assert report['initial_inverse_refresh_event']['phase']=='after_completed_initial_reciprocal'
    for row in report['softmax_K']:
        a,b=map(exact,row['original_K_domain']);lo,hi=map(unpack,row['next_shared_square_domain'])
        assert a<=lo<=hi<=b

@pytest.mark.parametrize('name',[n for n,r in M['nonlinear'].items() if n.endswith('.silu') and r['method']=='rational_tanh'])
def test_tanh_actual_power_to_cheb_matches_exact_horner(name,report):
    row=next(r for r in report['rational_silu'] if r['site']==name);power=list(map(exact,M['nonlinear'][name]['base']['evaluation_coefficients']));cheb=list(map(unpack,row['exact_power_to_chebyshev']))
    assert sum(map(abs,cheb),F())==unpack(row['base_abs_upper'])<1
    # More distinct exact points than degree independently establish conversion.
    for i in range(14):
        x=F(2*i,13)-1
        assert horner(power,x)==clenshaw(cheb,x)
        assert abs(horner(power,x))<=unpack(row['base_abs_upper'])
    assert len(row['rounds'])==M['nonlinear'][name]['doublings'] and row['original_reciprocal_iterations']==5

@pytest.mark.parametrize('value',[F(-1),F(-7,8),F(-1,4),F(),F(1,4),F(7,8),F(1)])
def test_shared_square_exact_tanh_round(value):
    rule=next(r['reciprocal'] for n,r in M['nonlinear'].items() if n.endswith('.silu') and r['method']=='rational_tanh')
    D=1+value*value;y=exact(rule['rounded_first_weight'])*D+exact(rule['rounded_first_bias'])
    for _ in range(rule['iterations']-1):y=y*(2-D*y)
    out=2*value*y
    assert 0<y<=1/D and abs(out)<=1 and 1<=1+out*out<=2

def test_B4_counterexample_retains_existing_larger_bound(report):
    witness=report['B4_not_universal'];assert witness['B4_insufficient']
    rule=M['reciprocal_by_keys']['8'];D=F(1,8);y=exact(rule['rounded_first_weight'])*D+exact(rule['rounded_first_bias'])
    for _ in range(witness['completed_updates_before_existing_refresh']-1):y=y*(2-D*y)
    assert y>=unpack(witness['inverse_lower'])>4
    assert witness['existing_bound']==PLAN['static_softmax_round_level_plans']['8']['conditional_inverse_refresh_bound']

def test_amplitude_is_not_domain_or_perturbation_certificate(report):
    zero=[r['layer'] for r in report['softmax_layers'] if unpack(r['additive_domain_margin'])==0]
    assert zero==[3,5,12,15]
    assert all(unpack(r['normalized_t_abs_upper'])*F(9,8)<=2 for r in report['softmax_layers'])
    assert 1+F(9,8)**2>exact(2.1) # B2 headroom alone does not preserve tanh denominator domain.

@pytest.mark.parametrize('value',[F(),F(1,3),F(2,3),F(1,2**300),F(7,8)])
def test_dyadic_rounding_is_upward_and_bounded(value):
    out=dyadic_upper(value)
    assert value<=out<value+F(1,2**256)

def test_original_manifest_required():
    changed=json.loads(RAW);changed['softmax']['0']['squarings']+=1
    with pytest.raises(ValueError):derive(json.dumps(changed).encode(),PLAN)

def test_invalid_reciprocal_cannot_be_silently_certified():
    rule=dict(M['initial_reciprocal']);rule['rounded_first_weight']=-100.
    with pytest.raises(ValueError):reciprocal_certificate(rule)

def test_first_affine_only_cannot_claim_completed_reciprocal_bound():
    rule=dict(M['initial_reciprocal']);rule['iterations']=1
    with pytest.raises(ValueError):reciprocal_certificate(rule)

def test_continuation_schedule_or_amplitude_changes_rejected():
    changed=json.loads(json.dumps(PLAN));changed['static_softmax_round_level_plans']['8']['conditional_inverse_refresh_bound']=4
    with pytest.raises(ValueError):derive(RAW,changed)

def test_no_backend_or_model_imports():
    import sys
    assert not any(name in sys.modules for name in ('jax','jaxlib','torch','transformers','safetensors'))
