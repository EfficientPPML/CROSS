from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import json,math
import numpy as np
import numeric
P=Path(__file__).resolve().parent
M=json.loads((P/'candidate-v1/manifest.json').read_text())

def test_domains_derive_only_from_frozen_calibration_inputs():
    rows=json.loads((P/'selection_inputs.json').read_text())
    assert len(M['nonlinear'])==141 and len(M['softmax'])==28
    for label,rule in M['nonlinear'].items():
        r=rows[label];lo,hi=r['minimum'],r['maximum']
        expected=[.8*lo,1.2*hi] if label.endswith('inverse_sqrt') else [math.floor(lo-.1*(hi-lo)),math.ceil(hi+.1*(hi-lo))]
        assert rule['domain']==expected
    for layer,rule in M['softmax'].items():
        r=rows[layer+'.softmax.centered'];lo,hi=r['minimum'],r['maximum']
        assert rule['centered_domain']==[math.floor(lo-.1*(hi-lo)),math.ceil(hi+.1*(hi-lo))]
        assert max(map(abs,rule['centered_domain']))/2**rule['squarings']<=.5

def test_every_selected_explicit_polynomial_or_fallback_meets_unchanged_grid_target():
    for label,r in M['nonlinear'].items():
        grid=np.linspace(*r['domain'],16385)
        if r['method']=='polynomial':actual=numeric.polynomial(grid,r)
        elif r['method']=='newton':actual=numeric.rsqrt(grid,r)
        else:actual=numeric.rational_silu(grid,r)
        assert np.isfinite(actual).all()
        if label.endswith('inverse_sqrt'):assert np.max(np.abs(actual*np.sqrt(grid)-1))<=1e-6,label
        else:assert np.max(np.abs(actual-grid*np.exp(-np.logaddexp(0,-grid))))<=1e-4,label

def test_frozen_basis_and_balanced_selection_policy():
    for r in M['nonlinear'].values():
        if r['method']!='polynomial':continue
        c=np.asarray(r['chebyshev_coefficients']);l1=np.abs(np.polynomial.chebyshev.cheb2poly(c)).sum()
        assert np.array_equal(numeric.basis_coefficients(c,r['strategy']),r['evaluation_coefficients'])
        if r['kind']=='inverse_sqrt':assert r['strategy']==('power' if r['degree']<=32 and l1<=1e8 else 'chebyshev')
        assert r['depth']==numeric.polynomial_depth(r)

def test_newton_includes_rounded_first_affine_residual_and_cost():
    for r in M['nonlinear'].values():
        if r['method']!='newton':continue
        a,b,w,bias=map(F.from_float,(*r['domain'],r['rounded_first_weight'],r['rounded_first_bias']))
        points=[a,b]+[x for x in (-bias/w,-bias/(3*w)) if a<=x<=b]
        R=max(abs(1-x*(w*x+bias)**2) for x in points);cert=r['first_exact_residual_abs_bound']
        assert R==F(int(cert['numerator']),int(cert['denominator']))<1
        assert min(w*a+bias,w*b+bias)>0
        assert Decimal(r['final_relative_error_bound'])<=Decimal('1e-6')
        assert r['depth']==4*r['iterations']-3
        for j in range(65):
            x=a+(b-a)*F(j,64);assert abs(1-x*(w*x+bias)**2)<=R

def test_all_causal_key_contracts_and_no_coefficient_rounding_of_counts():
    saved=json.loads((P/'reciprocal_schedule.json').read_text())
    assert set(M['reciprocal_by_keys'])=={str(k) for k in range(1,33)}
    for old in saved['round_contracts']:
        r=M['reciprocal_by_keys'][str(old['valid_keys'])]
        for key in ('domain','initial_estimate','rounded_first_weight','rounded_first_bias','iterations'):assert r[key]==old[key]
        assert r['depth']==3*r['iterations']-2
        assert Decimal(r['final_residual_bound'])<=Decimal('1e-12')

def test_rational_tanh_intermediates_stay_inside_declared_reciprocal_domain():
    for r in M['nonlinear'].values():
        if r['method']!='rational_tanh':continue
        x=np.linspace(*r['domain'],16385);t=numeric.polynomial(x,r['base'],2.**-(r['doublings']+1))
        for _ in range(r['doublings']):
            d=1+t*t;assert np.all((d>=1)&(d<=2.1));t=2*(t*numeric.reciprocal(d,r['reciprocal']))
        assert r['depth']==r['base']['depth']+r['doublings']*(r['reciprocal']['depth']+4)+2

def test_exp_polynomial_is_frozen_and_positive_on_declared_grid():
    r=M['exp'];x=np.linspace(-.5,.5,16385);actual=numeric.polynomial(x,r)
    assert r['degree']==8 and r['depth']==5 and np.all(actual>0)
    assert np.max(np.abs(actual-np.exp(x)))<1e-9
