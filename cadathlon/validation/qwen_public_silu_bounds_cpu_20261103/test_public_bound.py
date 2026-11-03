from fractions import Fraction
import math
import numpy as np
import pytest
from public_bound import weighted_rms_gate_bound, sqrt_upper


def test_exact_squared_bound_and_outward_sqrt():
    w=np.array([.1, -.3, 2., 0.],dtype=np.float32)
    g=np.array([1.3, .5, -.25, 7.],dtype=np.float32)
    result=weighted_rms_gate_bound(w,g)
    exact=4*sum((Fraction.from_float(float(a))*Fraction.from_float(float(b)))**2 for a,b in zip(w,g))
    assert result['squared_native_bound_fraction']==[str(exact.numerator),str(exact.denominator)]
    upper=result['native_exact_real_upper']
    assert Fraction.from_float(upper)**2>=exact
    if upper:assert Fraction.from_float(math.nextafter(upper,-math.inf))**2<exact


def test_arbitrary_input_and_tight_cauchy_direction():
    w=np.array([.25,-.5,1.,2.],dtype=np.float32);g=np.array([1.,2.,.5,-1.],dtype=np.float32)
    bound=weighted_rms_gate_bound(w,g)['native_exact_real_upper']
    direction=w.astype(np.float64)*g
    for x in (direction, -direction, np.array([1e4,-2e3,0.,7.]), np.zeros(4)):
        for eps in (1e-6,.1,10.):
            gate=np.dot(direction,x)/np.sqrt(np.mean(x*x)+eps)
            assert abs(gate)<=bound
    assert abs(np.dot(direction,direction)/np.sqrt(np.mean(direction*direction)+1e-12)-bound)<1e-10


def test_uniform_multiplier_is_rounded_outward_and_zero_is_exact():
    w=np.array([.1,.2],dtype=np.float32);g=np.array([.3,.4],dtype=np.float32);delta=Fraction(1,10**6)
    result=weighted_rms_gate_bound(w,g,relative_inverse_error=delta)
    assert Fraction.from_float(result['conditional_polynomial_upper'])>=Fraction.from_float(result['native_exact_real_upper'])*(1+delta)
    assert weighted_rms_gate_bound(np.zeros(2,dtype=np.float32),g)['native_exact_real_upper']==0.


@pytest.mark.parametrize('w,g,delta',[
    (np.ones(2,dtype=np.float64),np.ones(2,dtype=np.float32),Fraction(0)),
    (np.ones(2,dtype=np.float32),np.ones(3,dtype=np.float32),Fraction(0)),
    (np.array([np.nan],dtype=np.float32),np.ones(1,dtype=np.float32),Fraction(0)),
    (np.ones(2,dtype=np.float32),np.ones(2,dtype=np.float32),1e-6),
    (np.ones(2,dtype=np.float32),np.ones(2,dtype=np.float32),Fraction(-1,10)),
])
def test_unsupported_public_contract_rejected(w,g,delta):
    with pytest.raises((TypeError,ValueError)):weighted_rms_gate_bound(w,g,relative_inverse_error=delta)
