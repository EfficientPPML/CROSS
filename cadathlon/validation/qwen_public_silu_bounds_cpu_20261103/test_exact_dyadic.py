from fractions import Fraction
import math
import numpy as np
import pytest
import exact_dyadic


def expected(row,gamma):
    return len(gamma)*sum((Fraction.from_float(float(w))*Fraction.from_float(float(g)))**2
                          for w,g in zip(row,gamma))


def test_random_binary32_and_signed_zeros_exactly_match_independent_fractions():
    rng=np.random.default_rng(674025)
    weight=rng.normal(size=(31,37)).astype(np.float32)
    gamma=rng.normal(size=37).astype(np.float32)
    weight[0]=0.;weight[1]=-0.
    for row,(square,upper) in zip(weight,exact_dyadic.rows(weight,gamma)):
        assert square==expected(row,gamma)
        assert Fraction.from_float(upper)**2>=square


def test_subnormal_extremes_and_large_exponent_spread_remain_exact():
    f=np.finfo(np.float32)
    values=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),f.tiny,f.max,-f.max,1.,-.5],dtype=np.float32)
    weight=np.stack([values,np.roll(values,2),-values]);gamma=values[::-1].copy()
    for row,(square,upper) in zip(weight,exact_dyadic.rows(weight,gamma)):
        assert square==expected(row,gamma)
        assert math.isfinite(upper) and Fraction.from_float(upper)**2>=square


@pytest.mark.parametrize('weight,gamma',[
    (np.ones((2,3),dtype=np.float64),np.ones(3,dtype=np.float32)),
    (np.ones((2,3),dtype=np.float32),np.ones(2,dtype=np.float32)),
    (np.full((2,3),np.inf,dtype=np.float32),np.ones(3,dtype=np.float32)),
])
def test_unsupported_contracts_rejected(weight,gamma):
    with pytest.raises(ValueError):list(exact_dyadic.rows(weight,gamma))
