from fractions import Fraction as F
import pytest
from certify import value_at,envelope,power_two_cover,upper

@pytest.mark.parametrize('n',[0,1,2,3,4,5,8,16,64,192])
def test_exact_chebyshev_endpoints(n):
    c=[F(0)]*(n+1);c[n]=F(1)
    assert value_at(c,F(1))==1 and value_at(c,F(-1))==(-1)**n

@pytest.mark.parametrize('x',[F(1,3),F(-1,3),F(0),F(1,2**100),F(2**100)])
def test_outward_conversion(x):assert F(upper(x))>=x

@pytest.mark.parametrize('x',[F(1,3),F(1),F(2),F(3),F(100),F(1,2**50)])
def test_smallest_power_two_cover(x):
    b=power_two_cover(x);assert b>=x and b/2<x

def test_short_rounded_map_extension_is_explicitly_covered():
    eps=F(1,2**52);c=[F(0),F(0),F(1)]
    r=envelope(c,mapped_endpoints=[-1-eps,1+eps],count=16)
    assert F(int(r['mapped_extent']['numerator']),int(r['mapped_extent']['denominator']))==1+eps
    b=r['polynomial_abs_upper'];bound=F(int(b['numerator']),int(b['denominator']))
    assert bound>=2*(1+eps)**2-1

def test_grid_plus_derivative_controls_interior_peak():
    # 1-T2=2(1-x^2): endpoints vanish while the interior reaches two.
    r=envelope([F(1),F(0),F(-1)],mapped_endpoints=[F(-1),F(1)],count=16)
    assert r['grid_intervals']==16 and r['grid_max_abs']['upper_binary64']==2
    assert r['polynomial_abs_upper']['upper_binary64']>=2 and r['refresh_bound']==4

def test_subunit_polynomial_gets_positive_minimum_refresh_bound():
    r=envelope([F(0),F(1,10)],mapped_endpoints=[F(-1),F(1)])
    assert r['refresh_bound']==1

@pytest.mark.parametrize('count',[True,1,0,2.])
def test_bad_grid_count(count):
    with pytest.raises(ValueError):envelope([F(0),F(1)],mapped_endpoints=[F(-1),F(1)],count=count)
