from fractions import Fraction
import numpy as np
import pytest
import dyadic_envelope as d
from exact_dyadic import rows
from bounds import head_variance_bounds, normalized_linear_bounds, rms_variance_upper


def test_exact_linear_against_independent_fraction():
    rng=np.random.default_rng(52)
    for dtype in (np.float32,np.float64):
        a=rng.normal(size=(7,31)).astype(dtype);b=np.abs(rng.normal(size=31))
        actual=d.exact_linear(a,b)
        for i in range(7):
            expected=sum((abs(Fraction(float(x)))*Fraction(float(y)) for x,y in zip(a[i],b)),Fraction())
            assert Fraction(float(actual[i]))>=expected
            assert Fraction(float(np.nextafter(actual[i],-np.inf)))<expected


def test_extreme_subnormals_and_zeros():
    for dtype in (np.float32,np.float64):
        a=np.array([[0,np.nextafter(dtype(0),dtype(1)),dtype(1)]],dtype=dtype)
        values,p=d.integerize(a)
        for actual,x in zip(values.flat,a.flat):
            v=Fraction(int(actual)<<p) if p>=0 else Fraction(int(actual),1<<-p)
            assert v==Fraction(float(x))
        assert np.array_equal(d.exact_linear(np.zeros((2,3),dtype=dtype),np.ones(3)),[0,0])


def test_exact_normalized_rows_and_heads():
    rng=np.random.default_rng(1);w=rng.normal(size=(8,4)).astype(np.float32);g=rng.normal(size=4).astype(np.float32)
    got=list(rows(w,g));expected=normalized_linear_bounds(w,g)
    assert [Fraction(float(x[1])) for x in got]==list(expected)
    heads=head_variance_bounds(w,g,2,1e-6)
    assert tuple(sum((v[0] for v in got[i:i+2]),Fraction())/2+Fraction(1e-6) for i in range(0,8,2))==heads


@pytest.mark.parametrize('value',[np.array([True]),np.array([1],dtype=object),np.array([1j]),np.array([-1.]),np.array([np.inf]),np.array([np.nan]),np.array([],dtype=float)])
def test_bad_exact_input(value):
    with pytest.raises(ValueError):d.integerize(value)


def test_add_product_are_outward_and_variance_positive():
    a=np.array([1.,1e-200,1e100]);b=np.array([np.nextafter(1.,2.),1e-100,1e-100])
    for out,op in [(d.exact_product(a,b),lambda x,y:x*y),(d.exact_add(a,b),lambda x,y:x+y)]:
        for actual,x,y in zip(out,a,b):assert Fraction(float(actual))>=op(Fraction(float(x)),Fraction(float(y)))
    assert rms_variance_upper(np.zeros(4),1e-6)==Fraction(1e-6)
