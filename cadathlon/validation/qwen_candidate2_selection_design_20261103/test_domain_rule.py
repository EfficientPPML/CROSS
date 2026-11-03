from fractions import Fraction
import math
import pytest
from domain_rule import silu_domain,rational_cost_preview


def test_rounding_is_exact_ceiling_of_certified_product():
    for bound in (41.80374204545915,677.6913892458547,3200.237701238064):
        for delta in (Fraction(0),Fraction(1,10**6),Fraction(3,100)):
            row=silu_domain(bound,delta);radius=row['domain'][1]
            exact=Fraction.from_float(bound)*(1+delta)*Fraction(9,8)
            assert radius-1<exact<=radius and row['domain'][0]==-radius


def test_margin_is_not_silently_a_power_of_two_cover():
    assert silu_domain(3200.237701238064,Fraction(1,10**6))['domain']==[-3601,3601]
    assert silu_domain(677.6913892458547,Fraction(1,10**6))['domain']==[-763,763]


def test_exact_doubling_bound_and_existing_depth_contract():
    for radius in (1,2,3,48,763,3601):
        row=rational_cost_preview(radius);n=row['doublings']
        assert radius<=2**(n+1) and (n==0 or radius>2**n)
    assert rational_cost_preview(3601)['depth_if_base_degree24_passes']==196


@pytest.mark.parametrize('bound,error',[(True,Fraction(0)),(float('nan'),Fraction(0)),(0.,Fraction(0)),(2.,.000001),(2.,Fraction(-1,10)),(2.,Fraction(1))])
def test_unproved_or_invalid_public_contract_rejected(bound,error):
    with pytest.raises(ValueError):silu_domain(bound,error)
