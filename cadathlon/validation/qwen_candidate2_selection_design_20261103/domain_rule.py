"""Public candidate2 domain rule; not coefficient selection or evaluation."""
from fractions import Fraction
import math

MARGIN=Fraction(9,8)


def silu_domain(native_upper, relative_norm_error):
    if type(native_upper) is not float or not math.isfinite(native_upper) or native_upper<=0:
        raise ValueError('positive outward binary64 native bound required')
    if type(relative_norm_error) is not Fraction or not 0<=relative_norm_error<1:
        raise ValueError('a positive-norm uniform rational error certificate below one is required')
    exact=Fraction.from_float(native_upper)*(1+relative_norm_error)*MARGIN
    limit=(exact.numerator+exact.denominator-1)//exact.denominator
    return {'domain':[-limit,limit],'margin':[9,8],
            'pre_round_upper_fraction':[str(exact.numerator),str(exact.denominator)],
            'norm_relative_error_fraction':[str(relative_norm_error.numerator),str(relative_norm_error.denominator)]}


def rational_cost_preview(limit):
    if type(limit) is not int or limit<=0:raise ValueError('positive integer radius required')
    doublings=0
    while 2**(doublings+1)<limit:doublings+=1
    # Existing public reciprocal on[1,2.1] has five updates/depth13.
    return {'doublings':doublings,'reciprocal_depth':13,
            'depth_if_base_degree8_passes':5+doublings*17+2,
            'depth_if_base_degree24_passes':7+doublings*17+2,
            'new_domain_target_tested':False}
