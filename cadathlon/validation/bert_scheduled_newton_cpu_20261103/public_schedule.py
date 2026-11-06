"""Public stage contracts from exact coefficient/Taylor and ratio bounds.

Bounds concern an ideal positive real circuit. Floating-point observations and
all future CKKS/refresh errors require separate checks; they are not certified
by the contraction bound below.
"""
from decimal import Decimal,localcontext,ROUND_CEILING,ROUND_FLOOR
from fractions import Fraction
import math
import numpy as np

RELATIVE_RESIDUAL=Fraction(1,10**12)
SCALE=4.

def outward_float(value,up):
    actual=float(value)
    if (Fraction.from_float(actual)<value if up else Fraction.from_float(actual)>value):
        actual=float(np.nextafter(actual,np.inf if up else -np.inf))
    return actual

def rational(value):return {'numerator':str(value.numerator),'denominator':str(value.denominator)}

def certify_iterations(a,b):
    rho=(b-a)/(b+a)
    with localcontext() as ctx:
        ctx.prec=100;ctx.rounding=ROUND_CEILING
        upper=Decimal(rho.numerator)/Decimal(rho.denominator)
    with localcontext() as ctx:
        ctx.prec=100;ctx.rounding=ROUND_FLOOR
        lower=Decimal(rho.numerator)/Decimal(rho.denominator)
    gate=Decimal('1e-12');previous_lower=lower
    for n in range(1,20):
        previous_lower=lower
        with localcontext() as ctx:
            ctx.prec=100;ctx.rounding=ROUND_CEILING;upper=upper*upper
        with localcontext() as ctx:
            ctx.prec=100;ctx.rounding=ROUND_FLOOR;lower=lower*lower
        if upper<=gate:
            assert n==1 or previous_lower>gate
            return n,{'rho_exact':rational(rho),'residual_upper':str(upper),
                      'residual_lower':str(lower),'previous_iteration_residual_lower':str(previous_lower),
                      'directed_decimal_precision':100,'minimal_for_this_contract':True}
    raise AssertionError('public contraction did not converge in expected range')

class PublicSchedule:
    def __init__(self,coefficients):
        coefficients=tuple(map(float,coefficients))
        if len(coefficients)!=9 or not all(math.isfinite(x) for x in coefficients):
            raise ValueError('nine finite public power coefficients required')
        # exp(1/2) positive series, with a geometric upper bound on its tail.
        # After term24, successive ratios are at most1/(2*26).
        partial=sum((Fraction(1,2**n*math.factorial(n)) for n in range(25)),Fraction(0))
        first_tail=Fraction(1,2**25*math.factorial(25))
        exp_upper=partial+first_tail/(1-Fraction(1,52))
        exp_minus_lower=1/exp_upper
        # P(x)=sum c_n*(2x)^n. On |2x|<=1, coefficient L1 error
        # from the degree8 Taylor polynomial plus its Lagrange remainder
        # gives a whole-interval error bound, independent of samples.
        coefficient_difference=sum((abs(Fraction.from_float(c)-Fraction(1,2**n*math.factorial(n)))
                                    for n,c in enumerate(coefficients)),Fraction(0))
        remainder=exp_upper/Fraction(2**9*math.factorial(9))
        error=coefficient_difference+remainder
        assert exp_minus_lower>error
        ratio=(exp_upper+error)/(exp_minus_lower-error)
        initial_error=Fraction(3,5)**64
        assert initial_error<RELATIVE_RESIDUAL
        self.ratio=ratio;self.error=error;self.rows={}
        self.certificate={'scope':__doc__,'coefficients':list(coefficients),
            'exp_half_upper':rational(exp_upper),'exp_minus_half_lower':rational(exp_minus_lower),
            'coefficient_L1_difference_to_Taylor8':rational(coefficient_difference),
            'Taylor8_remainder_upper':rational(remainder),'whole_interval_error_E':rational(error),
            'whole_interval_error_E_float_upper':outward_float(error,True),
            'base_ratio_R0':rational(ratio),'base_ratio_R0_float_upper':outward_float(ratio,True),
            'relative_residual_gate':'1e-12','initial_mean_inverse_residual_upper':rational(initial_error),
            'mass_lower':'1-1e-12','mass_upper':1,
            'conditional_proof':'Positive common normalizers preserve ratios; squaring squares the ratio. For mass m<=1 and max/min<=R, sum p^2<=m^2*R/(K-1+R); Cauchy gives sum p^2>=m^2/K.',
            'iterations_selected_before_activations':True,'fixed_s':SCALE,
            'numerical_scope':'The exact contraction assumes ideal real Newton arithmetic. Binary64 public constants/operations and CKKS/refresh perturbations are not proved by it.'}
        mass_lower=1-RELATIVE_RESIDUAL
        for k in range(1,129):
            stages=[]
            for stage in range(5):
                r=ratio**(2**stage)
                exact_upper=r/(k-1+r)
                a=outward_float(Fraction(1,2*k),False)
                b=outward_float(min(Fraction(2),2*exact_upper),True)
                af,bf=map(Fraction.from_float,(a,b))
                assert af<=mass_lower*mass_lower/k and bf>=2*exact_upper
                assert 0<af<bf<=2
                n,contraction=certify_iterations(af,bf)
                assert n<=13
                dummy=(a+b)/2
                assert a<=dummy<=b
                stages.append({'stage':stage,'a':a,'b':b,'iterations':n,'inactive_dummy':dummy,
                    'ideal_ratio_upper':outward_float(r,True),
                    'ideal_denominator_upper_under_unit_mass':outward_float(exact_upper,True),
                    'lower_mass_assumption':'m>=1-1e-12','contraction':contraction})
            self.rows[k]=stages
    def stage(self,k,j):
        if type(k) is not int or not 1<=k<=128 or type(j) is not int or not 0<=j<5:
            raise ValueError('public K/stage outside the128-key/five-round contract')
        return self.rows[k][j]

    def report(self):
        return dict(self.certificate,all_public_K={str(k):v for k,v in self.rows.items()})
