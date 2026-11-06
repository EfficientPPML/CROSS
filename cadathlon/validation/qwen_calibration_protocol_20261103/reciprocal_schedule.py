"""Fixed public C1 reciprocal schedules for every required causal key count."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext,ROUND_CEILING,ROUND_FLOOR
import hashlib,json,math
HERE=Path(__file__).resolve().parent

def contract(K):
    exact_a=F(1,2*K);a=float(exact_a)
    if F.from_float(a)>exact_a:a=math.nextafter(a,-math.inf)
    b=2.;initial=2./(a+b);w=(-initial)*initial;bias=2.*initial
    af,bf,wf,biasf=map(F.from_float,(a,b,w,bias))
    residual=lambda x:1-x*(wf*x+biasf)
    points=[af,bf];vertex=-biasf/(2*wf)
    if af<=vertex<=bf:points.append(vertex)
    values=list(map(residual,points));worst=max(map(abs,values))
    assert worst<1
    with localcontext() as c:
        c.prec=100;c.rounding=ROUND_CEILING;upper=Decimal(worst.numerator)/Decimal(worst.denominator)
    with localcontext() as c:
        c.prec=100;c.rounding=ROUND_FLOOR;lower=Decimal(worst.numerator)/Decimal(worst.denominator)
    n=1;previous_lower=None
    while upper>Decimal('1e-12'):
        previous_lower=lower
        with localcontext() as c:c.prec=100;c.rounding=ROUND_CEILING;upper=upper*upper
        with localcontext() as c:c.prec=100;c.rounding=ROUND_FLOOR;lower=lower*lower
        n+=1
        assert n<=64
    assert n==1 or previous_lower>Decimal('1e-12')
    return {'valid_keys':K,'domain':[a,b],'initial_estimate':initial,'rounded_first_weight':w,'rounded_first_bias':bias,
        'first_residual_minimum':float(min(values)),'first_residual_maximum':float(max(values)),
        'exact_first_absolute_residual_bound':{'numerator':str(worst.numerator),'denominator':str(worst.denominator)},
        'iterations':n,'residual_upper':str(upper),'previous_iteration_residual_lower':None if previous_lower is None else str(previous_lower),
        'minimal_for_actual_rounded_initial_affine_and_exact_following_newton':True,'inactive_midpoint':(a+b)/2}

if __name__=='__main__':
    rows=[contract(K) for K in range(1,33)]
    report={'scope':__doc__,'passed':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'normalization_factor':1,'relative_residual_gate':'1e-12','directed_decimal_precision':100,
        'initial_mean_exp_contract':rows[0],'round_contracts':rows,
        'scope_limit':'Public exact-real Newton recurrence after actual rounded initial affine. No subsequent floating-point, CKKS, refresh, positivity/mass composition or model guarantee.'}
    (HERE/'reciprocal_schedule.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'iterations_by_keys':{r['valid_keys']:r['iterations'] for r in rows}}))
