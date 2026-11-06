"""Exact-real public bounds for unchanged Qwen candidate3 continuations.

Fractions represent the actual stored binary64 coefficients. Fixed256-bit
upward dyadic rounding controls certificate size without weakening soundness.
No ciphertext, activations, model weights or outcome data are consumed.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math
MANIFEST='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
BITS=256
TARGET=F(1,10**12)
def exact(value):return F.from_float(float(value))
def record(value):
    value=F(value);upper=float(value)
    if F.from_float(upper)<value:upper=math.nextafter(upper,math.inf)
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),upper_binary64=upper)
def unpack(row):return F(int(row['numerator']),int(row['denominator']))
def dyadic_upper(value,bits=BITS):
    value=F(value)
    if value<0 or type(bits)is not int or bits<1:raise ValueError('nonnegative value and positive exact precision required')
    scaled=value*(1<<bits);return F((scaled.numerator+scaled.denominator-1)//scaled.denominator,1<<bits)
def horner(coefficients,x):
    y=F()
    for c in reversed(coefficients):y=y*x+c
    return y
def power_to_chebyshev(coefficients):
    out=[F() for _ in coefficients]
    for n,coefficient in enumerate(coefficients):
        if n==0:out[0]+=coefficient;continue
        for j in range((n+1)//2):out[n-2*j]+=coefficient*F(math.comb(n,j),1<<(n-1))
        if n%2==0:out[0]+=coefficient*F(math.comb(n,n//2),1<<n)
    return out
def reciprocal_certificate(rule):
    a,b=map(exact,rule['domain']);w=exact(rule['rounded_first_weight']);bias=exact(rule['rounded_first_bias']);n=rule['iterations']
    if not 0<a<=b or w>=0 or type(n)is not int or n<2:raise ValueError('positive static reciprocal contract required')
    points=[a,b];vertex=-bias/(2*w)
    if a<=vertex<=b:points.append(vertex)
    residual=lambda x:1-x*(w*x+bias)
    radius=max(abs(residual(x)) for x in points)
    if radius>=1 or min(w*a+bias,w*b+bias)<=0:raise ValueError('positive first affine and residual contraction required')
    bounds=[dyadic_upper(radius)]
    for _ in range(n-1):bounds.append(dyadic_upper(bounds[-1]**2))
    return dict(domain=[record(a),record(b)],actual_weight=record(w),actual_bias=record(bias),
        weight_hex=float(w).hex(),bias_hex=float(bias).hex(),iterations=n,extrema_candidates=[record(x) for x in points],
        first_exact_residual_abs=record(radius),upward_dyadic_bits=BITS,residual_upper_by_iteration=[record(x) for x in bounds],
        final_residual_upper=record(bounds[-1]),passes_unchanged_1e_12=bounds[-1]<=TARGET,
        first_affine_positive=True,iterations2_and_later_positive_and_at_most_reciprocal=True,
        premise='Original actual rounded first affine and original exact-real Newton updates; later floating/CKKS/refresh error excluded.')

def exp_certificate(rule):
    if rule['domain']!=[-.5,.5] or rule['strategy']!='power' or rule['degree']!=8:raise ValueError('unchanged exp8 contract required')
    coefficients=list(map(exact,rule['evaluation_coefficients']))
    if len(coefficients)!=9:raise ValueError('fixed nine power coefficients required')
    derivative_lower=coefficients[1]-sum((n*abs(coefficients[n]) for n in range(2,9)),F())
    if derivative_lower<=0:raise ValueError('simple exact whole-interval monotonicity certificate failed')
    lo=horner(coefficients,F(-1));hi=horner(coefficients,F(1))
    if not F(1,2)<lo<=hi<2:raise ValueError('actual exp polynomial leaves initial reciprocal domain')
    return dict(actual_power_hex=[float(c).hex() for c in coefficients],mapped_domain=[-1,1],
        derivative_positive_lower=record(derivative_lower),minimum=record(lo),maximum=record(hi),
        proof='P derivative >=a1-sum(n*abs(an))>0 on[-1,1]; exact endpoint Horner bounds the actual stored power polynomial.')

def derive(raw,continuation):
    if type(raw)is not bytes or hashlib.sha256(raw).hexdigest()!=MANIFEST:raise ValueError('exact frozen candidate3 manifest required')
    manifest=json.loads(raw);exp=exp_certificate(manifest['exp']);exp_lo=unpack(exp['minimum']);exp_hi=unpack(exp['maximum'])
    initial=reciprocal_certificate(manifest['initial_reciprocal']);initial_rho=unpack(initial['final_residual_upper'])
    initial_lo,initial_hi=map(unpack,initial['domain']);assert initial['passes_unchanged_1e_12']
    layers=[]
    for layer in range(28):
        rule=manifest['softmax'][str(layer)];s=rule['squarings'];lo,hi=map(exact,rule['centered_domain'])
        # Exactly the frozen Arithmetic.polynomial's public affine expression.
        A=exact((2/(manifest['exp']['domain'][1]-manifest['exp']['domain'][0]))*2.**-s)
        B=exact(-(sum(manifest['exp']['domain']))/(manifest['exp']['domain'][1]-manifest['exp']['domain'][0]))
        endpoints=[A*lo+B,A*hi+B];radius=max(map(abs,endpoints))
        if radius>1:raise ValueError('frozen layer centered domain exceeds exp map domain')
        layers.append(dict(layer=layer,centered_domain=rule['centered_domain'],squarings=s,
            actual_map_weight=record(A),actual_map_bias=record(B),mapped_endpoints=[record(x) for x in endpoints],
            normalized_t_abs_upper=record(radius),guarded_upper=record(F(9,8)*radius),refresh_bound=2,
            refresh_bound_certified=F(9,8)*radius<=2,
            additive_domain_margin=record(min(1+endpoints[0],1-endpoints[1]))))
    keys=[];reciprocals={};all_rho=initial_rho
    for K in range(1,33):
        rule=manifest['reciprocal_by_keys'][str(K)];cert=reciprocal_certificate(rule);reciprocals[str(K)]=cert
        if not cert['passes_unchanged_1e_12']:raise ValueError('frozen reciprocal target cannot be certified')
        all_rho=max(all_rho,unpack(cert['final_residual_upper']))
    for K in range(1,33):
        cert=reciprocals[str(K)];a,b=map(unpack,cert['domain']);alpha=exact(1./K);beta=K*alpha
        den_lo,den_hi=beta*exp_lo,beta*exp_hi
        mass_lo=1-all_rho;square_lo=mass_lo**2/K;square_hi=F(1)
        if not initial_lo<=den_lo<=den_hi<=initial_hi:raise ValueError('actual rounded mean coefficient violates initial domain')
        if not a<=square_lo<=square_hi<=b:raise ValueError('shared-square denominator induction exceeds fixed K domain')
        events=continuation['static_softmax_round_level_plans'].get(str(K),{}).get('refresh_events',[])
        bound=continuation['static_softmax_round_level_plans'].get(str(K),{}).get('conditional_inverse_refresh_bound')
        if events and not (all(e['before_iteration']>=3 for e in events) and F(9,8)/a<=bound):raise ValueError('existing inverse continuation amplitude cannot be certified')
        keys.append(dict(K=K,original_iterations=cert['iterations'],rounded_alpha=record(alpha),beta=record(beta),
            initial_mean_domain=[record(den_lo),record(den_hi)],mass_domain=[record(mass_lo),record(F(1))],
            probability_abs_upper=record(F(1)),probability_guarded_upper=record(F(9,8)),probability_refresh_bound=2,
            next_shared_square_domain=[record(square_lo),record(square_hi)],original_K_domain=manifest['reciprocal_by_keys'][str(K)]['domain'],
            inverse_abs_upper_after_iteration2=record(1/a),existing_inverse_refresh_events=events,
            existing_inverse_refresh_bound=bound,existing_inverse_refresh_bound_certified=not events or F(9,8)/a<=bound,
            K1_original_runtime_bypass=(K==1)))
    rational=[]
    for name,rule in sorted(manifest['nonlinear'].items()):
        if not name.endswith('.silu') or rule['method']!='rational_tanh':continue
        base=rule['base'];m=rule['doublings'];coefficients=list(map(exact,base['evaluation_coefficients']))
        if base['degree']!=12 or base['strategy']!='power' or base['domain']!=[-1.,1.]:raise ValueError('frozen degree12 power tanh required')
        A=exact((2/(base['domain'][1]-base['domain'][0]))*2.**-(m+1));B=exact(-sum(base['domain'])/(base['domain'][1]-base['domain'][0]))
        endpoints=[A*exact(x)+B for x in rule['domain']];radius=max(map(abs,endpoints))
        if radius>1:raise ValueError('frozen SiLU base domain leaves[-1,1]')
        converted=power_to_chebyshev(coefficients);base_bound=sum(map(abs,converted),F())
        if base_bound>=1:raise ValueError('actual converted base unit envelope not established')
        inverse=reciprocal_certificate(rule['reciprocal']);a,b=map(unpack,inverse['domain'])
        if inverse['iterations']!=5 or not inverse['passes_unchanged_1e_12'] or not a<=1<=2<=b:raise ValueError('frozen five-update rational denominator contract differs')
        rational.append(dict(site=name,domain=rule['domain'],doublings=m,original_reciprocal_iterations=5,
            base_declared_degree=12,actual_power_degree=len(coefficients)-1,actual_power_hex=[float(c).hex() for c in coefficients],
            actual_map_weight=record(A),actual_map_bias=record(B),mapped_endpoints=[record(x) for x in endpoints],
            exact_power_to_chebyshev=[record(x) for x in converted],base_abs_upper=record(base_bound),base_guarded_upper=record(F(9,8)*base_bound),
            base_refresh_bound=2,reciprocal=inverse,
            rounds=[dict(round=i+1,input_abs_upper=record(base_bound if i==0 else F(1)),
                denominator_domain=[record(F(1)),record(1+(base_bound if i==0 else F(1))**2)],
                output_abs_upper=record(F(1)),guarded_upper=record(F(9,8)),refresh_bound=2) for i in range(m)]))
    assert len(rational)==22 and sum(x['doublings'] for x in rational)==145
    # A concrete domain-level witness against an overbroad B4 claim. This is
    # not an activation/example selection and does not change the planned bound.
    witness_rule=manifest['reciprocal_by_keys']['8'];x=F(1,8);w=exact(witness_rule['rounded_first_weight']);bias=exact(witness_rule['rounded_first_bias'])
    stopped=continuation['static_softmax_round_level_plans']['8']['refresh_events'][0]['before_iteration']-1
    R=dyadic_upper(abs(1-x*(w*x+bias)))
    for _ in range(stopped-1):R=dyadic_upper(R**2)
    witness_lower=(1-R)/x;assert witness_lower>4
    return dict(passed=True,scope='Exact-real envelopes of frozen stored public polynomials/first affines and original iterations. No input/model/HE or coefficient selection.',
        manifest_sha256=MANIFEST,exp=exp,initial_reciprocal=initial,initial_inverse_abs_upper=record(1/initial_lo),
        initial_inverse_guarded_upper=record(F(9,8)/initial_lo),initial_inverse_refresh_bound=4,
        softmax_layers=layers,softmax_K=keys,reciprocal_certificates=reciprocals,uniform_mass_residual_upper=record(all_rho),rational_silu=rational,
        B4_not_universal=dict(K=8,denominator=record(x),completed_updates_before_existing_refresh=stopped,inverse_lower=record(witness_lower),B4_insufficient=True,
            existing_bound=continuation['static_softmax_round_level_plans']['8']['conditional_inverse_refresh_bound'],scope='Uniform-probability/domain-level witness; no model activation or fixture claim.'),
        mass_proof='The SAME stored alpha=fl(1/K) appears in D=alpha*sum(base) and p_i=alpha*base_i*y, so sum(p)=D*y=1-r. For every later original reciprocal count n>=2,0<=r<=rho, y>0. Thus p>=0 and mass in[1-rho,1]; Cauchy gives sum(p²)>=mass²/K and nonnegativity gives sum(p²)<=mass²<=1. This inductively stays inside every frozen[.5/K,2].',
        tanh_proof='Actual stored power coefficients are converted to Chebyshev exactly over[-1,1], giving base|y|<1. With shared d=1+y² and the original reciprocal n=5,0<inverse<=1/d. Then|2*y*inverse|<=2|y|/(1+y²)<=1. Every next d lies in[1,2], inside the unchanged[1,2.1] contract.',
        limitations=['All original centered/SiLU input domains remain caller assumptions.','B2 amplitude headroom does not prove exp-domain preservation: some frozen normalized_t endpoints equal1 and have zero positive perturbation margin.',
            'Inserted inverse/probability/tanh refresh errors are excluded. Exact input/output correlation and shared-square relations are required for the induction.',
            'Later binary64, plaintext encoding, CKKS, refresh error and downstream native/logit gates are not certified.',
            'B4 certifies the initial mean-exp inverse only; K-specific inverse events retain their existing larger bounds without change.'])
