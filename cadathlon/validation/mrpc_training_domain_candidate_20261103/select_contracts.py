"""Fix analytic-function candidates from training-only ranges before validation."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal,localcontext,ROUND_CEILING
import hashlib,json,math,sys,io,tarfile,importlib.metadata,time
import numpy as np
from audit import sha,write,REPO
sys.path.insert(0,str(REPO/'cadathlon/common'))
from encrypted_polynomial import nonlinear_approximation
from public_schedule import PublicSchedule

HERE=Path(__file__).resolve().parent

def reciprocal_contract(domain,tolerance=Fraction(1,10**12)):
    a,b=map(float,domain)
    if not math.isfinite(a) or not math.isfinite(b) or not 0<a<b:raise ValueError('invalid reciprocal domain')
    seed=2/(a+b)
    af,bf,s=map(Fraction.from_float,(a,b,seed))
    rho=max(abs(1-af*s),abs(1-bf*s))
    if not 0<=rho<1:raise ValueError('invalid public Newton seed')
    with localcontext() as ctx:
        ctx.prec=100;ctx.rounding=ROUND_CEILING
        upper=Decimal(rho.numerator)/Decimal(rho.denominator)
        gate=Decimal(tolerance.numerator)/Decimal(tolerance.denominator)
        count=0
        while upper>gate:
            upper*=upper;count+=1
            if count>30:raise ValueError('unexpected reciprocal schedule')
    return dict(domain=[a,b],seed=seed,iterations=max(1,count),
        relative_residual_upper=str(upper),rounding='100-digit Decimal ceiling; exact binary64 seed endpoint residual',
        assumption='Ideal real Newton updates on values inside domain; floating-point/CKKS trajectory error excluded.')

def inverse(values,rule):
    result=np.full_like(values,rule['seed'])
    for _ in range(rule['iterations']):result=result*(2-values*result)
    return result

def sqrt_enclosure(value):
    """Exact rational checks enclose a public, representable positive root."""
    approximate=float(value)
    if not math.isfinite(approximate) or approximate<=0:
        raise ValueError('normalized squared seed outside public planning envelope')
    lo=hi=math.sqrt(approximate)
    while Fraction.from_float(lo)**2>value:lo=math.nextafter(lo,0.)
    while Fraction.from_float(hi)**2<value:hi=math.nextafter(hi,math.inf)
    return Fraction.from_float(lo),Fraction.from_float(hi)

def inverse_sqrt_contract(domain,tolerance=Decimal('0.000001')):
    a,b=map(float,domain)
    if not math.isfinite(a) or not math.isfinite(b) or not 0<a<b:raise ValueError('invalid inverse-sqrt domain')
    seed=1/math.sqrt(b);s=Fraction.from_float(seed)
    lo,_=sqrt_enclosure(Fraction.from_float(a)*s*s)
    _,hi=sqrt_enclosure(Fraction.from_float(b)*s*s)
    positive=max(Fraction(0),1-lo);negative=max(Fraction(0),hi-1)
    if positive>=1 or negative>=1:raise ValueError('inverse-sqrt seed outside convergence envelope')
    with localcontext() as ctx:
        ctx.prec=100;ctx.rounding=ROUND_CEILING
        # Evaluate each polynomial exactly as a rational before rounding
        # upward; separate rounded cubic subtraction would not be outward.
        first=max(Fraction(3,2)*positive**2-Fraction(1,2)*positive**3,
                  Fraction(3,2)*negative**2+Fraction(1,2)*negative**3)
        upper=Decimal(first.numerator)/Decimal(first.denominator)
        iterations=1
        while upper>tolerance:
            e=Fraction(upper)
            exact=Fraction(3,2)*e*e-Fraction(1,2)*e*e*e
            upper=Decimal(exact.numerator)/Decimal(exact.denominator)
            iterations+=1
            if iterations>100:raise ValueError('unexpected inverse-sqrt schedule')
    # Independent public numerical checks are kept distinct from the ideal
    # interval certificate and use exactly the serialized binary64 seed.
    grid=np.linspace(a,b,16385);y=np.full_like(grid,seed)
    for _ in range(iterations):y=.5*y*(3-grid*y*y)
    error=float(np.max(np.abs(y*np.sqrt(grid)-1)))
    if error>float(tolerance)+8*np.finfo(float).eps:raise ValueError('sampled inverse-sqrt schedule failed')
    return dict(domain=[a,b],seed=seed,seed_hex=seed.hex(),iterations=iterations,
        ideal_relative_error_upper=str(upper),sampled_relative_error=error,
        certificate='Exact rational outward sqrt endpoint checks for the actual binary64 seed, then100-digit directed-ceiling recurrence; ideal real updates only.',
        normalized_seed_lower=[str(lo.numerator),str(lo.denominator)],normalized_seed_upper=[str(hi.numerator),str(hi.denominator)])

def serialize(contract):
    raw=np.asarray(contract.coefficients,dtype='<f8').tobytes()
    return dict(name=contract.name,domain=list(contract.domain),degree=len(contract.coefficients)-1,
        coefficients=list(contract.coefficients),coefficient_f64le_sha256=hashlib.sha256(raw).hexdigest(),
        coefficient_rule='Existing nonlinear_approximation: Chebyshev.interpolate of analytic function; original small-coefficient zeroing unchanged.')

def choose_function(kind,domain,protocol):
    x=np.linspace(*domain,protocol['public_analytic_grid_points']);sweep=[]
    if kind=='inverse_sqrt':exact=1/np.sqrt(x);gate=protocol['inverse_sqrt_max_relative_target']
    else:exact=.5*x*(1+np.vectorize(math.erf)(x/np.sqrt(2)));gate=protocol['gelu_max_abs_target']
    for degree in protocol['degree_candidates']:
        contract=nonlinear_approximation(kind,domain,degree);actual=contract.evaluate_public(x)
        absolute=float(np.max(np.abs(actual-exact)))
        error=float(np.max(np.abs(actual/exact-1))) if kind=='inverse_sqrt' else absolute
        powers=np.polynomial.chebyshev.cheb2poly(contract.coefficients)
        l1=float(np.abs(powers).sum());met=error<=gate
        sweep.append(dict(degree=degree,max_abs_grid_error=absolute,selection_error=error,target=gate,
            target_met=bool(met),normalized_monomial_coefficient_l1=l1))
        if met:break
    if not met and kind!='inverse_sqrt':raise ValueError(f'No unchanged analytic degree target met: {kind} {domain}')
    return contract,sweep

def main():
    training=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False);start=time.monotonic()
    protocol=json.loads((HERE/'selection_protocol.json').read_text());r=json.loads(training.read_text())
    assert r['passed'] and r['completed_examples']==3668 and r['training_labels_loaded'] is False
    assert r['selection_protocol']==protocol
    selected=[Path(__file__),HERE/'test_select.py',HERE/'selection_protocol.json',HERE/'public_schedule.py',HERE/'audit.py',training,
        REPO/'cadathlon/common/encrypted_polynomial.py']
    before={str(p.resolve()):sha(p) for p in selected}
    with tarfile.open(out/'sources.tar.gz','w:gz') as archive:
        for name in before:
            raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    proposed=r['proposed_domains'];norms={};gelu={};softmax={};evidence={}
    for layer in range(12):
        for phase in ('attention','mlp'):
            label=f'{layer}.{phase}.inverse_sqrt';domain=proposed[label]['domain']
            contract,sweep=choose_function('inverse_sqrt',domain,protocol)
            power_supported=(sweep[-1]['target_met'] and len(contract.coefficients)-1<=32 and sweep[-1]['normalized_monomial_coefficient_l1']<=1e8)
            newton=inverse_sqrt_contract(domain,Decimal(str(protocol['inverse_sqrt_newton_ideal_relative_target'])))
            norms[f'{layer}.{phase}']=dict(domain=domain,method='power' if power_supported else 'newton',
                degree=len(contract.coefficients)-1,iterations=newton['iterations'],newton_seed=newton['seed'],
                polynomial=serialize(contract),newton_certificate=newton,polynomial_grid_target_met=sweep[-1]['target_met'])
            evidence[label]=sweep
        label=f'{layer}.gelu';domain=proposed[label]['domain'];contract,sweep=choose_function('gelu',domain,protocol)
        gelu[str(layer)]=dict(domain=domain,degree=len(contract.coefficients)-1,strategy='balanced Chebyshev',polynomial=serialize(contract))
        evidence[label]=sweep
        domain=proposed[f'{layer}.softmax.centered_scores']['domain'];squarings=max(0,math.ceil(math.log2(max(map(abs,domain))/.5)))
        softmax[str(layer)]=dict(centered_domain=domain,squarings=squarings)
    exp=nonlinear_approximation('exp',(-.5,.5),8)
    # The unchanged reviewed coefficient/Taylor certificate proves positive
    # base polynomial on the entire declared interval, independently of data.
    positive=PublicSchedule(np.polynomial.chebyshev.cheb2poly(exp.coefficients)).certificate
    reciprocal_by_K={str(k):reciprocal_contract((.5/k,2.)) for k in range(1,129)}
    initial=reciprocal_contract((.5,2.));doubling_inverse=reciprocal_contract((1.,2.1))
    domain=proposed['pooler.tanh']['domain'];rounds=max(0,math.ceil(math.log2(max(map(abs,domain)))))
    grid=np.linspace(*domain,protocol['public_analytic_grid_points']);tanh_evidence=[]
    for degree in (8,12,16):
        base=nonlinear_approximation('tanh',(-1.,1.),degree);value=base.evaluate_public(grid/(2**rounds))
        for _ in range(rounds):value=2*value*inverse(1+value*value,doubling_inverse)
        error=float(np.max(np.abs(value-np.tanh(grid))));tanh_evidence.append(dict(base_degree=degree,max_abs_grid_error=error))
        if error<=protocol['tanh_max_abs_target']:break
    if error>protocol['tanh_max_abs_target']:raise ValueError('No unchanged pooler target met')
    tanh=dict(domain=domain,base_degree=degree,doublings=rounds,
        reciprocal_iterations=doubling_inverse['iterations'],base_polynomial=serialize(base))
    manifest=dict(norms=norms,gelu=gelu,softmax=softmax,tanh=tanh,
        base_exp=serialize(exp),initial_reciprocal=initial,normalization_reciprocal_by_public_K=reciprocal_by_K,
        tanh_reciprocal=doubling_inverse,
        selection='Fixed using official training-only native extrema, unchanged historical domain construction/analytic targets, and endpoint-derived reciprocal counts before approximate validation.')
    write(out/'candidate_manifest.json',manifest)
    after={n:sha(n) for n in before};assert before==after
    with tarfile.open(out/'sources.tar.gz') as archive:
        assert len(archive.getmembers())==len(before)
        assert all(hashlib.sha256(archive.extractfile(m).read()).hexdigest()==before['/'+m.name] for m in archive.getmembers())
    report=dict(scope=__doc__,passed=True,selection_protocol=protocol,training_report_sha256=sha(training),
        candidate_manifest_sha256=sha(out/'candidate_manifest.json'),function_sweeps=evidence,tanh_sweep=tanh_evidence,
        base_exp_positivity=positive,source_hashes_before=before,source_hashes_after=after,source_files_unchanged=True,
        source_archive_sha256=sha(out/'sources.tar.gz'),numpy_version=np.__version__,elapsed_s=time.monotonic()-start,
        no_jax_imported=not any(n=='jax' or n.startswith('jax.') for n in sys.modules),
        validation_examples_or_labels_used=False,approximate_validation_evaluated=False,encrypted_execution=False,
        limitations=['Empirical training ranges plus fixed slack do not certify unseen inputs or approximate/CKKS trajectories.',
          'Function errors are sampled analytic-grid checks; inverse endpoint contractions assume ideal real arithmetic.',
          'Power/Newton and balanced-Chebyshev manifests are arithmetic planning, not a compiled full-model encrypted implementation.'])
    assert report['no_jax_imported'];write(out/'report.json',report)
    print(json.dumps(dict(passed=True,manifest_sha256=report['candidate_manifest_sha256'],gelu_degrees={k:v['degree'] for k,v in gelu.items()},
        softmax_squarings={k:v['squarings'] for k,v in softmax.items()},max_reciprocal_iterations=max(v['iterations'] for v in reciprocal_by_K.values()),tanh=tanh_evidence[-1],elapsed_s=report['elapsed_s']),indent=2))
if __name__=='__main__':main()
