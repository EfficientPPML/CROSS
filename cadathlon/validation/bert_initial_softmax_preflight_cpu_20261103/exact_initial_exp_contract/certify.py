"""Exact ideal initial-exp/denominator certificate; no HE or observed activations."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext,ROUND_CEILING
import hashlib,io,json,math,struct,tarfile
import numpy as np
P=Path(__file__).resolve().parent

def digest(raw):return hashlib.sha256(raw).hexdigest()
def upward(x):
    y=float(x)
    return math.nextafter(y,math.inf) if F.from_float(y)<x else y
def downward(x):return -upward(-x)
def multiply_x(c):
    result=[F(0)]*(len(c)+1)
    for n,v in enumerate(c):
        if n==0:result[1]+=v
        else:result[n-1]+=v/2;result[n+1]+=v/2
    return result

def chebval(c,t):
    a,b=F(1),t;result=c[0]+c[1]*t
    for v in c[2:]:a,b=b,2*t*b-a;result+=v*b
    return result

def main():
    files=[P/'certify.py',P/'candidate_manifest.json',P/'owner.py.txt']
    before={p.name:digest(p.read_bytes()) for p in files}
    assert before['candidate_manifest.json']=='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
    model=json.loads((P/'candidate_manifest.json').read_text());original=np.array(model['base_exp']['coefficients'],dtype=np.float64)
    assert digest(original.astype('<f8').tobytes())=='64c193dd47fbbc6f850b14a35dad72803d35a2353497aba4db8061b217fe41dd'
    power=np.polynomial.chebyshev.cheb2poly(original)
    assert digest(power.astype('<f8').tobytes())=='5173fba9463c2962090f8ddeb64953ef60ad8ba700b2cd37573aba1e11b6c59a'
    # Pull actual rounded power coefficients back exactly, rather than assume
    # their polynomial equals the original rounded Chebyshev coefficient list.
    actual=[F(0)]*9;basis=[F(1)]
    for v in power:
        for n,t in enumerate(basis):actual[n]+=F.from_float(v)*t
        basis=multiply_x(basis)
    for t in (F(-1),F(-7,8),F(-2,3),F(-1,2),F(-1,4),F(0),F(1,4),F(1,2),F(2,3),F(7,8),F(1)):
        assert chebval(actual,t)==sum((F.from_float(v)*t**n for n,v in enumerate(power)),F(0))
    intervals=512;grid=[chebval(actual,F(-1)+F(2*j,intervals)) for j in range(intervals+1)]
    derivative=sum((n*n*abs(c) for n,c in enumerate(actual)),F(0))
    lo=min(grid)-derivative/intervals;hi=max(grid)+derivative/intervals
    assert lo>F(1,2) and hi<2
    rows=[]
    for K in range(1,129):
        f=F.from_float(1./K);weight=K*f;lower=lo*weight;upper=hi*weight
        assert lower>=F(1,2) and upper<=2
        rows.append(dict(K=K,actual_sum_of_mean_weights=float(weight),denominator_lower=downward(lower),denominator_upper=upward(upper)))
    # Actual rounded initial affine, followed by five exact-real Newton updates.
    a,b=F(1,2),F(2);w=F.from_float(-.8*.8);bias=F.from_float(2*.8)
    residual=lambda x:1-w*x*x-bias*x
    vertex=-bias/(2*w);values=[residual(a),residual(b)]
    if a<=vertex<=b:values.append(residual(vertex))
    r=max(map(abs,values));assert min(values)>=0
    for _ in range(5):r*=r
    assert r<F(1,10**12)
    report=dict(passed=True,scope=__doc__,no_backend_import=True,no_key_generation=True,no_encrypted_execution=True,
        manifest_sha256=before['candidate_manifest.json'],power_coefficient_sha256=digest(power.astype('<f8').tobytes()),
        source_hashes_before=before,numpy_version=np.__version__,domain_t=[-1,1],grid_points=513,
        exact_rational_grid_plus_derivative=True,actual_power_polynomial_lower=downward(lo),actual_power_polynomial_upper=upward(hi),
        cheb_to_power_rounding_difference_upper=upward(sum((abs(x-F.from_float(y)) for x,y in zip(actual,original)),F(0))),
        all_K_denominator_domain=[.5,2.],all_K_rows=rows,inactive_denominator=1.,
        actual_rounded_first_affine_residual_nonnegative=True,six_update_residual_upper=upward(r),
        ideal_probability_mass_lower=downward(1-r),ideal_probability_mass_upper=1.,
        requirements=['Actual centered scores using the circuit mean satisfy[-16,16].',
            'Public source masks and anchor reduction are the separately tested layout.',
            'This is exact-real arithmetic of actual rounded public coefficients; CKKS, plaintext encoding, rescale and later floating arithmetic errors are excluded.',
            'No observed native inputs, calibration extrema or labels are used; no coefficient/domain/count changes.',
            'The later five normalized-squaring rounds and their inverse denominators are outside this certificate.'])
    with tarfile.open(P/'sources.tar.gz','w:gz') as t:
        for p in files:
            raw=p.read_bytes();info=tarfile.TarInfo(p.name);info.size=len(raw);t.addfile(info,io.BytesIO(raw))
    with tarfile.open(P/'sources.tar.gz') as t:assert {m.name:digest(t.extractfile(m).read()) for m in t.getmembers()}==before
    after={p.name:digest(p.read_bytes()) for p in files};assert before==after
    report.update(source_hashes_after=after,source_files_unchanged=True,source_archive_verified=True,source_archive_sha256=digest((P/'sources.tar.gz').read_bytes()))
    (P/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','actual_power_polynomial_lower','actual_power_polynomial_upper','six_update_residual_upper','ideal_probability_mass_lower','source_archive_sha256')}))
if __name__=='__main__':main()
