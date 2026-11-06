"""Independent exact public coefficient review; no model, tokenizer, NumPy or HE."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import hashlib,json,math,struct,tarfile,time
LAB=Path('/tmp/qwen-candidate3-polynomial-selection-20261103')
OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def unpack(x):return F(int(x['numerator_hex'],16),int(x['denominator_hex'],16))
def cheb(c,x):
    b1=b2=F(0)
    for a in reversed(c[1:]):b1,b2=a+2*x*b1-b2,b1
    return c[0]+x*b1-b2

def residual(c,A,B):
    # Independently enumerate all ordered products (no triangular cross factor).
    denominator=max(v.denominator for v in c);a=[v.numerator*(denominator//v.denominator) for v in c]
    raw=[0]*(2*len(c)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(a):raw[i+j]+=x*y;raw[abs(i-j)]+=x*y
    q=[F(v,2*denominator**2) for v in raw];h=[F(0)]*(len(q)+1)
    for k,v in enumerate(q):
        h[k]-=B/A*v
        if k==0:h[1]+=v/A
        else:h[k-1]+=v/(2*A);h[k+1]+=v/(2*A)
    h[0]-=1
    while len(h)>1 and h[-1]==0:h.pop()
    return h

def T(n,M):
    a,b=F(1),M
    if n==0:return a
    for _ in range(1,n):a,b=b,2*M*b-a
    return b

def main():
    start=time.monotonic();r=json.loads((LAB/'source-bound-v1/report.json').read_text());before=r['source_hashes_before']
    assert r['passed'] and len(before)==12 and before==r['source_hashes_after']
    assert all(sha((LAB/n).read_bytes())==v for n,v in before.items())
    with tarfile.open(LAB/'source-bound-v1/sources.tar.gz') as tar:
        rows=tar.getmembers();assert len(rows)==len({m.name for m in rows})==12 and all(m.isfile() for m in rows)
        assert {m.name:sha(tar.extractfile(m).read()) for m in rows}==before
    assert sha((LAB/'source-bound-v1/sources.tar.gz').read_bytes())==r['source_archive_sha256']
    selected=LAB/'source-bound-v1/candidate-v3';report=json.loads((selected/'report.json').read_text());m=json.loads((selected/'manifest.json').read_text());assess=json.loads((selected/'assessments.json').read_text())
    assert sha((selected/'report.json').read_bytes())==r['selection_report_sha256']
    assert sha((selected/'manifest.json').read_bytes())==r['manifest_sha256']==report['manifest_sha256']
    assert sha((selected/'assessments.json').read_bytes())==report['assessments_sha256']
    base=json.loads((LAB/'inputs/candidate2_manifest.json').read_text());fb=json.loads((LAB/'inputs/factor4_fallback_report.json').read_text());config=json.loads((LAB/'inputs/qwen_config.json').read_text());multipliers=json.loads((LAB/'inputs/norm_multipliers.json').read_text())
    protocol=json.loads((LAB/'protocol.json').read_text());ladder=[16,32,64,128,256,512];assert protocol['degree_ladder']==ladder and protocol['uniform_relative_target']==1e-6
    names=sorted(k for k in base['nonlinear'] if k.endswith('.inverse_sqrt'));assert len(names)==len(assess)==113 and set(names)==set(assess)
    polynomial=[];newton=0;failed_attempts=0
    for name in names:
        old=base['nonlinear'][name];a=assess[name];rule=m['nonlinear'][name]
        expected=[max(config['rms_norm_eps'],old['domain'][0]/4),old['domain'][1]*4]
        assert expected==fb['sites'][name]['expanded_domain']==a['domain']==rule['domain']
        attempts=a['attempts'];assert [x['degree'] for x in attempts]==ladder[:len(attempts)]
        assert all(x['selected']==bool(x['uniform_certificate']['certified'] and x['binary64_diagnostic']['passed']) for x in attempts)
        assert all(not x['selected'] for x in attempts[:-1]);failed_attempts+=sum(not x['selected'] for x in attempts)
        if rule['method']=='newton':
            newton+=1;assert len(attempts)==6 and not attempts[-1]['selected']
            assert rule==dict(kind='inverse_sqrt',method='newton',**fb['sites'][name]['rule'])
            assert fb['sites'][name]['certificate']['verified'] and F(Decimal(rule['final_relative_error_bound']))<=F(1,10**6)
            continue
        assert attempts[-1]['selected'] and rule['strategy']=='chebyshev' and rule['degree']==attempts[-1]['degree']
        assert rule['chebyshev_coefficients']==rule['evaluation_coefficients']==attempts[-1]['coefficients']
        lo,hi=rule['domain'];A,B=F(2/(hi-lo)),F(-(hi+lo)/(hi-lo));c=list(map(F,rule['evaluation_coefficients']));cert=a['certificate']
        assert cert['coefficient_sha256']==sha(struct.pack('<'+str(len(c))+'d',*rule['evaluation_coefficients']))
        left,right=A*F(lo)+B,A*F(hi)+B;M=max(F(1),abs(left),abs(right));h=residual(c,A,B)
        # Independent exact Clenshaw identity checks at endpoints and midpoint.
        for t in (left,F(0),right):assert cheb(h,t)==((t-B)/A)*cheb(c,t)**2-1
        envelope=T(len(h)-1,M);l1=sum(map(abs,h),F(0));delta=l1*envelope;relative=delta/(2-delta)
        assert 0<F(lo)<F(hi) and A>0 and left<=0<=right and cheb(c,F(0))>0 and delta<1
        assert relative<=F(1,10**6) and cert['certified']
        assert unpack(cert['extent'])==M and unpack(cert['H_L1'])==l1 and unpack(cert['chebyshev_extent_envelope'])==envelope and unpack(cert['delta'])==delta and unpack(cert['relative'])==relative
        assert cert['H_degree']==len(h)-1 and cert['actual_A_hex']==float(A).hex() and cert['actual_B_hex']==float(B).hex()
        assert F(cert['delta_upper_binary64'])>=delta and F(cert['relative_upper_binary64'])>=relative
        polynomial.append(dict(site=name,degree=rule['degree'],certified_relative_upper=cert['relative_upper_binary64'],endpoint_extension=M>1))
    assert polynomial and newton==77 and len(polynomial)==36 and failed_attempts==610
    for name,rule in base['nonlinear'].items():
        if name not in names:assert m['nonlinear'][name]==rule
    for name in ('softmax','exp','initial_reciprocal','reciprocal_by_keys'):assert m[name]==base[name]
    for i in range(28):
        a=assess[f'{i}.post.inverse_sqrt'];cert=a['certificate'];delta=unpack(cert['relative']) if a['selected_method']=='polynomial' else F(Decimal(cert['relative']))
        bound=F(9,8)*(1+delta)*F(multipliers[str(i)]['native_gate_upper']);radius=-(-bound.numerator//bound.denominator);d=base['nonlinear'][f'{i}.silu']['domain']
        assert d[0]<=-radius and d[1]>=radius and report['silu_radius_recheck'][str(i)]['required_conditional_radius']==radius
    after={n:sha((LAB/n).read_bytes()) for n in before};assert before==after
    result=dict(passed=True,scope='Independent exact public coefficient/map residual and frozen selection checks only; no model/calibration/fresh56/heldout/HE run.',
        source_hashes_before=before,source_hashes_after=after,source_files_unchanged=True,review_script_sha256=sha(Path(__file__).read_bytes()),
        selected_manifest_sha256=r['manifest_sha256'],parent_source_archive_sha256=r['source_archive_sha256'],
        polynomial_certificates_independently_recomputed=36,newton_rules_byte_field_equivalent_to_pinned_fallback=77,
        factor4_domains_checked=113,first_passing_ladder_and_prior_failures_checked=True,unsuccessful_polynomial_attempts_retained=610,
        unchanged_other_rules_verified=True,public_downstream_radius_checks=28,polynomial_sites=polynomial,
        mathematical_basis='For |t|<=M, each |T_j(t)|<=T_degree(H)(M). Positive v and |vP^2-1|<1 prohibit zero-crossing; mapped0 lies in interval and P(0)>0. Thus P>0 and relative error<=delta/(1+sqrt(1-delta))<=delta/(2-delta).',
        limitations='Ideal-real polynomial with actual binary64 coefficients and map constants only. Initial affine arithmetic rounding, later binary64 operations, CKKS/refresh noise, and actual variance-domain coverage are not certified. Fallback certificate arithmetic was previously reviewed; this review checks exact fallback preservation.',elapsed_s=time.monotonic()-start)
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({n:result[n] for n in ('passed','polynomial_certificates_independently_recomputed','factor4_domains_checked','elapsed_s')}))
if __name__=='__main__':main()
