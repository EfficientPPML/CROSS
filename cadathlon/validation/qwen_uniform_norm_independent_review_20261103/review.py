"""Independent readback/algebra audit; no model, fitting, JAX or HE."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,struct,tarfile,time

LAB=Path('/tmp/qwen-rms-polynomial-certificate-20261103')
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def dyadic(row):return F(int(row['numerator_hex'],16),1<<row['denominator_power_of_two'])


def evaluate(coefficients,t,strategy):
    coefficients=list(map(F,coefficients))
    if strategy=='power':
        result=F(0)
        for c in reversed(coefficients):result=result*t+c
        return result
    # Independent backward Clenshaw recurrence in exact rationals.
    next_one=next_two=F(0)
    for c in coefficients[:0:-1]:
        current=c+2*t*next_one-next_two
        next_two,next_one=next_one,current
    return coefficients[0]+t*next_one-next_two


def main():
    start=time.perf_counter();report_path=LAB/'report.json';report=json.loads(report_path.read_text())
    before=report['source_hashes_before']
    assert report['passed'] and report['source_files_unchanged'] and before==report['source_hashes_after']
    assert all(sha(Path(n))==h for n,h in before.items())
    archive=LAB/'sources.tar.gz';assert sha(archive)==report['source_archive_sha256']=='2545fb88ed41d2e3b592cba68dc7a9f267ddfe9a7b39a8fbc240d7a414cd4f41'
    with tarfile.open(archive) as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==4 and all(m.isfile() for m in members)
        assert {'/'+m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==before
    manifest_path=next(Path(n) for n in before if n.endswith('/manifest.json'));manifest=json.loads(manifest_path.read_text())
    expected=[f'{i}.post.inverse_sqrt' for i in (0,1,2,23,24,25,26,27)]
    assert set(report['results'])==set(expected)
    results={}
    for name in expected:
        certified=report['results'][name];rule=manifest['nonlinear'][name];lo,hi=rule['domain']
        a=F(2/(hi-lo));b=F(-(hi+lo)/(hi-lo));mid=(F(lo)+F(hi))/2;half=(F(hi)-F(lo))/2
        assert float(a).hex()==certified['actual_affine']['weight_float_hex']
        assert float(b).hex()==certified['actual_affine']['bias_float_hex']
        assert a*half==dyadic(certified['actual_affine']['alpha'])
        assert a*mid+b==dyadic(certified['actual_affine']['beta'])
        coeff=rule['evaluation_coefficients'];assert hashlib.sha256(struct.pack('<'+'d'*len(coeff),*coeff)).hexdigest()==certified['evaluation_coefficient_f64le_sha256']
        assert certified['domain']==rule['domain'] and certified['strategy']==rule['strategy']
        delta=dyadic(certified['exact_H_coefficient_L1']);assert 0<=delta<1 and certified['certified']
        relative=delta/(2-delta);saved=certified['exact_relative_bound']
        assert relative==F(int(saved['numerator_hex'],16),int(saved['denominator_hex'],16))
        midpoint_positive=evaluate(coeff,a*mid+b,rule['strategy'])>0
        assert midpoint_positive and certified['positive_polynomial_midpoint']
        # Directed endpoints must enclose their exact rationals, including
        # values between adjacent binary64 numbers around one.
        lower,upper=certified['normalized_multiplier_bounds']
        assert F(lower)<=1-relative and F(upper)>=1+relative
        assert F(certified['uniform_relative_error_upper'])>=relative
        assert 0<lower<=upper and certified['uniform_relative_error_upper']<1e-6
        results[name]={'verified':True,'actual_rounded_affine_and_coefficient_binding':True,
                       'independent_exact_midpoint_positive':True,'relative_error_upper':certified['uniform_relative_error_upper'],
                       'directed_multiplier_interval':[lower,upper]}
    paths={**{n:Path(n) for n in before},str(report_path):report_path,str(archive):archive,str(Path(__file__)):Path(__file__)}
    hashes={n:sha(p) for n,p in paths.items()}
    output={'passed':True,'scope':__doc__,'certificate_source_archive_sha256':sha(archive),'certificate_report_sha256':sha(report_path),
            'results':results,'source_hashes':hashes,'elapsed_s':time.perf_counter()-start,
            'reviewed_exact_identities':['T_i*T_j=(T_(i+j)+T_abs(i-j))/2; diagonal/cross factors in integer convolution are correct.',
              'Affine pullback uses rounded binary64 A=2/(hi-lo), B=-(hi+lo)/(hi-lo), composed exactly with v=mid+half*u.',
              '|v*p(v)^2-1|<=delta<1 excludes zeros; independently positive midpoint fixes p positive over the entire connected interval.',
              '|sqrt(v)*p(v)-1|<=delta/(sqrt(1-delta)+1)<=delta/(2-delta); v2 avoids numerical square-root lower bounds.',
              'Lower and upper binary64 multiplier endpoints are directed with exact Fraction comparisons.'],
            'limits':['Does not recompute the full high-degree Chebyshev L1 convolution; its exact implementation and source-bound results were reviewed.',
              'Later floating operation, CKKS, refresh and variance-domain coverage remain excluded.','No coefficient changes or model/validation evaluation.']}
    (OUT/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'passed':True,'sites':8,'max_relative_upper':max(x['relative_error_upper'] for x in results.values()),'elapsed_s':output['elapsed_s']}))

if __name__=='__main__':main()
