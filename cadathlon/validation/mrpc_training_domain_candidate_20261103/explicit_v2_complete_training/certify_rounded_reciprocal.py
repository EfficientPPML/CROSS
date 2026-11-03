"""Separate certificate for unchanged BERT v2 reciprocal schedules.

Certifies ideal real updates after the actual binary64-rounded first affine.
Later floating-point arithmetic and CKKS encoding/rescale error are excluded.
No coefficient, seed, interval or iteration count is selected or changed.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal,localcontext,ROUND_CEILING
import hashlib,json,math,tarfile,io

HERE=Path(__file__).resolve().parent

def certify(rule):
    lo,hi=map(float,rule['domain']);seed=float(rule['seed']);iterations=rule['iterations']
    if not 0<lo<hi or not all(map(math.isfinite,(lo,hi,seed))) or seed<=0 or type(iterations) is not int or iterations<1:
        raise ValueError('positive finite interval/seed and integer update count required')
    weight=-seed*seed;bias=2*seed
    if not math.isfinite(weight) or not math.isfinite(bias) or weight==0:
        raise ValueError('rounded affine coefficients outside representable envelope')
    a,b,w,c=map(Fraction.from_float,(lo,hi,weight,bias))
    points=[a,b];vertex=-c/(2*w)
    if a<vertex<b:points.append(vertex)
    residuals=[1-w*x*x-c*x for x in points]
    initial=max(map(abs,residuals))
    if initial>=1:raise ValueError('rounded initial affine is outside Newton convergence envelope')
    with localcontext() as context:
        context.prec=100;context.rounding=ROUND_CEILING
        upper=Decimal(initial.numerator)/Decimal(initial.denominator)
        for _ in range(iterations-1):upper=upper*upper
    return dict(domain=[lo,hi],seed_hex=seed.hex(),weight_hex=weight.hex(),bias_hex=bias.hex(),iterations=iterations,
        exact_stationary_point_in_domain=a<vertex<b,
        exact_initial_residual_upper=[str(initial.numerator),str(initial.denominator)],
        final_relative_residual_upper=str(upper),fixed_target='1e-12',passed=upper<=Decimal('1e-12'))

def main():
    output=HERE/'rounded-reciprocal-v2';output.mkdir(exist_ok=False)
    manifest=HERE/'candidate-v2/candidate_manifest.json';m=json.loads(manifest.read_text())
    rules={'initial':m['initial_reciprocal'],'tanh':m['tanh_reciprocal']}
    rules.update({f'K{k}':v for k,v in m['normalization_reciprocal_by_public_K'].items()})
    measured=[Path(__file__),HERE/'test_rounded_reciprocal.py',manifest,HERE/'approximate.py']
    before={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in measured}
    results={name:certify(rule) for name,rule in rules.items()}
    worst=max(results,key=lambda key:Decimal(results[key]['final_relative_residual_upper']))
    with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
        for name in before:
            raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    with tarfile.open(output/'sources.tar.gz') as archive:
        members=archive.getmembers();assert len(members)==len(before)
        assert all(hashlib.sha256(archive.extractfile(member).read()).hexdigest()==before['/'+member.name] for member in members)
    after={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in before}
    report=dict(scope=__doc__,passed=len(results)==130 and all(r['passed'] for r in results.values()) and before==after,
        rule_count=len(results),rules=results,worst_rule=worst,worst_upper=results[worst]['final_relative_residual_upper'],
        arithmetic='Exact Fraction extrema of 1-x*(weight*x+bias), including its rational stationary point; 100-digit Decimal ceiling conversion and squaring for each remaining update.',
        candidate_unchanged=True,candidate_manifest_sha256=before[str(manifest)],source_hashes_before=before,source_hashes_after=after,
        source_files_unchanged=before==after,source_archive_verified=True,
        source_archive_sha256=hashlib.sha256((output/'sources.tar.gz').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','rule_count','worst_rule','worst_upper','candidate_manifest_sha256')},indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
