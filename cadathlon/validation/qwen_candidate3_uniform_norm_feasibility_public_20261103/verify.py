"""Offline exact residual and directed recurrence checks; no model/backend."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext,ROUND_CEILING
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent

def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in pins.items())
    r=json.loads((ROOT/'public-v1/report.json').read_text());assert r['passed'] and not r['model_evaluated'] and not r['HE_executed'] and not r['replacement_candidate_selected']
    assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==12
    archive=ROOT/'public-v1/sources.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['source_archive_sha256']
    with tarfile.open(archive) as t:
        ms=t.getmembers();assert len(ms)==12==len({m.name for m in ms}) and all(m.isfile() for m in ms)
        raw={m.name:t.extractfile(m).read() for m in ms};assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==r['source_hashes_before']
    original=json.loads(raw['inputs/candidate2_manifest.json']);multipliers=json.loads(raw['inputs/norm_multipliers.json'])
    assert len(r['sites'])==113 and r['summary']['proposed_newton_updates']==[9,24]
    for name,x in r['sites'].items():
        old=original['nonlinear'][name]['domain'];a,b=map(F,x['expanded_domain'])
        assert a<=max(F(1e-6),F(old[0])/4) and b>=4*F(old[1]) and a>=F(1e-6)
        rule=x['rule'];w,bias=F(rule['rounded_first_weight']),F(rule['rounded_first_bias'])
        probes=[a,b]+[v for v in (-bias/w,-bias/(3*w)) if a<=v<=b]
        first=max(abs(1-v*(w*v+bias)**2) for v in probes)
        saved=rule['first_exact_residual_abs_bound'];assert first==F(int(saved['numerator']),int(saved['denominator']))<1
        assert min(w*a+bias,w*b+bias)>0
        with localcontext() as c:
            c.prec=180;c.rounding=ROUND_CEILING;R=Decimal(first.numerator)/Decimal(first.denominator)
            for _ in range(rule['iterations']-1):R=R*R*(3+R)/4
        assert R<=Decimal(rule['final_residual_bound'])
        # The recorded relative upper bound must dominate the exact residual
        # inequality R <= delta*(2-delta), using rational decimal values.
        delta=F(Decimal(rule['final_relative_error_bound']));assert F(R)<=delta*(2-delta) and 0<delta<=F(1,10**6)
        assert rule['depth']==4*rule['iterations']-3 and x['counts']['complete_rms_depth']==rule['depth']+4
        assert F(x['inverse_abs_bound_conditional'])**2*a>=(1+delta)**2
    for layer,record in r['silu_radius_recheck'].items():
        delta=F(Decimal(r['sites'][layer+'.post.inverse_sqrt']['rule']['final_relative_error_bound']))
        z=F(9,8)*(1+delta)*F(multipliers[layer]['native_gate_upper']);radius=-(-z.numerator//z.denominator)
        assert radius==record['conditional_public_radius'] and record['covered']
    assert not list(ROOT.rglob('*.npy')) and not list(ROOT.rglob('*.npz'))
    print(json.dumps(dict(passed=True,uniform_certificates=113,unchanged_silu_radius_checks=28,model_evaluated=False,HE_executed=False)))

if __name__=='__main__':main()
