"""Offline artifact/interval/schedule verification; no backend or model reads."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import hashlib,json,tarfile,math
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fraction(x):return F(int(x['numerator']),int(x['denominator']))
def source_archive(path,before):
    with tarfile.open(path) as t:
        members=t.getmembers()
        assert len(members)==len({m.name for m in members})==len(before)
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        data={m.name:t.extractfile(m).read() for m in members}
        assert {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}==before
    return data

def main():
    artifacts=json.loads((ROOT/'artifact_sha256.json').read_text())
    for name,digest in artifacts.items():assert sha(ROOT/name)==digest,name
    report=json.loads((ROOT/'public-v2/report.json').read_text())
    prep=json.loads((ROOT/'source-bound-v2/report.json').read_text())
    assert report['passed'] and prep['passed']
    before=report['source_hashes_before']
    assert report['source_hashes_after']==before==prep['source_hashes_before']==prep['source_hashes_after']
    assert len(before)==9
    assert sha(ROOT/'public-v2/sources.tar.gz')==report['source_archive_sha256']
    data=source_archive(ROOT/'public-v2/sources.tar.gz',before)
    old=json.loads((ROOT/'historical-public-v1/report.json').read_text())
    assert sha(ROOT/'historical-public-v1/sources.tar.gz')==old['source_archive_sha256']
    source_archive(ROOT/'historical-public-v1/sources.tar.gz',old['source_hashes_before'])
    manifest=json.loads(data['inputs/candidate2_manifest.json']);outer=json.loads(data['inputs/outer_domains.json'])
    assert hashlib.sha256(data['inputs/candidate2_manifest.json']).hexdigest()==report['candidate2_sha256']=='0af45a8fc6d13b9d429ee1af2311b2cb6ac0aa4cde02eca56478472762482489'
    assert len(report['sites'])==113 and set(report['sites'])==set(outer['sites'])
    updates=[];schedules=0
    for label,row in report['sites'].items():
        a=F(*map(int,outer['sites'][label]['lower_fraction']));b=F(*map(int,outer['sites'][label]['upper_fraction']))
        k=row['seed_exponent_rule_k'];seed=F(2)**(-k)
        assert seed*seed*b<=1 and seed*seed*b*4>1 and a*1024**2>=1
        residual=1-a*seed*seed
        assert fraction(row['initial_residual_upper'])==residual
        assert fraction(row['first_scaled_residual_gap'])==a*seed*seed/4
        bounds=row['exact_real_residual_bounds'];n=row['iterations'];updates.append(n)
        assert len(bounds)==n and n>=2
        for text in bounds:
            bound=F(Decimal(text));assert residual*residual*(3+residual)/4<=bound<1
            residual=bound
        rel=F(Decimal(row['final_relative_error_upper']))
        assert residual/(2-residual)<=rel<=F(1,10**6)
        assert row['tracked_aligned_state_depth']==3*n+1 and row['complete_rms_aligned_depth']==3*n+5
        assert row['ciphertext_products_inverse']==3*n-1 and row['weighted_operations_inverse']==4*n
        for schedule in row['schedules'].values():
            capacity=schedule['refresh_capacity'];variance=schedule['variance_level'];w=variance-1;z=None;events=[]
            for i in range(n):
                if w<7 or (z is not None and z<7):
                    assert w>=2 and (z is None or z>=2)
                    events.append(dict(before_update=i+1,w_input=w,z_input=z,ciphertext_refresh_calls=1+(z is not None)))
                    w=capacity;z=capacity if z is not None else None
                new_z=w-1 if z is None else min(z,w-1)-2
                w=w-3;z=new_z;assert min(w,z)>=4
            assert schedule['events']==events
            assert schedule['state_refresh_calls']==sum(x['ciphertext_refresh_calls'] for x in events)
            assert schedule['final_w_level']==w and schedule['final_z_level']==z
            assert schedule['final_norm_output_level']==min(variance+2,z)-2>=2
            schedules+=1
    soft=report['softmax_premises']
    coeff=list(map(F,manifest['exp']['evaluation_coefficients']))
    E=sum((abs(c-F(1,2**i*math.factorial(i))) for i,c in enumerate(coeff)),F())+F(2,2**9*math.factorial(9))
    assert fraction(soft['exp_uniform_abs_error'])==E
    assert F(1,2)<F(3,5)-E and F(5,3)+E<2
    for key,check in soft['reciprocal_checks'].items():
        rule=manifest['initial_reciprocal'] if key=='initial' else manifest['reciprocal_by_keys'][key]
        a,b=map(F,rule['domain']);weight=F(rule['rounded_first_weight']);bias=F(rule['rounded_first_bias'])
        points=[a,b];vertex=-bias/(2*weight)
        if a<=vertex<=b:points.append(vertex)
        residual=max(abs(1-x*(weight*x+bias)) for x in points)
        assert residual==fraction(check['first_exact_residual_abs']) and min(weight*a+bias,weight*b+bias)>0
        final=F(Decimal(check['final_residual_upper']))
        assert residual**(2**(rule['iterations']-1))<=final<=F(1,10**12)
    for K in range(1,33):
        beta=K*F(1./K);mass_low=beta*(1-F(1,10**12))
        assert mass_low*mass_low/K>=F(1,2*K) and beta*beta<=2
    assert report['summary']['minimum_updates']==min(updates)==18 and max(updates)==60
    assert not report['model_evaluated'] and not report['HE_executed'] and not report['coefficient_selection']
    assert not report['uniform_floating_or_HE_repair_certified']
    print(json.dumps(dict(passed=True,source_files=9,public_native_sites=113,uniform_exact_real_interval_checks=113,explicit_level_schedules=schedules,rounded_reciprocal_checks=33,scope='Offline artifact/public-math verification; no weights, model, tokens or HE loaded.')))
if __name__=='__main__':main()
