"""Offline exact bound verification; no ciphertext/model/activation reads."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,math,struct,tarfile

def q(x):return F(*map(int,x))
def verify(base):
    r=json.loads((base/'report.json').read_text());assert r['passed'];c=r['certificate']
    assert r['source_hashes_before']==r['source_hashes_after']
    assert hashlib.sha256((base/'sources.tar.gz').read_bytes()).hexdigest()==r['source_archive_sha256']
    with tarfile.open(base/'sources.tar.gz') as t:
        rows=t.getmembers();assert len(rows)==len(r['source_hashes_before'])==len({m.name for m in rows})
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in rows}==r['source_hashes_before']
        raw=t.extractfile('candidate_manifest.json').read();manifest=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==r['manifest_sha256']
    a=[F(float.fromhex(x)) for x in c['actual_power_coefficients']]
    assert hashlib.sha256(b''.join(struct.pack('<d',float(x)) for x in a)).hexdigest()==c['actual_power_coefficient_sha256']
    assert q(c['base_input_scale'])==F(1,64) and q(c['rounded_map_A'])==1 and q(c['rounded_map_B'])==0
    assert list(map(q,c['actual_mapped_endpoints']))==[F(-17,32),F(17,32)]
    base_bound=sum(abs(v)*F(17,32)**i for i,v in enumerate(a))
    assert base_bound==q(c['base_absolute_upper'])<1
    inv=c['reciprocal'];lo,hi=map(q,inv['domain']);seed=manifest['tanh_reciprocal']['seed']
    w,b=F(-seed*seed),F(2*seed);assert w==q(inv['first_weight']) and b==q(inv['first_bias'])
    points=[lo,hi];vertex=-b/(2*w)
    if lo<=vertex<=hi:points.append(vertex)
    R=max(abs(1-x*(w*x+b)) for x in points)
    assert R==q(inv['first_residual_absolute_upper'])<1 and min(w*lo+b,w*hi+b)>0
    assert q(inv['final_residual_upper'])==R**16<1 and inv['iterations']==5
    assert len(c['rounds'])==6
    prior=base_bound
    for row in c['rounds']:
        assert q(row['input_absolute_upper'])==prior
        assert list(map(q,row['denominator_domain']))==[F(1),1+prior*prior]
        assert 1+prior*prior<=hi and q(row['output_absolute_upper'])==1
        assert q(row['guarded_output_upper'])==F(9,8)<row['refresh_bound']==2
        prior=F(1)
    p=c['conditional_refresh_perturbation'];assert q(p['next_denominator_upper'])==1+(1+q(p['additive_error_upper']))**2==F(2113,1024)<hi
    assert not p['measured_or_certified_CKKS_error'] and not c['HE_error_certified'] and not r['HE_executed']
    print(json.dumps(dict(passed=True,six_exact_induction_steps=True,rounded_first_affine_checked=True,conditional_B2_only=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);verify(p.parse_args().base)
