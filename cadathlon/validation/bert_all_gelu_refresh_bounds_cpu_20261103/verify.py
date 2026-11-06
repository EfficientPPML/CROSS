"""Offline exact certificate/source verification; no backend or model run."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math,struct,tarfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exact(row):
    result=F(int(row['numerator']),int(row['denominator']))
    assert F(row['upper_binary64'])>=result
    return result
def clenshaw(c,t):
    one=two=F(0)
    for a in reversed(c[1:]):one,two=a+2*t*one-two,one
    return c[0]+t*one-two
def main():
    for n,h in json.loads((HERE/'artifact_sha256.json').read_text()).items():assert sha(HERE/n)==h,n
    reports=[]
    for name in ('historical-v1','current-v2'):
        p=HERE/name;r=json.loads((p/'report.json').read_text());reports.append(r)
        assert r['passed'] and r['source_files_unchanged'] and not r['model_evaluated'] and not r['encrypted_execution']
        assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==3
        assert {n:sha(p/n) for n in r['source_hashes_before']}==r['source_hashes_before']
        assert sha(p/'sources.tar.gz')==r['source_archive_sha256']
        with tarfile.open(p/'sources.tar.gz') as t:
            ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==3 and all(m.isfile() for m in ms)
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']
        terminal=json.loads((p/'terminal.json').read_text());assert terminal['observed_exit_code']==0 and terminal['child_reaped'] and terminal['source_files_unchanged']
    r=reports[1];assert reports[0]['sites']==r['sites'] and len(r['sites'])==12
    manifest=json.loads((HERE/'current-v2/candidate_manifest.json').read_text())
    for key,row in r['sites'].items():
        p=manifest['gelu'][key]['polynomial'];c=list(map(F,p['coefficients']));n=len(c)-1
        assert hashlib.sha256(struct.pack('<'+str(n+1)+'d',*p['coefficients'])).hexdigest()==row['coefficient_sha256']
        lo,hi=map(float,p['domain']);a,b=F(2/(hi-lo)),F(-(hi+lo)/(hi-lo))
        assert exact(row['rounded_map']['weight'])==a and exact(row['rounded_map']['bias'])==b
        endpoints=[F(lo)*a+b,F(hi)*a+b];assert list(map(exact,row['rounded_map']['endpoints']))==endpoints
        M=max(F(1),*map(abs,endpoints));assert exact(row['mapped_extent'])==M
        growth=[F(1),M]
        for i in range(2,n+1):growth.append(2*M*growth[-1]-growth[-2])
        l1=sum((abs(x)*g for x,g in zip(c,growth)),F());D=sum((i*i*abs(x)*growth[i] for i,x in enumerate(c)),F())
        assert exact(row['coefficient_envelope'])==l1 and exact(row['derivative_envelope'])==D
        assert exact(row['endpoint_max_abs'])==max(abs(clenshaw(c,t)) for t in endpoints)
        bound=l1
        if row['grid_intervals']:
            count=row['grid_intervals'];assert count==4096
            largest=max(abs(clenshaw(c,F(-1)+F(2*j,count))) for j in range(count+1))
            assert exact(row['grid_max_abs'])==largest
            distance=F(1,count)+M-1;assert exact(row['nearest_grid_distance'])==distance
            bound=min(l1,largest+D*distance)
        assert exact(row['polynomial_abs_upper'])==bound and exact(row['explicit_margin'])==F(9,8)
        assert exact(row['with_margin'])==bound*F(9,8)
        B=row['refresh_bound'];assert type(B)is int and B>=1 and B&(B-1)==0 and B>=bound*F(9,8)
        assert B==1 or B/2<bound*F(9,8)
        assert row['balanced_levels']==2*math.ceil(math.log2(n))+2
    print('Verified12 exact-real envelopes by independent Clenshaw readback, both3-source archives and artifact hashes; no HE claim.')
if __name__=='__main__':main()
