"""Exact-real bounds for all frozen BERT-v2 GELUs; no model or HE execution."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,io,json,math,os,struct,tarfile,time

HERE=Path(__file__).resolve().parent
MANIFEST_SHA='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'

def sha(data):return hashlib.sha256(data).hexdigest()
def upper(x):
    out=float(x)
    return math.nextafter(out,math.inf) if F(out)<x else out
def record(x):return {'numerator':str(x.numerator),'denominator':str(x.denominator),'upper_binary64':upper(x)}
def power_two_cover(x):
    if x<=0:raise ValueError('positive bound required')
    result=F(1)
    while result<x:result*=2
    while result/2>=x:result/=2
    return result

def value_at(c,t):
    before,current=F(1),t
    result=c[0]
    if len(c)>1:result+=c[1]*t
    for a in c[2:]:
        before,current=current,2*t*current-before
        result+=a*current
    return result

def envelope(c,*,mapped_endpoints,count=4096):
    if type(count)is not int or count<2:raise ValueError('at least two grid intervals required')
    if len(c)<2 or len(mapped_endpoints)!=2 or mapped_endpoints[0]>=mapped_endpoints[1]:raise ValueError('ordered domain and nonconstant polynomial required')
    # Cover the actual rounded domain map, including an ULP beyond [-1,1].
    M=max(F(1),*(abs(x) for x in mapped_endpoints));growth=[F(1),M]
    for n in range(2,len(c)):growth.append(2*M*growth[-1]-growth[-2])
    # |T_n(x)| <= T_n(M). U_(n-1) is a positive sum of T_j with
    # coefficient sum n; T_n'=n U_(n-1), so |T_n'| <= n^2 T_n(M).
    l1=sum((abs(a)*g for a,g in zip(c,growth)),F())
    derivative=sum((n*n*abs(a)*growth[n] for n,a in enumerate(c)),F())
    endpoint_max=max(abs(value_at(c,x)) for x in mapped_endpoints)
    margin=F(9,8)
    mode='extended Chebyshev coefficient envelope'
    largest=argmax=None;distance=None;bound=l1
    if power_two_cover(max(endpoint_max*margin,F(1,2**100)))<power_two_cover(l1*margin):
        largest=F(0);argmax=0
        for j in range(count+1):
            observed=abs(value_at(c,F(-1)+F(2*j,count)))
            if observed>largest:largest,argmax=observed,j
        # The extra M-1 covers the short portion outside the unit interval.
        distance=F(1,count)+(M-1)
        bound=min(l1,largest+derivative*distance)
        mode='exact rational grid plus extended global derivative envelope'
    return dict(method=mode,mapped_extent=record(M),coefficient_envelope=record(l1),
        derivative_envelope=record(derivative),endpoint_max_abs=record(endpoint_max),
        grid_intervals=count if largest is not None else 0,grid_max_abs=None if largest is None else record(largest),
        argmax_index=argmax,nearest_grid_distance=None if distance is None else record(distance),
        polynomial_abs_upper=record(bound),with_margin=record(bound*margin),
        explicit_margin=record(margin),refresh_bound=int(power_two_cover(bound*margin)))

def main():
    start=time.monotonic();paths=[HERE/n for n in ('certify.py','test_certificate.py','candidate_manifest.json')]
    before={p.name:sha(p.read_bytes()) for p in paths}
    assert before['candidate_manifest.json']==MANIFEST_SHA
    manifest=json.loads((HERE/'candidate_manifest.json').read_text());sites={}
    assert set(manifest['gelu'])==set(map(str,range(12)))
    for key,entry in manifest['gelu'].items():
        poly=entry['polynomial'];floats=poly['coefficients'];degree=entry['degree']
        assert entry['strategy']=='balanced Chebyshev' and len(floats)==degree+1
        assert sha(struct.pack('<'+str(len(floats))+'d',*floats))==poly['coefficient_f64le_sha256']
        assert entry['domain']==poly['domain'] and all(type(x)is float and math.isfinite(x) for x in floats)
        lo,hi=map(float,poly['domain']);weight=2./(hi-lo);bias=-(hi+lo)/(hi-lo)
        endpoints=[F(lo)*F(weight)+F(bias),F(hi)*F(weight)+F(bias)]
        row=envelope(list(map(F,floats)),mapped_endpoints=endpoints)
        row.update(degree=degree,domain=[lo,hi],coefficient_sha256=poly['coefficient_f64le_sha256'],
            rounded_map=dict(weight=record(F(weight)),bias=record(F(bias)),endpoints=list(map(record,endpoints))),
            balanced_levels=2*math.ceil(math.log2(degree))+2)
        sites[key]=row
        print(json.dumps(dict(layer=key,bound=row['refresh_bound'],method=row['method'],envelope=row['polynomial_abs_upper']['upper_binary64'])),flush=True)
    after={p.name:sha(p.read_bytes()) for p in paths};assert before==after
    with tarfile.open(HERE/'sources.tar.gz','w:gz') as archive:
        for p in paths:
            b=p.read_bytes();info=tarfile.TarInfo(p.name);info.size=len(b);archive.addfile(info,io.BytesIO(b))
    with tarfile.open(HERE/'sources.tar.gz') as archive:
        assert {m.name:sha(archive.extractfile(m).read()) for m in archive.getmembers()}==before
    result=dict(passed=True,scope=__doc__,manifest_sha256=MANIFEST_SHA,sites=sites,
        source_hashes_before=before,source_hashes_after=after,source_files_unchanged=True,
        source_archive_sha256=sha((HERE/'sources.tar.gz').read_bytes()),cpu_affinity=sorted(os.sched_getaffinity(0)),
        elapsed_s=time.monotonic()-start,model_evaluated=False,encrypted_execution=False,
        limits=['Exact binary64 public coefficients and affine map interpreted as real numbers only.',
          'The 9/8 margin is explicit headroom, not a proved CKKS or floating-arithmetic error allowance.',
          'Actual encrypted inputs must remain in each frozen activation domain; no clipping or observed-range selection occurs.',
          'These bounds do not certify refresh accuracy, repeated composition, native model accuracy or practical latency.'])
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
