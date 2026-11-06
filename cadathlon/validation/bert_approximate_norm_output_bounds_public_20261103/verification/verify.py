"""Independent exact power-basis certificate verification; no model forward."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,math,struct,sys,tarfile

def trim(x):
    while len(x)>1 and x[-1]==0:x.pop()
    return x
def add(a,b):
    out=[F()]*max(len(a),len(b))
    for i,v in enumerate(a):out[i]+=v
    for i,v in enumerate(b):out[i]+=v
    return trim(out)
def mul(a,b):
    out=[F()]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return trim(out)
def cp(c):
    before=[F(1)];current=[F(),F(1)];out=[c[0]]
    for degree,v in enumerate(c[1:],1):
        out=add(out,[v*x for x in current])
        before,current=current,add([F()]+[2*x for x in current],[-x for x in before])
    return trim(out)
def clenshaw(c,x):
    b1=b2=F()
    for v in reversed(c[1:]):b1,b2=v+2*x*b1-b2,b1
    return c[0]+x*b1-b2

def frac(record):return F(*map(int,record))
def rec(record):return frac(record['exact_fraction'])
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(base,checkpoint=None):
    sys.set_int_max_str_digits(100000)
    r=json.loads((base/'report.json').read_text());assert r['passed']
    assert r['source_hashes_before']==r['source_hashes_after'] and r['source_files_unchanged']
    assert digest(base/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(base/'sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len(r['source_hashes_before'])==len({m.name for m in members})
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
        raw=t.extractfile('candidate_manifest.json').read();m=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==r['manifest_sha256']
    assert len(r['sites'])==24
    for site,row in r['sites'].items():
        rule=m['norms'][site];p=rule['polynomial'];lo,hi=map(float,p['domain'])
        A,B=F(2/(hi-lo)),F(-(hi+lo)/(hi-lo));assert A==rec(row['rounded_map_A']) and B==rec(row['rounded_map_B'])
        P=list(map(frac,row['exact_P_chebyshev']));H=list(map(frac,row['exact_H_chebyshev']))
        evaluation=[F(float.fromhex(s)) for s in row['actual_evaluated_coefficients']]
        packed=b''.join(struct.pack('<d',float(x)) for x in evaluation)
        assert hashlib.sha256(packed).hexdigest()==row['actual_evaluated_coefficient_sha256']
        if rule['method']=='power':assert cp(P)==trim(evaluation)
        else:assert P==list(map(F,p['coefficients']))
        # Independent ordinary-power multiplication, not the certificate's
        #Chebyshev product identity or conversion routine.
        assert cp(H)==add(mul([-B/A,1/A],mul(cp(P),cp(P))),[F(-1)])
        M=max(F(1),abs(A*F(lo)+B),abs(A*F(hi)+B));assert M==rec(row['extension_M'])
        delta=clenshaw([abs(v) for v in H],M);assert delta==rec(row['squared_relative_envelope'])
        assert rec(row['inverse_energy_factor'])**2>=1+delta
        alpha=F(1./768);factor=rec(row['normalized_coordinate_factor'])
        assert rec(row['variance_weight'])==alpha and factor**2*alpha>=1+delta
        bound=rec(row['output_absolute_bound']);guarded=rec(row['guarded_output_bound'])
        assert guarded==bound*F(9,8) and row['refresh_bound']>=guarded
    affine_checked=False
    if checkpoint is not None:
        assert digest(checkpoint)==r['checkpoint_sha256']
        with checkpoint.open('rb') as f:
            n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n));offset=n+8
            for site,row in r['sites'].items():
                arrays=[]
                for name in row['public_affine_tensor_sha256']:
                    desc=header[name];assert desc['dtype']=='F32' and desc['shape']==[768]
                    a,b=desc['data_offsets'];f.seek(offset+a);raw=f.read(b-a)
                    assert hashlib.sha256(raw).hexdigest()==row['public_affine_tensor_sha256'][name]
                    arrays.append([F(v) for (v,) in struct.iter_unpack('<f',raw)])
                gamma,beta=arrays;factor=rec(row['normalized_coordinate_factor'])
                terms=[factor*abs(g)+abs(b) for g,b in zip(gamma,beta)]
                assert max(terms)==rec(row['output_absolute_bound']) and terms.index(max(terms))==row['maximum_coordinate']
        assert digest(checkpoint)==r['checkpoint_sha256'];affine_checked=True
    assert r['all_sites_fit_B128_with_9_over_8_headroom'] and max(v['refresh_bound'] for v in r['sites'].values())<=128
    assert not r['HE_executed'] and not r['model_evaluated'] and not r['actual_floating_trajectory_or_CKKS_error_certified']
    result=dict(passed=True,sites=24,exact_polynomial_identities=24,exact_extension_envelopes=24,public_affine_bytes_checked=affine_checked,scope='Conditional ideal coefficients, not ciphertext precision/domain propagation.')
    print(json.dumps(result));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('--checkpoint',type=Path);a=p.parse_args();verify(a.base,a.checkpoint)
