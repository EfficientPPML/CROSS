"""Explicit public learned-weight NumPy fixtures, never a model forward."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from join_ideal import Arithmetic as JoinArithmetic,Linear,OuterIdentityRefresh as JoinIdentityRefresh,install_matrix,Value,DELTA
from encrypted_qwen_attention_join import QwenAttentionJoinResult
from encrypted_qwen_decoder_tail import QwenDecoderTail,QwenDecoderTailContract,PARAMETERS
P=Path(__file__).resolve().parent
BODY=Path('/tmp/qwen-decoder-parameters-20261103/actual-v1/public_matrices')
GAMMA=Path('/tmp/qwen-rmsnorm-owner-20261103/client_gamma_fixtures.npz')
class Arithmetic(JoinArithmetic):
    """Same ideal arithmetic, with mandatory finite checks at each primitive."""
    @staticmethod
    def finite(value):
        if value.polynomial.shape!=(65536,) or not np.isfinite(value.polynomial).all():raise FloatingPointError('nonfinite ideal primitive payload')
        return value
    def weighted_sum(self,*args,**kwargs):return self.finite(super().weighted_sum(*args,**kwargs))
    def multiply(self,*args,**kwargs):return self.finite(super().multiply(*args,**kwargs))
    def add(self,*args,**kwargs):return self.finite(super().add(*args,**kwargs))
    def rotate(self,*args,**kwargs):return self.finite(super().rotate(*args,**kwargs))
class OuterIdentityRefresh(JoinIdentityRefresh):
    def refresh(self,value,*,public_bound):
        Arithmetic.finite(value)
        if np.max(np.abs(value.polynomial))>public_bound:raise ValueError('ideal refresh input exceeds declared public bound')
        result=super().refresh(value,public_bound=public_bound)
        Arithmetic.finite(result)
        return result
def contract(layer=0,sequence=1):
    return QwenDecoderTailContract(*( (P/'inputs'/n).read_bytes() for n in
        ('candidate_manifest.json','site_metadata.json','certificates.json','join_plan.json')),layer,sequence)
def parameters(layer):
    raw=(P/'inputs/decoder_parameters.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARAMETERS:raise ValueError('pinned parameter manifest differs')
    rows=json.loads(raw)['matrices'];weights={}
    for role,name in (('o','self_attn.o_proj'),('gate','mlp.gate_proj'),('up','mlp.up_proj'),('down','mlp.down_proj')):
        d=rows[f'model.layers.{layer}.{name}.weight'];p=BODY/d['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=d['file_sha256']:raise ValueError('pinned matrix file differs')
        weights[role]=np.array(np.load(p,mmap_mode='r',allow_pickle=False),copy=True)
        if hashlib.sha256(weights[role].astype('<f8',copy=False).tobytes()).hexdigest()!=d['sha256']:raise ValueError('pinned matrix payload differs')
    if hashlib.sha256(GAMMA.read_bytes()).hexdigest()!='5f40efb5fac32037999ac8aff5f01bae7f937dd0f1d308b2a694883d01b4ef30':raise ValueError('gamma fixture differs')
    with np.load(GAMMA,allow_pickle=False) as g:gamma=g[f'{layer}.post.inverse_sqrt']
    return raw,weights,gamma
def build(monkeypatch,layer=0,positions=(0,),progress=None,arithmetic=None,linear=None,refresher=None):
    install_matrix(monkeypatch);a=Arithmetic() if arithmetic is None else arithmetic
    linear=Linear(a) if linear is None else linear;r=OuterIdentityRefresh(a) if refresher is None else refresher
    raw,w,g=parameters(layer)
    o=QwenDecoderTail(a,linear,contract(layer,len(positions)),(P/'inputs/config.json').read_bytes(),parameter_manifest_bytes=raw,
        weights=w,gamma=g,positions=positions,refresher=r,progress=progress)
    return o,a,r,w,g
def synthetic_inputs(layer,positions):
    """Domain-midpoint source and tiny nonzero attended rows, fixed public rule.

    These attended stand-ins are distinct from the separately joined layer0
    cases. No native/model output or validation trajectory is read.
    """
    m=json.loads((P/'inputs/candidate_manifest.json').read_bytes());lo,hi=m['nonlinear'][f'{layer}.post.inverse_sqrt']['domain']
    base=np.sin(np.arange(1024)*.137)+.1*np.cos(np.arange(1024)*.061)
    base*=math.sqrt(math.sqrt(lo*hi)/np.mean(base*base))
    source=np.stack([base*(1.+.001*i) for i in range(len(positions))])
    attended=np.stack([1e-4*np.sin(np.arange(2048)*.127+i*.071) for i in range(len(positions))])
    return source,attended
def pack(source,attended,positions,layer):
    inputs=[];outputs=[];meta=[];level=22 if len(positions)==1 else 20
    for start in range(0,len(positions),16):
        n=min(16,len(positions)-start);x=np.full((16,4096),-321.25);x[:n,:1024]=source[start:start+n]
        v=np.full((16,4096),471.5);v[:n,:2048]=attended[start:start+n]
        inputs.append(Value(x.ravel(),25,math.nextafter(DELTA,math.inf)))
        outputs.append(Value(v.ravel(),level,math.nextafter(DELTA,-math.inf)))
        meta.append(dict(tile=start//16,level=level,scale=outputs[-1]._ckks_scale))
    return tuple(inputs),QwenAttentionJoinResult(tuple(outputs),positions,layer,tuple(meta),())
def reference(source,attended,weights,gamma,layer):
    """Independent dense original-Newton/Chebyshev path; no owner methods."""
    m=json.loads((P/'inputs/candidate_manifest.json').read_bytes());domains=[]
    def check(label,x,domain):
        row=dict(site=label,lo=float(np.min(x)),hi=float(np.max(x)),domain=list(domain),passed=bool(np.isfinite(x).all() and np.all(x>=domain[0]) and np.all(x<=domain[1])))
        domains.append(row)
        if not row['passed']:raise ValueError('independent declared domain failure: '+str(row))
    residual=source+attended@weights['o'].T
    site=f'{layer}.post.inverse_sqrt';rule=m['nonlinear'][site];variance=np.mean(residual*residual,axis=-1)+1e-6;check(site,variance,rule['domain'])
    if rule['method']=='polynomial':
        lo,hi=rule['domain'];t=variance*(2/(hi-lo))-(hi+lo)/(hi-lo)
        inv=np.polynomial.chebyshev.chebval(t,rule['evaluation_coefficients'])
    else:
        inv=variance*rule['rounded_first_weight']+rule['rounded_first_bias']
        for _ in range(1,rule['iterations']):inv=inv*(1.5-.5*(variance*(inv*inv)))
    normalized=residual*inv[:,None]*gamma;gate=normalized@weights['gate'].T;up=normalized@weights['up'].T
    rule=m['nonlinear'][f'{layer}.silu'];check(f'{layer}.silu',gate,rule['domain'])
    def polynomial(x,p,factor=1.):
        lo,hi=p['domain'];t=x*(2/(hi-lo)*factor)-(hi+lo)/(hi-lo)
        return np.polynomial.chebyshev.chebval(t,p['chebyshev_coefficients']) if p['strategy']=='chebyshev' else np.polynomial.polynomial.polyval(t,p['evaluation_coefficients'])
    if rule['method']=='polynomial':activation=polynomial(gate,rule)
    else:
        y=polynomial(gate,rule['base'],2.**-(rule['doublings']+1));ir=rule['reciprocal']
        for j in range(rule['doublings']):
            d=1.+y*y;check(f'{layer}.silu.d{j}',d,ir['domain']);inverse=d*ir['rounded_first_weight']+ir['rounded_first_bias']
            for _ in range(1,ir['iterations']):inverse=inverse*(2.-d*inverse)
            y=2.*(y*inverse)
        activation=gate*(.5*y+.5)
    output=residual+(activation*up)@weights['down'].T
    return output,domains
def expected_counts(owner):
    from collections import Counter
    from encrypted_packed_qwen import MatrixDiagonals
    from encrypted_chained_streaming_linear import linear_rotation_plan
    count=Counter(weighted=1,multiply=1,add=2) # original mask, gated product, residual adds
    for w in owner.weights:
        p=linear_rotation_plan(MatrixDiagonals(w,65536,4096),strategy='chained')
        count.update(rotate=p.nonzero_rotation_calls,weighted=len(p.giant_indices),add=len(p.giant_indices)-1)
    n=owner.norm.contract;count.update(rotate=20,add=20,weighted=n.degree+3 if n.method=='polynomial' else n.iterations+2,
        multiply=n.degree+1 if n.method=='polynomial' else 3*n.iterations-1)
    c=owner.silu.contract
    count.update(weighted=3+7*c.doublings if c.doublings else c.degree+1,multiply=9+10*c.doublings if c.doublings else c.degree-1)
    return Counter({k:v*owner.layout.tile_count for k,v in count.items()})
