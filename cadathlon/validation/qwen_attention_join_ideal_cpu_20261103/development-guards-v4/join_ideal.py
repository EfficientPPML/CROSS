"""Explicit ideal test seams; real owner/chained loops, no HE backend."""
from pathlib import Path
from numbers import Integral,Real
from typing import Optional,Tuple
import ast,hashlib,json,math,sys
import numpy as np
from ideal_primitives import Arithmetic as BaseArithmetic,Value,Q,PP,DELTA
from ideal_refresh import IdentityRefresh
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_streaming_linear import RowTileLayout
P=Path(__file__).resolve().parent

class Arithmetic(BaseArithmetic):
    def __init__(self):
        super().__init__()
        # Public keys are placeholders; no key payload or encrypted execution.
        union=set(json.loads((P/'inputs/join_plan.json').read_bytes())['resources']['preliminary_dense_case_key_union'])
        union|={s%65536 for s in range(-4096,4097)}
        for key in union:
            self.ctx._param_cache.raw_rotation_keys[key]={'a':object(),'b':object()}
            self.ctx._param_cache.coef_maps[key]=object()
        self.ctx._param_cache.num_q_at_level=lambda level:2*(level+1)
    def _ensure_capacity(self,values,required):
        if any(v.level<required for v in values):raise ValueError('ideal static capacity exhausted')
        return tuple(values)
    def _prepare(self,level):pass
    def rotate(self,value,shift):
        if type(shift)is not int:raise ValueError('integer rotation required')
        if shift%65536==0:return value
        return super().rotate(value,shift)

class Linear(ChainedStreamingLinear):
    def __init__(self,a):self.arithmetic=a;self.ctx=a.ctx;self.layout=RowTileLayout(65536)

class OuterIdentityRefresh(IdentityRefresh):
    def refresh(self,value,*,public_bound):
        if type(public_bound)is not float or public_bound not in tuple(float(2**n) for n in range(16)) or value.level<2:
            raise ValueError('explicit power-of-two B1..32768 ideal refresh required')
        self.calls.append((value.level,public_bound))
        out=Value(value.polynomial,25,math.nextafter(DELTA,math.inf if len(self.calls)%2 else -math.inf))
        if self.inject is not None:self.inject(value,out)
        return out

def install_matrix(monkeypatch):
    fn=next(n for n in ast.parse((P/'checkout/jaxite_word/bsgs.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_bsgs_params')
    cls=next(n for n in ast.parse((P/'reference_sources/encrypted_packed_qwen.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MatrixDiagonals')
    ns=dict(Integral=Integral,Real=Real,Optional=Optional,Tuple=Tuple,math=math,np=np)
    exec(compile(ast.Module(body=[fn,cls],type_ignores=[]),'frozen_public_matrix_planner','exec'),ns)
    monkeypatch.setattr(sys.modules['encrypted_packed_qwen'],'MatrixDiagonals',ns['MatrixDiagonals'],raising=False)
    import encrypted_packed_transformer as packed
    # The NumPy payload has no RNS axes. Only the queue size seam is replaced;
    # actual diagonal masks, baby/giant loops and all arithmetic calls execute.
    monkeypatch.setattr(packed,'output_payload_bytes',lambda graph,value:2*131072*(2*value.level)*4)

def gamma(layer):
    path=Path('/tmp/qwen-rmsnorm-owner-20261103/client_gamma_fixtures.npz')
    expected='5f40efb5fac32037999ac8aff5f01bae7f937dd0f1d308b2a694883d01b4ef30'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('public gamma fixture differs')
    with np.load(path,allow_pickle=False) as values:
        return {kind:values[f'{layer}.{kind}.inverse_sqrt'] for kind in ('input','q','k')}

def weights(layer):
    manifest=(P/'qkv-export-v1/parameter_manifest.json').read_bytes()
    metadata=json.loads(manifest);result={}
    for kind in ('q','k','v'):
        row=metadata['matrices'][f'model.layers.{layer}.self_attn.{kind}_proj.weight'];path=P/'public_qkv_parameters'/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['file_sha256']:raise ValueError('public tensor file differs')
        result[kind]=np.array(np.load(path,mmap_mode='r',allow_pickle=False),copy=True)
    return manifest,result

def dense_reference(rows,weights,gammas,layer,positions):
    """Independent dense original-Newton/Chebyshev oracle, with domain records."""
    m=json.loads((P/'inputs/candidate_manifest.json').read_bytes());domains=[]
    def check(site,value,domain):
        domains.append(dict(site=site,lo=float(np.min(value)),hi=float(np.max(value)),domain=list(domain),
            passed=bool(np.isfinite(value).all() and np.all(value>=domain[0]) and np.all(value<=domain[1]))))
    def rms(x,kind):
        site=f'{layer}.{kind}.inverse_sqrt';rule=m['nonlinear'][site];variance=np.mean(x*x,axis=-1)+1e-6;check(site,variance,rule['domain'])
        if rule['method']=='polynomial':
            a,b=rule['domain'];t=variance*(2./(b-a))-(a+b)/(b-a)
            inverse=np.polynomial.chebyshev.chebval(t,rule['evaluation_coefficients'])
        else:
            inverse=rule['rounded_first_weight']*variance+rule['rounded_first_bias']
            for _ in range(1,rule['iterations']):inverse=inverse*(1.5-.5*(variance*(inverse*inverse)))
        return x*inverse[...,None]*gammas[kind]
    x=rms(rows,'input');q=rms((x@weights['q'].T).reshape(len(rows),16,128),'q');k=rms((x@weights['k'].T).reshape(len(rows),8,128),'k')
    v=(x@weights['v'].T).reshape(len(rows),8,128)
    frequencies=1e6**(-np.arange(0,128,2)/128)
    for row,pos in enumerate(positions):
        angle=pos*frequencies;c=np.cos(angle);s=np.sin(angle)
        for array in (q,k):
            left=array[row,:,:64].copy();right=array[row,:,64:].copy()
            array[row,:,:64]=left*c-right*s;array[row,:,64:]=right*c+left*s
    k=np.repeat(k,2,axis=1);v=np.repeat(v,2,axis=1);out=np.empty((len(rows),16,128))
    for query in range(len(rows)):
        if query==0:out[query]=v[0];continue
        count=query+1;scores=np.einsum('hd,khd->hk',q[query],k[:count])/math.sqrt(128)
        centered=scores-scores.sum(axis=-1,keepdims=True)*(1./count)
        check(f'{layer}.centered.K{count}',centered,m['softmax'][str(layer)]['centered_domain'])
        squarings=m['softmax'][str(layer)]['squarings'];mapped=centered/(2**(squarings-1))
        base=np.polynomial.polynomial.polyval(mapped,m['exp']['evaluation_coefficients']);alpha=1./count
        denominator=base.sum(axis=-1)*alpha;rule=m['initial_reciprocal'];check('initial',denominator,rule['domain'])
        def inverse(d,rule):
            y=rule['rounded_first_weight']*d+rule['rounded_first_bias']
            for _ in range(1,rule['iterations']):y=y*(2.-d*y)
            return y
        p=base*inverse(denominator,rule)[:,None]*alpha
        rule=m['reciprocal_by_keys'][str(count)]
        for round_ in range(squarings):
            square=p*p;denominator=square.sum(axis=-1);check(f'K{count}.round{round_}',denominator,rule['domain'])
            p=square*inverse(denominator,rule)[:,None]
        out[query]=np.einsum('hk,khd->hd',p,v[:count])
    return out.reshape(len(rows),2048),domains
