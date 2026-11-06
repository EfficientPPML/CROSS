"""Explicit NumPy payload/identity refresh test seams; no HE or model."""
from pathlib import Path
from collections import OrderedDict
import json,numpy as np
from ideal_primitives import Arithmetic as BaseArithmetic,Value,Q,DELTA
from ideal_refresh import IdentityRefresh
from encrypted_compilation import ExplicitArrayCompiler
from qwen_silu_contract import QwenSiLUContract
from encrypted_qwen_silu import QwenSiLU
P=Path(__file__).resolve().parent
class Arithmetic(BaseArithmetic):
    def __init__(self):
        super().__init__();self._multiplication_compiler=ExplicitArrayCompiler();self._multiplication_bindings={};self._constant_cache=OrderedDict();self._active_level=None
        self.rotation._compiler=ExplicitArrayCompiler();self.rotation._binding=None
        for helper in (self._plaintext_multiply,self._rescale):helper.compiler=ExplicitArrayCompiler();helper.binding=None

def contract(layer):return QwenSiLUContract((P/'inputs/candidate_manifest.json').read_bytes(),(P/'inputs/certificates.json').read_bytes(),layer)
def owner(layer,*,positions=(0,),progress=None):
    a=Arithmetic();c=contract(layer);r=IdentityRefresh(a) if c.doublings else None
    return QwenSiLU(a,c,(P/'inputs/config.json').read_bytes(),positions=positions,refresher=r,progress=progress),a,r

def reference(x,layer):
    rule=json.loads((P/'inputs/candidate_manifest.json').read_bytes())['nonlinear'][f'{layer}.silu'];domains=[]
    def check(name,v,domain):
        if not np.isfinite(v).all() or v.min()<domain[0] or v.max()>domain[1]:raise ValueError('independent declared domain failure: '+name)
        domains.append(dict(label=name,minimum=float(v.min()),maximum=float(v.max()),domain=domain))
    check('input',x,rule['domain'])
    def polynomial(v,p,factor=1.):
        lo,hi=p['domain'];t=v*(2/(hi-lo)*factor)-(hi+lo)/(hi-lo)
        return np.polynomial.chebyshev.chebval(t,p['chebyshev_coefficients']) if p['strategy']=='chebyshev' else np.polynomial.polynomial.polyval(t,p['evaluation_coefficients'])
    if rule['method']=='polynomial':return polynomial(x,rule),domains
    y=polynomial(x,rule['base'],2.**-(rule['doublings']+1));ir=rule['reciprocal']
    for j in range(rule['doublings']):
        denominator=1.+y*y;check('denominator'+str(j),denominator,ir['domain'])
        inverse=denominator*ir['rounded_first_weight']+ir['rounded_first_bias']
        for _ in range(ir['iterations']-1):inverse=inverse*(2.-denominator*inverse)
        y=2.*(y*inverse)
    return x*(.5*y+.5),domains
