"""Explicit component stand-ins for metadata/lifecycle only, no fullmodel oracle.

Real public contracts, scales, final masks, rotation and parent loop execute.
Attention/tail/norm/matrix payloads here are deliberately synthetic substitutes;
these tests cannot establish the frozen28-layer numerical trajectory.
"""
from types import SimpleNamespace
import json,math
import numpy as np
from tail_ideal import Arithmetic,OuterIdentityRefresh,Value,DELTA
from encrypted_scale_planning import weighted_sum_scale,drop_divisor
from encrypted_qwen_rmsnorm import rms_scale_plan
from encrypted_qwen_attention_join import QwenAttentionJoinResult
from encrypted_qwen_decoder_tail import QwenDecoderTailResult
from encrypted_streaming_linear import RowTileLayout
import encrypted_qwen_full_model as runtime
from plan_full import contract

class Parameters:
    def __init__(self,c):self.contract=c;self.calls=[]
    def check(self):pass
    def matrix(self,name):
        self.calls.append(('matrix',name));rows={**json.loads(self.contract.qkv_bytes)['matrices'],**json.loads(self.contract.decoder_bytes)['matrices']}
        return np.broadcast_to(np.float64(1.),tuple(rows[name]['shape']))
    def gamma(self,site):
        self.calls.append(('gamma',site));width=128 if '.q.' in site or '.k.' in site else 1024
        return np.ones(width)

class Linear:
    def __init__(self,a):self.arithmetic=a;self.ctx=a.ctx;self.layout=RowTileLayout(65536);self.calls=[]
    def required_rotation_indices(self,*args):return ()
    def _validate_ciphertext(self,*args):pass
    def linear_tile(self,value,weights,*,valid_rows):
        self.calls.append((value,weights.shape,valid_rows));self.arithmetic.calls['synthetic_linear']+=1
        x=np.zeros((16,4096));x[0,:weights.shape[0]]=value.polynomial[0]
        scale=weighted_sum_scale([value._ckks_scale],self.arithmetic.scale,drop_divisor(self.ctx._param_cache.q_towers,value.level))['output_scale']
        return Value(x.ravel(),value.level-1,scale)

class Attention:
    records=[]
    def __init__(self,a,linear,c,config,*,positions,progress,**kwargs):
        self.arithmetic=a;self.ctx=a.ctx;self.contract=c;self.positions=positions;self.progress=progress
    def _join_guard(self):pass
    def evaluate(self,inputs):
        Attention.records.append((self.contract.layer,inputs,self));out=[];level=22 if len(self.positions)==1 else 20
        for i,x in enumerate(inputs):
            self.arithmetic.calls['synthetic_attention']+=1;v=Value(x.polynomial,level,x._ckks_scale)
            self.progress('synthetic',v,dict(level=level,scale=v._ckks_scale));out.append(v)
        return QwenAttentionJoinResult(tuple(out),self.positions,self.contract.layer,tuple(dict(tile=i,level=level,scale=v._ckks_scale) for i,v in enumerate(out)),())

class Tail:
    records=[]
    def __init__(self,a,linear,c,config,*,positions,progress,**kwargs):
        self.arithmetic=a;self.ctx=a.ctx;self.contract=c;self.positions=positions;self.progress=progress
    def _join_guard(self):pass
    def evaluate(self,attention,inputs):
        Tail.records.append((self.contract.layer,attention,inputs,self));out=[]
        for i,x in enumerate(inputs):
            self.arithmetic.calls['synthetic_tail']+=1
            scale=weighted_sum_scale([x._ckks_scale],self.arithmetic.scale,drop_divisor(self.ctx._param_cache.q_towers,25))['output_scale']
            payload=np.zeros((16,4096));rows=min(16,len(self.positions)-i*16)
            payload[:rows,:1024]=x.polynomial.reshape(16,4096)[:rows,:1024]+.001*(self.contract.layer+1)
            v=Value(payload.ravel(),4,scale);self.progress('synthetic',v,dict(level=4,scale=scale));out.append(v)
        return QwenDecoderTailResult(tuple(out),self.positions,self.contract.layer,tuple(dict(tile=i,level=4,scale=v._ckks_scale) for i,v in enumerate(out)),())

class Final:
    records=[]
    def __init__(self,a,c,config,*,positions,progress,**kwargs):self.arithmetic=a;self.ctx=a.ctx;self.contract=c;self.positions=positions;self.progress=progress
    def _rms_guard(self):pass
    def evaluate_tile(self,value,*,tile):
        Final.records.append((value,self.positions,tile));self.arithmetic.calls['synthetic_norm']+=1
        plan=rms_scale_plan(self.ctx._param_cache.q_towers,self.contract,input_scale=value._ckks_scale)
        out=Value(value.polynomial,7,plan['output_scale']);self.progress('synthetic',out,dict(level=7,scale=out._ckks_scale))
        return SimpleNamespace(output=out)

def build(monkeypatch,positions=(0,),progress=None,distinct=False):
    for name,cls in [('QwenParameterSource',Parameters),('ChainedStreamingLinear',Linear),('QwenAttentionJoin',Attention),('QwenDecoderTail',Tail),('_CompleteRMS',Final)]:monkeypatch.setattr(runtime,name,cls)
    Attention.records=[];Tail.records=[];Final.records=[]
    c=contract(positions);a=Arithmetic();b=Arithmetic() if distinct else a
    if distinct:
        b.ctx=a.ctx
        for n in ('rotation','_plaintext_multiply','_rescale'):getattr(b,n).ctx=a.ctx
    for key in c.required_keys:
        a.ctx._param_cache.raw_rotation_keys.setdefault(key,{'a':object(),'b':object()});a.ctx._param_cache.coef_maps.setdefault(key,object())
    p=Parameters(c);linear=Linear(b);r=OuterIdentityRefresh(a)
    o=runtime.QwenFullModel(a,linear,c,p,refresher=r,progress=progress)
    inputs=[]
    for start in range(0,len(positions),16):
        x=np.full((16,4096),4321.);n=min(16,len(positions)-start)
        for row in range(n):x[row,:1024]=.01*(start+row+1)
        inputs.append(Value(x.ravel(),25,math.nextafter(DELTA,math.inf)))
    return o,a,b,r,p,tuple(inputs)
