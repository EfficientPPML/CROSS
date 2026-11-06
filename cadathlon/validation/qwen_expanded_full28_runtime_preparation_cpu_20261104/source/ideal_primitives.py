"""Ideal fullslot payload tests of actual owner loops; no encrypted claim."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from collections import Counter
import json,sys,math
import numpy as np
import pytest
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from encrypted_scale_planning import weighted_sum_scale,multiply_scale,drop_divisor

class Value:
    def __init__(self,data,level=25,scale=None):
        self.polynomial=np.array(data,dtype=np.float64,copy=True);self.level=level
        self._ckks_scale=DELTA if scale is None else scale;self._ckks_nsd=1
        self.batch=1;self.num_elements=2;self.degree=131072;self.degree_layout=(256,512)
        self.moduli=tuple(Q[:2*(level+1)]);self.num_moduli=len(self.moduli);self.ntt_ctx=None
        self.precision=32;self.r=256;self.c=512;self.modulus_dtype=np.uint32
        self.shape_in_ntt_all_limbs=(-1,self.r,self.c,self.num_moduli)
        self.moduli_array=np.asarray(self.moduli,dtype=np.uint32)

class Arithmetic:
    def __init__(self):
        self.scale=DELTA;self.strategy='power';self.refresher=None;self.progress=None;self.calls=Counter()
        self.low_memory=True
        # Supply every possible public rotation in the double. Tests separately
        # remove/replace entries from the owner's exact inventory.
        keys={i:{'a':object(),'b':object()} for i in ROTATIONS};maps={i:object() for i in keys}
        k=SimpleNamespace(degree=131072,batch=1,composite_degree=2,dnum=8,degree_layout=(256,512),q_towers=Q,p_towers=PP,secret_key=None,raw_rotation_keys=keys,coef_maps=maps,per_level_rotation_keys=False)
        self.ctx=SimpleNamespace(degree=131072,num_slots=65536,composite_degree=2,dnum=8,degree_layout=(256,512),secret_key=None,parameters={},_param_cache=k,max_level=49,scaling_factor=DELTA,q_towers=Q,p_towers=PP)
        self.rotation=SimpleNamespace(ctx=self.ctx,rotate=lambda *a:None)
        self._plaintext_multiply=SimpleNamespace(ctx=self.ctx,multiply=lambda *a:None)
        self._rescale=SimpleNamespace(ctx=self.ctx,rescale=lambda *a:None)
        self._multiplication_compiler=object()
    def _level(self,x):return x.level
    def _finish(self,value):
        # NumPy operations in this explicit ideal double are already complete.
        # Separate tests execute the actual production barrier AST.
        return value
    def rotate(self,x,s):self.calls['rotate']+=1;return Value(np.roll(x.polynomial,-s),x.level,x._ckks_scale)
    def weighted_sum(self,values,weights,bias=0.):
        self.calls['weighted']+=1;l=min(x.level for x in values);d=drop_divisor(Q,l)
        data=values[0].polynomial*weights[0]
        for x,w in zip(values[1:],weights[1:]):data=data+x.polynomial*w
        return Value(data+bias,l-1,weighted_sum_scale([x._ckks_scale for x in values],DELTA,d)['output_scale'])
    def multiply(self,x,y):
        self.calls['multiply']+=1;l=min(x.level,y.level)
        return Value(x.polynomial*y.polynomial,l-1,multiply_scale(x._ckks_scale,y._ckks_scale,drop_divisor(Q,l)))
    def add(self,x,y,*,subtract=False):
        self.calls['add']+=1
        assert abs(x._ckks_scale-y._ckks_scale)<=max(x._ckks_scale,y._ckks_scale)*1e-12
        return Value(x.polynomial-y.polynomial if subtract else x.polynomial+y.polynomial,min(x.level,y.level),x._ckks_scale)

profile=next(r for r in json.loads((P/'inputs/profile_audit.json').read_bytes())['profiles'] if r['name']=='depth49_p60_d8')
Q=tuple(profile['q']);PP=tuple(profile['p']);DELTA=profile['canonical_scale']
ROTATIONS=set(json.loads((P/'inputs/profile_audit.json').read_bytes())['bootstrap_rotation_indices'])|{shift%65536 for bit in range(10) for shift in (1<<bit,-(1<<bit))}
for name,attrs in [('encrypted_packed_qwen',{'CompiledArithmetic':Arithmetic}),('jaxite_word',{'Polynomial':Value}),('support',{'require_secure_execution':lambda c,v:None})]:
    m=ModuleType(name);m.__dict__.update(attrs);sys.modules[name]=m
from encrypted_qwen_head_routing import QwenHeadRouting,QwenHeadLayout
