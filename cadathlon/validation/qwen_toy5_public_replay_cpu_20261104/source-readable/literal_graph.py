"""Execute selected unchanged original AST over explicit plaintext/metadata doubles.

Construction is lazy. No original backend module is imported. Refresh copies
plaintext identically and returns nominal scale/capacity; no CKKS/noise proof.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from collections import deque,OrderedDict
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from numbers import Integral,Real
from types import SimpleNamespace
from typing import Callable,Optional,Tuple,List,Sequence
import weakref

def selected(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
    if len(nodes)!=len(names)or {n.name for n in nodes}!=set(names):raise ValueError('exact selected AST names')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec',dont_inherit=True),namespace)

def build(np,root):
    common=root/'frozen/cadathlon/common'
    ns=dict(np=np,math=math,dataclass=dataclass,Callable=Callable,Optional=Optional,Tuple=Tuple,List=List,Sequence=Sequence,Integral=Integral,Real=Real,deque=deque,closing=closing,Fraction=Fraction,lru_cache=lru_cache,OrderedDict=OrderedDict,weakref=weakref,__name__='literal_graph',DEFAULT_EXTRA_PAYLOAD_BYTES=256*1024**2,DEFAULT_MAX_PENDING=8)
    selected(common/'encrypted_polynomial.py',['PolynomialApproximation','EncryptedPolynomialEvaluator','nonlinear_approximation'],ns)
    base=ns['EncryptedPolynomialEvaluator']
    class Value:
        def __init__(self,slots,level,scale):
            self.slots=np.asarray(slots,dtype=np.float64);self.level=level;self._ckks_scale=float(scale);self._ckks_nsd=1
            if self.slots.shape!=(65536,):raise ValueError('full65536 required')
    def clone(source,slots,level,scale=None):
        result=Value(slots,level,source._ckks_scale if scale is None else scale)
        saved=dict(source.__dict__);saved.update(result.__dict__);result.__dict__.update(saved);return result
    class Primitives(base):
        def _level(self,value):return value.level
        def _lower(self,value,level):
            if level>value.level:raise ValueError('cannot increase level')
            return value if level==value.level else clone(value,value.slots,level)
        def _prepare(self,level):pass
        def _finish(self,value):
            if not np.isfinite(value.slots).all()or not math.isfinite(value._ckks_scale):
                if self.first_nonfinite is None:self.first_nonfinite={'stack':list(self.stack),'level':value.level,'finite_slots':int(np.isfinite(value.slots).sum()),'nan_slots':int(np.isnan(value.slots).sum()),'infinite_slots':int(np.isinf(value.slots).sum()),'scale_finite':math.isfinite(value._ckks_scale)}
                raise FloatingPointError('nonfinite ideal stage')
            return value
        def checkpoint(self,value,label):return self.record('phase',label,value)
        def rotate(self,value,shift):
            if shift%65536:self.rotations.add(shift%65536)
            return clone(value,np.roll(value.slots,-shift),value.level)
        def add(self,left,right,*,subtract=False):
            self.counts['add']+=1
            if not math.isclose(left._ckks_scale,right._ckks_scale,rel_tol=1e-12):raise ValueError('scale mismatch')
            return self._finish(clone(left,left.slots-right.slots if subtract else left.slots+right.slots,min(left.level,right.level)))
        def weighted_sum(self,values,weights,*,bias=0.):
            values=self._ensure_capacity(tuple(values),1);self.counts['weighted']+=1;level=min(v.level for v in values);D=math.prod(self.q[2*level:2*(level+1)]);scale=values[0]._ckks_scale
            return self._finish(clone(values[0],sum(v.slots*w for v,w in zip(values,weights,strict=True))+bias,level-1,scale*((self.scale*D)/scale)/D))
        def multiply(self,left,right):
            left,right=self._ensure_capacity((left,right),1);self.counts['multiply']+=1;level=min(left.level,right.level);D=math.prod(self.q[2*level:2*(level+1)])
            return self._finish(clone(left,left.slots*right.slots,level-1,left._ckks_scale*right._ckks_scale/D))
        def _encode_public(self,slots,level,scale):return self.declare_bound(Value(slots,level,scale),float(np.max(np.abs(slots))),'public bias')
    ns['CompiledArithmetic']=Primitives
    selected(root/'frozen/jaxite_word/bsgs.py',['compute_bsgs_params'],ns)
    selected(common/'encrypted_packed_qwen.py',['MatrixDiagonals'],ns)
    selected(common/'encrypted_packed_transformer.py',['pending_limit','output_payload_bytes','giant_chunks','_power_two','packed_feature_width','PackedPairLayout','PackedTransformer'],ns)
    selected(common/'encrypted_bounded_arithmetic.py',['_up','_fraction_upper','public_power_coefficients','public_polynomial_difference','refresh_public_bound','normalized_inverse_bound','polynomial_envelope','reciprocal_ratio_bound','BoundedArithmetic'],ns)
    selected(common/'encrypted_bounded_transformer.py',['_upper_float','_upper_sqrt','_product','_norm','_matrix_norm_bound','_affine_bounds','public_affine_bounds','BoundedPackedTransformer'],ns)
    selected(common/'encrypted_transformer.py',['public_toy_approximations'],ns)
    tree=ast.parse((root/'frozen/jaxite_word/bootstrapping.py').read_text());bootstrap=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Bootstrap')
    method=next(n for n in bootstrap.body if isinstance(n,ast.FunctionDef)and n.name=='compute_bootstrap_nq')
    shell=ast.ClassDef(name='BootstrapDepth',bases=[],keywords=[],body=[method],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[shell],type_ignores=[])),'selected_bootstrap_depth','exec',dont_inherit=True),ns)
    selected(root/'frozen/jaxite_word/bootstrapping.py',['_select_layers','_reduce_rotation','_bootstrap_rotation_indices'],ns)
    class Arithmetic(ns['BoundedArithmetic']):
        def __init__(self,profile,reference):
            self.q=profile['ring']['q_towers'];self.scale=profile['scale'];self.strategy='power';self.refresh_events=[];self.rotations=set();self.trace=[];self.domains=[];self.counter={};self.stack=[];self.compared=set();self.reference=reference;self.first_nonfinite=None
            self.counts=dict(add=0,weighted=0,multiply=0);self.progress=None;self.ctx=SimpleNamespace(num_slots=65536);self.low_memory=True
            class Adds:
                def __getitem__(unused,level):return SimpleNamespace(add_plain=lambda a,b:self.add(a,b))
            self.ctx.he_add=Adds()
            depth=ns['BootstrapDepth'].compute_bootstrap_nq(2,profile['level_budget'],secret_key_dist='uniform_ternary')
            self.refresh_capacity=profile['max_level']-depth['boot_depth']-2
            self.refresh_ranges=[]
            def refresh(value,*,public_bound):
                label=str(len(self.refresh_events));self.record('refresh_before',label,value,public_bound=public_bound)
                self.refresh_ranges.append(dict(index=int(label),magnitude=float(np.max(np.abs(value.slots))),public_bound=public_bound,passed=bool(np.max(np.abs(value.slots))<=public_bound)))
                result=Value(value.slots.copy(),self.refresh_capacity,self.scale)
                return self.record('refresh_after',label,result,public_bound=public_bound)
            self.refresher=SimpleNamespace(input_level=profile['refresh_input_level'],output_level=self.refresh_capacity,refresh=refresh)
        def record(self,kind,label,value,**extra):
            self._finish(value);key=(kind,str(label));error=None
            if key in self.reference:
                if key in self.compared:raise ValueError('duplicate semantic comparison')
                self.compared.add(key);error=float(np.max(np.abs(value.slots.reshape(64,1024)-self.reference[key])))
            periodic_error=float(np.max(np.abs(value.slots.reshape(64,1024)-value.slots[:1024])))
            self.trace.append(dict(index=len(self.trace),kind=kind,label=str(label),stack=list(self.stack),level=value.level,scale=value._ckks_scale,nsd=value._ckks_nsd,minimum=float(value.slots.min()),maximum=float(value.slots.max()),periodicity_max_error=periodic_error,independent_dense_max_error=error,array_sha256=hashlib.sha256(value.slots.tobytes()).hexdigest(),ciphertext_payload_bytes=2*131072*2*(value.level+1)*4,**extra));return value
        def evaluate(self,value,approximation,*,output_weights=1.):
            index=self.counter.get('polynomial',0);self.counter['polynomial']=index+1;label=f'{index}:{approximation.name}'
            lo,hi=approximation.domain
            self.record('polynomial_before',label,value,domain=list(approximation.domain))
            self.domains.append(dict(label=label,domain=[lo,hi],minimum=float(value.slots.min()),maximum=float(value.slots.max()),passed=bool(np.all((value.slots>=lo)&(value.slots<=hi)))))
            # Continue finite out-of-domain values only to diagnose the fixed
            # circuit. Every resulting bound/refresh schedule is conditional.
            self.stack.append('polynomial:'+label)
            try:result=super().evaluate(value,approximation,output_weights=output_weights)
            finally:self.stack.pop()
            return self.record('polynomial_after',label,result)
    class Graph(ns['BoundedPackedTransformer']):
        def __init__(self,layout,arithmetic):self.layout=layout;self.features=layout.features;self.arithmetic=arithmetic;self.ctx=arithmetic.ctx;self._head_contract=None
        def event(self,kind,function,*args,**kwargs):
            a=self.arithmetic;index=a.counter.get(kind,0);a.counter[kind]=index+1;label=str(index);a.record(kind+'_before',label,args[0]);a.stack.append(kind+':'+label)
            try:result=function(*args,**kwargs)
            finally:a.stack.pop()
            return a.record(kind+'_after',label,result)
        def linear(self,*args,**kwargs):return self.event('linear',super().linear,*args,**kwargs)
        def normalize(self,*args,**kwargs):return self.event('normalize',super().normalize,*args,**kwargs)
        def normalize_heads(self,*args,**kwargs):return self.event('normalize_heads',super().normalize_heads,*args,**kwargs)
        def rope(self,*args,**kwargs):return self.event('rope',super().rope,*args,**kwargs)
        def attention(self,*args,**kwargs):return self.event('attention',super().attention,*args,**kwargs)
    ns.update(Value=Value,Arithmetic=Arithmetic,Graph=Graph)
    return ns
