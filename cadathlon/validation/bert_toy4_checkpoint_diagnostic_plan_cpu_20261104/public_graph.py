"""Literal frozen graph over explicit NumPy stand-ins; never imports a backend.

These doubles are public planning/reference infrastructure, not HE primitives.
Refresh is the identity on slots with a declared metadata return, so no noise or
range transfer is proved. Runtime diagnostic transport remains future work.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json,math,sys,types
from numbers import Integral,Real
from typing import Optional,Tuple,List,Sequence
from types import SimpleNamespace
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P/'predecessor';COMMON=ROOT/'cadathlon/common'
sys.path.insert(0,str(COMMON))
from encrypted_polynomial import EncryptedPolynomialEvaluator

def selected(path,names,namespace):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('exact public AST inventory missing')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
 return namespace

class Value:
 def __init__(self,slots,level,scale):self.slots=np.asarray(slots,dtype=np.float64);self.level=level;self._ckks_scale=float(scale);self._ckks_nsd=1

def clone(source,slots,level,scale=None):
 result=Value(slots,level,source._ckks_scale if scale is None else scale);saved=dict(source.__dict__);saved.update(result.__dict__);result.__dict__.update(saved);return result

class Primitives(EncryptedPolynomialEvaluator):
 def _level(self,v):return v.level
 def _lower(self,v,l):
  if l>v.level:raise ValueError('cannot raise level')
  return v if l==v.level else clone(v,v.slots,l)
 def _prepare(self,l):pass
 def _finish(self,v):
  if not np.isfinite(v.slots).all() or not math.isfinite(v._ckks_scale):raise ValueError('nonfinite public graph stage')
  return v
 def checkpoint(self,v,label):return self.record('phase',label,v)
 def rotate(self,v,shift):
  if shift%65536:self.rotations.add(shift%65536)
  return clone(v,np.roll(v.slots,-shift),v.level)
 def add(self,a,b,subtract=False):
  self.counts['add']+=1
  if not math.isclose(a._ckks_scale,b._ckks_scale,rel_tol=1e-12):raise ValueError('ordinary addition scale mismatch')
  return self._finish(clone(a,a.slots-b.slots if subtract else a.slots+b.slots,min(a.level,b.level)))
 def weighted_sum(self,values,weights,*,bias=0.):
  values=self._ensure_capacity(tuple(values),1);self.counts['weighted']+=1;level=min(v.level for v in values);D=math.prod(self.q[:2*(level+1)][-2:]);s=values[0]._ckks_scale
  return self._finish(clone(values[0],sum(v.slots*w for v,w in zip(values,weights))+bias,level-1,s*((self.scale*D)/s)/D))
 def multiply(self,a,b):
  a,b=self._ensure_capacity((a,b),1);self.counts['multiply']+=1;level=min(a.level,b.level);D=math.prod(self.q[:2*(level+1)][-2:])
  return self._finish(clone(a,a.slots*b.slots,level-1,a._ckks_scale*b._ckks_scale/D))
 def _encode_public(self,slots,level,scale):
  return self.declare_bound(Value(slots,level,scale),float(np.max(np.abs(slots))),'public bias constant')

namespace=dict(np=np,math=math,Integral=Integral,Real=Real,Optional=Optional,Tuple=Tuple,List=List,Sequence=Sequence)
selected(ROOT/'jaxite_word/bsgs.py',['compute_bsgs_params'],namespace)
selected(COMMON/'encrypted_packed_qwen.py',['MatrixDiagonals'],namespace)
stub=types.ModuleType('encrypted_packed_qwen');stub.CompiledArithmetic=Primitives;stub.MatrixDiagonals=namespace['MatrixDiagonals'];sys.modules[stub.__name__]=stub
from encrypted_packed_transformer import PackedPairLayout,PackedTransformer,packed_feature_width,plan_packed_transformer
from encrypted_bounded_arithmetic import BoundedArithmetic
from encrypted_bounded_transformer import BoundedPackedTransformer
from encrypted_transformer import public_toy_approximations

class Arithmetic(BoundedArithmetic):
 def __init__(self,q,scale,refresh_scale=None):
  self.q=q;self.scale=scale;self.strategy='power';self.refresh_events=[];self.rotations=set();self.trace=[];self.values=[];self.domains=[];self.counter={};self.stack=[]
  self.counts=dict(add=0,weighted=0,multiply=0);self.progress=None
  self.ctx=SimpleNamespace(num_slots=65536)
  class Adds:
   def __getitem__(unused,l):return SimpleNamespace(add_plain=lambda a,b:self.add(a,b))
  self.ctx.he_add=Adds()
  def refresh(value,*,public_bound):
   index=len(self.refresh_events);self.record('refresh_before',str(index),value,public_bound=public_bound)
   if np.max(np.abs(value.slots))>public_bound:raise ValueError('ideal refresh range exceeds declared public bound')
   result=Value(value.slots.copy(),25,self.scale if refresh_scale is None else refresh_scale)
   self.record('refresh_after',str(index),result,public_bound=public_bound)
   return result
  self.refresher=SimpleNamespace(input_level=2,output_level=25,refresh=refresh)
 def record(self,kind,label,v,**extra):
  self._finish(v);row=dict(index=len(self.trace),kind=kind,label=label,stack=list(self.stack),level=v.level,scale=v._ckks_scale,nsd=v._ckks_nsd,
    expected_array_sha256=hashlib.sha256(v.slots.tobytes()).hexdigest(),minimum=float(v.slots.min()),maximum=float(v.slots.max()),
    ciphertext_payload_bytes=2*131072*2*(v.level+1)*4,**extra)
  self.trace.append(row);self.values.append(v.slots.copy());return v
 def evaluate(self,value,approximation,*,output_weights=1.):
  index=self.counter.get('polynomial',0);self.counter['polynomial']=index+1;label=f'{index}:{approximation.name}'
  self.record('polynomial_before',label,value,domain=list(approximation.domain))
  lo,hi=approximation.domain
  self.domains.append(dict(label=label,domain=[lo,hi],minimum=float(value.slots.min()),maximum=float(value.slots.max()),passed=bool(np.all((value.slots>=lo)&(value.slots<=hi)))))
  if not self.domains[-1]['passed']:raise ValueError('fixed toy clear input domain violated')
  self.stack.append('polynomial:'+label)
  try:result=super().evaluate(value,approximation,output_weights=output_weights)
  finally:self.stack.pop()
  return self.record('polynomial_after',label,result)

class Graph(BoundedPackedTransformer):
 def __init__(self,layout,arithmetic):self.layout=layout;self.features=layout.features;self.arithmetic=arithmetic;self.ctx=arithmetic.ctx;self._head_contract=None
 def event(self,kind,function,*args,**kwargs):
  a=self.arithmetic;index=a.counter.get(kind,0);a.counter[kind]=index+1;label=str(index);a.record(kind+'_before',label,args[0]);a.stack.append(kind+':'+label)
  try:result=function(*args,**kwargs)
  finally:a.stack.pop()
  return a.record(kind+'_after',label,result)
 def linear(self,*args,**kwargs):return self.event('linear',super().linear,*args,**kwargs)
 def normalize(self,*args,**kwargs):return self.event('normalize',super().normalize,*args,**kwargs)
 def attention(self,*args,**kwargs):return self.event('attention',super().attention,*args,**kwargs)

def run(refresh_scale=None):
 case=json.loads((ROOT/'cadathlon/agent_pset/p2_bert/test/test4.in').read_bytes())
 spec=importlib.util.spec_from_file_location('fixed_toy4_reference',ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
 weights,embeddings,mask,native_functions=reference.make_case(case)
 security=json.loads((P/'failed_checkpoint/security_device.json').read_bytes());ring=security['events'][0]['actual_rings'][1];q=ring['q_towers'];scale=float(q[-2])*q[-1]
 layout=PackedPairLayout(2,8,64,65536);a=Arithmetic(q,scale,refresh_scale);graph=Graph(layout,a);initial=Value(layout.pack_client_embeddings(embeddings)[0],49,scale)
 a.record('input','original_embeddings',initial);result=graph.evaluate(initial,'bert',case['config'],weights,public_toy_approximations('bert'),mask);a.record('output','classifier',result)
 output=layout.unpack_client_features([result.slots],2)[:,0];golden=np.asarray(json.loads((ROOT/'cadathlon/agent_pset/p2_bert/test/test4.out').read_bytes())['logits'])
 if not np.allclose(native_functions,golden,atol=1e-10,rtol=1e-10):raise ValueError('fixed clear exact-function fixture drift')
 gate=bool(np.all(np.abs(output-golden)<=.02+.01*np.abs(golden)) and np.array_equal(output[:,1]>=output[:,0],golden[:,1]>=golden[:,0]))
 return dict(case=case,weights=weights,embeddings=embeddings,mask=mask,approximations=public_toy_approximations('bert'),ring=ring,scale=scale,trace=a.trace,values=a.values,refreshes=a.refresh_events,domains=a.domains,
    rotations=sorted(a.rotations),counts=a.counts,output=output,original_toy_clear_gate_passed=gate,clear_max_logit_error=float(np.max(np.abs(output-golden))),
    graph_plan=plan_packed_transformer('bert',case['config'],weights,batch=2,sequence=8,attention_mask=np.ones((2,8),bool)))
