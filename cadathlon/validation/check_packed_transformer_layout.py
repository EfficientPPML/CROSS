"""NumPy-only packed-feature algebra proof; no ciphertexts or TPU execution."""
import sys,json,math,importlib.util,hashlib,argparse,os
from pathlib import Path
import numpy as np
os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS'] != 'cpu': raise RuntimeError('Planning script requires JAX_PLATFORMS=cpu')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
from encrypted_transformer import public_toy_approximations
from jaxite_word.bsgs import compute_bsgs_params
from jaxite_word import he_params,util
from jaxite_word.composite_prime_gen import composite_prime_gen

SOURCES=[Path(__file__), ROOT/'cadathlon/common/encrypted_transformer.py', ROOT/'cadathlon/common/encrypted_polynomial.py', ROOT/'cadathlon/tests/test_transformer_layout.py', ROOT/'cadathlon/agent_pset/p4_qwen/src/reference.py', ROOT/'cadathlon/agent_pset/p4_qwen/test/test1.in', ROOT/'jaxite_word/bsgs.py', ROOT/'jaxite_word/hemul.py', ROOT/'jaxite_word/util.py']
def hashes(): return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES}
source_hashes=hashes()

class Packed:
 def __init__(self,b,s,f):
  self.b,self.s,self.f=b,s,f;self.n=b*s*s*f;self.multiplications=0;self.linear_shapes=[]
 def pack(self,x):
  out=np.zeros((self.b,self.s,self.s,self.f));out[...,:x.shape[-1]]=x[:,:,None,:];return out.reshape(-1)
 def unpack(self,x,h):return x.reshape(self.b,self.s,self.s,self.f)[:,:,0,:h]
 def tile_feature(self,x):return np.tile(np.pad(np.asarray(x), (0,self.f-len(x))),self.n//self.f)
 def linear(self,x,w,bias=None):
  rows,cols=w.shape;assert max(rows,cols)<=self.f;out=np.zeros(self.n);f=np.arange(self.n)%self.f
  self.linear_shapes.append((rows,cols))
  for shift in range(-(rows-1),cols):
   src=f+shift;valid=(f<rows)&(src>=0)&(src<cols);diagonal=np.zeros(self.n);diagonal[valid]=w[f[valid],src[valid]]
   out+=diagonal*np.roll(x,-shift)
  if bias is not None:out+=self.tile_feature(bias)
  return out
 def reduce(self,x,length,stride,factor=1.):
  result=x.copy()
  for j in range((length-1).bit_length()):result+=np.roll(result,-stride*(1<<j))
  result*=(((np.arange(self.n)//stride)%length)==0)*factor
  for j in range((length-1).bit_length()):result+=np.roll(result,stride*(1<<j))
  return result
 def transpose(self,x):
  q=np.arange(self.s)[:,None];k=np.arange(self.s)[None,:];out=np.zeros(self.n)
  for offset in range(-(self.s-1),self.s):
   mask=np.broadcast_to((k-q==offset)[None,:,:,None],(self.b,self.s,self.s,self.f)).reshape(-1)
   out+=mask*np.roll(x,-offset*(self.s-1)*self.f)
  return out
 def transpose_staged(self,x,width=2):
  assert self.s & (self.s-1) == 0
  q=np.arange(self.s)[:,None];k=np.arange(self.s)[None,:];rotations=0;stages=0
  for start in range(0,(self.s-1).bit_length(),width):
   bitmask=((1<<min(width,(self.s-1).bit_length()-start))-1)<<start
   offsets=(k&bitmask)-(q&bitmask);out=np.zeros(self.n)
   for offset in np.unique(offsets):
    mask=np.broadcast_to((offsets==offset)[None,:,:,None],(self.b,self.s,self.s,self.f)).reshape(-1)
    out+=mask*np.roll(x,-int(offset)*(self.s-1)*self.f)
    rotations+=int(offset!=0)
   x=out;stages+=1
  return x,rotations,stages
 def multiply(self,x,y):self.multiplications+=1;return x*y
 def polynomial(self,x,approx,mask=1.):
  lo,hi=approx.domain;assert np.all((x>=lo)&(x<=hi)),(approx.name,float(x.min()),float(x.max()),approx.domain)
  t=(2*x-lo-hi)/(hi-lo);coeff=np.polynomial.chebyshev.cheb2poly(approx.coefficients);terms={1:t}
  def power(n):
   if n not in terms:terms[n]=self.multiply(power(n//2),power((n+1)//2))
   return terms[n]
  return (sum(coeff[n]*power(n) for n in range(1,len(coeff)) if coeff[n]!=0)+coeff[0])*mask
 def rms(self,x,gamma,eps,approx):
  h=len(gamma);variance=self.reduce(self.multiply(x,x),self.f,1,1/h)+eps
  inverse=self.polynomial(variance,approx)
  return self.multiply(x,inverse)*self.tile_feature(gamma)

# Generic rectangular linear maps, Q/K transpose, segmented sums and GQA.
errors={}
rng=np.random.default_rng(333)
for b,s,f,h,m in [(3,8,64,24,48),(2,4,32,16,24)]:
 p=Packed(b,s,f);x=rng.normal(size=(b,s,h));w=rng.normal(size=(m,h));bias=rng.normal(size=m)
 actual=p.unpack(p.linear(p.pack(x),w,bias),m);expected=x@w.T+bias
 errors[f'linear_{b}_{s}_{h}_{m}']=float(np.max(np.abs(actual-expected)));np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
 labels=np.arange(p.n,dtype=float)
 np.testing.assert_array_equal(p.transpose(labels),labels.reshape(b,s,s,f).transpose(0,2,1,3).reshape(-1))
 np.testing.assert_allclose(p.reduce(labels,f,1),np.repeat(labels.reshape(-1,f).sum(-1),f),atol=1e-12,rtol=0)

for b,s,d,heads,kvheads,f,causal in [(3,8,6,4,4,64,False),(2,4,8,4,2,32,True)]:
 dp=1<<(d-1).bit_length();p=Packed(b,s,f)
 q=rng.normal(0,.1,(b,s,heads,d));k=rng.normal(0,.1,(b,s,kvheads,d));v=rng.normal(0,.1,(b,s,kvheads,d))
 def project(x):
  x=np.repeat(x,heads//x.shape[2],axis=2);a=np.zeros((b,s,f))
  for head in range(heads):a[:,:,head*dp:head*dp+d]=x[:,:,head]
  return p.pack(a)
 query,key,value=project(q),p.transpose(project(k)),p.transpose(project(v))
 scores=p.reduce(p.multiply(query,key),dp,1,1/math.sqrt(d))
 valid=np.ones((b,s,s),bool)
 if causal:valid&=np.tril(np.ones((s,s),bool))[None]
 else:valid[:,:,-2:]=False
 valid=np.broadcast_to(valid[...,None],(b,s,s,f)).reshape(-1)
 counts=np.broadcast_to(valid.reshape(b,s,s,f).sum(axis=2,keepdims=True),(b,s,s,f)).reshape(-1)
 active=np.tile(np.arange(f)<heads*dp,b*s*s)
 scores*=valid*active;mean=p.reduce(scores,s,f,1/counts)
 exponentials=np.exp(scores-mean)*valid*active
 denom=p.reduce(exponentials,s,f);denom+=1-active
 probabilities=exponentials/denom
 attended=p.reduce(p.multiply(probabilities,value),s,f).reshape(b,s,s,f)[:,:,0]
 actual=np.concatenate([attended[:,:,head*dp:head*dp+d] for head in range(heads)],-1)
 kr=np.repeat(k,heads//kvheads,axis=2);vr=np.repeat(v,heads//kvheads,axis=2)
 raw=np.einsum('bqhd,bkhd->bhqk',q,kr)/math.sqrt(d)
 mask=valid.reshape(b,s,s,f)[...,0][:,None]
 raw=np.where(mask,raw,-np.inf);ex=np.exp(raw-np.max(raw,axis=-1,keepdims=True));prob=ex/ex.sum(-1,keepdims=True)
 expected=np.einsum('bhqk,bkhd->bqhd',prob,vr).reshape(b,s,heads*d)
 errors[f'attention_d{d}_causal{causal}']=float(np.max(np.abs(actual-expected)));np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

staged_transpose=[]
for seq in (4,8,16):
 for features in (1,32):
  layout=Packed(2,seq,features);labels=np.arange(layout.n,dtype=float)
  actual,rotations,stages=layout.transpose_staged(labels)
  np.testing.assert_array_equal(actual,layout.transpose(labels))
  staged_transpose.append({'batch':2,'sequence_length':seq,'features':features,'exact_numpy_parity':True,'nonzero_rotations':rotations,'mask_multiply_stages':stages,'direct_nonzero_rotations':2*(seq-1),'direct_mask_multiply_stages':1})

# Complete original Qwen test1 with the current runtime polynomial contracts.
folder=ROOT/'cadathlon/agent_pset/p4_qwen';spec=importlib.util.spec_from_file_location('packed_qwen_oracle',folder/'src/reference.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
case=json.loads((folder/'test/test1.in').read_text());w,x,expected=oracle.make_case(case);c=case['config'];a=public_toy_approximations('qwen');p=Packed(1,1,32);hidden=p.pack(x);prefix='model.layers.0.';attn=prefix+'self_attn.';mlp=prefix+'mlp.'
z=p.rms(hidden,w[prefix+'input_layernorm.weight'],c['rms_norm_eps'],a['inverse_sqrt'])
expanded_v=np.concatenate([w[attn+'v_proj.weight'][(head//2)*8:(head//2+1)*8] for head in range(4)],axis=0)
v=p.linear(z,expanded_v);hidden=hidden+p.linear(v,w[attn+'o_proj.weight'])
z=p.rms(hidden,w[prefix+'post_attention_layernorm.weight'],c['rms_norm_eps'],a['inverse_sqrt'])
gate=p.polynomial(p.linear(z,w[mlp+'gate_proj.weight']),a['silu'],p.tile_feature(np.ones(c['intermediate_size'])))
up=p.linear(z,w[mlp+'up_proj.weight']);hidden=hidden+p.linear(p.multiply(gate,up),w[mlp+'down_proj.weight'])
hidden=p.rms(hidden,w['model.norm.weight'],c['rms_norm_eps'],a['inverse_sqrt']);actual=p.unpack(p.linear(hidden,w['model.embed_tokens.weight']),c['vocab_size'])[:,0]
errors['qwen1_polynomial_vs_exact_max_abs']=float(np.max(np.abs(actual-expected)))
assert errors['qwen1_polynomial_vs_exact_max_abs']<.2
# Compare separately to current one-ciphertext-per-feature planning implementation.
spec=importlib.util.spec_from_file_location('current_layout_planning',ROOT/'cadathlon/tests/test_transformer_layout.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
from encrypted_transformer import PairLayout,EncryptedTensorOps,qwen_ciphertext_forward
layout=PairLayout(1,1,1);ops=EncryptedTensorOps(t.PublicPlanningArithmetic(1),layout)
reference=layout.unpack_client_features(qwen_ciphertext_forward(ops,layout.pack_client_embeddings(x),c,w,a))[:,0]
errors['qwen1_packed_vs_current_polynomial_max_abs']=float(np.max(np.abs(actual-reference)));np.testing.assert_allclose(actual,reference,atol=1e-12,rtol=0)
# Sparse BSGS rotation/key inventory for the packed Qwen1 prototype.
slots=65536;rotations=set();linear_plans=[]
for rows,cols in sorted(set(p.linear_shapes)):
 active=sorted({shift%slots for shift in range(-(rows-1),cols)})
 n1,n2=compute_bsgs_params(slots,len(active),2.)
 keys={i%n1 for i in active}|{(i//n1)*n1 for i in active};keys.discard(0);rotations|=keys
 linear_plans.append({'rows':rows,'columns':cols,'diagonals':len(active),'n1':n1,'rotation_count':len(keys)})
rotations|={sign*(1<<j)%slots for sign in (-1,1) for j in range(5)}
q=composite_prime_gen(2,84,61,56,262144,31);aux=util.generate_p_towers(q,8,131072,aux_bits=30);part=(len(q)+7)//8;parts=(len(q)+part-1)//part
key_bytes=2*parts*(len(q)+len(aux))*131072*4
report={'scope':'NumPy packed algebra only; no HE execution, refresh, setup or speed measurement.','errors':errors,'packed_qwen1_ciphertext_multiplications':p.multiplications,'current_qwen1_ciphertext_multiplications':177,'packed_qwen1_ciphertexts_per_activation_tensor':1,'packed_qwen1_period':32,'linear_plans':linear_plans,'rotation_indices':sorted(rotations),'rotation_key_count':len(rotations),'original_qwen1_rotation_key_count':0,'actual_q_count':len(q),'actual_p_count':len(aux),'actual_qp_secure':he_params.qp_is_secure(q,aux,131072,he_params.SecurityTarget(128,'classical')),'raw_bytes_per_rotation_key_pair':key_bytes,'raw_rotation_key_store_bytes':len(rotations)*key_bytes,'key_memory_note':'Coefficient-format storage estimate only; formatted keys, NTT tables, compilation and diagonals add memory.'}
report['staged_transpose']=staged_transpose
report['source_sha256']=source_hashes
report['source_unchanged']=source_hashes==hashes()
assert report['source_unchanged']
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=Path(__file__).with_suffix('.json'));args=parser.parse_args()
args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
