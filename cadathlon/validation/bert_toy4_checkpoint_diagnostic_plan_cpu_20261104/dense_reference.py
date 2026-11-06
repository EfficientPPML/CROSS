"""Independent dense fixed-polynomial oracle for every selected toy4 checkpoint.

No packed graph/primitive/layout method is called. All physical padding and CLS
mask+bias effects are expanded explicitly, including invalid key exp inputs.
"""
import math
import numpy as np

def references(config,weights,embeddings,mask,approximations):
 B,S,H,F,D,HEADS=2,8,24,64,6,4
 expected={};contexts={};linear_index=poly_index=norm_index=0
 def grid_rows(x):
  grid=np.zeros((B,S,S,F));grid[...,:x.shape[-1]]=x[:,:,None,:];return grid
 def grid_heads(x):
  y=np.zeros((B,S,HEADS,8));y[...,:D]=x.reshape(B,S,HEADS,D);return grid_rows(y.reshape(B,S,32))
 def grid_scores(x): # B,H,Q,K -> B,Q,K,64, repeated within padded head8
  grid=np.zeros((B,S,S,F));grid[...,:32]=np.repeat(x.transpose(0,2,3,1),8,axis=-1);return grid
 def slots(grid):return np.tile(np.asarray(grid,dtype=np.float64).reshape(-1),8)
 def record(kind,label,grid):expected[(kind,str(label))]=slots(grid)
 def polynomial(grid,name,output_weights=1.):
  nonlocal poly_index
  i=poly_index;poly_index+=1;label=f'{i}:{name}';record('polynomial_before',label,grid)
  rule=approximations[name];lo,hi=rule.domain;t=grid*(2/(hi-lo))-(hi+lo)/(hi-lo)
  c=np.polynomial.chebyshev.cheb2poly(rule.coefficients)
  out=np.polynomial.polynomial.polyval(t,c)*output_weights;record('polynomial_after',label,out);return out
 def dense(x,name,inputgrid=None,outputhead=False,source_mask=None):
  nonlocal linear_index
  i=linear_index;linear_index+=1;record('linear_before',i,grid_rows(x)if inputgrid is None else inputgrid)
  value=x if source_mask is None else x*source_mask
  out=value@weights[name+'.weight'].T+weights[name+'.bias'];record('linear_after',i,grid_heads(out)if outputhead else grid_rows(out));return out
 def normalize(x,name):
  nonlocal norm_index,linear_index
  i=norm_index;norm_index+=1;before=grid_rows(x);record('normalize_before',i,before)
  index=linear_index;linear_index+=1;record('linear_before',index,before)
  mean=np.sum(x,axis=-1,keepdims=True)*(1./H);record('linear_after',index,grid_rows(np.broadcast_to(mean,x.shape)))
  centered=x-mean;variance=np.sum(centered*centered,axis=-1,keepdims=True)*(1./H)+config['layer_norm_eps']
  vg=np.broadcast_to(variance[:,:,None,:],(B,S,S,F)).copy()
  contexts['norm'+str(i)+'.variance']=vg;contexts['norm'+str(i)+'.centered']=grid_rows(centered)
  inverse=polynomial(vg,'inverse_sqrt')[:,:,0,0:1]
  out=centered*inverse*weights[name+'.weight']+weights[name+'.bias'];record('normalize_after',i,grid_rows(out));return out
 x=embeddings.copy();record('input','original_embeddings',grid_rows(x))
 counts=mask.sum(axis=-1)[:,None,None,None];active=np.zeros(F);active[:32]=1.
 publicmask=mask[:,None,:,None]*active[None,None,None,:]
 for layer in range(2):
  prefix=f'bert.encoder.layer.{layer}.'
  q,k,v=[dense(x,prefix+'attention.self.'+kind,outputhead=True) for kind in ('query','key','value')]
  record('attention_before',layer,grid_heads(q))
  qh=q.reshape(B,S,HEADS,D).transpose(0,2,1,3);kh=k.reshape(B,S,HEADS,D).transpose(0,2,1,3);vh=v.reshape(B,S,HEADS,D).transpose(0,2,1,3)
  raw=(qh@kh.swapaxes(-1,-2))*(1./math.sqrt(D));score=grid_scores(raw)*publicmask
  contexts['attention'+str(layer)+'.scores']=score
  mean=np.sum(score,axis=2,keepdims=True)/counts
  centered=score-mean
  exponentials=polynomial(centered,'exp',publicmask)
  denominator=np.broadcast_to(np.sum(exponentials,axis=2,keepdims=True)/counts,(B,S,S,F)).copy()+(1-active)
  inverse=polynomial(denominator,'reciprocal',active/counts)
  probability=exponentials*inverse
  # Pick the one scalar probability per head; padded dimensions are explicitzero.
  prob=probability[...,:32].reshape(B,S,S,HEADS,8)[...,0].transpose(0,3,1,2)
  attended=(prob@vh).transpose(0,2,1,3).reshape(B,S,H)
  contexts['attention'+str(layer)+'.attended']=grid_heads(attended);record('attention_after',layer,grid_heads(attended))
  branch=dense(attended,prefix+'attention.output.dense',inputgrid=grid_heads(attended))
  x=normalize(x+branch,prefix+'attention.output.LayerNorm');record('phase',f'packed BERT layer {layer}: attention/normalization ready',grid_rows(x))
  up=dense(x,prefix+'intermediate.dense');contexts['gelu'+str(layer)+'.input']=grid_rows(up)
  features=np.zeros(F);features[:48]=1.;middle=polynomial(grid_rows(up),'gelu',features)[:,:,0,:48]
  branch=dense(middle,prefix+'output.dense');x=normalize(x+branch,prefix+'output.LayerNorm');record('phase',f'packed BERT layer {layer}: output ready',grid_rows(x))
 cls=np.zeros((1,S,1));cls[:,0]=1
 pooler=dense(x,'bert.pooler.dense',source_mask=cls)
 active=np.zeros(F);active[:H]=1.;pooled=polynomial(grid_rows(pooler),'tanh',active)[:,:,0,:H]
 logits=dense(pooled,'classifier');record('output','classifier',grid_rows(logits))
 selected=[contexts['attention1.scores'],contexts['attention1.attended'],contexts['norm2.variance'],contexts['gelu1.input'],contexts['norm3.centered']]
 for i,value in enumerate(selected):
  record('refresh_before',i,value);record('refresh_after',i,value)
 # Refresh phase callbacks have duplicate labels; resolved by ordered index.
 return expected,[slots(x)for x in selected]
