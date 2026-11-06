"""Public random canonical arrays and synthetic keys only; not a secure HE test."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'checkout/jaxite_word'))

def source_ast_checks():
 before=ast.parse((HERE/'baseline_herot.py').read_text());after=ast.parse((HERE/'checkout/jaxite_word/herot.py').read_text())
 def methods(tree):return {n.name:n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='_HERotKernel' for n in c.body if isinstance(n,ast.FunctionDef)}
 old,new=methods(before),methods(after)
 assert set(new)-set(old)=={'_rotate_from_digits_array'}
 assert all(ast.dump(old[n])==ast.dump(new[n]) for n in old)
 class Rewrite(ast.NodeTransformer):
  def visit_Assign(self,node):
   if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_prepare_decomposition_inputs':return None
   return self.generic_visit(node)
  def visit_Call(self,node):
   if isinstance(node.func,ast.Attribute) and node.func.attr=='_decompose_part':return ast.Subscript(value=ast.Name(id='digits',ctx=ast.Load()),slice=copy.deepcopy(node.args[2]),ctx=ast.Load())
   return self.generic_visit(node)
 expected=Rewrite().visit(copy.deepcopy(old['_rotate_barrett_fused']));expected.body=expected.body[1:]
 actual=new['_rotate_from_digits_array'];index=next(i for i,n in enumerate(actual.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [x.id for x in n.targets[0].elts]==['eval_a','eval_b'])
 assert ast.dump(ast.Module(body=expected.body,type_ignores=[]))==ast.dump(ast.Module(body=actual.body[index:],type_ignores=[]))
 return dict(all_original_methods_ast_unchanged=True,completion_body_exact_after_only_decomposition_substitution=True)

def run(output,quick=False):
 output=Path(output);output.mkdir(exist_ok=False);start=time.monotonic()
 profile=json.loads((HERE/'p60_profile_and_refresh.json').read_bytes())['profile']
 report=dict(passed=False,scope='Actual JAX arithmetic on public random canonical residues and synthetic key operands only; no context/secret/keygen/encryption/decryption/private inputs, no secure-HE or performance claim.',
  fixtures=[],ast=source_ast_checks(),cpu_affinity=sorted(os.sched_getaffinity(0)),quick=quick)
 (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU required')
 import numpy as np
 import jax,jax.numpy as jnp
 import herot,finite_field,util
 assert not any(n in sys.modules for n in ('ckks_ctx','key_gen','jaxite_word.ckks_ctx','jaxite_word.key_gen'))
 fixtures=[(16,12)] if quick else [(n,q) for n in (16,256) for q in (12,14,26,28,52,100)]
 for N,Q in fixtures:
  t=time.monotonic();r=4 if N==16 else 16;c=N//r;ql=profile['q'][:Q];p=profile['p'];moduli=np.asarray(ql+p,dtype=np.uint32);parts=(Q+12)//13
  rng=np.random.default_rng(20261103+N+Q)
  ct=jnp.asarray(rng.integers(0,np.asarray(ql,dtype=np.uint32),size=(1,2,r,c,Q),dtype=np.uint32))
  op=herot._HERotKernel(r,c,8,ql,p,finite_field_context=finite_field.BarrettContext)
  op.control_gen(batch=1,degree_layout=(r,c),keygen_sizeQ=100)
  assert op.numPartQl==parts
  before=np.asarray(ct).copy();digits=jax.jit(op._decompose_array)(ct);jax.block_until_ready(digits);saved=np.asarray(digits).copy()
  ordinary=jax.jit(op._rotate_array);prepared=jax.jit(op._rotate_from_digits_array)
  groups=[('edges',(0,-5,3,3,-1))] if quick else [('head11',tuple(63*i for i in range(1,12))),('key15',tuple(-16*i for i in range(1,16))),('edges',(0,-5,3,3,-1))]
  keys={};observations=[];outputs={}
  for group,indices in groups:
   for index in indices:
    if index not in keys:
     a=jnp.asarray(rng.integers(0,moduli,size=(parts,r,c,Q+len(p)),dtype=np.uint32))
     b=jnp.asarray(rng.integers(0,moduli,size=(parts,r,c,Q+len(p)),dtype=np.uint32))
     mapping=jnp.asarray(util.precompute_auto_map(N,util.find_automorphism_index_2n_complex(index,2*N)),dtype=jnp.int32)
     keys[index]=(a,b,mapping)
    a,b,mapping=keys[index];fingerprints=[np.asarray(v).copy() for v in (a,b,mapping)]
    expected=ordinary(ct,a,b,mapping);got=prepared(ct,digits,a,b,mapping);jax.block_until_ready((expected,got))
    np.testing.assert_array_equal(np.asarray(got),np.asarray(expected))
    for value,original in zip((a,b,mapping),fingerprints):np.testing.assert_array_equal(np.asarray(value),original)
    if index in outputs:np.testing.assert_array_equal(np.asarray(got),outputs[index])
    outputs[index]=np.asarray(got).copy();observations.append(dict(group=group,index=index,equal=True,shape=list(got.shape),dtype=str(got.dtype)))
  np.testing.assert_array_equal(np.asarray(ct),before);np.testing.assert_array_equal(np.asarray(digits),saved)
  a,b,mapping=next(iter(keys.values()));negative=0
  bad=[(ct,digits.astype(jnp.uint64),a,b,mapping),(ct,digits[:-1],a,b,mapping),
       (ct.astype(jnp.uint64),digits,a,b,mapping),(ct,digits,a.astype(jnp.float64),b,mapping),
       (ct,digits,a,b,mapping.astype(jnp.int64)),(ct,digits,a,None,mapping),
       (ct,digits,a[:-1],b[:-1],mapping),(ct,digits,a,b,mapping[:-1]),
       (ct,digits,None,None,mapping),(ct,digits,a,b,None),(ct,digits,None,None,None)]
  for args in bad:
   try:op._rotate_from_digits_array(*args)
   except (ValueError,TypeError):negative+=1
   else:raise AssertionError('invalid prepared shape/dtype/paired key accepted')
  field=op.ff_context_cls;op.ff_context_cls=finite_field.MontgomeryContext
  try:
   try:op._rotate_from_digits_array(ct,digits,a,b,mapping)
   except ValueError:negative+=1
   else:raise AssertionError('nonstandard representation accepted')
  finally:op.ff_context_cls=field
  state=op._controls_ready;op._controls_ready=False
  try:
   try:op._rotate_from_digits_array(ct,digits,a,b,mapping)
   except RuntimeError:negative+=1
   else:raise AssertionError('unprepared controls accepted')
  finally:op._controls_ready=state
  original_moduli=op.overall_moduli;op.overall_moduli=[(1<<31)+1]+original_moduli[1:]
  try:
   try:op._rotate_from_digits_array(ct,digits,a,b,mapping)
   except ValueError:negative+=1
   else:raise AssertionError('full-width Barrett envelope accepted')
  finally:op.overall_moduli=original_moduli
  op._controls_ready=1
  try:
   try:op._rotate_from_digits_array(ct,digits,a,b,mapping)
   except ValueError:negative+=1
   else:raise AssertionError('non-bool controls readiness accepted')
  finally:op._controls_ready=state
  row=dict(N=N,Q=Q,P=len(p),dnum=8,keygen_Q=100,alpha=13,effective_parts=parts,comparisons=observations,
   negative_guards=negative,source_and_digits_unchanged=True,keys_unchanged=True,elapsed_s=time.monotonic()-t)
  report['fixtures'].append(row);(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(N=N,Q=Q,comparisons=len(observations),negative_guards=negative)),flush=True)
  del ordinary,prepared,op,digits,ct,keys,outputs
  jax.clear_caches()
 report.update(passed=True,total_comparisons=sum(len(x['comparisons']) for x in report['fixtures']),
  total_negative_guards=sum(x['negative_guards'] for x in report['fixtures']),elapsed_s=time.monotonic()-start,
  no_context_keygen_modules_loaded=not any(n in sys.modules for n in ('ckks_ctx','key_gen','jaxite_word.ckks_ctx','jaxite_word.key_gen')),
  versions=dict(numpy=np.__version__,jax=jax.__version__),zero_scope='Raw identity-automorphism key-switch parity. Public rotate(0) bypass/source alias belongs to a future group-owner test.')
 (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--quick',action='store_true')
 r=run(**vars(p.parse_args()));print(json.dumps(dict(passed=r['passed'],comparisons=r['total_comparisons'])))
