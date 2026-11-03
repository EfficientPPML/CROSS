"""Secure explicit-array lifetime audit of guarded opt-in dispatch-cache policy."""
from pathlib import Path
import copy,gc,hashlib,json,math,resource,sys,time,types,weakref
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout';sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import jax,numpy as np
from jaxite_word import CKKSContext,Polynomial,key_gen,he_params,ckks_ctx
from encrypted_packed_qwen import CompiledArithmetic
from encrypted_refresh import _install_compiled_primitives
import polynomial,ntt_mm
jax.config.update('jax_enable_x64',True)

class References:
 def __init__(self):self.entries=[];self.ids={}
 def add(self,value,name,kind):
  prior=self.ids.get(id(value))
  if prior is not None and prior() is value:return
  try:ref=weakref.ref(value)
  except TypeError:return
  self.ids[id(value)]=ref;self.entries.append((ref,name,kind,int(getattr(value,'nbytes',0))))
 def arrays(self,value,name):
  seen=set()
  def visit(x,path):
   if id(x) in seen:return
   seen.add(id(x))
   if isinstance(x,(jax.Array,np.ndarray)):
    self.add(x,path,'jax_array' if isinstance(x,jax.Array) else 'numpy_array')
    if isinstance(x,np.ndarray):
     root=x
     while isinstance(root.base,np.ndarray):root=root.base
     self.add(root,path+'.backing','numpy_backing')
   elif isinstance(x,dict):
    for k,v in x.items():visit(v,path+'.'+str(k))
   elif isinstance(x,(list,tuple)):
    for i,v in enumerate(x):visit(v,path+'.'+str(i))
   elif not isinstance(x,(type,types.ModuleType,types.FunctionType,types.MethodType)) and hasattr(x,'__dict__'):
    if type(x).__name__.startswith('NTT'):self.add(x,path,'ntt_context')
    for k,v in vars(x).items():visit(v,path+'.'+k)
  visit(value,name)
 def binding(self,binding,name):
  if binding is not None:self.arrays(binding.controls,name+'.controls')
 def status(self):
  live=[(name,kind,size) for ref,name,kind,size in self.entries if ref() is not None]
  return {'tracked':len(self.entries),'alive':len(live),'live_objects':live,
    'alive_logical_array_bytes_not_unique_buffers':sum(x[2] for x in live)}

def run():
 q=[4079617,1196033,1589249,1376257,2277377,1720321,1769473,1785857];p=[8273921,8257537];N=8192;dnum=4
 assert he_params.qp_is_secure(q,p,N,he_params.SecurityTarget(128,'classical'))
 keys=key_gen.gen_pke_pair(q,[],N);ek=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
 rots={i:key_gen.gen_rotation_key(keys['secret_key'],q,p,i,dnum=dnum)[i] for i in (3,4091)}
 S=float(q[-2])*q[-1];params=dict(degree=N,num_slots=N//2,scaling_factor=S,output_scale=S,q_towers=q,p_towers=p,composite_degree=2,degree_layout=(64,128),p=42,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
 client=CKKSContext(dict(params,**keys));server=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(ek[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
 server.program_initialization(list(rots),dnum=dnum,r=64,c=128,batch=1,pregenerated_rotation_keys=rots)
 arithmetic=CompiledArithmetic(server,low_memory=True,preserve_dispatch_cache=True)
 engine=types.SimpleNamespace(ctx=copy.copy(server));helpers=_install_compiled_primitives(engine,server)
 tracker=References();bound=References();report={'scope':__doc__,'degree':N,'q':q,'p':p,'log2_qp':sum(math.log2(x) for x in q+p),'security_ceiling':218,'cycles':[]}
 flatten_calls=[];original_flatten=Polynomial.tree_flatten
 def forbidden_flatten(self):
  flatten_calls.append(type(self).__name__)
  raise AssertionError('Polynomial crossed a JAX transform during explicit-array proof')
 Polynomial.tree_flatten=forbidden_flatten
 try:
  values=np.linspace(-.2,.2,N//2);ciphertext=client.encrypt_slots(values);ciphertext._ckks_nsd=1
  for cycle,level in enumerate((3,2,1,3,2,1)):
   value=ciphertext if level==3 else server.he_level_reduce[3,level].level_reduce(ciphertext)
   arithmetic._prepare(level)
   # Observe original keyed facade controls before the compiled path detaches
   # keys and removes the facade. Track weak references only, never values.
   facade=server.he_rot[level,3];tracker.arrays(facade._herot,f'cycle{cycle}.rotate');del facade
   rotated=arithmetic.rotate(value,3);bound.binding(arithmetic.rotation._binding[1],f'cycle{cycle}.rotate')
   expected=np.roll(values,-3);error=float(np.max(np.abs(client.decrypt_slots(rotated)-expected)));assert error<1e-6
   scaled=arithmetic.weighted_sum([rotated],[.75],bias=.125)
   for name,helper in [('ptct',arithmetic._plaintext_multiply),('rescale',arithmetic._rescale)]:bound.binding(helper.binding[1],f'cycle{cycle}.{name}')
   error=max(error,float(np.max(np.abs(client.decrypt_slots(scaled)-(.75*expected+.125)))));assert error<1e-6
   squared=arithmetic.multiply(value,value);bound.binding(arithmetic._multiplication_bindings[level],f'cycle{cycle}.multiply')
   error=max(error,float(np.max(np.abs(client.decrypt_slots(squared)-values**2))));assert error<1e-6
   # Exercise the same no-rescale tensor/relinearize adapter installed by
   # compiled SecureRefresh, without claiming a full bootstrap at this ring.
   tensor=engine.ctx.he_mul[level-1].hemul_no_relin(value,value)
   relin=engine.ctx.he_mul[level-1].relinearize(tensor)
   for name,helper in zip(('rotate','relin','rescale','ptct','tensor'),helpers):
    binding=getattr(helper,'binding',None) or getattr(helper,'_binding',None)
    if binding:bound.binding(binding[1],f'cycle{cycle}.bootstrap_{name}')
   del binding
   jax.block_until_ready([rotated.polynomial,scaled.polynomial,squared.polynomial,tensor.polynomial,relin.polynomial])
   del rotated,scaled,squared,tensor,relin,value
   for helper in helpers:helper.clear()
   arithmetic.checkpoint(ciphertext,'retire completed controls without JAX cache clear')
   assert not polynomial._POLYNOMIAL_REGISTRY and not ntt_mm._barrett_ntt_context_cache
   report['cycles'].append({'cycle':cycle,'level':level,'max_client_abs_error':error,'registry_entries':len(polynomial._POLYNOMIAL_REGISTRY),'binding_weakrefs':bound.status(),'facade_weakrefs':tracker.status(),'process_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
   print(json.dumps({'cycle':cycle,'level':level,'bound_alive':report['cycles'][-1]['binding_weakrefs']['alive'],'facade_alive':report['cycles'][-1]['facade_weakrefs']['alive']}),flush=True)
  # A context/operand's physical device must not be confused with a later
  # ambient default-device scope. Observe this separately from lifetime.
  device_check={'devices_available':len(jax.devices())}
  if len(jax.devices())>1:
   shifted=None
   try:
    with jax.default_device(jax.devices()[1]):
     shifted=arithmetic.rotate(ciphertext,3)
     bound.binding(arithmetic.rotation._binding[1],'default_device_switch.rotate')
     device_check['payload_device_preserved']=shifted.polynomial.devices()==ciphertext.polynomial.devices()
     device_check['max_client_abs_error']=float(np.max(np.abs(client.decrypt_slots(shifted)-np.roll(values,-3))))
   except Exception as error:
    device_check['error']=f'{type(error).__name__}: {error}'
   del shifted
   arithmetic.checkpoint(ciphertext,'retire default-device probe')
  report['default_device_switch']=device_check
  report['polynomial_tree_flatten_calls']=len(flatten_calls)
  # Keep executable code caches alive while dropping all bindings/contexts.
  code=[arithmetic.rotation._compiler,arithmetic._multiplication_compiler,arithmetic._plaintext_multiply.compiler,arithmetic._rescale.compiler]
  for helper in helpers:code.append(getattr(helper,'compiler',None) or getattr(helper,'_compiler',None))
  ckks_ctx.evict_context_caches(client);ckks_ctx.evict_context_caches(server)
  arithmetic.checkpoint(ciphertext,'last context checkpoint')
  for helper in helpers:helper.clear()
  owner_refs=[weakref.ref(server),weakref.ref(server._param_cache),weakref.ref(client)]
  del ciphertext,arithmetic,engine,helpers,server,client,helper
  polynomial._POLYNOMIAL_REGISTRY.clear();ntt_mm._barrett_ntt_context_cache.clear();gc.collect()
  report['owners_collected_without_jax_clear']=all(x() is None for x in owner_refs)
  report['after_owner_drop_without_jax_clear']={'binding':bound.status(),'facade':tracker.status(),'registry_entries':len(polynomial._POLYNOMIAL_REGISTRY)}
  # This contrast distinguishes dispatch/trace cache references from caller
  # ownership and ordinary immutable per-modulus NumPy caches.
  jax.clear_caches();polynomial._POLYNOMIAL_REGISTRY.clear();gc.collect()
  report['after_explicit_jax_clear']={'binding':bound.status(),'facade':tracker.status(),'registry_entries':len(polynomial._POLYNOMIAL_REGISTRY)}
  report['retained_executable_counts']=[getattr(item,'compile_count',None) for item in code]
  return report
 finally:Polynomial.tree_flatten=original_flatten

paths=[Path(__file__),*sorted((ROOT/'jaxite_word').rglob('*.py')),*sorted((ROOT/'cadathlon/common').rglob('*.py'))]
hashes=lambda:{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=hashes();started=time.monotonic();report=run();report['source_hashes_before']=before;report['source_hashes_after']=hashes();report['source_files_unchanged']=before==report['source_hashes_after'];report['elapsed_s']=time.monotonic()-started
report['passed']=report['source_files_unchanged'] and report['polynomial_tree_flatten_calls']==0
report['limitations']=['CPU object-lifetime and secure component audit only; no TPU HBM or timing claim.','Alive byte sums include aliases and views and are not unique allocated buffers.','Full bootstrap retains its original stage-level jax.clear_caches; only its compiled primitive adapter is exercised here.','A flat dispatch-cache preservation change remains incompatible with general Polynomial-returning JIT functions.']
(HERE/'secure_lifetime.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('cycles','source_hashes_before','source_hashes_after')}),flush=True)
