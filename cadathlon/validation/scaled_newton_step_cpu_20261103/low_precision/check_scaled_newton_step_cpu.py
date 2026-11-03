"""Secure CPU check of one scaled Newton update; no full inverse or refresh."""
import argparse,gc,hashlib,importlib.metadata,io,json,math,os
from pathlib import Path
import resource,sys,tarfile,time
from datetime import datetime,timezone

if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('Explicit CPU platform required')
ROOT=Path(os.environ['CROSS_ROOT']).resolve()
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import jax
import numpy as np
from jaxite_word import CKKSContext,ckks_ctx,he_params,key_gen,util
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_packed_qwen import CompiledArithmetic
from support import require_secure_execution

def sources():
 paths={Path(__file__).resolve()}
 for folder in ('jaxite_word','cadathlon/common'):paths.update((ROOT/folder).rglob('*.py'))
 return {str(p):p.read_bytes() for p in sorted(paths)}

def write(path,report):path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 initial=sources();before={p:hashlib.sha256(data).hexdigest() for p,data in initial.items()}
 with tarfile.open(args.output/'sources.tar.gz','w:gz') as tar:
  for name,data in initial.items():
   p=Path(name);member=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'external/'+p.name
   info=tarfile.TarInfo(member);info.size=len(data);tar.addfile(info,io.BytesIO(data))
 report={'scope':__doc__,'passed':False,'cpu_only':True,'full_model_evaluated':False,
  'full_reciprocal_evaluated':False,'refresh_evaluated':False,'fixed_max_abs_error_gate':1e-6,
  'started_utc':datetime.now(timezone.utc).isoformat(),'source_hashes_start':before,
  'versions':{name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy')},
  'cpu_affinity':sorted(os.sched_getaffinity(0)),'checks':{}}
 write(args.output/'report.json',report)
 client=server=arithmetic=None;patched=[];caught=None;started=time.perf_counter()
 keys=evaluation=encrypted_x=encrypted_z=product=correction=updated=ciphertext=None
 try:
  degree,depth,dnum=8192,3,8;slots=degree//2
  q=composite_prime_gen(2,2*(depth+1),48,40,2*degree,31)
  p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
  target=he_params.SecurityTarget(128,'classical')
  if not he_params.qp_is_secure(q,p,degree,target):raise ValueError('Insecure actual Q/P profile')
  he_params.assert_p_coverage(q,p,dnum)
  scale=float(q[-2])*q[-1]
  report['ring']={'degree':degree,'depth':depth,'dnum':dnum,'q_towers':q,'p_towers':p,
   'actual_log2_qp':sum(map(math.log2,q+p)),'scale':scale,'security_bits':128,'slots':slots}
  # Public test points cover key counts, interval endpoints and initial errors.
  choices=[]
  for count in (1,2,11,15,16,64,128):
   lower=.5/count
   for denominator in (lower,2*lower,(lower+2)/2,1.,2.):
    for relative_error in (.01,.25,.75):
     coefficient=count/4
     estimate=(1-relative_error)/(coefficient*denominator)
     choices.append((count,denominator,estimate,relative_error))
  values=np.asarray(choices)[np.arange(slots)%len(choices)]
  counts,x,z,error=values.T;coefficient=counts/4
  expected_product=x*z;expected_correction=2-coefficient*expected_product
  expected_update=z*expected_correction
  assert np.max(z)<=8 and np.max(expected_product)<=4
  assert np.all((expected_correction>=1)&(expected_correction<2))
  assert np.max(expected_update)<=8
  assert np.max(np.abs(1-coefficient*x*expected_update-error**2))<2e-15
  keys=key_gen.gen_pke_pair(q,[],degree)
  evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
  params=dict(degree=degree,num_slots=slots,scaling_factor=scale,output_scale=scale,
   q_towers=q,p_towers=p,composite_degree=2,degree_layout=(64,128),p=40,
   CKKS_M_FACTOR=1,max_bits_in_word=48,noise_scale_degree=1)
  client=CKKSContext(dict(params,**keys))
  server=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(evaluation[k],dtype='uint32').transpose(0,2,1) for k in ('a','b')]))
  def forbidden(*a,**kw):raise AssertionError('Server secret/key generation/encryption/decryption forbidden')
  for module in {key_gen,sys.modules.get('key_gen')}-{None}:
   for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):
    patched.append((module,name,getattr(module,name)));setattr(module,name,forbidden)
  report['key_generation_guarded_modules']=sorted({module.__name__ for module,_,_ in patched})
  assert set(report['key_generation_guarded_modules'])=={'key_gen','jaxite_word.key_gen'}
  server.program_initialization([],dnum=dnum,r=64,c=128,batch=1,pregenerated_rotation_keys={})
  client._polynomial_ntt_ctx=server._param_cache.ntt_q_max
  assert server.secret_key is None and server._param_cache.secret_key is None
  assert server.parameters.get('secret_key') is None
  for name in ('encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch'):
   setattr(server,name,forbidden)
  arithmetic=CompiledArithmetic(server,low_memory=True)
  encrypted_x=client.encrypt_slots(x);encrypted_z=client.encrypt_slots(z)
  require_secure_execution(server,[encrypted_x,encrypted_z])
  report['setup_s']=time.perf_counter()-started
  report['keyless_server']=True;report['public_test_cases']=len(choices)
  begin=time.perf_counter()
  product=arithmetic.multiply(encrypted_x,encrypted_z)
  correction=arithmetic.weighted_sum([product],[-coefficient],bias=2.)
  updated=arithmetic.multiply(encrypted_z,correction)
  updated.polynomial.block_until_ready()
  report['server_cpu_s']=time.perf_counter()-begin
  # All client decoding occurs after the fixed server circuit has finished.
  for name,ciphertext,expected,level in (
    ('product',product,expected_product,2),('correction',correction,expected_correction,1),
    ('newton_update',updated,expected_update,0)):
   rings=require_secure_execution(server,ciphertext)
   decoded=np.asarray(client.decrypt_slots(ciphertext))
   absolute_error=float(np.max(np.abs(decoded-expected)))
   metadata=(arithmetic._level(ciphertext)==level and ciphertext._ckks_nsd==1
             and math.isfinite(ciphertext._ckks_scale) and ciphertext._ckks_scale>0)
   if name in ('product','correction'):
    metadata=metadata and math.isclose(ciphertext._ckks_scale,scale,rel_tol=1e-12)
   else:
    expected_scale=scale*scale/math.prod(q[2:4])
    metadata=metadata and math.isclose(ciphertext._ckks_scale,expected_scale,rel_tol=1e-12)
   report['checks'][name]={'passed':bool(metadata and absolute_error<1e-6),'max_abs_error':absolute_error,
     'level':level,'scale':ciphertext._ckks_scale,'metadata_ok':metadata,
     'all_slots_checked':slots,'expected_max_abs':float(np.max(np.abs(expected))),'execution_rings':rings}
   print(json.dumps({name:report['checks'][name]}),flush=True)
  report['passed']=set(report['checks'])=={'product','correction','newton_update'} and all(v['passed'] for v in report['checks'].values())
 except BaseException as error:
  caught=error;report['error']=f'{type(error).__name__}: {error}'
  raise
 finally:
  cleanup_errors=[];cleanup_exceptions=[]
  def attempt(label,operation):
   try:return operation()
   except BaseException as error:
    cleanup_errors.append(f'{label}: {type(error).__name__}: {error}');cleanup_exceptions.append(error)
  for context in (server,client):
   if context is not None:attempt('evict context',lambda:ckks_ctx.evict_context_caches(context))
  context=arithmetic=server=client=None
  keys=evaluation=encrypted_x=encrypted_z=product=correction=updated=ciphertext=None
  attempt('JAX cache clear',jax.clear_caches);attempt('GC',gc.collect)
  for module,name,original in patched:attempt('restore '+name,lambda:setattr(module,name,original))
  final=attempt('source snapshot',sources)
  after={p:hashlib.sha256(data).hexdigest() for p,data in final.items()} if final is not None else {}
  report.update(source_hashes_end=after,source_files_unchanged=before==after,
   elapsed_s=time.perf_counter()-started,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   cleanup_errors=cleanup_errors,finished_utc=datetime.now(timezone.utc).isoformat())
  report['passed']=bool(report['passed'] and before==after and not cleanup_errors)
  cleanup_cancel=next((e for e in cleanup_exceptions if not isinstance(e,Exception)),None)
  try:write(args.output/'report.json',report)
  except BaseException as error:
   primary=caught if caught is not None else cleanup_cancel
   if primary is None:raise
   primary.add_note(f'Final evidence write failed: {error}')
  if caught is None and cleanup_cancel is not None:raise cleanup_cancel
 return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
