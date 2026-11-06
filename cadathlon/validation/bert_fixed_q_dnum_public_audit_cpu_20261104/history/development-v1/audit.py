"""Actual public he_params checks with fixed Q; never creates HE state."""
import argparse,hashlib,json,math,os,sys,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'core')]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def sources():
 origins=json.loads((HERE/'source_origins.json').read_bytes());names=[*origins,'source_origins.json','audit.py','test_public_profiles.py']
 return {n:sha(HERE/n)for n in sorted(names)}
def install_guards():
 import jaxite_word.he_params as hp
 import jaxite_word.ckks_ctx as ckks
 import jaxite_word.polynomial as polynomial
 import jaxite_word.key_gen as keys
 import key_gen
 calls=[];old=[]
 def deny(*args,**kwargs):calls.append('forbidden HE/state entrypoint');raise RuntimeError('public profile audit forbids HE state')
 targets=[(hp.HEParameterCache,'__init__'),(ckks.CKKSContext,'__init__'),(ckks.CKKSContext,'program_initialization'),(polynomial.Polynomial,'__init__')]
 for module in {keys,key_gen}:
  targets.extend((module,n)for n in ('gen_evaluation_key','gen_rotation_key','gen_pke_pair','key_switch_gen','gen_ternary_uniform_polynomial','gen_uniform_polynomial','gen_gaussian_polynomial'))
 for name in ('encrypt','decrypt','encode','decode','batch_encrypt','batch_decrypt','decode_from_slots'):
  if hasattr(ckks.CKKSContext,name):targets.append((ckks.CKKSContext,name))
 for owner,name in targets:old.append((owner,name,getattr(owner,name)));setattr(owner,name,deny)
 return hp,calls,old

def assess(hp,q,p,degree,dnum,target):
 product=math.prod((*q,*p));p_product=math.prod(p)
 try:hp.validate_kernel_moduli(q,p,dnum);kernel=True;kernel_error=None
 except ValueError as e:kernel=False;kernel_error=str(e)
 coverage=hp.p_covers_q(q,p,dnum)
 try:hp.assert_p_coverage(q,p,dnum);coverage_assert=True
 except ValueError:coverage_assert=False
 independent_parts=[math.prod(q[i:i+math.ceil(len(q)/dnum)])for i in range(0,len(q),math.ceil(len(q)/dnum))]
 assert coverage==coverage_assert==(p_product>=max(independent_parts))
 secure=hp.qp_is_secure(q,p,degree,target);ceiling=hp.security_ceiling(degree,target)
 assert secure==(product.bit_length()<=ceiling)and hp.openfhe_logq_bits(product)==product.bit_length()
 return dict(q_count=len(q),cd2_level=len(q)//2-1 if len(q)%2==0 else None,
   exact_log2_QP=math.log2(product),QP_bitlength=product.bit_length(),security_ceiling=ceiling,secure_128_classical=secure,
   kernel_valid=kernel,kernel_error=kernel_error,P_coverage=coverage,max_partition_bitlength=max(independent_parts).bit_length(),
   P_product_bitlength=p_product.bit_length(),P_margin_ratio=p_product/max(independent_parts),
   NTT_congruence=all(v%(2*degree)==1 for v in (*q,*p)))

def resource(degree,q_count,p_count,dnum,*,keys):
 pair=2*dnum*degree*(q_count+p_count)*4
 return dict(rotation_count=keys,rotation_pair_raw_bytes=pair,rotation_keys_raw_bytes=keys*pair,
   rotation_keys_raw_GiB=keys*pair/(1<<30),relinearization_pair_raw_bytes=pair,
   public_key_raw_bytes=2*degree*q_count*4,one_ciphertext_raw_bytes=2*degree*q_count*4,
   public_plus_relinearization_plus_rotation_raw_bytes=(keys+1)*pair+2*degree*q_count*4,
   excludes='NPY headers, secret keys, mmap/page cache, duplicate/transient/device buffers, ciphertext inventory, refresh/NTT controls, JAX/compiler/allocator memory')

def evaluate(hp):
 raw=json.loads((HERE/'inputs/profile.json').read_bytes());base=raw['profile'];q=list(base['q']);N=base['degree'];target=hp.SecurityTarget(128,hp.CostModel.CLASSICAL)
 assert (N,len(q),base['depth'],base['dnum'])==(131072,100,49,8)
 assert hp.generate_p_towers(q,8,N)==base['p']
 baseline=dict(default_generated_P_matches=True,q_sha256=hashlib.sha256(json.dumps(q).encode()).hexdigest(),q=q,P=base['p'],security_ceiling=hp.security_ceiling(N,target))
 rows=[]
 for dnum in (5,6,7,8):
  for aux in (29,30,31):
   p=hp.generate_p_towers(q,dnum,N,aux_bits=aux)
   prefixes=[assess(hp,q[:n],p,N,dnum,target)for n in range(1,101)]
   full=prefixes[-1];assert all(x['kernel_valid']and x['P_coverage']and x['NTT_congruence']for x in prefixes)
   row=dict(dnum=dnum,aux_bits=aux,P=p,p_count=len(p),q_unchanged=True,full=full,prefixes=prefixes,
     public_valid_full_profile=all(full[k]for k in ('kernel_valid','P_coverage','NTT_congruence','secure_128_classical')),
     resources={name:resource(N,len(q),len(p),dnum,keys=count)for name,count in [('bert118',118),('qwen166',166)]})
   rows.append(row)
 return baseline,rows

def run(output):
 output=Path(output).resolve();output.mkdir(exist_ok=False);before=sources();start=time.monotonic();old=[];calls=[]
 report=dict(scope='Fixed100Q public parameter/kernel/security/resource audit using actual he_params entrypoints; no key/context/ciphertext/HE/model',passed=False,pid=os.getpid(),source_hashes_before=before,profiles=[])
 write(output/'report.json',report)
 try:
  with tarfile.open(output/'sources.tar.gz','w:gz')as t:
   for n in before:t.add(HERE/n,arcname=n)
  report['source_archive_sha256']=sha(output/'sources.tar.gz')
  hp,calls,old=install_guards()
  import jax
  from jax._src import xla_bridge
  report['cpu_backend_initialized_after_import']=xla_bridge.backends_are_initialized()
  devices=jax.devices();assert devices and all(d.platform=='cpu'for d in devices)
  report['backend_initialization']='CPU initialization is permitted and recorded; no CKKS/parameter-cache/Polynomial object construction'
  report['devices']=[str(d)for d in devices]
  report['actual_entrypoint_sources']={n:dict(module=getattr(hp,n).__module__,path=getattr(hp,n).__code__.co_filename,sha256=sha(Path(getattr(hp,n).__code__.co_filename)))for n in ('generate_p_towers','qp_is_secure','p_covers_q','assert_p_coverage','validate_kernel_moduli','security_ceiling')}
  assert all(Path(v['path']).resolve().is_relative_to(HERE/'core')for v in report['actual_entrypoint_sources'].values())
  report['baseline'],report['profiles']=evaluate(hp)
  report['public_kernel_primality_calls_use_actual_module']=True
  report['forbidden_calls']=calls;assert not calls
  report['loaded_local_modules']={n:dict(path=str(Path(m.__file__).resolve().relative_to(HERE)),sha256=sha(Path(m.__file__)))for n,m in list(sys.modules.items())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(HERE)and Path(m.__file__).suffix=='.py'}
  report['BERT_count_basis']='Completed authentic74-matrix audit supplied118-key union; copied public report'
  report['Qwen_count_basis']='Root-requested prospective166-key resource scenario; this audit does not derive Qwen key inventory'
  report['passed']=True
 except BaseException as error:report['error']=dict(type=type(error).__name__,message=str(error));raise
 finally:
  active=sys.exception();errors=[]
  for owner,name,value in reversed(old):
   try:setattr(owner,name,value)
   except BaseException as error:errors.append(error)
  report['forbidden_calls']=calls;report['guards_restored']=not errors
  report['source_hashes_after']=sources();report['source_files_unchanged']=before==report['source_hashes_after']
  with tarfile.open(output/'sources.tar.gz','r:gz')as t:archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in t.getmembers()}
  report['source_archive_verified']=archived==before;report['elapsed_s']=time.monotonic()-start
  report['passed']=report['passed']and active is None and not errors and not calls and report['source_files_unchanged']and report['source_archive_verified']
  try:write(output/'report.json',report)
  except BaseException as error:
   if active is None:raise
   active.add_note('public report publication: '+type(error).__name__)
  if active is None and errors:raise errors[0]
 return report
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args();r=run(args.output)
 print(json.dumps(dict(passed=r['passed'],sources=len(r['source_hashes_before']),profiles=[dict(dnum=x['dnum'],aux_bits=x['aux_bits'],P=x['p_count'],logQP=x['full']['exact_log2_QP'],secure=x['full']['secure_128_classical'])for x in r['profiles']])))
 raise SystemExit(0 if r['passed']else 1)
