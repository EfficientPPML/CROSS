"""Public actual-prime/resource audit for depth49 p60; no CKKS context/keygen/ciphertext/HE."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,math,os,shutil,sys,tarfile,time
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout';sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
from jaxite_word import CKKSContext,he_params,key_gen,util
from jaxite_word.composite_prime_gen import composite_prime_gen
from jaxite_word.bootstrapping import Bootstrap
from encrypted_refresh import refresh_plan,RefreshProfile
import jax
import jax._src.xla_bridge as bridge
FORBIDDEN=[]
def forbidden(*args,**kwargs):FORBIDDEN.append('context/keygen/backend call');raise AssertionError(FORBIDDEN[-1])
CKKSContext.__init__=Bootstrap.__init__=forbidden
for module in {key_gen,sys.modules.get('key_gen')}-{None}:
 for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):setattr(module,name,forbidden)
jax.devices=jax.local_devices=jax.default_backend=bridge.get_backend=forbidden
IMPORT_INITIALIZED_CPU_BACKEND=bool(bridge._backends)
N=131072;SLOTS=65536;DEPTH=49

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 start=time.monotonic();paths=[Path(__file__),*sorted((HERE/'inputs').glob('*.json')),*sorted(ROOT.rglob('*.py'))]
 before={str(p.relative_to(HERE)):sha(p.read_bytes()) for p in paths}
 baseline=json.loads((HERE/'inputs/baseline_report.json').read_text());oldchain=json.loads((HERE/'inputs/chained_inventory_report.json').read_text())
 groups=baseline['key_inventory']['groups'];dense={1,64,61760,64064}
 graph=set().union(*(set(v) for n,v in groups.items() if n!='dense_QKV_attention_output_FFN'))|dense
 refresh=refresh_plan(N,SLOTS,(4,4),max_level=DEPTH);boot=set(refresh['rotations']);union=graph|boot
 assert len(graph)==56 and len(boot)==106 and len(union)==116 and refresh['output_level']==25
 assert all(set(p['prospective_chained_resources']['graph_bootstrap_keys'])==union for p in oldchain['profiles'] if p.get('accepted'))
 baseline_profile=next(p for p in baseline['profiles'] if p['name']=='current49_p56_d4')
 profiles=[]
 for pbits,dnum in [(56,4)]+[(60,d) for d in (4,5,6,7,8,9,10)]:
  q=composite_prime_gen(2,100,61,pbits,2*N,31);p=util.generate_p_towers(q,dnum,N,aux_bits=30)
  he_params.assert_p_coverage(q,p,dnum)
  secure=he_params.qp_is_secure(q,p,N,he_params.SecurityTarget(128,'classical'))
  alpha=math.ceil(len(q)/dnum);parts=math.ceil(len(q)/alpha);pair=2*parts*(len(q)+len(p))*N*4
  coverage=[math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha)]
  if pbits==56:assert q==baseline_profile['q'] and p==baseline_profile['p']
  if secure:
   validated=RefreshProfile(N,49,pbits,61,dnum).parameters();assert validated['q_towers']==q and validated['p_towers']==p
  else:
   try:RefreshProfile(N,49,pbits,61,dnum).parameters()
   except ValueError:pass
   else:raise AssertionError('rejected profile accepted by production parameters')
  raw=len(union)*pair;public=2*len(q)*N*4
  profiles.append(dict(name=f'depth49_p{pbits}_d{dnum}',degree=N,slots=SLOTS,depth=49,scale_bits=pbits,first_bits=61,dnum=dnum,
   q=q,p=p,Q_count=len(q),P_count=len(p),log2_QP=sum(map(math.log2,q+p)),QP_bit_length=math.prod(q+p).bit_length(),security_ceiling=3523,
   actual_128_bit_QP_secure=secure,P_covers_each_partition=math.prod(p)>=max(coverage),alpha=alpha,actual_partitions=parts,
   P_product_bits=math.prod(p).bit_length(),maximum_partition_bits=max(coverage).bit_length(),rotation_pair_shape=[parts,len(q)+len(p),N],
   rotation_pair_bytes=pair,rotation_pair_MiB=pair/2**20,prospective_116_rotation_bytes=raw,prospective_116_rotation_GiB=raw/2**30,
   direct_224_rotation_GiB=224*pair/2**30,relinearization_pair_bytes=pair,public_key_bytes=public,client_secret_bytes=len(q)*N*4,
   rotation_headers_allowance_bytes=116*8192,minimum_key_store_preflight_bytes=raw+116*8192+64*2**20,
   public_rotation_relin_pk_bytes=raw+pair+public,top_ciphertext_bytes=100*N*8,refresh_output25_ciphertext_bytes=52*N*8,
   eight_top_row_tiles_bytes=8*100*N*8,X_Q_K_V_eight_tiles_raw_bytes=32*100*N*8,canonical_scale=float(q[-2])*q[-1],
   normalized_refresh_output_level=25,fullslot_period=65536,aliases=1,level_budget=[4,4],meta_precision=10))
 fs={str(p):dict(total=shutil.disk_usage(p).total,used=shutil.disk_usage(p).used,free=shutil.disk_usage(p).free) for p in (Path('/tmp'),Path('/dev/shm'))}
 for row in profiles:row['currently_fits_key_store_minimum']={p:row['minimum_key_store_preflight_bytes']<=v['free'] for p,v in fs.items()}
 report=dict(scope=__doc__,passed=not FORBIDDEN,created_utc=datetime.now(timezone.utc).isoformat(),
  cpu_affinity=sorted(os.sched_getaffinity(0)),forbidden_calls=FORBIDDEN,backends_initialized=bool(bridge._backends),import_initialized_cpu_backend=IMPORT_INITIALIZED_CPU_BACKEND,
  graph_rotation_indices=sorted(graph),bootstrap_rotation_indices=sorted(boot),prospective_union_indices=sorted(union),
  chained_dense_indices=sorted(dense),graph_count=len(graph),bootstrap_count=len(boot),union_count=len(union),profiles=profiles,
  filesystem_snapshot=fs,filesystem_scope='Current live free space, not projected post-cleanup capacity; raw estimates exclude transient and unrelated workload use.',
  caveats=['Actual-prime security/P coverage and resource planning only; package import initialized the CPU backend, but no CKKS context, keygen, ciphertext or HE was constructed.',
   'The116-key union replaces only dense direct linears with the prospective chained/Horner contract. Small-ring secure proof does not establish N131072 noise, memory or runtime.',
   'Full-slot output25 capacity is unchanged by p60 but actual Q/P prime chains and scales differ. The active p56/d4 refresh cannot be relabeled as this profile.',
   'Full-slot refresh with p60/dnum7 or8, meta10 and bound16/attention-normalized bound128 requires independent numerical proof.',
   'Top level49 and normalized output25 payload counts stay100MiB and52MiB respectively, despite larger p60 key stores.',
   'The proposed tracked inverse plus GELU refresh makes a conditional level schedule; it does not certify propagated domains/CKKS/correlated softmax error.',
   'Raw rotation files exclude keygen temporaries, formatted controls, NTT/FFT tables, JIT/compiler, allocator and concurrent ciphertexts. No peak RSS/HBM/time claim.'],
  source_hashes_before=before,elapsed_s=time.monotonic()-start)
 with tarfile.open(HERE/'sources.tar.gz','w:gz') as t:
  for p in paths:
   data=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(HERE)));info.size=len(data);t.addfile(info,io.BytesIO(data))
 after={str(p.relative_to(HERE)):sha(p.read_bytes()) for p in paths};assert before==after
 with tarfile.open(HERE/'sources.tar.gz') as t:assert {m.name:sha(t.extractfile(m).read()) for m in t.getmembers()}==before
 report.update(source_hashes_after=after,source_files_unchanged=True,source_archive_verified=True,source_archive_sha256=sha((HERE/'sources.tar.gz').read_bytes()))
 (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(passed=report['passed'],profiles=[{k:r[k] for k in ('name','actual_128_bit_QP_secure','P_count','log2_QP','prospective_116_rotation_GiB','rotation_pair_MiB')} for r in profiles],elapsed_s=report['elapsed_s']),indent=2))
if __name__=='__main__':main()
