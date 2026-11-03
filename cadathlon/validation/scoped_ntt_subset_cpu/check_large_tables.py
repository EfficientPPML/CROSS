"""Production-shape CPU public NTT tables only; no HE/model/TPU performance claim."""
from pathlib import Path
import gc,hashlib,json,resource,sys,time,weakref
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(HERE)]
import jax,numpy as np
from jaxite_word import he_params
from encrypted_refresh import RefreshProfile
from scoped_ntt import ScopedNTTSubsets,FIELDS
jax.config.update('jax_enable_x64',True)
ntt=he_params.ntt_mm
class Owner:pass
sources=[Path(__file__),HERE/'scoped_ntt.py',*sorted((ROOT/'jaxite_word').glob('*.py')),*sorted((ROOT/'cadathlon/common').glob('*.py'))]
hashes=lambda:{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
before=hashes();start=time.monotonic();parameters=RefreshProfile(degree=131072,depth=49,dnum=4).parameters();q=parameters['q_towers'];p=parameters['p_towers'];owner=Owner();owner.q_towers=q;owner.p_towers=p;owner.r=256;owner.c=512;owner.degree=131072;owner.ff_context_cls=he_params.BarrettContext;owner.perf_test=False

def args(moduli):return {'r':256,'c':512,'finite_field_context':he_params.BarrettContext(moduli)}
def ready(context):jax.block_until_ready([getattr(context,f) for f in FIELDS])
def memory(context):
 arrays=[getattr(context,f) for f in FIELDS]
 return {'logical_six_table_bytes':sum(x.nbytes for x in arrays),'numpy_host_logical_bytes':sum(x.nbytes for x in arrays if isinstance(x,np.ndarray)),'jax_array_logical_bytes':sum(x.nbytes for x in arrays if isinstance(x,jax.Array))}
for attribute,moduli in [('ntt_q_max',q),('ntt_p',p)]:
 t=time.monotonic();context=ntt.NTTCiphertextBarrettContext(moduli,args(moduli));ready(context);setattr(owner,attribute,context);print(attribute,'setup',time.monotonic()-t,memory(context),flush=True)
del context
provider=ScopedNTTSubsets(owner,max_entries=2);device=jax.devices()[0]
report={'scope':__doc__,'profile':parameters,'source_sha256_before':before,'masters':{name:memory(getattr(owner,name)) for name in ('ntt_q_max','ntt_p')},'cases':[],'limitations':['CPU development measurements only; no accelerator timing or total-HBM claim.','Baseline rebuild reuses the existing per-modulus NumPy twiddle cache, like repeated current control construction.','Provider keeps weak derived contexts; host slices may retain full base arrays while a caller holds the view.','Logical array bytes exclude FFT/twiddle cache, temporary arrays, executables, keys, ciphertexts, field controls and allocator overhead.','JAX CPU arrays consume host RAM; logical JAX bytes are not TPU HBM measurements.']}
choices=[('rescale_odd_q99',q[:-1]),('first_partition_complement_q75_p24',q[25:]+p),('middle_partition_complement_q75_p24',q[:25]+q[50:]+p),('lower_q52_p24',q[:52]+p),('arbitrary_order', [q[99],p[23],q[0],p[0],q[50]]),('full_q100_p24',q+p)]
for name,moduli in choices:
 jax.clear_caches();ntt._barrett_ntt_context_cache.clear();gc.collect();t=time.monotonic();expected=ntt.NTTCiphertextBarrettContext(moduli,args(moduli));ready(expected);fresh=time.monotonic()-t
 jax.clear_caches();ntt._barrett_ntt_context_cache.clear();gc.collect();t=time.monotonic();actual=provider.get(moduli,args(moduli),device=device);assert actual is not None;ready(actual);reuse=time.monotonic()-t
 for f in FIELDS:np.testing.assert_array_equal(getattr(actual,f),getattr(expected,f))
 again=provider.get(moduli,args(moduli),device=device);assert again is actual
 row={'name':name,'ordered_moduli':moduli,'six_fields_exact':True,'fresh_seconds':fresh,'subset_seconds':reuse,**memory(actual)}
 refs=[weakref.ref(expected),weakref.ref(actual)];del expected,actual,again
 jax.clear_caches();ntt._barrett_ntt_context_cache.clear();gc.collect();row['contexts_collected_after_payload_barrier']=all(ref() is None for ref in refs);assert row['contexts_collected_after_payload_barrier']
 report['cases'].append(row);print(json.dumps(row),flush=True)
report['provider_stats']=dict(provider.stats);report['max_weak_entries']=provider.max_entries;report['live_entries_after_cases']=provider.live_entries;assert provider.live_entries==0
owner_ref=weakref.ref(owner);q_ref=weakref.ref(owner.ntt_q_max);p_ref=weakref.ref(owner.ntt_p);del owner
jax.clear_caches();gc.collect();report['owner_and_masters_collected']=owner_ref() is None and q_ref() is None and p_ref() is None;assert report['owner_and_masters_collected']
provider.close();report['source_sha256_after']=hashes();report['source_files_unchanged']=before==report['source_sha256_after'];report['elapsed_s']=time.monotonic()-start;report['process_maxrss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;report['passed']=report['source_files_unchanged']
(HERE/'large_tables.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('source') and k not in ('cases','profile')}),flush=True)
raise SystemExit(0 if report['passed'] else 1)
