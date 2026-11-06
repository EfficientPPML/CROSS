"""Public-only N131072 Q100/P24 rescale controls; no keys, HE, TPU or speedup claim."""
from pathlib import Path
import gc,hashlib,json,math,os,resource,sys,time,weakref
ROOT=Path(__file__).resolve().parent/'checkout';OUT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('explicit CPU platform required')
import jax,numpy as np
from jaxite_word import he_params,util,key_gen
from jaxite_word.composite_prime_gen import composite_prime_gen
import rescale
from encrypted_ntt import ScopedNTTSubsets,FIELDS
jax.config.update('jax_enable_x64',True)
ntt=he_params.ntt_mm
paths=[*ROOT.rglob('*.py'),ROOT/'pytest.ini',Path(__file__)];paths=[p for p in paths if '__pycache__' not in p.parts]
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
def clear():jax.clear_caches();ntt._barrett_ntt_context_cache.clear();gc.collect()
def sync(kernel):
 jax.block_until_ready([kernel.gammas_stacked,kernel.betas_stacked,kernel.lift_add_stacked,
  *[getattr(v.ntt_ctx,n) for v in kernel.ct_last_list+kernel.ct_work_list for n in FIELDS]])
def compare(a,b):
 assert a.moduli==b.moduli and a.output_moduli==b.output_moduli and a._lift_reduction_modes==b._lift_reduction_modes
 for name in ('moduli_threshold_all','gammas_stacked','betas_stacked','lift_add_stacked'):
  np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
 checks=0
 for x,y in zip(a.ct_last_list+a.ct_work_list,b.ct_last_list+b.ct_work_list):
  assert x.moduli==y.moduli and x.shape==y.shape
  for name in FIELDS:np.testing.assert_array_equal(getattr(x.ntt_ctx,name),getattr(y.ntt_ctx,name));checks+=1
 return checks
before=hashes();started=time.perf_counter();records={};caught=None
try:
 # No secret/key constructor is permitted anywhere in this public-only probe.
 def forbidden(*a,**kw):raise AssertionError('key generation is outside public-table scope')
 for module in {key_gen,sys.modules.get('key_gen')}-{None}:
  for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):setattr(module,name,forbidden)
 q=composite_prime_gen(2,100,61,56,262144,31);p=util.generate_p_towers(q,4,131072,aux_bits=30)
 assert len(p)==24 and he_params.qp_is_secure(q,p,131072,he_params.SecurityTarget(128,'classical'))
 he_params.assert_p_coverage(q,p,4)
 records['ring']={'degree':131072,'q_towers':q,'p_towers':p,'dnum':4,'actual_log2_qp':sum(math.log2(x) for x in q+p),'security_bits':128}
 class Owner:pass
 def execute():
  owner=Owner();owner.q_towers=q;owner.p_towers=p;owner.r=256;owner.c=512;owner.degree=131072;owner.ff_context_cls=he_params.BarrettContext;owner.perf_test=False
  began=time.perf_counter()
  for attr,moduli in [('ntt_q_max',q),('ntt_p',p)]:
   field=owner.ff_context_cls(moduli)
   setattr(owner,attr,ntt.NTTCiphertextBarrettContext(moduli,{'r':256,'c':512,'finite_field_context':field}))
  jax.block_until_ready([getattr(getattr(owner,attr),name) for attr in ('ntt_q_max','ntt_p') for name in FIELDS])
  records['master_setup_s']=time.perf_counter()-began;print('masters',records['master_setup_s'],flush=True)
  provider=ScopedNTTSubsets(owner,max_entries=16);args=dict(batch=1,num_elements=2,moduli=q,r=256,c=512,degree_layout=(256,512))
  clear();began=time.perf_counter();ordinary=rescale._HERescaleKernel(**args);ordinary.control_gen(composite_degree=2);sync(ordinary)
  records['ordinary_control_s']=time.perf_counter()-began;print('ordinary',records['ordinary_control_s'],flush=True)
  clear();began=time.perf_counter();selected=rescale._HERescaleKernel(**args,polynomial_factory=provider.polynomial_factory);selected.control_gen(composite_degree=2);sync(selected)
  records['scoped_control_s']=time.perf_counter()-began;print('scoped',records['scoped_control_s'],flush=True)
  records['six_field_checks']=compare(selected,ordinary)
  records['requested_moduli']=[v.moduli for v in selected.ct_last_list+selected.ct_work_list]
  records['requested_limb_counts']=[v.num_moduli for v in selected.ct_last_list+selected.ct_work_list]
  records['provider_stats']=dict(provider.stats)
  refs=[weakref.ref(owner),weakref.ref(owner.ntt_q_max),weakref.ref(owner.ntt_p),*[weakref.ref(v.ntt_ctx) for v in selected.ct_last_list+selected.ct_work_list]]
  provider.close();assert compare(selected,ordinary)==records['six_field_checks']
  return refs
 refs=execute();clear();records['retired_contexts_collected']=all(r() is None for r in refs)
 assert records['six_field_checks']==24 and records['requested_limb_counts']==[1,1,99,98] and records['retired_contexts_collected']
except BaseException as error:
 caught=error;records['error']=f'{type(error).__name__}: {error}'
finally:
 clear();after=hashes()
 report={'scope':__doc__,'passed':caught is None and before==after,'source_hashes_start':before,'source_hashes_end':after,'source_files_unchanged':before==after,'elapsed_s':time.perf_counter()-started,'peak_host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'records':records,'measurement_scope':'Each control build starts after JAX and shared NTT cache clearing while public masters remain live. Ordinary precedes scoped; shared-host load/order are uncontrolled. These setup observations do not establish TPU performance, HBM savings or whole-model speedup.'}
 (OUT/'fullring_controls.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if not k.startswith('source_hashes')},indent=2),flush=True)
if caught is not None:raise caught
raise SystemExit(0 if report['passed'] else 1)
