"""128-bit-secure ciphertext parity for scoped public NTT table injection."""
from pathlib import Path
import gc,hashlib,json,math,sys,time,weakref
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(HERE)]
import jax,numpy as np
from jaxite_word import CKKSContext,key_gen,he_params
from scoped_ntt import ScopedNTTSubsets,FIELDS
from encrypted_polynomial import EncryptedPolynomialEvaluator
import herot,ntt_mm,polynomial
jax.config.update('jax_enable_x64',True)
sources=[Path(__file__),HERE/'scoped_ntt.py',*sorted((ROOT/'jaxite_word').glob('*.py')),*sorted((ROOT/'cadathlon/common').glob('*.py'))]
hashes=lambda:{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
before=hashes();start=time.monotonic();q=[4079617,1196033,1589249,1376257,2277377,1720321,1769473,1785857];p=[8273921,8257537];N=8192;dnum=4;batch=2
assert he_params.qp_is_secure(q,p,N,he_params.SecurityTarget(128,'classical'))
keys=key_gen.gen_pke_pair(q,[],N)
ek=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
indices=(3,N//2-5)
rotations={index:key_gen.gen_rotation_key(keys['secret_key'],q,p,index,dnum=dnum)[index] for index in indices}
S=float(q[-2])*q[-1];params=dict(degree=N,num_slots=N//2,scaling_factor=S,output_scale=S,q_towers=q,p_towers=p,composite_degree=2,degree_layout=(64,128),p=42,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
client=CKKSContext(dict(params,**keys));server=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(ek[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
server.program_initialization(list(indices),dnum=dnum,r=64,c=128,batch=batch,pregenerated_rotation_keys=rotations)
assert server.secret_key is None and server._param_cache.secret_key is None and server.parameters.get('secret_key') is None
cache=server._param_cache;provider=ScopedNTTSubsets(cache,max_entries=8);evaluator=EncryptedPolynomialEvaluator(server,low_memory=True)
values=np.stack([np.linspace(-.25,.25,N//2),.12*np.sin(np.arange(N//2)/13)])
ct=client.encrypt_slots_batch(values);ct._ckks_nsd=1
report={'scope':__doc__,'degree':N,'q':q,'p':p,'dnum':dnum,'batch':batch,'log2_qp':sum(math.log2(x) for x in q+p),'security_ceiling':218,'cases':[],'source_sha256_before':before}
for level,index in [(3,3),(3,N//2-5),(1,3),(1,N//2-5)]:
 value=ct if level==3 else server.he_level_reduce[3,level].level_reduce(ct)
 evaluator.checkpoint(value,'before baseline scoped NTT comparison')
 t=time.monotonic();facade=server.he_rot[level,index];expected=facade.rotate(value);jax.block_until_ready(expected.polynomial);baseline_s=time.monotonic()-t
 baseline_controls=facade._herot
 state=cache.get_rot_key(index,level);alpha=math.ceil(len(q)/dnum);parts=math.ceil(value.num_moduli/alpha)
 t=time.monotonic();actual_kernel=herot._HERotKernel(64,128,dnum,list(value.moduli),p,finite_field_context=cache.ff_context_cls,polynomial_factory=provider.polynomial_factory)
 actual_kernel.control_gen(batch=batch,degree_layout=(64,128),keygen_sizeQ=len(q))
 actual_kernel.setup_rotation(state[0][:parts],state[1][:parts],state[2])
 setup_s=time.monotonic()-t
 a_contexts=[actual_kernel.ct_in,actual_kernel.ct_single,*actual_kernel.ct_parts,actual_kernel.ct_full,actual_kernel.ct_approx]
 b_contexts=[baseline_controls.ct_in,baseline_controls.ct_single,*baseline_controls.ct_parts,baseline_controls.ct_full,baseline_controls.ct_approx]
 for a,b in zip(a_contexts,b_contexts):
  for field in FIELDS:np.testing.assert_array_equal(getattr(a.ntt_ctx,field),getattr(b.ntt_ctx,field))
 t=time.monotonic();actual=value._clone_with_payload(actual_kernel._rotate_array(value.polynomial));jax.block_until_ready(actual.polynomial);runtime_s=time.monotonic()-t
 np.testing.assert_array_equal(actual.polynomial,expected.polynomial)
 assert actual._ckks_scale==expected._ckks_scale and actual._ckks_nsd==expected._ckks_nsd and actual.moduli==expected.moduli
 decoded=client.decrypt_slots_batch(actual);oracle=np.roll(values,-index,axis=-1);error=float(np.max(np.abs(decoded-oracle)));assert error<1e-6,error
 refs=[weakref.ref(x.ntt_ctx) for x in a_contexts]
 row={'level':level,'rotation':index,'payload_and_metadata_exact':True,'all_control_six_tables_exact':True,'max_abs_error':error,'baseline_construct_plus_rotate_s':baseline_s,'subset_control_setup_s':setup_s,'subset_eager_rotate_s':runtime_s}
 del actual_kernel,a_contexts,b_contexts,a,b,baseline_controls,facade,actual,expected,state
 evaluator.checkpoint(value,'after scoped NTT comparison')
 gc.collect();row['retired_control_contexts_collected']=all(ref() is None for ref in refs)
 # ct_in may be retained by an earlier live ciphertext; the provider itself
 # never owns it. All weak references must clear after dropping current value.
 if not row['retired_control_contexts_collected']:
  row['live_control_count']=sum(ref() is not None for ref in refs)
 report['cases'].append(row);print(json.dumps(row),flush=True)
report['provider_stats']=dict(provider.stats);provider.close()
report['source_sha256_after']=hashes();report['source_files_unchanged']=before==report['source_sha256_after'];report['elapsed_s']=time.monotonic()-start
report['passed']=report['source_files_unchanged'] and all(row['retired_control_contexts_collected'] for row in report['cases'])
(HERE/'secure.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('source_') and k!='cases'}),flush=True)
raise SystemExit(0 if report['passed'] else 1)
