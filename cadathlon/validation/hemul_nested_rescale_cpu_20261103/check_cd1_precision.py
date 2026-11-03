"""Two genuine encryption precisions on the same secure CD1 core-hook fixture.

This is an exploratory numerical diagnostic at an unchanged 2e-6 gate, not a
production CD1 opt-in (the experimental evaluator still requires CD2).
"""
from pathlib import Path
import gc,hashlib,json,math,os,sys,time
LAB=Path(__file__).resolve().parent;ROOT=LAB/'checkout'
os.chdir(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(ROOT/'cadathlon/tests')]
import jax
import numpy as np
from jaxite_word import CKKSContext,ckks_ctx,he_params
from encrypted_ntt import ScopedNTTSubsets
from test_encrypted_tensor_only import ring
from test_scoped_nested_rescale import same
paths=[*ROOT.rglob('*.py'),ROOT/'pytest.ini',Path(__file__)]
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if '__pycache__' not in p.parts}
start=hashes();started=time.perf_counter();record={'scope':__doc__,'gate':2e-6,'source_hashes_start':start,'trials':[],'device_kinds':[d.device_kind for d in jax.devices()]}
client,params,key,dnum=ring.__wrapped__()
record['profile']={'degree':client.degree,'dnum':dnum,'q_towers':params['q_towers'],'p_towers':params['p_towers'],'actual_log2_qp':math.log2(math.prod(params['q_towers']+params['p_towers']))}
def server():
 ctx=CKKSContext(dict(params,composite_degree=1,evaluation_key=[jax.numpy.asarray(key[k],dtype='uint32').transpose(0,2,1) for k in ('a','b')]))
 ctx.program_initialization([],dnum=dnum,r=64,c=128,batch=2,pregenerated_rotation_keys={})
 assert ctx.secret_key is None and ctx._param_cache.secret_key is None
 assert he_params.qp_is_secure(ctx.q_towers,ctx.p_towers,ctx.degree,he_params.SecurityTarget(128,'classical'))
 return ctx
ordinary,selected=server(),server();owner=ScopedNTTSubsets(selected._param_cache)
selected._param_cache.multiplication_polynomial_factory=owner.polynomial_factory
selected._param_cache.rescale_polynomial_factory=owner.polynomial_factory
x=np.stack([np.linspace(-.3,.3,client.num_slots),.2*np.sin(np.arange(client.num_slots)/17)])
for bits in (48,56):
 left=client.encrypt_slots_batch(x,scale=float(1<<bits));right=client.encrypt_slots_batch(-.7*x,scale=float(1<<bits))
 expected=ordinary.he_mul[ordinary.max_level-1].mul(left,right)
 actual=selected.he_mul[selected.max_level-1].mul(left,right);same(actual,expected)
 error=float(np.max(np.abs(client.decrypt_slots_batch(actual)-x*(-.7*x))))
 trial={'input_scale_bits':bits,'max_abs_error':error,'gate_passed':error<record['gate'],'baseline_payload_metadata_exact':True}
 record['trials'].append(trial);print(json.dumps(trial),flush=True)
 del left,right,actual,expected
owner.close();ckks_ctx.evict_context_caches(ordinary);ckks_ctx.evict_context_caches(selected);jax.clear_caches();gc.collect()
record['source_hashes_end']=hashes();record['source_files_unchanged']=start==record['source_hashes_end'];record['elapsed_s']=time.perf_counter()-started
record['diagnostic_completed']=record['source_files_unchanged'];record['all_precision_gates_passed']=all(t['gate_passed'] for t in record['trials'])
(LAB/'cd1_precision.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
raise SystemExit(0 if record['diagnostic_completed'] else 1)
