"""Actual CPU CKKS packed components; no encrypted full-Qwen result."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import argparse
from pathlib import Path
import sys, json, math, time, hashlib, importlib.util, resource
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
LAB=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import jax
import jax.numpy as jnp
from jaxite_word import CKKSContext, he_params, key_gen, util
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_refresh import ClientRotationKeyStore
from encrypted_transformer import public_toy_approximations
from encrypted_packed_qwen import PackedQwen1, MatrixDiagonals
jax.config.update('jax_enable_x64',True)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report',type=Path,default=LAB/'packed_qwen_secure_components.json')
args=parser.parse_args()
started=time.monotonic()
def progress(label): print(f'[{time.monotonic()-started:.2f}s] {label}',flush=True)
paths=[Path(__file__),*[ROOT/'cadathlon/common'/name for name in ('encrypted_packed_qwen.py','encrypted_polynomial.py','encrypted_rotation.py','encrypted_primitives.py','encrypted_compilation.py')]]
paths += [ROOT/'jaxite_word'/name for name in ('bsgs.py','ckks_ctx.py','hemul.py','he_ops.py','herot.py','he_param_cache.py') if (ROOT/'jaxite_word'/name).exists()]
paths += [ROOT/'cadathlon/common'/name for name in ('support.py','encrypted_transformer.py','encrypted_refresh.py')]
paths += [ROOT/'cadathlon/agent_pset/p4_qwen/src/reference.py',ROOT/'cadathlon/agent_pset/p4_qwen/test/test1.in']
def hashes(): return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
source_hashes=hashes()
report={'scope':__doc__,'passed':False,'source_sha256':source_hashes,'tests':{},'cpu_only':True}
key_store=None
try:
 degree=16384;depth=4;dnum=3;slots=degree//2
 q=composite_prime_gen(2,2*(depth+1),45,44,2*degree,31)
 p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
 assert he_params.qp_is_secure(q,p,degree,he_params.SecurityTarget(128,'classical'))
 he_params.assert_p_coverage(q,p,dnum)
 folder=ROOT/'cadathlon/agent_pset/p4_qwen'
 spec=importlib.util.spec_from_file_location('public_qwen_fixture',folder/'src/reference.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
 case=json.loads((folder/'test/test1.in').read_text());weights,embeddings,_=oracle.make_case(case)
 prefix='model.layers.0.';attn=prefix+'self_attn.';mlp=prefix+'mlp.'
 raw_v=weights[attn+'v_proj.weight'];expanded_v=np.concatenate([raw_v[(head//2)*8:(head//2+1)*8] for head in range(4)])
 matrices={'gate24x16':weights[mlp+'gate_proj.weight'],'value32x16_fused_gqa':expanded_v,
           'attention16x32':weights[attn+'o_proj.weight'],'down16x24':weights[mlp+'down_proj.weight']}
 rotations={sign*(1<<j)%slots for sign in (-1,1) for j in range(5)}
 for weight in matrices.values(): rotations.update(MatrixDiagonals(weight,slots).rotations)
 rotations=tuple(sorted(rotations))
 keys=key_gen.gen_pke_pair(q,[],degree)
 evk=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum,noise_std=he_params.DEFAULT_SIGMA,noise_scale=1)
 key_store=ClientRotationKeyStore.generate(keys['secret_key'],q,p,degree=degree,dnum=dnum,rotations=rotations)
 progress('client generated evaluation/rotation keys')
 scale=float(q[-2])*q[-1]
 params=dict(degree=degree,num_slots=slots,scaling_factor=scale,output_scale=scale,q_towers=q,p_towers=p,
             composite_degree=2,degree_layout=(128,128),p=44,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
 client=CKKSContext(dict(params,**keys))
 server=CKKSContext(dict(params,evaluation_key=[jnp.asarray(evk[name],dtype=jnp.uint32).transpose(0,2,1) for name in ('a','b')]))
 server.program_initialization(rotations,dnum=dnum,r=128,c=128,batch=1,pregenerated_rotation_keys=key_store.keys)
 client._polynomial_ntt_ctx=server._param_cache.ntt_q_max
 assert server.secret_key is None and server._param_cache.secret_key is None and server.parameters.get('secret_key') is None
 def forbidden(*a,**kw): raise AssertionError('evaluator cannot encrypt/decrypt')
 server.encrypt_slots=server.decrypt_slots=server.encrypt=server.decrypt=forbidden
 evaluator=PackedQwen1(server,low_memory=True)
 progress('separate keyless evaluator initialized')
 report['ring']={'degree':degree,'q_count':len(q),'p_count':len(p),'dnum':dnum,'actual_log2_qp':sum(math.log2(x) for x in q+p),'security_bits':128,'rotations':rotations}
 report['keyless_evaluator']=True
 rng=np.random.default_rng(761)
 def pack(x):
  padded=np.pad(np.asarray(x,dtype=float),((0,0),(0,32-x.shape[1])))
  return np.tile(padded.reshape(-1),slots//padded.size)
 def verify(name,value,expected,level):
  value.polynomial.block_until_ready()
  actual=client.decrypt_slots(value)
  error=float(np.max(np.abs(actual-expected)))
  assert error < 5e-7,(name,error)
  assert evaluator.arithmetic._level(value)==level,(name,evaluator.arithmetic._level(value),level)
  assert value._ckks_nsd==1
  report['tests'][name]={'passed':True,'max_abs_error':error,'output_level':level,'noise_scale_degree':value._ckks_nsd,'scale':value._ckks_scale}
  progress(f'{name}: max error {error:.3e}, level {level}')
 for name,weight in matrices.items():
  # Multiple different F32 blocks also test isolation across block boundaries.
  x=rng.normal(0,.2,(4,weight.shape[1]));inputs=client.encrypt_slots(pack(x))
  output=evaluator.linear(inputs,weight)
  verify(name,output,pack(x@weight.T),depth-1)
  del inputs,output
 # Original Qwen public input with its actual gate projection, then SiLU.
 x=embeddings.reshape(1,-1);input_ct=client.encrypt_slots(pack(x))
 gate=evaluator.linear(input_ct,matrices['gate24x16'])
 approx=public_toy_approximations('qwen')['silu']
 plain_gate=x@matrices['gate24x16'].T
 assert np.all((plain_gate>=approx.domain[0])&(plain_gate<=approx.domain[1]))
 output=evaluator.arithmetic.evaluate(gate,approx,output_weights=evaluator.public_feature(np.ones(24)))
 verify('actual_fixture_gate_then_silu',output,pack(approx.evaluate_public(plain_gate)),1)
 del input_ct,gate,output
 # Actual encrypted square plus one-level segmented variance reduction.
 x=rng.normal(0,.2,(4,16));inputs=client.encrypt_slots(pack(x))
 squares=evaluator.arithmetic.multiply(inputs,inputs)
 variance=evaluator.reduce_features(squares,factor=1/16,bias=case['config']['rms_norm_eps'])
 expected=np.repeat((x*x).mean(axis=1)+case['config']['rms_norm_eps'],32)
 expected=np.tile(expected,slots//len(expected))
 verify('square_then_segmented_variance',variance,expected,depth-2)
 report['phase_results']=evaluator.phase_results
 # Parameter generation only: no full-ring keys, context or execution here.
 fq=composite_prime_gen(2,78,61,56,262144,31);fp=util.generate_p_towers(fq,2,131072,aux_bits=30)
 assert he_params.qp_is_secure(fq,fp,131072,he_params.SecurityTarget(128,'classical'))
 he_params.assert_p_coverage(fq,fp,2)
 report['future_full_qwen1_profile']={'degree':131072,'q_count':len(fq),'p_count':len(fp),'dnum':2,'depth':38,'actual_log2_qp':sum(math.log2(x) for x in fq+fp),'security_bits':128,'executed':False}
 report['passed']=True
except BaseException as error:
 import traceback
 report['error']=f'{type(error).__name__}: {error}';traceback.print_exc()
finally:
 from jaxite_word import ckks_ctx
 import gc
 for context in (locals().get('client'),locals().get('server')):
  if context is not None: ckks_ctx.evict_context_caches(context)
 for name in ('client','server','evaluator','inputs','input_ct','squares','variance','output','gate','context'):
  globals().pop(name,None)
 jax.clear_caches();gc.collect()
 if key_store is not None: key_store.close()
 report['wall_s']=time.monotonic()-started;report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 report['source_unchanged']=source_hashes==hashes();report['passed']=report['passed'] and report['source_unchanged']
 report['jax_version']=jax.__version__;report['numpy_version']=np.__version__
 args.report.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
raise SystemExit(0 if report['passed'] else 1)
