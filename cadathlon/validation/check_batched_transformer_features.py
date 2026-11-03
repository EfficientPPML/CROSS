"""Real secure CPU batching proof only; no TPU/full-transformer/refresh claim."""
import sys,json,time,os,argparse,hashlib,resource
from pathlib import Path
import numpy as np
os.environ.setdefault('JAX_PLATFORMS','cpu')
if os.environ['JAX_PLATFORMS'] != 'cpu': raise RuntimeError('This proof requires JAX_PLATFORMS=cpu')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import jax
import jax.numpy as jnp
from jaxite_word import CKKSContext,Polynomial,he_params,key_gen,util
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_polynomial import EncryptedPolynomialEvaluator,nonlinear_approximation
jax.config.update('jax_enable_x64',True)
SOURCES=[Path(__file__), ROOT/'cadathlon/common/encrypted_polynomial.py', ROOT/'cadathlon/common/support.py', ROOT/'jaxite_word/ckks_ctx.py', ROOT/'jaxite_word/he_ops.py', ROOT/'jaxite_word/hemul.py', ROOT/'jaxite_word/polynomial.py']
def hashes(): return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES}
source_hashes=hashes()
started=time.monotonic();degree=16384;depth=3;dnum=3
q=composite_prime_gen(2,2*(depth+1),45,44,2*degree,31);p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
assert he_params.qp_is_secure(q,p,degree,he_params.SecurityTarget(128,'classical'));he_params.assert_p_coverage(q,p,dnum)
keys=key_gen.gen_pke_pair(q,[],degree);evk=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum,noise_std=he_params.DEFAULT_SIGMA,noise_scale=1)
scale=float(q[-2])*q[-1];params=dict(degree=degree,num_slots=degree//2,scaling_factor=scale,output_scale=scale,q_towers=q,p_towers=p,composite_degree=2,degree_layout=(128,128),p=44,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
client=CKKSContext(dict(params,**keys))
def server(batch):
 c=CKKSContext(dict(params,evaluation_key=[jnp.asarray(evk[name],dtype=jnp.uint32).transpose(0,2,1) for name in ('a','b')]))
 c.program_initialization([],dnum=dnum,r=128,c=128,batch=batch)
 assert c.secret_key is None and c._param_cache.secret_key is None
 return c
single=server(1);client._polynomial_ntt_ctx=single._param_cache.ntt_q_max
values=[np.tile([-.4+i*.1,.1+i*.1],degree//4) for i in range(4)]
inputs=[client.encrypt_slots(x) for x in values]
def wrap(data,template):
 out=Polynomial.from_array(data,{'batch':data.shape[0],'num_elements':template.num_elements,'degree':template.degree,'num_moduli':template.num_moduli,'precision':32,'degree_layout':template.degree_layout},{'moduli':template.moduli,'ntt_ctx':template.ntt_ctx})
 for name in ('_ckks_scale','_ckks_nsd'):
  if hasattr(template,name): setattr(out,name,getattr(template,name))
 return out
approx=nonlinear_approximation('silu',(-2,2),2)
scalar=EncryptedPolynomialEvaluator(single,strategy='power')
expected=[scalar.evaluate(value,approx) for value in inputs]
report={'scope':__doc__,'degree':degree,'q_count':len(q),'p_count':len(p),'dnum':dnum,'qp_secure':True,'extra_rotation_keys':0,'batches':{}}
for batch in (2,4):
 context=server(batch);evaluator=EncryptedPolynomialEvaluator(context,strategy='power')
 combined=wrap(jnp.concatenate([x.polynomial for x in inputs[:batch]],axis=0),inputs[0])
 out=evaluator.evaluate(combined,approx);out.polynomial.block_until_ready()
 concat=jnp.concatenate([x.polynomial for x in expected[:batch]],axis=0)
 np.testing.assert_array_equal(out.polynomial,concat)
 errors=[]
 for i in range(batch):
  piece=wrap(out.polynomial[i:i+1],out)
  decoded=client.decrypt_slots(piece)
  errors.append(float(np.max(np.abs(decoded-approx.evaluate_public(values[i])))))
 assert out._ckks_scale==expected[0]._ckks_scale and out._ckks_nsd==expected[0]._ckks_nsd
 report['batches'][str(batch)]={'bit_exact_vs_individual':True,'max_error_vs_public_polynomial':max(errors),'shape':list(out.shape),'output_level':evaluator._level(out),'scale':out._ckks_scale,'noise_scale_degree':out._ckks_nsd}
report['cpu_wall_s']=time.monotonic()-started
report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report['jax_version']=jax.__version__;report['numpy_version']=np.__version__;report['backend']=jax.default_backend()
report['source_sha256']=source_hashes;report['source_unchanged']=source_hashes==hashes();assert report['source_unchanged']
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=Path(__file__).with_suffix('.json'));args=parser.parse_args()
args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
