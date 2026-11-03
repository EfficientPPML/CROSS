"""Actual public p60 profile/resource preparation; no contexts, keys or ciphertexts."""
from pathlib import Path
import hashlib,json,math,os,shutil,sys,time
LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
import proof

def prepare():
 import jax
 from jaxite_word import CKKSContext,key_gen,he_params
 from jaxite_word.bootstrapping import Bootstrap
 from encrypted_refresh import RefreshProfile,refresh_plan
 forbidden_calls=[]
 def forbidden(*a,**kw):forbidden_calls.append('forbidden crypto call');raise AssertionError('no contexts, keys, or ciphertexts in public preflight')
 CKKSContext.__init__=forbidden;Bootstrap.__init__=forbidden
 for module in {sys.modules[name] for name in ('key_gen','jaxite_word.key_gen')}:
  for name in proof.KEYGEN_METHODS:setattr(module,name,forbidden)
 params=RefreshProfile(131072,49,60,61,8).parameters();plan=refresh_plan(131072,65536,(4,4),max_level=49)
 q,p=params['q_towers'],params['p_towers'];he_params.assert_p_coverage(q,p,8)
 audit=json.loads((LAB/'profile_audit.json').read_text());pinned=next(r for r in audit['profiles'] if r['name']=='depth49_p60_d8')
 assert q==pinned['q'] and p==pinned['p'] and params['scaling_factor']==pinned['canonical_scale']
 assert len(q)==100 and len(p)==14 and len(plan['rotations'])==106 and plan['output_level']==25
 assert he_params.qp_is_secure(q,p,131072,he_params.SecurityTarget(128,'classical'))
 alpha=(len(q)+7)//8;parts=(len(q)+alpha-1)//alpha;pair=2*parts*(len(q)+len(p))*131072*4
 usage=shutil.disk_usage('/dev/shm');required=proof.required_key_space(params,plan['rotations'])
 result={'passed':True,'scope':__doc__,'keygen_context_ciphertext_or_HE_launched':False,'forbidden_calls':forbidden_calls,
 'profile':{'degree':131072,'slots':65536,'depth':49,'scale_bits':60,'first_bits':61,'dnum':8,'q':q,'p':p,'canonical_scale':params['scaling_factor'],
  'actual_log2_QP':math.log2(math.prod(q+p)),'QP_bits':math.prod(q+p).bit_length(),'security_ceiling':3523,'actual_128_classical_secure':True,'P_coverage':True,'alpha':alpha,'parts':parts},
 'refresh_plan':plan,'proof':{'bounds':[16.,128.],'meta_precision':10,'full_slot_aliases':1,'level_budget':[4,4],'input_level':2,'output_level':25,'square_level':24,
 'all_four_outputs_before_first_decode':True,'strict_absolute_gate_refresh_and_square':1e-3,'CPU_affinity':proof.CPU_AFFINITY,'threads':4,'healthy_deadline':None},
 'resources':{'rotation_pair_bytes':pair,'rotation_count':106,'raw_rotation_bytes':106*pair,'raw_rotation_GiB':106*pair/2**30,
 'relinearization_pair_bytes':pair,'public_key_bytes':2*len(q)*131072*4,'secret_key_bytes':len(q)*131072*4,
 'headers_allowance_bytes':106*8192,'extra_free_margin_bytes':4*1024**3,'required_free_bytes':required,
 'tmpfs_total_bytes':usage.total,'tmpfs_free_bytes':usage.free,'raw_keys_plus_margin_fit_observation':usage.free>=required,
 'two_level2_input_payload_bytes':2*2*6*131072*4,'two_refresh_and_two_square_payload_bytes':2*2*(52+50)*131072*4,
 'scope':'Public storage lower bounds only; controls/compiler/transient arrays/host peak and run time are unmeasured.'},
 'pattern_sha256':{str(int(b)):hashlib.sha256(proof.input_pattern(b).astype('<f8').tobytes()).hexdigest() for b in proof.BOUNDS},
 'backend_initialization':'CPU backend may initialize through public package import; no CKKS context/key/ciphertext calls.',
 'source_py_sha256':{str(p.relative_to(LAB)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(LAB.rglob('*.py'))}}
 (LAB/'public_plan.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('passed','proof','resources')},indent=2))
if __name__=='__main__':prepare()
