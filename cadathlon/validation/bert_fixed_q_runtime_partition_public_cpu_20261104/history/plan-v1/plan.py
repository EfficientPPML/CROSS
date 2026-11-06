"""Public static acceptance plan only; no backend or runtime module imports."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).parent

def frozen_partitions(num_q,*,max_q,dnum):
 if any(type(v)is not int for v in (num_q,max_q,dnum))or not 0<num_q<=max_q or not 0<dnum<=max_q:
  raise ValueError('exact positive fixed-Q partition dimensions required')
 alpha=(max_q+dnum-1)//dnum
 return tuple(tuple(range(i,min(i+alpha,num_q)))for i in range(0,num_q,alpha))

def build(profile,prior):
 q=profile['q'];N=profile['degree'];assert len(q)==100 and N==131072 and profile['depth']==49
 plans=[]
 for dnum,aux in ((8,30),(6,30),(6,31)):
  source=next(r for r in prior['profiles']if (r['dnum'],r['aux_bits'])==(dnum,aux));p=source['P']
  rows=[]
  for level in range(50):
   n=2*(level+1);parts=frozen_partitions(n,max_q=len(q),dnum=dnum)
   rows.append(dict(level=level,num_q=n,fixed_alpha=(100+dnum-1)//dnum,active_digits=len(parts),
     partition_indices=[list(x)for x in parts],P_covers_fixed_partitions=math.prod(p)>=max(math.prod(q[i]for i in part)for part in parts),
     raw_key_shape=[dnum,N,100+len(p)],formatted_key_shape=[dnum,256,512,n+len(p)],
     active_key_shape=[len(parts),256,512,n+len(p)],Q_slice=[0,n],P_slice=[100,100+len(p)]))
  plans.append(dict(dnum=dnum,aux_bits=aux,P=p,rows=rows,
   registered_full_config_fields=dict(degree=N,num_slots=65536,q_towers=q,p_towers=p,
    scaling_factor=math.prod(q[-2:]),dnum=dnum,composite_degree=2,first_mod_size=61,
    scaling_mod_size=60,register_word_size=aux+1,sigma=3.190000057220458984375,secret_distribution='uniform_ternary'),
   registration_interpretation='Exact full fixed-Q candidate; register_word_size=aux_bits+1 is the actual validator coupling. Do not replace primes/counts to obtain acceptance.'))
 return dict(scope='Plan only, no registered or abstract-kernel function executed',actual_backend=False,profiles=plans,
  registered_scope='One full-chain registered validator result per profile; lower-level Q prefixes remain views of that fixed-key profile, not independent newly registered rings.',
  abstract_scope='All50CD2levels: actual get_rot_key formatting and HERot key shape validation under abstract JAX shape tracing; no key/ciphertext arrays or compiled HE execution.',
  pending=['root plan review','actual imported validator outcomes','actual source method abstract traces','exact source-bound regressions/readback'])

if __name__=='__main__':
 profile=json.loads((HERE/'inputs/profile.json').read_bytes())['profile'];prior=json.loads((HERE/'inputs/public_profile_audit.json').read_bytes())
 result=build(profile,prior);(HERE/'public_plan.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'profiles':len(result['profiles']),'fixed_prefixes':sum(len(r['rows'])for r in result['profiles']),'backend':False}))
