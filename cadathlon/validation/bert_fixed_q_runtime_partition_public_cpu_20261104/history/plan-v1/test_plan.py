import json,math,sys
import pytest
import plan

def test_frozen_alpha_never_repartitions_lower_Q_view():
 parts=plan.frozen_partitions(36,max_q=100,dnum=6)
 assert tuple(map(len,parts))==(17,17,2)
 assert parts[0]==tuple(range(17))and parts[1]==tuple(range(17,34))and parts[2]==(34,35)
 assert len(plan.frozen_partitions(36,max_q=36,dnum=6))==6

@pytest.mark.parametrize('dnum,alpha',((6,17),(8,13)))
def test_all50_fixed_prefixes_cover_once_and_fit_original_key(dnum,alpha):
 for level in range(50):
  n=2*(level+1);p=plan.frozen_partitions(n,max_q=100,dnum=dnum)
  assert sum((list(x)for x in p),[])==list(range(n))
  assert len(p)==math.ceil(n/alpha)<=dnum
  assert all(len(x)==alpha for x in p[:-1])and 1<=len(p[-1])<=alpha

@pytest.mark.parametrize('values',((True,100,6),(2,100,6.),(0,100,6),(102,100,6),(2,100,0)))
def test_invalid_partition_dimensions_reject(values):
 with pytest.raises(ValueError):plan.frozen_partitions(values[0],max_q=values[1],dnum=values[2])

def test_plan_retains_exact_public_primes_and_current_d8_baseline():
 profile=json.loads((plan.HERE/'inputs/profile.json').read_bytes())['profile'];prior=json.loads((plan.HERE/'inputs/public_profile_audit.json').read_bytes());result=plan.build(profile,prior)
 assert [(x['dnum'],x['aux_bits'])for x in result['profiles']]==[(8,30),(6,30),(6,31)]
 assert result['profiles'][0]['P']==profile['p']
 for row in result['profiles']:
  assert row['registered_full_config_fields']['q_towers']==profile['q']
  assert row['registered_full_config_fields']['register_word_size']==row['aux_bits']+1
  assert all(x['P_covers_fixed_partitions']for x in row['rows'])
  assert len(row['rows'])==50
 assert result['actual_backend']is False
 assert not any(n in sys.modules for n in ('numpy','jax','jaxite_word','torch','transformers'))
