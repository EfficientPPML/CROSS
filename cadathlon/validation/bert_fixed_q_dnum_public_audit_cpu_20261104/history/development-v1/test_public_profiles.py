"""Public accounting/validation regressions; no state construction."""
import math
import pytest
import audit

@pytest.mark.parametrize('d,p,pair',[(6,18,742391808),(8,14,956301312),(6,17,736100352)])
def test_public_raw_shapes(d,p,pair):
 r=audit.resource(131072,100,p,d,keys=118)
 assert r['rotation_pair_raw_bytes']==pair and r['rotation_keys_raw_bytes']==118*pair
 assert r['one_ciphertext_raw_bytes']==104857600


def test_fixed_budget_scenarios_do_not_claim_physical_peak():
 r=audit.resource(131072,100,18,6,keys=166)
 assert r['rotation_keys_raw_GiB']==114.7734375 and 'transient'in r['excludes']and 'JAX'in r['excludes']


def test_actual_real_entrypoints_reject_uncovered_and_nonprime():
 hp,calls,old=audit.install_guards()
 try:
  with pytest.raises(ValueError):hp.validate_kernel_moduli([17,97],[193],1)
  assert not hp.p_covers_q([17,97],[193],1)
  with pytest.raises(ValueError):hp.assert_p_coverage([17,97],[193],1)
  with pytest.raises(ValueError):hp.validate_kernel_moduli([9],[17],1)
  assert not calls
 finally:
  for owner,name,value in reversed(old):setattr(owner,name,value)
