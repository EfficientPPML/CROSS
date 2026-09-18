#!/usr/bin/env python3
"""Barrett vs Montgomery full-bootstrap parity gate (N=256).

Runs the SAME depleted ciphertext (identical keys, identical encryption
randomness) through a Barrett-backed and a Montgomery/BAT-backed bootstrap
and requires bit-canonical equality of the output residues plus matching
tracked scale. The pipeline is a deterministic composition of the
Phase-2-verified level operators, form-preserving raw CRT arithmetic, and
host-side encodes, so exact agreement is the expected behavior — any
divergence indicates a representation-boundary bug.

Runtime: two full N=256 (nq=44) setups + bootstraps on CPU — roughly an
hour. Run directly:
    python3 bootstrapping_montgomery_test.py
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from absl.testing import absltest

import bootstrapping
import ckks_ctx
import finite_field as ff_context
import key_gen as kg
import util
from bootstrapping import decrypt_decode
from composite_prime_gen import composite_prime_gen

# Bootstrap intermediates legitimately exceed the decode approximation limit.
# Scoped so the guard is restored for anything running later in this process.
_decode_guard_bypass = ckks_ctx.bypass_decode_stddev_check()


def setUpModule():
  _decode_guard_bypass.__enter__()


def tearDownModule():
  _decode_guard_bypass.__exit__(None, None, None)

TEST_X = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
SIGMA = 3.190000057220458984375

DEGREE = 256
NQ = 44
CD = 2
LEVEL_BUDGET = [1, 1]
DNUM = 3


def _build_backend_pair():
  """One chain + one key set; two fully initialized (ctx, Bootstrap) pairs."""
  num_slots = DEGREE // 2
  cycl_order = 2 * DEGREE
  r = 16
  c = DEGREE // r
  q_towers = composite_prime_gen(CD, NQ, 61, 56, cycl_order, 31)
  p_towers = util.generate_p_towers(q_towers, dnum=DNUM, degree=DEGREE)
  key_pair = kg.gen_pke_pair(q_towers, [], DEGREE)
  sf0 = float(q_towers[-2]) * float(q_towers[-1])
  ek = kg.gen_evaluation_key(
      key_pair["secret_key"], q=q_towers, P=p_towers,
      noise_std=SIGMA, noise_scale=1, dnum=DNUM)
  eval_key = (
      jnp.array(ek["a"], dtype=jnp.uint32).transpose(0, 2, 1),
      jnp.array(ek["b"], dtype=jnp.uint32).transpose(0, 2, 1),
  )
  params = dict(
      degree=DEGREE, num_slots=num_slots, scaling_factor=sf0,
      output_scale=sf0, q_towers=q_towers, p_towers=p_towers,
      p=round(math.log2(sf0)), CKKS_M_FACTOR=1, max_bits_in_word=61,
      noise_scale_degree=1, composite_degree=CD,
      public_key=key_pair["public_key"], secret_key=key_pair["secret_key"],
      evaluation_key=eval_key)

  # Discover the bootstrap rotation indices with a throwaway context.
  ctx_tmp = ckks_ctx.CKKSContext(dict(params))
  ctx_tmp.program_initialization(
      total_rotation_indices=[1, cycl_order - 1], dnum=DNUM, r=r, c=c, batch=1)
  bs_tmp = bootstrapping.Bootstrap(ctx_tmp)
  bs_tmp.control_gen(level_budget=LEVEL_BUDGET)
  ri = sorted(set(list(bs_tmp._all_rot_indices) + [cycl_order - 1]))
  del ctx_tmp, bs_tmp

  # Generate every rotation key once so both backends share key material
  # (setup_key only registers indices; key generation draws fresh noise).
  rot_keys = {}
  for rot_idx in ri:
    rk = kg.gen_rotation_key(
        key_pair["secret_key"], q_towers, p_towers, rot_idx,
        dnum=DNUM, noise_std=SIGMA, noise_scale=1)
    rot_keys[rot_idx] = rk[rot_idx]

  def make(finite_field_cls):
    ctx = ckks_ctx.CKKSContext(dict(params))
    ctx.program_initialization(
        total_rotation_indices=ri, dnum=DNUM, r=r, c=c, batch=1,
        pregenerated_rotation_keys=rot_keys,
        finite_field_context=finite_field_cls)
    bs = bootstrapping.Bootstrap(ctx)
    bs.control_gen(level_budget=LEVEL_BUDGET)
    bs.setup_key()
    return ctx, bs

  return make, q_towers, sf0, num_slots


class BootstrapMontgomeryParityTest(absltest.TestCase):

  def test_full_bootstrap_parity_n256(self):
    make, q_towers, sf0, num_slots = _build_backend_pair()
    ctx_b, bs_b = make(None)  # Barrett default
    ctx_m, bs_m = make(ff_context.MontgomeryContext)

    # Encrypt ONCE (encryption noise is a CSPRNG draw); mirror the identical
    # standard-form payload into the Montgomery pipeline via its encrypt
    # boundary conversion.
    slots = [complex(v, 0) for v in TEST_X] + \
            [complex(0, 0)] * (num_slots - len(TEST_X))
    ct_b = ctx_b.encrypt(ctx_b.encode(slots))
    ct_b.validate()
    payload_std = ct_b.polynomial
    cache_m = ctx_m._param_cache
    payload_mont = cache_m.ff_q_max.to_computation_format(
        jnp.asarray(payload_std, jnp.uint64))
    ct_m = bs_m._new_ciphertext(payload_mont, cache_m.max_level)

    for bs, ct in ((bs_b, ct_b), (bs_m, ct_m)):
      bs._set_scale(ct, sf0)
      bs._set_nsd(ct, 1)
    ct_b_dep, _ = bs_b._level_reduce(ct_b, bs_b._infer_level(ct_b), 1)
    ct_m_dep, _ = bs_m._level_reduce(ct_m, bs_m._infer_level(ct_m), 1)

    out_b = bs_b.bootstrap(ct_b_dep)
    out_m = bs_m.bootstrap(ct_m_dep)

    # Bit-canonical residue equality after conversion to standard form.
    self.assertEqual(out_b.polynomial.shape, out_m.polynomial.shape)
    nq_out = out_b.polynomial.shape[-1]
    q_out = np.array(q_towers[:nq_out], dtype=np.uint64)
    std_b = np.asarray(out_b.polynomial, dtype=np.uint64) % q_out
    ff_out = ff_context.MontgomeryContext(moduli=q_towers[:nq_out])
    std_m = np.asarray(ff_out.to_original_format(
        jnp.asarray(out_m.polynomial, jnp.uint64)))
    np.testing.assert_array_equal(std_b, std_m)

    # Tracked scale metadata must agree (same float pipeline on both sides).
    self.assertEqual(bs_b._get_scale(out_b), bs_m._get_scale(out_m))

    # Both decode to the expected message with the usual functional quality.
    dec_b = decrypt_decode(ctx_b, bs_b, out_b, len(TEST_X))
    dec_m = decrypt_decode(ctx_m, bs_m, out_m, len(TEST_X))
    mn = 99.0
    print("\n[montgomery-parity N=256] per-slot precision:")
    for i, expected in enumerate(TEST_X):
      vb, vm = dec_b[i].real, dec_m[i].real
      self.assertAlmostEqual(vb, vm, places=12)
      err = abs(vm - expected)
      bits = -math.log2(err / abs(expected)) if err > 0 else 99.0
      mn = min(mn, bits)
      print(f"    exp={expected:5.2f}  got={vm:12.8f}  bits={bits:.1f}")
    print(f"    >>> min = {mn:.1f} bits")
    self.assertGreater(mn, 3.0)


if __name__ == "__main__":
  absltest.main()
