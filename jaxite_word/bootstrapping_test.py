#!/usr/bin/env python3
"""Functional and schedule tests for bootstrapping.py.

The default suite covers low-cost arithmetic/schedule invariants and the
matched N=256, scalingMod=56 bootstrap.  Precision parity against independently
randomized OpenFHE runs is tested by bootstrapping_openfhe_crosscheck.py; the
absolute thresholds here are deliberately looser functional-quality guards.

Optional long-running Meta-BTS and larger-ring quality tests remain gated by
environment variables; they are not the OpenFHE parity criterion.

Each case depletes a freshly-encrypted ciphertext to a low level, refreshes it
via bootstrap / meta_bootstrap, and checks per-slot precision (relative-error
bits) against the known plaintext.

Run:
    python bootstrapping_test.py                 # N=256 checks (default)
    BOOTSTRAP_TEST_N4096=1 python bootstrapping_test.py   # + N=4096 Meta-BTS
"""
from __future__ import annotations

import copy
import math
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from absl.testing import absltest
from absl.testing import parameterized

import bootstrapping
import ckks_ctx
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

# Test message spans small (0.25) to large (5.0), so both absolute and relative
# error behavior are exercised.
TEST_X = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]


def _build(degree, nq, cd, level_budget, dnum=3):
  """Construct a CKKSContext + Bootstrap ready for a full bootstrap."""
  num_slots = degree // 2
  cycl_order = 2 * degree
  r = 16 if degree <= 256 else 64
  c = degree // r
  q_towers = composite_prime_gen(cd, nq, 61, 56, cycl_order, 31)
  p_towers = util.generate_p_towers(q_towers, dnum=dnum, degree=degree)
  key_pair = kg.gen_pke_pair(q_towers, [], degree)
  sf0 = float(q_towers[-2]) * float(q_towers[-1])
  params = dict(
      degree=degree, num_slots=num_slots, scaling_factor=sf0, output_scale=sf0,
      q_towers=q_towers, p_towers=p_towers, p=round(math.log2(sf0)),
      CKKS_M_FACTOR=1, max_bits_in_word=61, noise_scale_degree=1,
      composite_degree=cd, public_key=key_pair["public_key"],
      secret_key=key_pair["secret_key"])

  # Discover bootstrap rotation indices, then full init.
  ctx_tmp = ckks_ctx.CKKSContext(params)
  ctx_tmp.program_initialization(
      total_rotation_indices=[1, cycl_order - 1], dnum=dnum, r=r, c=c, batch=1)
  bs_tmp = bootstrapping.Bootstrap(ctx_tmp)
  bs_tmp.control_gen(level_budget=level_budget)
  ri = sorted(set(list(bs_tmp._all_rot_indices) + [cycl_order - 1]))
  del ctx_tmp, bs_tmp

  ctx = ckks_ctx.CKKSContext(params)
  ctx.program_initialization(
      total_rotation_indices=ri,
      dnum=dnum, r=r, c=c, batch=1)
  bs = bootstrapping.Bootstrap(ctx)
  bs.control_gen(level_budget=level_budget)
  bs.setup_key()
  return ctx, bs, sf0, r, c, num_slots


def _encrypt_depleted(ctx, bs, sf0, r, c, num_slots, deplete_to=1):
  """Encrypt TEST_X and deplete to a low level (simulating a used-up ct)."""
  slots = [complex(v, 0) for v in TEST_X] + \
          [complex(0, 0)] * (num_slots - len(TEST_X))
  ct = ctx.encrypt(ctx.encode(slots))
  ct.validate()
  bs._set_scale(ct, sf0)
  bs._set_nsd(ct, 1)
  ct_dep, _ = bs._level_reduce(ct, bs._infer_level(ct), deplete_to)
  return ct_dep


def _min_precision_bits(decoded):
  """Minimum per-slot relative-error precision (bits) over TEST_X."""
  mn = 99.0
  per_slot = []
  for i, expected in enumerate(TEST_X):
    v = decoded[i].real
    err = abs(v - expected)
    bits = -math.log2(err / abs(expected)) if err > 0 else 99.0
    per_slot.append((expected, v, bits))
    mn = min(mn, bits)
  return mn, per_slot


class BootstrappingFunctionalTest(parameterized.TestCase):

  def _report(self, tag, per_slot, mn):
    print(f"\n[{tag}] per-slot precision:")
    for expected, v, bits in per_slot:
      print(f"    exp={expected:5.2f}  got={v:12.8f}  bits={bits:.1f}")
    print(f"    >>> min = {mn:.1f} bits")

  def test_meta_precision_validation(self):
    self.assertEqual(bootstrapping.Bootstrap._meta_precision(8), 8)
    with self.assertRaises(TypeError):
      bootstrapping.Bootstrap._meta_precision(8.0)
    with self.assertRaises(ValueError):
      bootstrapping.Bootstrap._meta_precision(0)

  def test_ckks_dft_matrix_stays_on_host(self):
    """Control-generation DFT constants must not compile on TPU."""
    matrix = bootstrapping._build_ckks_dft_matrix(
        8, 32, conjugate_transpose=True
    )
    self.assertIsInstance(matrix, np.ndarray)
    self.assertEqual(matrix.dtype, np.complex128)
    rot_group = np.asarray(
        bootstrapping._ckks_rotation_group(32, 8), dtype=np.float64
    )
    indices = np.arange(8, dtype=np.float64)
    expected = np.exp(
        -2.0j * np.pi * np.outer(indices, rot_group) / 32
    )
    np.testing.assert_allclose(np.asarray(matrix), expected, rtol=0, atol=1e-15)
    diagonals = bootstrapping._extract_diagonals(matrix)
    self.assertLen(diagonals, 8)
    self.assertTrue(all(isinstance(d, np.ndarray) for d in diagonals.values()))

    collapsed = bootstrapping._decompose_dft_into_levels(
        8, 32, 3, inverse=False
    )
    self.assertTrue(all(
        isinstance(d, np.ndarray)
        for level in collapsed
        for d in level.values()
    ))
    hoisted = bootstrapping._decompose_dft_into_hoisted_groups(
        8, 32, 3, inverse=False
    )
    self.assertTrue(all(
        isinstance(d, np.ndarray)
        for level in hoisted
        for group in level
        for _, d in group['entries']
    ))

  def test_large_integer_multiplier_reduces_before_uint64_product(self):
    modulus = 2147483489
    precision_factor = 1 << 62
    bs = object.__new__(bootstrapping.Bootstrap)
    bs._cache = types.SimpleNamespace(
        q_moduli_at_level=lambda unused_level: [modulus]
    )
    bs._ensure_canonical = lambda ciphertext, unused_level=None: ciphertext
    ct = types.SimpleNamespace(
        polynomial=jnp.asarray([[[[[modulus - 2]]]]], dtype=jnp.uint32),
        _ckks_scale=1.0,
        _ckks_nsd=1,
    )

    result = bs._mult_by_integer(ct, precision_factor, level=0)

    expected = ((modulus - 2) * (precision_factor % modulus)) % modulus
    self.assertEqual(int(result.polynomial.reshape(-1)[0]), expected)

  def test_mod_raise_retains_openfhe_unreduced_crt_multiple(self):
    """Composite ModRaise must preserve OpenFHE's intentional +v*Q lift."""
    degree = 4
    src_moduli = [17, 41]
    dst_moduli = [17, 41, 73]
    coefficients = [-346, 0, 1, 173]

    limbs = []
    for modulus in src_moduli:
      psi = util.root_of_unity(2 * degree, modulus)
      evaluation = util.ntt_negacyclic_bit_reverse(
          [value % modulus for value in coefficients], modulus, psi)
      limbs.append(util.bit_reverse_array(evaluation))
    payload = jnp.asarray(limbs, dtype=jnp.uint32).T.reshape(
        1, 1, 1, degree, len(src_moduli))

    raised = bootstrapping._mod_raise_array(
        payload, src_moduli, dst_moduli)

    # Existing limbs are retained byte-for-byte.
    self.assertTrue(jnp.array_equal(raised[..., :2], payload))

    # For coefficient -346 in basis {17, 41}, OpenFHE centers each CRT digit
    # separately and extends their unreduced sum 351.  A globally centered CRT
    # reconstruction would instead extend -346; the two differ by Q=697.
    modulus = dst_moduli[-1]
    raised_limb = [int(v) for v in raised.reshape(degree, 3)[:, -1]]
    decoded = util.intt_negacyclic_bit_reverse(
        util.bit_reverse_array(raised_limb), modulus,
        util.root_of_unity(2 * degree, modulus))
    q_product = math.prod(src_moduli)
    q_hats = [q_product // q for q in src_moduli]
    expected = []
    for value in coefficients:
      lift = 0
      for q, q_hat in zip(src_moduli, q_hats):
        digit = (value % q) * pow(q_hat, -1, q) % q
        if digit > q // 2:
          digit -= q
        lift += digit * q_hat
      expected.append(lift % modulus)
    self.assertEqual(decoded, expected)
    self.assertEqual(expected[0], 351 % modulus)
    self.assertNotEqual(expected[0], (-346) % modulus)

  def test_auxiliary_plaintexts_use_recursive_openfhe_scales(self):
    recursive_scales = {0: 101.0, 1: 103.0, 2: 107.0, 3: 109.0}
    cache = types.SimpleNamespace(
        max_level=3,
        q_moduli_at_level=lambda level: [17, 41][:level + 1],
        scaling_factor_recursive=lambda level: recursive_scales[level],
    )
    bs = object.__new__(bootstrapping.Bootstrap)
    bs.ctx = types.SimpleNamespace(_param_cache=cache)
    bs._cache = cache
    seen_scales = []

    def fake_encode(diagonal, moduli, num_q, scale):
      del diagonal, moduli, num_q
      seen_scales.append(scale)
      return scale

    bs._encode_diagonal = fake_encode
    encoded = bs._pre_encode_diags([{0: [1.0]}, {0: [1.0]}], 2)

    self.assertEqual(seen_scales, [107.0, 103.0])
    self.assertEqual(encoded, [{0: 107.0}, {0: 103.0}])

  def test_hybrid_p_basis_matches_openfhe_partition_sizing(self):
    q44 = composite_prime_gen(2, 44, 61, 56, 512, 31)
    q58 = composite_prime_gen(2, 58, 61, 56, 1024, 31)

    self.assertEqual(util.compute_num_p_towers(q44, dnum=3), 15)
    self.assertEqual(util.compute_num_p_towers(q58, dnum=3), 19)
    self.assertLen(util.generate_p_towers(q44, 3, degree=256), 15)
    self.assertLen(util.generate_p_towers(q58, 3, degree=512), 19)

  def test_bootstrap_nq_matches_openfhe_depth_schedule(self):
    direct = bootstrapping.Bootstrap.compute_bootstrap_nq(
        2, level_budget=[1, 1], headroom=5
    )
    self.assertEqual(direct['boot_depth'], 16)
    self.assertEqual(direct['nq'], 44)
    self.assertEqual(direct['max_level'], 21)
    self.assertEqual(direct['expected_output_level'], 5)
    self.assertEqual(direct['expected_output_nq'], 12)

    collapsed = bootstrapping.Bootstrap.compute_bootstrap_nq(
        2, level_budget=[4, 4], headroom=6
    )
    self.assertEqual(collapsed['boot_depth'], 22)
    self.assertEqual(collapsed['nq'], 58)
    self.assertEqual(collapsed['max_level'], 28)
    self.assertEqual(collapsed['expected_output_level'], 6)
    self.assertEqual(collapsed['expected_output_nq'], 14)

  def test_structural_rotation_plan_matches_numeric_transforms(self):
    for degree, num_slots, budget in (
        (16, 8, (1, 1)),
        (32, 16, (2, 3)),
        (32, 4, (2, 2)),
    ):
      sparse = num_slots < degree // 2
      key_mod = 2 * num_slots if sparse else num_slots
      rotations = set()
      for levels, inverse in zip(budget, (False, True)):
        transforms = bootstrapping._decompose_dft_into_levels(
            num_slots, 4 * num_slots, levels, inverse=inverse,
            key_mod=key_mod,
        )
        rotations.update(
            int(index)
            for transform in transforms
            for index in transform
            if index
        )
        if not sparse:
          groups = bootstrapping._decompose_dft_into_hoisted_groups(
              num_slots, 4 * num_slots, levels, inverse=inverse
          )
          if groups is None:
            rotations.update(
                int(index)
                for transform in transforms
                for giant, entries in (
                    bootstrapping._collapsed_hoisted_plan(
                        transform, num_slots
                    )
                )
                for index in (giant, *(baby for _, baby in entries))
                if index
            )
          else:
            rotations.update(
                int(index)
                for level_groups in groups
                for group in level_groups
                for index in (
                    group['giant'],
                    *(baby for baby, _ in group['entries']),
                )
                if index
              )
      if sparse:
        gap = (degree // 2) // num_slots
        rotations.update(
            shift * num_slots
            for shift in (1 << exponent for exponent in range(gap.bit_length()))
            if shift < gap
        )
        rotations.add(num_slots)

      self.assertEqual(
          bootstrapping._bootstrap_rotation_indices(
              degree, num_slots, budget
          ),
          tuple(sorted(rotations)),
      )
      self.assertEqual(
          bootstrapping.Bootstrap.required_rotation_indices(
              degree, num_slots, budget
          ),
          tuple(sorted({*rotations, 2 * degree - 1})),
      )

  def test_collapsed_hoisted_plan_preserves_each_fft_level(self):
    """Centered baby/giant factorization equals the dense diagonal sum."""
    n = 256
    rng = np.random.default_rng(7)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    for inverse in (False, True):
      levels = bootstrapping._decompose_dft_into_levels(
          n, 4 * n, 2, inverse=inverse
      )
      for diags in levels:
        dense = sum(
            np.asarray(diag) * np.roll(x, -int(k))
            for k, diag in diags.items()
        )
        factored = np.zeros(n, dtype=np.complex128)
        for giant, entries in bootstrapping._collapsed_hoisted_plan(
            diags, n
        ):
          inner = np.zeros(n, dtype=np.complex128)
          for diag_key, baby in entries:
            clear = np.asarray(diags[diag_key])
            if giant:
              clear = np.roll(clear, giant)
            inner += clear * np.roll(x, -baby)
          factored += np.roll(inner, -giant)
        np.testing.assert_allclose(factored, dense, atol=1e-12, rtol=0)

  def test_exact_hoisted_groups_preserve_cyclic_wrap_boundaries(self):
    """Exact groups retain OpenFHE's separate zero-giant ModDowns."""
    n = 256  # Fully packed slot count for the matched ring dimension N=512.
    rng = np.random.default_rng(19)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)

    for inverse in (False, True):
      levels = bootstrapping._decompose_dft_into_levels(
          n, 4 * n, 2, inverse=inverse
      )
      exact_levels = bootstrapping._decompose_dft_into_hoisted_groups(
          n, 4 * n, 2, inverse=inverse
      )
      self.assertLen(exact_levels, len(levels))

      # C2S wraps in its first level and S2C in its last.  The ordinary
      # diagonal map collapses these equal total rotations, but OpenFHE keeps
      # the two outer groups separate because each c0 component is ModDown'd.
      wrap_level = 0 if not inverse else len(levels) - 1
      wrap_groups = exact_levels[wrap_level]
      self.assertEqual([group['giant'] for group in wrap_groups], [0, 0])
      self.assertEqual(
          [len(group['entries']) for group in wrap_groups], [16, 15]
      )
      self.assertLen(
          bootstrapping._collapsed_hoisted_plan(levels[wrap_level], n), 1
      )

      # Group preservation changes only the QP/ModDown schedule, not the clear
      # transform.  Reconstruct every level from its exact baby/giant groups.
      for diags, groups in zip(levels, exact_levels):
        dense = sum(
            np.asarray(diag) * np.roll(x, -int(k))
            for k, diag in diags.items()
        )
        factored = np.zeros(n, dtype=np.complex128)
        for group in groups:
          giant = int(group['giant'])
          inner = np.zeros(n, dtype=np.complex128)
          for baby, diagonal in group['entries']:
            clear = np.asarray(diagonal)
            if giant:
              clear = np.roll(clear, giant)
            inner += clear * np.roll(x, -int(baby))
          factored += np.roll(inner, -giant)
        np.testing.assert_allclose(factored, dense, atol=1e-12, rtol=0)

  def test_n256_single_pass_gate(self):
    """N=256 single-pass bootstrap must retain useful precision."""
    ctx, bs, sf0, r, c, ns = _build(256, nq=44, cd=2, level_budget=[1, 1])
    ct = _encrypt_depleted(ctx, bs, sf0, r, c, ns)
    out = bs.bootstrap(ct)
    # Matched OpenFHE schedule: direct S2C returns q12 at nsd=2 and the final
    # integer correction does not consume a level or alter NSD.
    self.assertEqual(out.num_moduli, 12)
    self.assertEqual(bs._get_nsd(out), 2)
    # Decode only the slots we actually check.  decrypt_decode's pure-Python
    # fallback is O(slots x degree), so decoding all `ns` slots makes the test
    # appear to hang at large N; we only assert on TEST_X (the first len(TEST_X)).
    mn, per_slot = _min_precision_bits(
        decrypt_decode(ctx, bs, out, len(TEST_X)))
    self._report("N=256 single-pass", per_slot, mn)
    self.assertGreaterEqual(mn, 12.0,
                            f"N=256 single-pass below quality floor: "
                            f"{mn:.1f} bits")

  def test_n256_meta_improves(self):
    """N=256 Meta-BTS must improve precision over single-pass."""
    ctx, bs, sf0, r, c, ns = _build(256, nq=44, cd=2, level_budget=[1, 1])
    ct = _encrypt_depleted(ctx, bs, sf0, r, c, ns)

    ct_a = copy.copy(ct); ct_a.polynomial = ct.polynomial.copy()
    bs._copy_scale(ct_a, ct); bs._copy_nsd(ct_a, ct)
    mn_single, _ = _min_precision_bits(
        decrypt_decode(ctx, bs, bs.bootstrap(ct_a), len(TEST_X)))

    ct_b = copy.copy(ct); ct_b.polynomial = ct.polynomial.copy()
    bs._copy_scale(ct_b, ct); bs._copy_nsd(ct_b, ct)
    mn_meta, per_slot = _min_precision_bits(
        decrypt_decode(ctx, bs, bs.meta_bootstrap(ct_b, precision=8),
                       len(TEST_X)))
    self._report("N=256 Meta-BTS", per_slot, mn_meta)
    print(f"    single-pass min = {mn_single:.1f} bits; "
          f"meta min = {mn_meta:.1f} bits")
    self.assertGreater(mn_meta, mn_single + 1.0,
                       "Meta-BTS did not improve precision")
    self.assertGreaterEqual(mn_meta, 15.0)

  @absltest.skipUnless(
      os.environ.get("BOOTSTRAP_TEST_N4096") == "1",
      "N=4096 runs two full bootstraps (~70 min); set BOOTSTRAP_TEST_N4096=1")
  def test_n4096_meta_gate(self):
    """N=4096 Meta-BTS (NQ=58 headroom) must clear the 15-bit gate."""
    ctx, bs, sf0, r, c, ns = _build(4096, nq=58, cd=2, level_budget=[4, 4])
    ct = _encrypt_depleted(ctx, bs, sf0, r, c, ns)
    out = bs.meta_bootstrap(ct, precision=9)
    mn, per_slot = _min_precision_bits(
        decrypt_decode(ctx, bs, out, len(TEST_X)))
    self._report("N=4096 Meta-BTS", per_slot, mn)
    self.assertGreaterEqual(mn, 15.0,
                            f"N=4096 Meta-BTS below gate: {mn:.1f} bits")

  @absltest.skipUnless(
      os.environ.get("BOOTSTRAP_TEST_N8192") == "1",
      "N=8192 runs a full bootstrap (~2 h, ~37 GB); set BOOTSTRAP_TEST_N8192=1")
  def test_n8192_single_pass_correctness(self):
    """N=8192 single-pass must produce CORRECT (non-garbage) output.

    This is a coarse correctness guard for the large collapsed-transform path,
    not an OpenFHE precision-parity assertion.
    """
    ctx, bs, sf0, r, c, ns = _build(8192, nq=58, cd=2, level_budget=[4, 4])
    ct = _encrypt_depleted(ctx, bs, sf0, r, c, ns)
    out = bs.bootstrap(ct)
    mn, per_slot = _min_precision_bits(
        decrypt_decode(ctx, bs, out, len(TEST_X)))
    self._report("N=8192 single-pass (correctness)", per_slot, mn)
    self.assertGreaterEqual(mn, 3.0,
                            f"N=8192 single-pass is garbage: {mn:.1f} bits")


if __name__ == "__main__":
  absltest.main()
