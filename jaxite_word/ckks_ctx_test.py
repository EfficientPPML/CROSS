import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import ckks_ctx
import hemul
import herot
import polynomial
import rescale
import util
import key_gen as kg
from absl.testing import absltest
from absl.testing import parameterized

# `FastEncryptCorrectness` uses `LoLAHE` from the demos directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "demos"))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

HEMul = hemul.HEMul
Polynomial = polynomial.Polynomial
HERot = herot.HERot
HERescale = rescale.HERescale

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

testing_params = [
  {
    'testcase_name': '0',
  }
]

@parameterized.named_parameters(testing_params)
class CKKSContextTest(parameterized.TestCase):
  def setUp(self):
    self.degree = 16
    self.num_slots = 8
    self.dnum = 3
    self.r, self.c = 4, 4

    self.scaling_factor = 563019763943521
    self.q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
    self.p_towers = [1073740609, 1073739937, 1073739649]
    self.q = 696985728458547852910430465530901300664961
    self.qt = 1329229981441028949792278227703286337
    self.p = 30
    self.CKKS_M_FACTOR = 1
    self.noise_scale_degree = 1
    self.max_bits_in_word = 61
    self.sigma = 3.190000057220458984375
    self.degree_layout = (self.r, self.c)

    key_pair = kg.gen_pke_pair(self.q_towers, self.p_towers, self.degree)
    self.params = {
        "degree": self.degree,
        "num_slots": self.num_slots,
        "scaling_factor": self.scaling_factor,
        "output_scale": self.scaling_factor,
        "q_towers": self.q_towers,
        "p_towers": self.p_towers,
        "p": self.p,
        "CKKS_M_FACTOR": self.CKKS_M_FACTOR,
        "max_bits_in_word": self.max_bits_in_word,
        "noise_scale_degree": self.noise_scale_degree,
        "public_key": key_pair["public_key"],
        "secret_key": key_pair["secret_key"]
    }
    self.real_values_input_in1 = [
        complex(0.25, 0), complex(0.5, 0), complex(0.75, 0), complex(1, 0),
        complex(2, 0), complex(3, 0), complex(4, 0), complex(5, 0),
    ]
    self.real_values_input_in2 = [
        complex(5, 0), complex(4, 0), complex(3, 0), complex(2, 0),
        complex(1, 0), complex(0.75, 0), complex(0.5, 0), complex(0.25, 0),
    ]
    self.real_values_multiply_result = [
        complex(1.25, 0), complex(2, 0), complex(2.25, 0), complex(2, 0),
        complex(2, 0), complex(2.25, 0), complex(2, 0), complex(1.25, 0),
    ]
    self.real_values_rotate_result = [
        complex(0.5, 0), complex(0.75, 0), complex(1, 0), complex(2, 0),
        complex(3, 0), complex(4, 0), complex(5, 0), complex(0.25, 0),
    ]

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encode_decode(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Decoding
    decoded_values = ctx.decode(encoded_ct, is_ntt=True)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_decrypt(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Decryption
    decrypted_ct = ctx.decrypt(encrypted_ct)
    # Step 4: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rotate_decrypt(self):
    # Paramters Setup
    rotate_idx = 1
    coef_map = util.precompute_auto_map(self.degree, kg.find_automorphism_index_2n_complex(rotate_idx, 2 * self.degree))
    # initialization
    herot_obj = HERot(self.r, self.c, self.dnum, self.q_towers, self.p_towers)
    ek_dict = kg.gen_rotation_key(self.params["secret_key"], self.q_towers, self.p_towers, rotate_idx, dnum=self.dnum, noise_std=self.sigma, noise_scale=self.noise_scale_degree)
    ek = ek_dict[rotate_idx]
    herot_obj.setup_rotate(jnp.array(ek["a"], jnp.uint64).transpose(0,2,1).reshape(self.dnum,*self.degree_layout,-1), jnp.array(ek["b"], jnp.uint64).transpose(0,2,1).reshape(self.dnum,*self.degree_layout,-1), coef_map)
    herot_obj.control_gen(batch=1, degree_layout=self.degree_layout)
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Rotate
    encrypted_ct.polynomial = encrypted_ct.polynomial.reshape(1, 2, *self.degree_layout, len(self.q_towers))
    result_ct = herot_obj.rotate(encrypted_ct)
    # Step 4: Decryption
    decrypted_ct = ctx.decrypt(result_ct)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_rotate_result, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rescale_decrypt(self):
    # Paramters Setup
    batch, num_elements, degree, num_moduli = 1, 2, 16, 5
    ct_shapes = {'batch': 1, 'num_elements': 2, 'degree': 16, 'num_moduli': 5, 'precision': 32, 'degree_layout': self.degree_layout}
    ct_params = {'moduli': self.q_towers, 'r': self.r, 'c': self.c}
    params = self.params.copy()
    params.update({
        "output_scale": (self.scaling_factor/self.q_towers[-1]),
    })
    # Initialization
    ctx = ckks_ctx.CKKSContext(params)
    he_rescale = HERescale(batch=batch, num_elements=num_elements, moduli=self.q_towers, r=self.r, c=self.c, degree_layout=self.degree_layout)
    he_rescale.control_gen()

    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Rescale
    in_data = encrypted_ct.polynomial.reshape(batch, num_elements, *self.degree_layout, num_moduli)
    rescaled_data = he_rescale.rescale(in_data)
    # Step 4: Decryption — wrap result in a Polynomial for decrypt
    ct_out = Polynomial(
        {'batch': batch, 'num_elements': num_elements, 'degree': degree,
         'num_moduli': num_moduli - 1, 'precision': 32, 'degree_layout': self.degree_layout},
        {'moduli': self.q_towers[:-1], 'r': self.r, 'c': self.c})
    ct_out.polynomial = rescaled_data.reshape(batch, num_elements, degree, num_moduli - 1)
    decrypted_ct = ctx.decrypt(ct_out)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_multiply_decrypt(self):
    """
    Test the encryption, multiplication, and decryption of the CKKS context.
    See hemul_test.py for the debugging version
    """
    # Paramters Setup
    r, c = 4, 4
    assert (r*c==self.degree)
    batch, num_elements, dnum, num_eval_mult = 1, 2, self.dnum, 1
    self.ek = kg.gen_evaluation_key(self.params["secret_key"], q=self.q_towers, P=self.p_towers, noise_std=self.sigma, noise_scale=1, dnum=3)
    eval_key_a, eval_key_b = jnp.array(self.ek["a"], dtype=jnp.uint32).transpose(0,2,1), jnp.array(self.ek["b"], dtype=jnp.uint32).transpose(0,2,1)
    params = self.params.copy()
    params.update({
        "evaluation_key": [eval_key_a, eval_key_b],
        "output_scale": (self.scaling_factor/self.q_towers[-1])**2,
    })
    # Initialization
    ctx = ckks_ctx.CKKSContext(params)
    he_mul = HEMul(batch, r, c, dnum, num_eval_mult, self.q_towers, self.p_towers)
    he_mul.control_gen(degree_layout=self.degree_layout)
    he_mul.setup_relinearization(eval_key_a, eval_key_b)
    # Step 1: Encoding
    encoded_ct1 = ctx.encode(self.real_values_input_in1)
    encoded_ct2 = ctx.encode(self.real_values_input_in2)
    # Step 2: Encryption
    encrypted_ct1 = ctx.encrypt(encoded_ct1)
    encrypted_ct2 = ctx.encrypt(encoded_ct2)
    # Step 3: Homomorphic Multiplication
    in_cts_array = jnp.concatenate([encrypted_ct1.polynomial, encrypted_ct2.polynomial], axis=1).reshape(batch, 2*num_elements, r, c, len(self.q_towers)).astype(jnp.uint32)
    ct_in_shapes = {'batch': batch, 'num_elements': 2*num_elements, 'degree': self.degree, 'precision': 32, 'num_moduli': len(self.q_towers), 'degree_layout': (r, c)}
    ct_in = Polynomial(ct_in_shapes, parameters={'moduli': self.q_towers})
    ct_in.polynomial = in_cts_array
    encrypted_result = he_mul.mul(ct_in)
    # Step 4: Decryption
    encrypted_ct1.drop_last_modulus()
    encrypted_ct1.set_batch_polynomial(encrypted_result.polynomial.reshape(batch, 2, self.degree, len(self.q_towers)-1))
    decrypted_result = ctx.decrypt(encrypted_ct1)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_result, is_ntt=False)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_multiply_result, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_composite_rescale(self):
    """
    Test that composite rescaling correctly calculates scale factors.

    This test verifies the mathematical correctness of composite scaling:
    - composite_degree=k groups k moduli into a single logical scale
    - composite_scale_factor = product of last k moduli
    - After rescale: effective_scale = original_scale / composite_scale_factor

    This follows the algorithm from ePrint 2023/1462:
    "High-precision RNS-CKKS on fixed but smaller word-size architectures"
    """
    # Test with different composite_degree values
    for composite_degree in [1, 2]:
        # Calculate expected composite scale
        expected_composite_scale = 1
        for i in range(composite_degree):
            expected_composite_scale *= self.q_towers[-(i + 1)]

        params = self.params.copy()
        params.update({
            "composite_degree": composite_degree,
            "output_scale": self.scaling_factor,
        })

        ctx = ckks_ctx.CKKSContext(params)

        # Verify composite scale factor
        self.assertEqual(
            ctx.composite_scale_factor,
            expected_composite_scale,
            f"Failed for composite_degree={composite_degree}"
        )

        # Calculate effective scale bits
        composite_bits = expected_composite_scale.bit_length()
        expected_bits_per_modulus = composite_bits / composite_degree

        # For 30-bit moduli, composite degree k should give ~k*30 bits
        self.assertGreater(
            expected_bits_per_modulus,
            15,  # Each modulus contributes at least 15 bits
            f"composite_degree={composite_degree} should use reasonable moduli"
        )


# ===========================================================================
# Bit-exact correctness gates for the vectorized fast encrypt/encode and
# decrypt/decode paths (formerly in `encrypt_fast_test.py` and
# `decrypt_fast_test.py`).
#
# Encrypt side — reference is `ckks_ctx.ckks_encrypt_fall_back` +
# `ckks_ctx.ckks_encode_fall_back` (pure-Python). We require:
#   * forward NTT (negacyclic, bit-reversed output) bit-equal to the reference
#     for random and edge inputs;
#   * `fast_encrypt_from_plaintext` bit-equal to `ckks_encrypt_fall_back` when
#     given the same `(v, e)`;
#   * full `fast_encode_encrypt` round-trip through `fast_decrypt_decode`
#     recovers the input slot vector within CKKS noise tolerance.
#
# Decrypt side — reference is `CKKSContext.decrypt` (pure-Python triple-loop +
# util.intt_negacyclic_bit_reverse) and `CKKSContext.decode`. We require all
# RNS coefficients to be **bit-equal**; decoded slot values must be within
# CKKS noise tolerance.
#
# Both classes use `LoLAHE.from_cache(...)` for a fully-initialized
# CKKSContext; they skip themselves if the cache file is absent.
# ===========================================================================
_LOLA_CACHE_PATH = os.path.join(_DEMO_DIR, "log", "lola_cache_7q3p.pkl")
_LOLA_MODEL = None
_LOLA_DATA = None


def _load_lola_model():
    global _LOLA_MODEL
    if _LOLA_MODEL is None:
        from lola_he import LoLAHE
        _LOLA_MODEL = LoLAHE.from_cache(_LOLA_CACHE_PATH)
        ckks_ctx.BYPASS_DECODE_STDDEV_CHECK = True
    return _LOLA_MODEL


def _load_lola_data():
    """Cached lazy-load of the LoLA evaluation set used by decrypt tests."""
    global _LOLA_DATA
    if _LOLA_DATA is None:
        from lola_he import load_or_generate_data
        _LOLA_DATA, _ = load_or_generate_data()
    return _LOLA_DATA


def _build_ct(ct_np: np.ndarray, q_towers, deg) -> Polynomial:
    """Wrap a numpy ciphertext into a Polynomial for the legacy decrypt path."""
    nq = ct_np.shape[-1]
    dc = Polynomial(
        {"batch": 1, "num_elements": 2, "degree": deg,
         "precision": 32, "num_moduli": nq, "degree_layout": (deg,)},
        {"moduli": q_towers[:nq]},
    )
    dc.polynomial = jnp.array(ct_np).reshape(1, 2, deg, nq)
    return dc


def _ref_ntt(coeffs_per_tower, q_towers, psi_pairs, N):
    ref = np.empty((N, len(q_towers)), dtype=np.uint64)
    for m, (qm, psi) in enumerate(zip(q_towers, psi_pairs)):
        nt = util.ntt_negacyclic_bit_reverse(coeffs_per_tower[m], int(qm), psi)
        rev = util.bit_reverse_array(nt)
        ref[:, m] = np.array(rev, dtype=np.uint64)
    return ref


class FastEncryptCorrectness(absltest.TestCase):
    """Bit-exact tests for the vectorized fast encrypt / encode path."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_LOLA_CACHE_PATH):
            raise absltest.SkipTest(
                f"LoLA cache missing at {_LOLA_CACHE_PATH}; run "
                f"`python3 demos/lola_he.py` to build it (~6 min) "
                f"before running FastEncryptCorrectness.")

    def test_forward_ntt_random(self):
        m = _load_lola_model()
        N = m.ctx.degree
        M = len(m.ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in m.ctx.q_towers]
        rng = np.random.default_rng(0xBEEF)
        coeffs = []
        for j in range(M):
            qj = int(m.ctx.q_towers[j])
            coeffs.append(
                [int(c) for c in rng.integers(0, qj, size=N, dtype=np.uint64)]
            )
        ref = _ref_ntt(coeffs, m.ctx.q_towers, psi_pairs, N)
        rns = np.zeros((N, M), dtype=np.uint64)
        for j in range(M):
            rns[:, j] = np.array(coeffs[j], dtype=np.uint64)
        cache = ckks_ctx._get_encrypt_cache(m.ctx, M)
        fast = ckks_ctx.vectorized_ntt(rns, cache)
        self.assertTrue(np.array_equal(ref, fast))

    def test_forward_ntt_edge(self):
        m = _load_lola_model()
        N = m.ctx.degree
        M = len(m.ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in m.ctx.q_towers]
        # Edge: zeros, ones, q-1, alternating.
        coeffs = []
        for j in range(M):
            qj = int(m.ctx.q_towers[j])
            row = ([0, 1, qj - 1, qj // 2] * (N // 4 + 1))[:N]
            coeffs.append(row)
        ref = _ref_ntt(coeffs, m.ctx.q_towers, psi_pairs, N)
        rns = np.zeros((N, M), dtype=np.uint64)
        for j in range(M):
            rns[:, j] = np.array(coeffs[j], dtype=np.uint64)
        cache = ckks_ctx._get_encrypt_cache(m.ctx, M)
        fast = ckks_ctx.vectorized_ntt(rns, cache)
        self.assertTrue(np.array_equal(ref, fast))

    def test_encrypt_bit_equal_with_seeded_v_e(self):
        m = _load_lola_model()
        from lola_he import NUM_SLOTS
        N = m.ctx.degree
        M = len(m.ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in m.ctx.q_towers]
        # Deterministic v ∈ {0, 1, ..} (small, mirrors sampling shape; the
        # bit-exact equivalence holds for arbitrary v including ternary).
        v_coeffs = [(i * 31 + 7) & 1 for i in range(N)]
        e0_coeffs = [(i * 13 - 5) % 11 - 5 for i in range(N)]
        e1_coeffs = [(i * 17 + 3) % 11 - 5 for i in range(N)]
        v_rns, e0_rns, e1_rns = [], [], []
        for j in range(M):
            qj = int(m.ctx.q_towers[j])
            psi = psi_pairs[j]
            v_rns.append(list(util.bit_reverse_array(
                util.ntt_negacyclic_bit_reverse(
                    [c % qj for c in v_coeffs], qj, psi))))
            e0_rns.append(list(util.bit_reverse_array(
                util.ntt_negacyclic_bit_reverse(
                    [c % qj for c in e0_coeffs], qj, psi))))
            e1_rns.append(list(util.bit_reverse_array(
                util.ntt_negacyclic_bit_reverse(
                    [c % qj for c in e1_coeffs], qj, psi))))
        slots = [complex(i * 0.1, 0.0) for i in range(NUM_SLOTS)]
        pt = m.ctx.encode(slots)
        # Use the pure-Python reference (`_fall_back`) for bit-exact baseline.
        # The default `ckks_encrypt` is now itself the fast path, so a
        # comparison against it would be tautological.
        leg = ckks_ctx.ckks_encrypt_fall_back(
            plaintext=pt.polynomial[0, 0].tolist(),
            public_key=[
                [list(m.ctx.public_key[k][j]) for j in range(M)]
                for k in range(2)
            ],
            q_towers=list(m.ctx.q_towers),
            v=v_rns, e=[e0_rns, e1_rns],
        )
        leg_arr = np.array(leg, dtype=np.uint64)               # (2, N, M)
        pt_eval = np.asarray(pt.polynomial[0, 0], dtype=np.uint64)
        fast = ckks_ctx.fast_encrypt_from_plaintext(
            pt_eval, m.ctx, v=v_rns, e=[e0_rns, e1_rns],
        )
        self.assertTrue(np.array_equal(leg_arr, fast))

    def test_round_trip_random_slots(self):
        m = _load_lola_model()
        from lola_he import NUM_SLOTS, SF
        rng = np.random.default_rng(0xDEAD)
        for trial in range(3):
            slots = np.zeros(NUM_SLOTS, dtype=complex)
            slots[: 200] = rng.standard_normal(200) * 5.0
            ct_np = ckks_ctx.fast_encode_encrypt(
                [complex(v) for v in slots], m.ctx, scale=SF
            )
            recovered = ckks_ctx.fast_decrypt_decode(ct_np, m.ctx, SF)
            err = float(np.max(np.abs(recovered[:200] - slots[:200].real)))
            self.assertLess(err, 1e-3,
                            f"trial={trial} round-trip max-err={err:.2e}")


class FastDecryptCorrectness(absltest.TestCase):
    """Bit-exact tests for the vectorized fast decrypt / decode path."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_LOLA_CACHE_PATH):
            raise absltest.SkipTest(
                f"LoLA cache missing at {_LOLA_CACHE_PATH}; run "
                f"`python3 demos/lola_he.py` to build it (~6 min) "
                f"before running FastDecryptCorrectness.")

    def test_random_ciphertexts_all_levels(self):
        m = _load_lola_model()
        deg = m.ctx.degree
        q_full = m.ctx.q_towers
        for nq in range(1, len(q_full) + 1):
            with self.subTest(nq=nq):
                rng = np.random.default_rng(1234 + nq)
                ct = np.zeros((2, deg, nq), dtype=np.uint64)
                for j in range(nq):
                    ct[0, :, j] = rng.integers(0, int(q_full[j]),
                                               size=deg, dtype=np.uint64)
                    ct[1, :, j] = rng.integers(0, int(q_full[j]),
                                               size=deg, dtype=np.uint64)
                leg = np.asarray(
                    m.ctx.decrypt(_build_ct(ct, q_full, deg)).polynomial[0, 0],
                    dtype=np.uint64)
                fast = ckks_ctx.fast_decrypt_to_rns_coeffs(ct, m.ctx)
                self.assertTrue(
                    np.array_equal(leg, fast),
                    f"RNS mismatch at nq={nq}: max|diff|="
                    f"{int(np.max(np.abs(leg.astype(np.int64) - fast.astype(np.int64))))}"
                )

    def test_edge_coefficients(self):
        m = _load_lola_model()
        deg = m.ctx.degree
        q_full = m.ctx.q_towers
        for nq in (1, min(4, len(m.ctx.secret_key)),
                   len(m.ctx.secret_key)):
            with self.subTest(nq=nq):
                ct = np.zeros((2, deg, nq), dtype=np.uint64)
                for j in range(nq):
                    qv = int(q_full[j])
                    ct[0, :, j] = np.array(
                        [0, qv - 1, qv >> 1, (qv >> 1) - 1, (qv >> 1) + 1] *
                        (deg // 5 + 1), dtype=np.uint64)[:deg]
                    ct[1, :, j] = np.array(
                        [qv - 1, 0, 1, qv - 2, qv >> 2] *
                        (deg // 5 + 1), dtype=np.uint64)[:deg]
                leg = np.asarray(
                    m.ctx.decrypt(_build_ct(ct, q_full, deg)).polynomial[0, 0],
                    dtype=np.uint64)
                fast = ckks_ctx.fast_decrypt_to_rns_coeffs(ct, m.ctx)
                self.assertTrue(np.array_equal(leg, fast))

    def test_real_lola_output(self):
        m = _load_lola_model()
        from lola_he import (prepare_weights, pack_lola_input,
                             FC1_OUT, FC1_IN, FC2_OUT, FC2_IN, SF, DEGREE)
        data = _load_lola_data()
        weights = prepare_weights(data)
        img = data["imgs"][0]
        ct = m.encrypt(pack_lola_input(img))
        lv = m.ctx.max_level
        ct = m.conv1_lola(ct, *weights[:2], lv); lv -= 1
        ct = m.he_mul(ct, lv - 1); lv -= 1
        ct = m.matmul_he(ct, weights[2].reshape(FC1_OUT, FC1_IN),
                         FC1_OUT, FC1_IN, lv); lv -= 1
        ct = m._add_encoded_plaintext(ct, m._fc1_bias_pt)
        ct = m.he_mul(ct, lv - 1); lv -= 1
        ct = m.matmul_he(ct, weights[4].reshape(FC2_OUT, FC2_IN),
                         FC2_OUT, FC2_IN, lv); lv -= 1
        ct = m._add_encoded_plaintext(ct, m._fc2_bias_pt)
        nq = ct.num_moduli
        ct_np = np.asarray(ct.polynomial.reshape(2, DEGREE, nq), dtype=np.uint64)
        m.ctx.output_scale = SF
        leg_rns = np.asarray(
            m.ctx.decrypt(_build_ct(ct_np, m.ctx.q_towers, DEGREE)).polynomial[0, 0],
            dtype=np.uint64)
        fast_rns = ckks_ctx.fast_decrypt_to_rns_coeffs(ct_np, m.ctx)
        self.assertTrue(np.array_equal(leg_rns, fast_rns))

        leg_slots = np.asarray(
            m.ctx.decode(
                m.ctx.decrypt(_build_ct(ct_np, m.ctx.q_towers, DEGREE)),
                is_ntt=False),
            dtype=complex).real
        fast_slots = ckks_ctx.fast_decrypt_decode(ct_np, m.ctx, SF)
        self.assertLess(
            float(np.max(np.abs(leg_slots[:FC2_OUT] - fast_slots[:FC2_OUT]))),
            1e-6, "decoded slot values exceed CKKS tolerance",
        )

    def test_batched_repeated_calls(self):
        """Cache should stay correct across many decrypt calls at varying nq."""
        m = _load_lola_model()
        deg = m.ctx.degree
        q_full = m.ctx.q_towers
        rng = np.random.default_rng(0xAA)
        for trial in range(8):
            nq = 1 + (trial % 4) * 2          # 1, 3, 5, 7
            ct = np.zeros((2, deg, nq), dtype=np.uint64)
            for j in range(nq):
                ct[0, :, j] = rng.integers(0, int(q_full[j]), size=deg,
                                            dtype=np.uint64)
                ct[1, :, j] = rng.integers(0, int(q_full[j]), size=deg,
                                            dtype=np.uint64)
            leg = np.asarray(
                m.ctx.decrypt(_build_ct(ct, q_full, deg)).polynomial[0, 0],
                dtype=np.uint64)
            fast = ckks_ctx.fast_decrypt_to_rns_coeffs(ct, m.ctx)
            self.assertTrue(np.array_equal(leg, fast),
                            f"trial={trial} nq={nq} batched-call mismatch")

    def test_zero_ciphertext(self):
        m = _load_lola_model()
        deg = m.ctx.degree
        for nq in (1, min(4, len(m.ctx.secret_key)),
                   len(m.ctx.secret_key)):
            ct = np.zeros((2, deg, nq), dtype=np.uint64)
            leg = np.asarray(
                m.ctx.decrypt(_build_ct(ct, m.ctx.q_towers, deg)).polynomial[0, 0],
                dtype=np.uint64)
            fast = ckks_ctx.fast_decrypt_to_rns_coeffs(ct, m.ctx)
            self.assertTrue(np.array_equal(leg, fast))
            self.assertTrue(np.all(fast == 0))


if __name__ == "__main__":
  absltest.main()
