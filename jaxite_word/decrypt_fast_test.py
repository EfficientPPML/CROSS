"""End-to-end correctness tests for `decrypt_fast`.

The reference is `ckks_ctx.CKKSContext.decrypt` (pure-Python triple-loop +
util.intt_negacyclic_bit_reverse) and `ckks_ctx.CKKSContext.decode`
(_crt_combine_rns_plaintext + ckks_decode). All RNS coefficients are required
to be **bit-equal**; decoded slot values are within CKKS noise tolerance.

Cases covered:
  - random ciphertext at every reachable level (1..max_q moduli)
  - edge ciphertexts (zeros, q-1, near q/2)
  - real LoLA output ciphertext at L3 (4 moduli) — only when the LoLA-MNIST
    cache pickle is present; auto-skipped otherwise
  - batched calls with the same cache (decrypts called repeatedly across
    many ciphertexts to confirm cache reuse stays correct)

Decrypt is workload-independent. The fast tests build a bare CKKSContext +
keys directly; only `test_real_lola_output` requires the LoLA cache and is
gated with skipIf.
"""
from __future__ import annotations
import os, sys, types, unittest
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "demos"))
for p in (THIS_DIR, DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lola_he import (
    SF, DEGREE, Q_TOWERS, P_TOWERS, SIGMA, NUM_SLOTS,
    FC1_OUT, FC1_IN, FC2_OUT, FC2_IN,
)
import ckks_ctx as cc
import key_gen as kg
from polynomial import Polynomial
import decrypt_fast as df


_CACHE_PATH = os.path.join(DEMO_DIR, "log", "lola_cache_7q3p.pkl")
_HAS_LOLA_CACHE = os.path.exists(_CACHE_PATH)
_MODEL = None


def _model():
    """Build a minimal CKKSContext + keys (no LoLA-specific state)."""
    global _MODEL
    if _MODEL is None:
        kp = kg.gen_pke_pair(list(Q_TOWERS), list(P_TOWERS), DEGREE,
                             noise_std=SIGMA)
        ctx = cc.CKKSContext({
            "degree": DEGREE,
            "num_slots": NUM_SLOTS,
            "scaling_factor": float(SF),
            "output_scale": float(SF),
            "q_towers": list(Q_TOWERS),
            "p_towers": list(P_TOWERS),
            "p": 60,
            "CKKS_M_FACTOR": 1,
            "max_bits_in_word": 61,
            "noise_scale_degree": 1,
            "composite_degree": 1,
            "public_key": kp["public_key"],
            "secret_key": kp["secret_key"],
        })
        cc.BYPASS_DECODE_STDDEV_CHECK = True
        _MODEL = types.SimpleNamespace(ctx=ctx)
    return _MODEL


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


class FastDecryptCorrectness(unittest.TestCase):

    def test_random_ciphertexts_all_levels(self):
        m = _model()
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
                fast = df.fast_decrypt_to_rns_coeffs(ct, m.ctx)
                self.assertTrue(
                    np.array_equal(leg, fast),
                    f"RNS mismatch at nq={nq}: max|diff|="
                    f"{int(np.max(np.abs(leg.astype(np.int64) - fast.astype(np.int64))))}"
                )

    def test_edge_coefficients(self):
        m = _model()
        deg = m.ctx.degree
        q_full = m.ctx.q_towers
        # Range across all available towers (clipped at 9 historically — now
        # uses len(secret_key) so the test works at 7Q+3P or any other size).
        for nq in (1, min(4, len(_model().ctx.secret_key)),
                   len(_model().ctx.secret_key)):
            with self.subTest(nq=nq):
                ct = np.zeros((2, deg, nq), dtype=np.uint64)
                # First column: 0..q-1 sweep, every position
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
                fast = df.fast_decrypt_to_rns_coeffs(ct, m.ctx)
                self.assertTrue(np.array_equal(leg, fast))

    @unittest.skipIf(not _HAS_LOLA_CACHE,
                     f"LoLA cache pickle not present at {_CACHE_PATH}; "
                     "this test exercises a full LoLA forward pass and "
                     "needs the cached keys + plaintexts.")
    def test_real_lola_output(self):
        # Lazy LoLA imports — only needed for this test.
        from lola_he import (
            LoLAHE, load_or_generate_data, prepare_weights, pack_lola_input,
        )
        m = LoLAHE.from_cache(_CACHE_PATH)
        cc.BYPASS_DECODE_STDDEV_CHECK = True
        data, _ = load_or_generate_data()
        weights = prepare_weights(data)
        img = data["imgs"][0]
        # full inference up to logits ciphertext (L3)
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
        # legacy
        m.ctx.output_scale = SF
        leg_rns = np.asarray(
            m.ctx.decrypt(_build_ct(ct_np, m.ctx.q_towers, DEGREE)).polynomial[0, 0],
            dtype=np.uint64)
        fast_rns = df.fast_decrypt_to_rns_coeffs(ct_np, m.ctx)
        self.assertTrue(np.array_equal(leg_rns, fast_rns))

        leg_slots = np.asarray(
            m.ctx.decode(
                m.ctx.decrypt(_build_ct(ct_np, m.ctx.q_towers, DEGREE)),
                is_ntt=False),
            dtype=complex).real
        fast_slots = df.fast_decrypt_decode(ct_np, m.ctx, SF)
        self.assertLess(
            float(np.max(np.abs(leg_slots[:FC2_OUT] - fast_slots[:FC2_OUT]))),
            1e-6, "decoded slot values exceed CKKS tolerance",
        )

    def test_batched_repeated_calls(self):
        """Cache should stay correct across many decrypt calls at varying nq."""
        m = _model()
        deg = m.ctx.degree
        q_full = m.ctx.q_towers
        rng = np.random.default_rng(0xAA)
        # Mix nq values to exercise the (id(ctx), num_moduli) cache.
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
            fast = df.fast_decrypt_to_rns_coeffs(ct, m.ctx)
            self.assertTrue(np.array_equal(leg, fast),
                            f"trial={trial} nq={nq} batched-call mismatch")

    def test_zero_ciphertext(self):
        m = _model()
        deg = m.ctx.degree
        # Range across all available towers (clipped at 9 historically — now
        # uses len(secret_key) so the test works at 7Q+3P or any other size).
        for nq in (1, min(4, len(_model().ctx.secret_key)),
                   len(_model().ctx.secret_key)):
            ct = np.zeros((2, deg, nq), dtype=np.uint64)
            leg = np.asarray(
                m.ctx.decrypt(_build_ct(ct, m.ctx.q_towers, deg)).polynomial[0, 0],
                dtype=np.uint64)
            fast = df.fast_decrypt_to_rns_coeffs(ct, m.ctx)
            self.assertTrue(np.array_equal(leg, fast))
            self.assertTrue(np.all(fast == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
