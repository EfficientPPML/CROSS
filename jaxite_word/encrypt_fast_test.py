"""End-to-end correctness tests for `encrypt_fast`.

The reference is `ckks_ctx.CKKSContext.encode` + `ckks_ctx.ckks_encrypt`
(pure-Python). We require:

  * forward NTT (negacyclic, bit-reversed output) bit-equal to the reference
    for random and edge inputs;
  * `fast_encrypt_from_plaintext` bit-equal to `ckks_encrypt` when given the
    same `v` and `e`;
  * full `fast_encode_encrypt` round-trip through `decrypt_fast` recovers the
    input slot vector within CKKS noise tolerance.

Encrypt/decrypt are workload-independent. We build a bare `CKKSContext` +
keys here instead of loading the LoLA-MNIST cache pickle (which was only ever
a 5-10 minute setup shortcut for development).
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

from lola_he import NUM_SLOTS, SF, DEGREE, Q_TOWERS, P_TOWERS, SIGMA
import ckks_ctx as cc
import key_gen as kg
import encrypt_fast as ef
import decrypt_fast as df
import util


_MODEL = None


def _model():
    """Build a minimal CKKSContext + keys (no LoLA-specific state).

    Returned as a SimpleNamespace with `.ctx` so tests can keep using `m.ctx`.
    """
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


def _ref_ntt(coeffs_per_tower, q_towers, psi_pairs, N):
    ref = np.empty((N, len(q_towers)), dtype=np.uint64)
    for m, (qm, psi) in enumerate(zip(q_towers, psi_pairs)):
        nt = util.ntt_negacyclic_bit_reverse(coeffs_per_tower[m], int(qm), psi)
        rev = util.bit_reverse_array(nt)
        ref[:, m] = np.array(rev, dtype=np.uint64)
    return ref


class FastEncryptCorrectness(unittest.TestCase):

    def test_forward_ntt_random(self):
        m = _model()
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
        cache = ef._get_cache(m.ctx, M)
        fast = ef.vectorized_ntt(rns, cache)
        self.assertTrue(np.array_equal(ref, fast))

    def test_forward_ntt_edge(self):
        m = _model()
        N = m.ctx.degree
        M = len(m.ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in m.ctx.q_towers]
        # Edge: zeros, ones, q-1, alternating
        coeffs = []
        for j in range(M):
            qj = int(m.ctx.q_towers[j])
            row = ([0, 1, qj - 1, qj // 2] * (N // 4 + 1))[:N]
            coeffs.append(row)
        ref = _ref_ntt(coeffs, m.ctx.q_towers, psi_pairs, N)
        rns = np.zeros((N, M), dtype=np.uint64)
        for j in range(M):
            rns[:, j] = np.array(coeffs[j], dtype=np.uint64)
        cache = ef._get_cache(m.ctx, M)
        fast = ef.vectorized_ntt(rns, cache)
        self.assertTrue(np.array_equal(ref, fast))

    def test_encrypt_bit_equal_with_seeded_v_e(self):
        m = _model()
        N = m.ctx.degree
        M = len(m.ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in m.ctx.q_towers]
        # Deterministic v, e (small ints to mirror sampling shape)
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
        # Encode some slots once with the reference encoder (we only test
        # encrypt; encode is exercised by the round-trip test below).
        slots = [complex(i * 0.1, 0.0) for i in range(NUM_SLOTS)]
        pt = m.ctx.encode(slots)
        leg = cc.ckks_encrypt(
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
        fast = ef.fast_encrypt_from_plaintext(
            pt_eval, m.ctx, v=v_rns, e=[e0_rns, e1_rns],
        )
        self.assertTrue(np.array_equal(leg_arr, fast))

    def test_round_trip_random_slots(self):
        m = _model()
        rng = np.random.default_rng(0xDEAD)
        for trial in range(3):
            slots = np.zeros(NUM_SLOTS, dtype=complex)
            slots[: 200] = rng.standard_normal(200) * 5.0
            ct_np = ef.fast_encode_encrypt(
                [complex(v) for v in slots], m.ctx, scale=SF
            )
            recovered = df.fast_decrypt_decode(ct_np, m.ctx, SF)
            err = float(np.max(np.abs(recovered[:200] - slots[:200].real)))
            self.assertLess(err, 1e-3,
                            f"trial={trial} round-trip max-err={err:.2e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
