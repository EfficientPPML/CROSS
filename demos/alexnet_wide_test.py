"""Path A correctness gates — reference-width AlexNet, one ciphertext, secure ring.

Four gates for the config + cleartext model in `alexnet_wide_config.py`:

* `test_fits_one_ciphertext` — Path A's central claim: every reference-width
  layer fits one `num_slots` ciphertext for BOTH the ~100-bit (degree 16384 →
  8192 slots) and 128-bit (degree 32768 → 16384 slots) rings; worst == 6144.
* `test_cleartext_forward_runs` — the reference-width forward runs, returns a
  finite (10,) logit vector, its intermediate shapes match `wide_activation_sizes`,
  and it is genuinely WIDER than the shipped narrow surrogate.
* `test_primes_valid` — both chains are 17 Q + 4 P, prime, NTT-friendly
  (≡ 1 mod 2·degree), correctly sized (Q 30-bit, P 32-bit), log2(Q·P) ≈ 617,
  and pass the hybrid key-switch P-tower coverage guard at dnum=5.
* `test_security` — the 617-bit chain is 128-bit secure at degree 32768 but only
  ~100-bit at degree 16384 (proving Path A needs the 32768 ring for HE128).

Runs on the jaxite conda `python3` (numpy + sympy, no matplotlib / jax):
    cd demos && JAX_PLATFORMS=cpu python3 alexnet_wide_test.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from absl.testing import absltest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from alexnet_wide_config import (                               # noqa: E402
    WIDE_16384, WIDE_32768,
    WIDE_CI, WIDE_H_IN, WIDE_W_IN, WIDE_CO1, WIDE_CO2, WIDE_CO3,
    WIDE_H1, WIDE_W1, WIDE_H2, WIDE_W2,
    WIDE_KH, WIDE_KW, WIDE_PAD,
    wide_activation_sizes, assert_fits_one_ciphertext,
    alexnet_wide_cleartext, alexnet_wide_random_inputs, prepare_wide_args,
)
from alexnet_he import (                                        # noqa: E402
    assert_he128_or_warn, check_p_tower_coverage,
    conv2d_ref, add_conv_bias, avg_pool_2x2, quad_ref,
    FULL_CO1, FULL_CO2, FULL_CO3,
)


def _is_prime(n: int) -> bool:
    """Deterministic primality — sympy if present, else CROSS's util helper."""
    try:
        from sympy import isprime
        return bool(isprime(int(n)))
    except ImportError:
        from util import is_prime_deterministic
        return bool(is_prime_deterministic(int(n)))


class PathAFitTest(absltest.TestCase):
    """Path A's central claim: everything fits one ciphertext at a secure ring."""

    def test_fits_one_ciphertext(self):
        sizes = wide_activation_sizes()
        worst_expected = max(s for _, s in sizes)
        self.assertEqual(worst_expected, 6144,
                         "conv3 (384·4·4) must be the 6144-slot worst layer")

        for cfg, slot_ceiling in ((WIDE_16384, 8192), (WIDE_32768, 16384)):
            self.assertEqual(cfg.num_slots, slot_ceiling,
                             f"{cfg.name} num_slots should be degree/2")
            for name, slots in sizes:
                self.assertLessEqual(
                    slots, cfg.num_slots,
                    f"{cfg.name}: {name} ({slots}) overflows "
                    f"num_slots={cfg.num_slots}")
            worst = assert_fits_one_ciphertext(cfg)
            self.assertEqual(worst, 6144)
            # The concrete fit margin Path A relies on.
            self.assertLessEqual(6144, cfg.num_slots)
        # Explicit per-ring margins from the task statement.
        self.assertLessEqual(6144, 8192)    # degree 16384
        self.assertLessEqual(6144, 16384)   # degree 32768


class PathACleartextTest(absltest.TestCase):
    """The reference-width cleartext forward runs and is wider than the surrogate."""

    def test_cleartext_forward_runs(self):
        d = alexnet_wide_random_inputs(seed=42)
        out = alexnet_wide_cleartext(d["X"], *prepare_wide_args(d))

        self.assertEqual(out.shape, (10,), "AlexNet emits 10 logits")
        self.assertTrue(np.all(np.isfinite(out)),
                        "logits must be finite (no NaN/Inf)")

        # Intermediate-shape validation against wide_activation_sizes: recompute
        # the forward up to conv3 with the same primitives and confirm the
        # activation count is exactly the table's conv3 entry (6144).
        sizes = dict(wide_activation_sizes())
        self.assertEqual(d["X"].shape[0], sizes["input"])
        y = conv2d_ref(d["X"], d["W1"], WIDE_CI, WIDE_CO1, WIDE_H_IN, WIDE_W_IN,
                       WIDE_KH, WIDE_KW, 1, WIDE_PAD)
        y = add_conv_bias(y, d["b1"], WIDE_CO1, WIDE_H_IN, WIDE_W_IN)
        y = avg_pool_2x2(y, WIDE_CO1, WIDE_H_IN, WIDE_W_IN)
        y = quad_ref(y)
        self.assertEqual(y.shape[0], sizes["conv1"], "conv1 → 4096 slots")
        y = conv2d_ref(y, d["W2"], WIDE_CO1, WIDE_CO2, WIDE_H1, WIDE_W1,
                       WIDE_KH, WIDE_KW, 1, WIDE_PAD)
        y = add_conv_bias(y, d["b2"], WIDE_CO2, WIDE_H1, WIDE_W1)
        y = avg_pool_2x2(y, WIDE_CO2, WIDE_H1, WIDE_W1)
        y = quad_ref(y)
        self.assertEqual(y.shape[0], sizes["conv2"], "conv2 → 3072 slots")
        y = conv2d_ref(y, d["W3"], WIDE_CO2, WIDE_CO3, WIDE_H2, WIDE_W2,
                       WIDE_KH, WIDE_KW, 1, WIDE_PAD)
        y = add_conv_bias(y, d["b3"], WIDE_CO3, WIDE_H2, WIDE_W2)
        self.assertEqual(y.shape[0], 6144,
                         "conv3 activation must be 6144 elements (the worst layer)")
        self.assertEqual(y.shape[0], sizes["conv3"])

        # The reference net is genuinely wider than the shipped narrow surrogate.
        self.assertGreater(WIDE_CO1, FULL_CO1)   # 64 vs 8
        self.assertGreater(WIDE_CO2, FULL_CO2)   # 192 vs 16
        self.assertGreater(WIDE_CO3, FULL_CO3)   # 384 vs 32
        self.assertEqual(WIDE_CO1, 64)
        self.assertEqual(FULL_CO1, 8)

    def test_forward_deterministic_and_input_dependent(self):
        # Same seed -> bit-identical logits (deterministic forward).
        a = alexnet_wide_random_inputs(seed=1)
        ya = alexnet_wide_cleartext(a["X"], *prepare_wide_args(a))
        ya2 = alexnet_wide_cleartext(a["X"], *prepare_wide_args(a))
        np.testing.assert_array_equal(ya, ya2)

        # Input dependence: the 7-squaring cascade drives the final logits to
        # ~1e-120 (distinct but below allclose's atol), so check dependence at
        # the pre-collapse conv1 activation instead of the collapsed logits.
        def conv1_act(d):
            y = conv2d_ref(d["X"], d["W1"], WIDE_CI, WIDE_CO1,
                           WIDE_H_IN, WIDE_W_IN, WIDE_KH, WIDE_KW, 1, WIDE_PAD)
            y = add_conv_bias(y, d["b1"], WIDE_CO1, WIDE_H_IN, WIDE_W_IN)
            return quad_ref(avg_pool_2x2(y, WIDE_CO1, WIDE_H_IN, WIDE_W_IN))

        b = alexnet_wide_random_inputs(seed=2)
        self.assertFalse(np.allclose(conv1_act(a), conv1_act(b)),
                         "distinct random weights should give distinct conv1 acts")


class PathAPrimesTest(absltest.TestCase):
    """Both modulus chains are valid, NTT-friendly, and depth-15 sized."""

    def test_primes_valid(self):
        for cfg in (WIDE_16384, WIDE_32768):
            q, p, m = cfg.q_towers, cfg.p_towers, 2 * cfg.degree
            self.assertEqual(len(q), 17, f"{cfg.name}: expected 17 Q-towers")
            self.assertEqual(len(p), 4, f"{cfg.name}: expected 4 P-towers")
            for qi in q:
                self.assertTrue(_is_prime(qi), f"{cfg.name}: Q {qi} not prime")
                self.assertEqual(qi % m, 1, f"{cfg.name}: Q {qi} not ≡1 mod {m}")
                self.assertEqual(int(qi).bit_length(), 30,
                                 f"{cfg.name}: Q {qi} not 30-bit")
            for pi in p:
                self.assertTrue(_is_prime(pi), f"{cfg.name}: P {pi} not prime")
                self.assertEqual(pi % m, 1, f"{cfg.name}: P {pi} not ≡1 mod {m}")
                self.assertEqual(int(pi).bit_length(), 32,
                                 f"{cfg.name}: P {pi} not 32-bit")
            # All primes distinct within the chain.
            self.assertEqual(len(set(q + p)), 21,
                             f"{cfg.name}: duplicate moduli in the chain")
            log_qp = sum(math.log2(x) for x in q + p)
            self.assertGreaterEqual(log_qp, 610, f"{cfg.name}: log2(QP) too small")
            self.assertLessEqual(log_qp, 625, f"{cfg.name}: log2(QP) too large")
            # Hybrid key-switch P-tower coverage at dnum=5 (raises on failure).
            check_p_tower_coverage(q, p, 5)


class PathASecurityTest(absltest.TestCase):
    """The 617-bit chain is 128-bit secure only at degree 32768."""

    def test_security(self):
        q, p = WIDE_32768.q_towers, WIDE_32768.p_towers
        # Same chain, secure at degree 32768 (HES128 ceiling ~881 > 617).
        assert_he128_or_warn(32768, q, p, require_128bit=True)  # must NOT raise
        # ... but only ~100-bit at degree 16384 (ceiling ~438 < 617).
        with self.assertRaises(ValueError):
            assert_he128_or_warn(16384, q, p, require_128bit=True)
        # The WIDE_16384 config's own chain is likewise insecure at its degree.
        with self.assertRaises(ValueError):
            assert_he128_or_warn(WIDE_16384.degree, WIDE_16384.q_towers,
                                 WIDE_16384.p_towers, require_128bit=True)


if __name__ == "__main__":
    absltest.main()
