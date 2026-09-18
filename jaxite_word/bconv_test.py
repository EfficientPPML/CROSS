from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
import bconv
import numpy as np
from util import random_batched_ciphertext
import finite_field as ff_context

# Use 64-bit precision as in bconv.py
jax.config.update("jax_enable_x64", True)

TEST_PARAMS = [
    (
        "L2_to_L5",
        [[180089039, 904401266, 277587483, 381410246, 867235356, 971323117, 934942938, 338146069, 129667711, 97559399, 337422188, 364870460, 916966745, 312366062, 762079964, 605485434], [540094309, 1034680811, 1057648335, 677992674, 650354195, 558219774, 502221165, 503532224, 1049911792, 146837876, 560962740, 820076664, 58915608, 1034452760, 724437159, 68291682]],
        [1073741441, 1073740609],
        [268437409, 268436801, 268435361, 268435649, 524353],
        [249077041, 824663761],
        [[268428382, 268430206, 268434526, 268433662, 390018], [268429214, 268431038, 268435358, 268434494, 390850]],
        [[127196115, 177098281, 103398386, 262465714, 225857559, 213539642, 56845406, 173328911, 21637023, 13036123, 259867486, 247888119, 190104469, 18415021, 107052173, 152967426], [168457304, 81199027, 93565169, 186078678, 45587255, 266135885, 57716353, 256503901, 42940759, 230451532, 167299604, 7499360, 30178241, 217184571, 253380763, 263628678], [36832759, 236716332, 248966405, 220314486, 69166987, 6190101, 204761211, 194300152, 254116210, 106417675, 161103783, 244499620, 193481707, 80047389, 247286681, 101528753], [57170724, 202300871, 265775502, 100961596, 165644129, 105852026, 62793519, 256209638, 261792224, 19441022, 102172131, 193804848, 207565270, 260633034, 136301593, 126040258], [7075121, 3849323, 6595938, 5951409, 7277876, 7164094, 6139709, 4952812, 4506557, 5030408, 7544214, 3494637, 8199168, 9397516, 5558747, 9127873]],#, [258532, 178852, 303702, 183526, 461287, 347505, 371826, 233635, 311733, 311231, 203272, 348519, 333873, 483515, 315217, 213872]],
    ),
]

class BConvContextTest(parameterized.TestCase):
    # @absltest.skip("Skip a single test")
    @parameterized.named_parameters(*TEST_PARAMS)
    def test_barrett_context(self, partCtCloneCoef, original_moduli, target_moduli, QHatInvModq, QHatModp, reference_result):
        """
        Verifies that basis_change works with BarrettContext
        """
        key = jax.random.PRNGKey(0)
        in_tower = jax.numpy.array(partCtCloneCoef, dtype=jnp.uint64).T
        reference_result = jax.numpy.array(reference_result, dtype=jnp.uint64).T

        # New API setup
        overall_moduli = original_moduli + target_moduli
        original_index = list(range(len(original_moduli)))
        target_index = list(range(len(original_moduli), len(overall_moduli)))

        _bconv = bconv.BConvBarrett(overall_moduli)
        _bconv.control_gen([(original_index, target_index)])
        
        in_formatted = _bconv.ff_ctx_origin[0].to_computation_format(in_tower)
        out_formatted = _bconv.basis_change(in_formatted)
        out = _bconv.ff_ctx_target[0].to_original_format(out_formatted)

        # The historical golden stores the dense path's lazy representative
        # for small target limbs. Automatic routing may return the canonical
        # BAT representative, so compare the mathematical residues.
        target_moduli_arr = jnp.array(target_moduli, dtype=jnp.uint64)
        np.testing.assert_array_equal(
            reference_result % target_moduli_arr,
            jnp.asarray(out, dtype=jnp.uint64) % target_moduli_arr,
        )


    # @absltest.skip("Skip a single test")
    @parameterized.named_parameters(*TEST_PARAMS)
    def test_bat_lazy_context(self, partCtCloneCoef, original_moduli, target_moduli, QHatInvModq, QHatModp, reference_result):
        """
        Verifies that basis_change works with BATLazyContext
        """
        key = jax.random.PRNGKey(0)
        in_tower = jax.numpy.array(partCtCloneCoef, dtype=jnp.uint64).T
        reference_result = jax.numpy.array(reference_result, dtype=jnp.uint64).T

        overall_moduli = original_moduli + target_moduli
        original_index = list(range(len(original_moduli)))
        target_index = list(range(len(original_moduli), len(overall_moduli)))

        _bconv = bconv.BConvBATLazy(overall_moduli)
        _bconv.control_gen([(original_index, target_index)])
        
        in_formatted = _bconv.ff_ctx_origin[0].to_computation_format(in_tower)
        out_formatted = _bconv.basis_change(in_formatted)
        out = _bconv.ff_ctx_target[0].to_original_format(out_formatted)
        
        # BATLazy produces result congruent mod p, but not necessarily fully reduced.
        # Hence we check the post modular reduction results.
        target_moduli_arr = jnp.array(target_moduli, dtype=jnp.uint64)
        diff = (out.astype(jnp.int64) - reference_result.astype(jnp.int64)) % target_moduli_arr.astype(jnp.int64)
        np.testing.assert_array_equal(diff, jnp.zeros_like(diff))

    # @absltest.skip("Skip a single test")
    @parameterized.named_parameters(*TEST_PARAMS)
    def test_montgomery_context(self, partCtCloneCoef, original_moduli, target_moduli, QHatInvModq, QHatModp, reference_result):
        """
        Verifies that CRNS-based basis_change with MontgomeryContext matches
        the golden reference exactly (input/output in Montgomery format).
        """
        in_tower = jax.numpy.array(partCtCloneCoef, dtype=jnp.uint64).T
        reference_result = jax.numpy.array(reference_result, dtype=jnp.uint64).T

        overall_moduli = original_moduli + target_moduli
        original_index = list(range(len(original_moduli)))
        target_index = list(range(len(original_moduli), len(overall_moduli)))

        _bconv = bconv.BConvMontgomery(overall_moduli)
        _bconv.control_gen([(original_index, target_index)])

        in_formatted = _bconv.ff_ctx_origin[0].to_computation_format(in_tower)
        out_formatted = _bconv.basis_change(in_formatted)
        out = _bconv.ff_ctx_target[0].to_original_format(out_formatted)

        # Montgomery's to_original_format is strictly reduced, while the
        # golden reference stores BConvBarrett's lazy representative (its
        # Barrett reduction runs past the strict range for small p_j, e.g.
        # reference 7075121 = 258532 + 13*524353). Compare canonical residues.
        target_moduli_arr = jnp.array(target_moduli, dtype=jnp.uint64)
        np.testing.assert_array_equal(out, reference_result % target_moduli_arr)
        # And the Montgomery output itself must be canonical.
        self.assertTrue(bool(jnp.all(out < target_moduli_arr)))

    def test_multiple_control_gen(self):
        """
        Verifies that BConv supports multiple control generations.
        """
        # Define a simple setup
        # overall_moduli = [q0, q1, p0, p1]
        # Config 0: [q0] -> [p0]
        # Config 1: [q1] -> [p1]
        
        # Using small primes for easy verification
        q0, q1 = 17, 19
        p0, p1 = 23, 29
        overall_moduli = [q0, q1, p0, p1]
        
        # Config 0
        original_index_0 = [0]
        target_index_0 = [2]
        
        # Config 1
        original_index_1 = [1]
        target_index_1 = [3]
        
        _bconv = bconv.BConvBarrett(overall_moduli)
        _bconv.control_gen([
            (original_index_0, target_index_0),
            (original_index_1, target_index_1)
        ])
        
        # Test Config 0
        val = 15
        in_tower_0 = jnp.array([[val]], dtype=jnp.uint64) # shape (1, 1) to match (d, q) ? sizeQ=1
        in_tower = jnp.array([[[15]]], dtype=jnp.uint64) # (1, 1, 1)
        out_0 = _bconv.basis_change(in_tower, control_index=0)
        self.assertEqual(out_0[0,0,0], 15)
        
        in_tower_1 = jnp.array([[[20]]], dtype=jnp.uint64)
        out_1 = _bconv.basis_change(in_tower_1, control_index=1)
        self.assertEqual(out_1[0,0,0], 1)
        
        # Quick check for non-interference
        in_tower_0_b = jnp.array([[[20]]], dtype=jnp.uint64)
        out_0_b = _bconv.basis_change(in_tower_0_b, control_index=0) # 20 mod 17 -> 3 -> 3
        self.assertEqual(out_0_b[0,0,0], 3)

class BConvMontgomeryVsBarrettTest(parameterized.TestCase):
    """Goal test: BConvMontgomery (CRNS) == BConvBarrett.

    Same input values and security parameters (degree, moduli); each path
    converts to its own computation format, runs basis_change, and converts
    back. The canonical residues must be identical, including the alpha*Q
    overflow term of the approximate basis extension (a different alpha would
    shift the result by a multiple of Q modulo every p_j).
    """

    @parameterized.named_parameters(
        (
            "d8_q27q30x5_to_p22p30x4",
            8,
            [134219681, 134218433, 134219009, 1073741857, 1073740609],
            [268435361, 268435009, 6710893, 1067031829],
        ),
        (
            "d128_q30x2_to_p28x5",
            128,
            [1073741441, 1073740609],
            [268437409, 268436801, 268435361, 268435649, 524353],
        ),
        (
            "d2048_q30x5_to_p28x5",
            2048,
            [1073741441, 1073740609, 134219681, 134218433, 134219009],
            [268437409, 268436801, 268435361, 268435649, 524353],
        ),
    )
    def test_montgomery_matches_barrett(self, d, original_moduli, target_moduli):
        overall_moduli = original_moduli + target_moduli
        original_index = list(range(len(original_moduli)))
        target_index = list(range(len(original_moduli), len(overall_moduli)))

        batch, elements = 2, 2
        sizeQ = len(original_moduli)
        # uint64: MontgomeryContext.to_computation_format shifts by 32 bits.
        in_tower = random_batched_ciphertext(
            (batch, elements, d, sizeQ), original_moduli, jnp.uint32
        ).astype(jnp.uint64)

        _barrett = bconv.BConvBarrett(overall_moduli)
        _barrett.control_gen([(original_index, target_index)])
        _mont = bconv.BConvMontgomery(overall_moduli)
        _mont.control_gen([(original_index, target_index)])

        ref = _barrett.ff_ctx_target[0].to_original_format(
            _barrett.basis_change(
                _barrett.ff_ctx_origin[0].to_computation_format(in_tower)
            )
        )
        in_mont = _mont.ff_ctx_origin[0].to_computation_format(in_tower)
        out = _mont.ff_ctx_target[0].to_original_format(
            _mont._basis_change_dense(in_mont)
        )

        # Both paths compute the identical approximate extension
        # S = sum_i y_i * QHat_i (same digits, same alpha*Q term), but
        # BConvBarrett's output representative can exceed p_j (its Barrett
        # reduction is lazy past 2^(2*ceil(log2 p_j))), so equality is
        # asserted on canonical residues.
        target_moduli_arr = jnp.array(target_moduli, dtype=jnp.uint64)
        ref_canonical = jnp.asarray(ref, jnp.uint64) % target_moduli_arr
        out_canonical = jnp.asarray(out, jnp.uint64) % target_moduli_arr
        np.testing.assert_array_equal(ref_canonical, out_canonical)
        # The Montgomery output must already be canonical (strictly reduced).
        np.testing.assert_array_equal(out_canonical, jnp.asarray(out, jnp.uint64))

        # The MXU (BAT) path computes the same math with a different (lazy)
        # representative; canonical residues must be identical too.
        out_bat = _mont.ff_ctx_target[0].to_original_format(
            _mont.basis_change_bat(in_mont)
        )
        np.testing.assert_array_equal(
            ref_canonical, jnp.asarray(out_bat, jnp.uint64) % target_moduli_arr
        )

    def test_montgomery_multiple_control_gen(self):
        """BConvMontgomery supports multiple control configurations."""
        q0, q1 = 17, 19
        p0, p1 = 23, 29
        overall_moduli = [q0, q1, p0, p1]

        _bconv = bconv.BConvMontgomery(overall_moduli)
        _bconv.control_gen([
            ([0], [2]),
            ([1], [3]),
        ])

        def run(value, control_index):
            in_tower = jnp.array([[[value]]], dtype=jnp.uint64)
            in_formatted = _bconv.ff_ctx_origin[control_index].to_computation_format(in_tower)
            out_formatted = _bconv.basis_change(in_formatted, control_index=control_index)
            return _bconv.ff_ctx_target[control_index].to_original_format(out_formatted)

        # sizeQ=1: digit is value mod q, extended to the target modulus.
        self.assertEqual(run(15, 0)[0, 0, 0], 15)   # 15 mod 17 -> 15 mod 23
        self.assertEqual(run(20, 1)[0, 0, 0], 1)    # 20 mod 19 -> 1 mod 29
        self.assertEqual(run(20, 0)[0, 0, 0], 3)    # 20 mod 17 -> 3 mod 23

    def test_montgomery_bat_tiny_primes(self):
        """Explicit BAT and dense paths match on tiny (w < 32) moduli."""
        _bconv = bconv.BConvMontgomery([17, 19, 23, 29])
        _bconv.control_gen([([0, 1], [2, 3])])
        xs = jnp.array([[[a, b] for a in range(17) for b in range(19)]],
                       dtype=jnp.uint64)
        xf = _bconv.ff_ctx_origin[0].to_computation_format(xs)
        dense = _bconv.ff_ctx_target[0].to_original_format(
            _bconv._basis_change_dense(xf))
        bat = _bconv.ff_ctx_target[0].to_original_format(
            _bconv.basis_change_bat(xf))
        np.testing.assert_array_equal(np.asarray(dense), np.asarray(bat))


class BConvMontgomeryDeepTest(absltest.TestCase):
    """Regression anchors for BConvMontgomery vs. exact-arithmetic ground truth.

    Ground truth is computed with exact Python big-int arithmetic:
    y_i = x_i * QHatInv_i mod q_i, S = sum_i y_i * QHat_i, out_j = S mod p_j
    (the approximate extension, alpha*Q term included).
    """

    def _ground_truth(self, om, tm, xs):
        import math as _math
        from util import modinv as _modinv
        Q = _math.prod(om)
        qhat = [Q // q for q in om]
        qhatinv = [_modinv(h, q) for h, q in zip(qhat, om)]
        out = []
        for x in xs:
            S = sum(((xi * hi) % qi) * qh
                    for xi, hi, qi, qh in zip(x, qhatinv, om, qhat))
            out.append([S % p for p in tm])
        return out

    def _run_montgomery(self, om, tm, xs, method="_basis_change_dense"):
        _bconv = bconv.BConvMontgomery(om + tm)
        _bconv.control_gen([(list(range(len(om))),
                             list(range(len(om), len(om) + len(tm))))])
        x = jnp.array(xs, dtype=jnp.uint64)[None, ...]  # (1, n, sizeQ)
        xf = _bconv.ff_ctx_origin[0].to_computation_format(x)
        out = _bconv.ff_ctx_target[0].to_original_format(
            getattr(_bconv, method)(xf))
        return np.asarray(out)[0]

    def test_exhaustive_tiny_primes(self):
        """All 323 inputs of q={17,19} -> p={23,29} vs big-int ground truth."""
        om, tm = [17, 19], [23, 29]
        xs = [[a, b] for a in range(17) for b in range(19)]
        expected = self._ground_truth(om, tm, xs)
        got = self._run_montgomery(om, tm, xs)
        np.testing.assert_array_equal(np.array(expected, dtype=np.uint64), got)
        got_bat = self._run_montgomery(om, tm, xs, method="basis_change_bat")
        np.testing.assert_array_equal(
            np.array(expected, dtype=np.uint64), got_bat)

    def test_wrap_boundary_max_digits(self):
        """Near-2^30 primes at high sizeQ with worst-case (maximal) digits.

        Digits are steered directly: x_i = (y_i * QHat_i) mod q_i gives digit
        y_i, so y_i = q_i - 1 drives the accumulator toward its maximum
        (also maximizing the alpha overflow term, alpha = sizeQ - 1).
        """
        import math as _math
        from util import modinv as _modinv, find_moduli_ntt
        moduli = find_moduli_ntt(12, 30, 1024)  # 12 largest 30-bit NTT primes
        om, tm = moduli[:10], moduli[10:]
        # must sit inside the wrap-safe assert
        Q = _math.prod(om)
        qhat = [Q // q for q in om]
        qhatinv = [_modinv(h, q) for h, q in zip(qhat, om)]
        # x giving y_i = q_i - 1: invert the digit map y = x * qhatinv mod q
        x_max = [((q - 1) * _modinv(hi, q)) % q for q, hi in zip(om, qhatinv)]
        xs = [x_max, [0] * len(om), [q - 1 for q in om]]
        expected = self._ground_truth(om, tm, xs)
        got = self._run_montgomery(om, tm, xs)
        np.testing.assert_array_equal(np.array(expected, dtype=np.uint64), got)
        got_bat = self._run_montgomery(om, tm, xs, method="basis_change_bat")
        np.testing.assert_array_equal(
            np.array(expected, dtype=np.uint64), got_bat)

    def test_differential_random_28bit(self):
        """Random 28-bit NTT-prime config vs ground truth and BConvBarrett."""
        from util import find_moduli_ntt
        moduli = find_moduli_ntt(10, 28, 4096)
        om, tm = moduli[:5], moduli[5:]
        rng = np.random.default_rng(7)
        xs = [[int(rng.integers(0, q)) for q in om] for _ in range(64)]
        expected = np.array(self._ground_truth(om, tm, xs), dtype=np.uint64)
        got = self._run_montgomery(om, tm, xs)
        np.testing.assert_array_equal(expected, got)
        got_bat = self._run_montgomery(om, tm, xs, method="basis_change_bat")
        np.testing.assert_array_equal(expected, got_bat)

        _barrett = bconv.BConvBarrett(om + tm)
        _barrett.control_gen([(list(range(5)), list(range(5, 10)))])
        x = jnp.array(xs, dtype=jnp.uint64)[None, ...]
        ref = np.asarray(_barrett.basis_change(x))[0].astype(np.uint64)
        p_arr = np.array(tm, dtype=np.uint64)
        np.testing.assert_array_equal(ref % p_arr, got)

    def test_lazy_montgomery_inputs_tolerated(self):
        """Inputs lazy in [0, 2q) (upstream MontRed convention) give the same
        canonical result as strict inputs."""
        om, tm = [268435361, 268435009], [268437409, 524353]
        _bconv = bconv.BConvMontgomery(om + tm)
        _bconv.control_gen([(list(range(2)), [2, 3])])
        rng = np.random.default_rng(3)
        x = jnp.array([[int(rng.integers(0, q)) for q in om]
                       for _ in range(32)], dtype=jnp.uint64)[None, ...]
        strict = _bconv.ff_ctx_origin[0].to_computation_format(x).astype(jnp.uint64)
        lazy = strict + jnp.array(om, dtype=jnp.uint64)  # still < 2^32
        out_strict = _bconv.ff_ctx_target[0].to_original_format(
            _bconv._basis_change_dense(strict))
        out_lazy = _bconv.ff_ctx_target[0].to_original_format(
            _bconv._basis_change_dense(lazy))
        np.testing.assert_array_equal(np.asarray(out_strict), np.asarray(out_lazy))
        bat_strict = _bconv.ff_ctx_target[0].to_original_format(
            _bconv.basis_change_bat(strict))
        bat_lazy = _bconv.ff_ctx_target[0].to_original_format(
            _bconv.basis_change_bat(lazy))
        np.testing.assert_array_equal(np.asarray(out_strict), np.asarray(bat_strict))
        np.testing.assert_array_equal(np.asarray(out_strict), np.asarray(bat_lazy))


class BConvBATTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        # Define some example moduli. Both fit in 32 bits (required for BAT assumption).
        # These are from basis_change_test.py (approx 2^27)
        self.original_moduli = [134219681, 134218433, 134219009, 1073741857, 1073740609]
        self.target_moduli = [268435361, 268435009, 6710893, 1067031829]

        self.overall_moduli = self.original_moduli + self.target_moduli
        self.original_index = list(range(len(self.original_moduli)))
        self.target_index = list(range(len(self.original_moduli), len(self.overall_moduli)))

        self.bconv = bconv.BConvBarrett(self.overall_moduli)
        self.bconv.control_gen([(self.original_index, self.target_index)])

    def _exact_basis_change(self, in_tower):
        """Return canonical approximate-CRT residues using Python big ints."""
        import math as _math
        from util import modinv as _modinv

        modulus_product = _math.prod(self.original_moduli)
        q_hats = [modulus_product // q for q in self.original_moduli]
        q_hat_inverses = [
            _modinv(q_hat, q)
            for q_hat, q in zip(q_hats, self.original_moduli)
        ]
        flat = np.asarray(in_tower).reshape(-1, len(self.original_moduli))
        result = []
        for values in flat:
            total = sum(
                (int(value) * inverse % q) * q_hat
                for value, inverse, q, q_hat in zip(
                    values,
                    q_hat_inverses,
                    self.original_moduli,
                    q_hats,
                )
            )
            result.append([total % p for p in self.target_moduli])
        return np.asarray(result, dtype=np.uint64).reshape(
            *np.asarray(in_tower).shape[:-1], len(self.target_moduli)
        )

    # @absltest.skip("Skip a single test")
    def test_basis_change_bat_vs_standard(self):
        """
        Verifies that BAT agrees with exact approximate-CRT arithmetic.
        """
        key = jax.random.PRNGKey(0)

        # Dimensions
        batch = 1
        elements = 2
        d = 128 # small ring dim
        sizeQ = len(self.original_moduli)
        in_tower = random_batched_ciphertext((batch, elements, d, sizeQ), self.original_moduli, jnp.uint32)

        expected = self._exact_basis_change(in_tower)

        # Actual result (BAT)
        actual = self.bconv.basis_change_bat(in_tower)
        target_moduli_arr = jnp.array(self.target_moduli, dtype=jnp.uint64)
        diff = (
            actual.astype(jnp.int64) - jnp.asarray(expected, dtype=jnp.int64)
        ) % target_moduli_arr.astype(jnp.int64)
        np.testing.assert_array_equal(diff, jnp.zeros_like(diff))

    # @absltest.skip("Skip a single test")
    def test_basis_change_bat_random_big(self):
        """
        Test with larger shapes to ensure robustness.
        """
        key = jax.random.PRNGKey(1)
        batch = 4
        elements = 4
        d = 8
        sizeQ = len(self.original_moduli)
        shape = (batch, elements, d, sizeQ)

        in_tower = random_batched_ciphertext(shape, self.original_moduli, jnp.uint32)

        expected = self._exact_basis_change(in_tower)
        actual = self.bconv.basis_change_bat(in_tower)
        target_moduli_arr = jnp.array(self.target_moduli, dtype=jnp.uint64)
        diff = (
            actual.astype(jnp.int64) - jnp.asarray(expected, dtype=jnp.int64)
        ) % target_moduli_arr.astype(jnp.int64)
        np.testing.assert_array_equal(diff, jnp.zeros_like(diff))


class BConvMontgomery60BitChainTest(absltest.TestCase):
    """Split dense/BAT envelope gates on real 60-bit composite-scaling chains.

    Uses the genuine COMPOSITESCALINGAUTO configuration (composite_prime_gen
    Q chain + generate_p_towers P chain) with the he_params key-switch
    partition layout (dnum=3), not synthetic moduli. The dense one-shot
    contraction exceeds its wrap-safe REDC bound on every control here while
    BAT stays far inside its own; see README.md, "CKKS bootstrapping
    invariants."
    """

    DEGREE = 4096
    NQ = 58
    DNUM = 3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from composite_prime_gen import composite_prime_gen
        from util import generate_p_towers
        cls.q = composite_prime_gen(2, cls.NQ, 61, 60, 2 * cls.DEGREE, 31)
        cls.p = generate_p_towers(cls.q, dnum=cls.DNUM, degree=cls.DEGREE)

    @staticmethod
    def _controls(nq, num_p, dnum):
        """he_params._build_bconv_params layout: dnum Q-partition controls
        (selected towers -> all other QP towers) plus the P -> Q ModDown."""
        alpha = (nq + dnum - 1) // dnum
        controls = []
        for start in range(0, nq, alpha):
            sel = list(range(start, min(start + alpha, nq)))
            non_sel = [i for i in range(nq + num_p) if i not in sel]
            controls.append((sel, non_sel))
        controls.append((list(range(nq, nq + num_p)), list(range(nq))))
        return controls

    @staticmethod
    def _ground_truth(om, tm, xs):
        """Exact big-int approximate basis extension (alpha*Q term included)."""
        import math as _math
        from util import modinv as _modinv
        Q = _math.prod(om)
        qhat = [Q // q for q in om]
        qhatinv = [_modinv(h, q) for h, q in zip(qhat, om)]
        out = []
        for x in xs:
            S = sum(((xi * hi) % qi) * qh
                    for xi, hi, qi, qh in zip(x, qhatinv, om, qhat))
            out.append([S % p for p in tm])
        return out

    def _make_bconv(self):
        _bconv = bconv.BConvMontgomery(self.q + self.p)
        _bconv.control_gen(
            self._controls(len(self.q), len(self.p), self.DNUM))
        return _bconv

    def test_auto_route_uses_bat_when_dense_is_unsafe(self):
        """The production Q58/P21 chain routes through its only safe path."""
        _bconv = self._make_bconv()
        num_controls = self.DNUM + 1
        self.assertEqual(_bconv._use_bat, [True] * num_controls)
        x = _bconv.ff_ctx_origin[0].to_computation_format(
            jnp.array(
                [[q - 1 for q in _bconv.original_moduli[0]]],
                dtype=jnp.uint64,
            )
        )
        with mock.patch.object(
            _bconv,
            "_basis_change_bat",
            wraps=_bconv._basis_change_bat,
        ) as bat_path, mock.patch.object(
            _bconv,
            "_basis_change_dense",
            wraps=_bconv._basis_change_dense,
        ) as dense_path:
            result = _bconv.basis_change(x, 0)

        bat_path.assert_called_once()
        dense_path.assert_not_called()
        self.assertEqual(result.shape[-1], len(_bconv.target_moduli[0]))

    def test_q58_p21_bat_matches_exact_crt(self):
        """All four controls: random + adversarial inputs vs big-int CRT."""
        import math as _math
        from util import modinv as _modinv
        _bconv = self._make_bconv()
        rng = np.random.default_rng(60)
        for ci, (om, tm) in enumerate(
                zip(_bconv.original_moduli, _bconv.target_moduli)):
            Q = _math.prod(om)
            qhatinv = [_modinv(Q // q, q) for q in om]
            # x steering every digit to its maximum y_i = q_i - 1 (worst-case
            # accumulator and alpha overflow term).
            x_max = [((q - 1) * _modinv(hi, q)) % q
                     for q, hi in zip(om, qhatinv)]
            xs = [[int(rng.integers(0, q)) for q in om] for _ in range(4)]
            xs += [x_max, [0] * len(om), [q - 1 for q in om]]
            expected = self._ground_truth(om, tm, xs)
            xf = _bconv.ff_ctx_origin[ci].to_computation_format(
                jnp.array(xs, dtype=jnp.uint64)[None, ...])
            got = _bconv.ff_ctx_target[ci].to_original_format(
                _bconv.basis_change(xf, ci))
            np.testing.assert_array_equal(
                np.array(expected, dtype=np.uint64),
                np.asarray(got)[0],
                err_msg=f"control {ci}")

    def test_bat_accepts_58_q_towers_at_scaling_widths_56_58_60(self):
        """Production-like Q/P bases retain a safe BAT route at each width.

        The chain has 58 Q limbs. The values 56, 58, and 60 are composite
        scaling-modulus bit widths, split across two primes. With dnum=3 the
        generated P basis has about 21 limbs. Every key-switch partition and
        the final P-to-Q ModDown control must support BAT.
        """
        from composite_prime_gen import composite_prime_gen
        from util import generate_p_towers
        for width in (56, 58, 60):
            q = composite_prime_gen(
                2, self.NQ, width + 1, width, 2 * self.DEGREE, 31)
            p = generate_p_towers(q, dnum=self.DNUM, degree=self.DEGREE)
            _bconv = bconv.BConvMontgomery(q + p)
            _bconv.control_gen(self._controls(len(q), len(p), self.DNUM))
            self.assertTrue(all(_bconv._use_bat), msg=f"width={width}")

    def test_small_config_auto_selects_bat(self):
        """A small safe configuration selects BAT automatically."""
        from util import find_moduli_ntt
        moduli = find_moduli_ntt(12, 30, 1024)
        _bconv = bconv.BConvMontgomery(moduli)
        _bconv.control_gen([(list(range(10)), [10, 11])])
        self.assertEqual(_bconv._use_bat, [True])

        in_tower = _bconv.ff_ctx_origin[0].to_computation_format(
            jnp.ones((1, 1, 10), dtype=jnp.uint64)
        )
        with mock.patch.object(
            _bconv,
            "_basis_change_bat",
            wraps=_bconv._basis_change_bat,
        ) as bat_path, mock.patch.object(
            _bconv,
            "_basis_change_dense",
            wraps=_bconv._basis_change_dense,
        ) as dense_path:
            _bconv.basis_change(in_tower)

        bat_path.assert_called_once()
        dense_path.assert_not_called()

    def test_common_envelope_rejects_oversized_quotient(self):
        """A tiny target modulus that cannot hold Lambda still fails setup."""
        from util import find_moduli_ntt
        om = find_moduli_ntt(4, 30, 1024)
        _bconv = bconv.BConvMontgomery(om + [12289])
        with self.assertRaisesRegex(ValueError, "CRNS quotient"):
            _bconv.control_gen([(list(range(4)), [4])])


class BConvGuardrailTest(absltest.TestCase):

    def test_large_barrett_auto_route_uses_bat_and_matches_exact_crt(self):
        """BAT-first routing avoids dense uint64 accumulation overflow."""
        import math as _math
        from util import find_moduli_ntt

        moduli = find_moduli_ntt(59, 30, 1024)
        original_moduli = moduli[:58]
        target_modulus = moduli[58]
        _bconv = bconv.BConvBarrett(moduli)
        _bconv.control_gen([(list(range(58)), [58])])

        modulus_product = _math.prod(original_moduli)
        q_hats = [modulus_product // q for q in original_moduli]
        # x_i = -QHat_i makes every approximate-CRT digit
        # x_i * QHat_i^-1 mod q_i equal q_i - 1.
        max_digit_input = [
            (-q_hat) % q
            for q_hat, q in zip(q_hats, original_moduli)
        ]
        expected = sum(
            (q - 1) * q_hat
            for q, q_hat in zip(original_moduli, q_hats)
        ) % target_modulus
        in_tower = jnp.array([[max_digit_input]], dtype=jnp.uint64)

        self.assertEqual(_bconv._use_bat, [True])

        with mock.patch.object(
            _bconv,
            "_basis_change_bat",
            wraps=_bconv._basis_change_bat,
        ) as bat_path, mock.patch.object(
            _bconv,
            "_basis_change_dense",
            wraps=_bconv._basis_change_dense,
        ) as dense_path:
            actual = _bconv.basis_change(in_tower)

        bat_path.assert_called_once()
        dense_path.assert_not_called()
        self.assertEqual(
            int(np.asarray(actual)[0, 0, 0]) % target_modulus,
            expected,
        )

    def test_bat_lazy_strictifies_wide_origin_before_narrowing(self):
        """BAT routing preserves high bits needed by lazy canonicalization."""
        original_moduli = [
            134219681,
            134218433,
            134219009,
            1073741857,
            1073740609,
        ]
        target_moduli = [268435361, 268435009, 6710893, 1067031829]
        values = [114170512, 87771918, 49694804, 1028745852, 8803516]
        expected = [243057938, 24172461, 101247, 321557622]
        _bconv = bconv.BConvBATLazy(original_moduli + target_moduli)
        _bconv.control_gen(
            [
                (
                    list(range(len(original_moduli))),
                    list(
                        range(
                            len(original_moduli),
                            len(original_moduli) + len(target_moduli),
                        )
                    ),
                )
            ]
        )
        in_tower = jnp.array([[values]], dtype=jnp.uint64)

        with mock.patch.object(
            _bconv,
            "_basis_change_bat",
            wraps=_bconv._basis_change_bat,
        ) as bat_path:
            actual = _bconv.basis_change(in_tower)

        bat_path.assert_called_once()
        actual = _bconv.ff_ctx_target[0].to_original_format(actual)
        np.testing.assert_array_equal(
            np.asarray(actual)[0, 0], np.asarray(expected, dtype=np.uint32)
        )

    def test_barrett_guardrail_rejects_paths_outside_exact_target_range(self):
        """Target Barrett reduction must be exact for every contraction path."""
        _bconv = bconv.BConvBarrett([101, 5])
        with self.assertRaisesRegex(ValueError, "neither dense nor BAT"):
            _bconv.control_gen([([0], [1])])

    def test_backend_guardrails_reject_unrepresentable_moduli(self):
        """Each standard-residue backend enforces its storage envelope."""
        with self.assertRaisesRegex(ValueError, r"Barrett.*2\*\*32"):
            bconv.BConvBarrett([1 << 32, 17]).control_gen([([0], [1])])
        with self.assertRaisesRegex(ValueError, r"BAT-lazy.*2\*\*32"):
            bconv.BConvBATLazy([1 << 32, 17]).control_gen([([0], [1])])

    def test_bat_byte_accumulator_guardrail_is_shared(self):
        """The common uint32 BAT contraction bound gates every backend."""
        terms_per_modulus = 4 * 255 * 255
        first_unsafe_size = (
            (1 << 32) + terms_per_modulus - 1
        ) // terms_per_modulus

        target_modulus = 1073741441
        for bconv_cls in (bconv.BConvBarrett, bconv.BConvBATLazy):
            with self.subTest(bconv_cls=bconv_cls.__name__):
                _bconv = bconv_cls([3, target_modulus])
                bat_safe = _bconv._safe_guardrail(
                    [3] * (first_unsafe_size - 1), [target_modulus], 0
                )
                self.assertTrue(bat_safe)
                bat_unsafe = _bconv._safe_guardrail(
                    [3] * first_unsafe_size, [target_modulus], 0
                )
                self.assertFalse(bat_unsafe)

    def test_auto_route_falls_back_to_dense_when_bat_is_unavailable(self):
        """BConvBATLazy automatically falls back when BAT is preflighted out."""

        class DenseOnlyBConvBATLazy(bconv.BConvBATLazy):

            def _safe_guardrail(self, original_moduli, target_moduli, index):
                super()._safe_guardrail(
                    original_moduli, target_moduli, index
                )
                return False

        _bconv = DenseOnlyBConvBATLazy([17, 19, 23, 29])
        _bconv.control_gen([([0, 1], [2, 3])])
        in_tower = jnp.array([[[1, 2]]], dtype=jnp.uint64)

        self.assertEqual(_bconv._use_bat, [False])
        with self.assertRaisesRegex(ValueError, "BAT"):
            _bconv.basis_change_bat(in_tower)
        with mock.patch.object(
            _bconv,
            "_basis_change_dense",
            wraps=_bconv._basis_change_dense,
        ) as dense_path, mock.patch.object(
            _bconv,
            "_basis_change_bat",
            wraps=_bconv._basis_change_bat,
        ) as bat_path:
            result = _bconv.basis_change(in_tower)

        dense_path.assert_called_once()
        bat_path.assert_not_called()
        self.assertEqual(result.shape, (1, 1, 2))

    def test_guardrail_rejection_is_preflight_and_preserves_state(self):
        """An unsafe perf control cannot create or partially replace state."""
        from util import find_moduli_ntt

        unsafe_original = find_moduli_ntt(4, 30, 1024)
        overall_moduli = [17, 19, 23, 29] + unsafe_original + [12289]
        _bconv = bconv.BConvMontgomery(overall_moduli)
        _bconv.control_gen([([0, 1], [2, 3])])

        previous_original = list(_bconv.original_moduli)
        previous_target = list(_bconv.target_moduli)
        previous_route = list(_bconv._use_bat)
        previous_constants = list(_bconv.QHatInvModq)

        with mock.patch.object(
            _bconv, "_create_contexts", wraps=_bconv._create_contexts
        ) as create_contexts, mock.patch.object(
            _bconv,
            "_generate_constants_single",
            wraps=_bconv._generate_constants_single,
        ) as generate_constants:
            with self.assertRaisesRegex(ValueError, "CRNS quotient"):
                _bconv.control_gen(
                    [([4, 5, 6, 7], [8])], perf_test=True
                )

        create_contexts.assert_not_called()
        generate_constants.assert_not_called()
        self.assertEqual(_bconv.original_moduli, previous_original)
        self.assertEqual(_bconv.target_moduli, previous_target)
        self.assertEqual(_bconv._use_bat, previous_route)
        self.assertEqual(len(_bconv.QHatInvModq), len(previous_constants))
        for actual, expected in zip(_bconv.QHatInvModq, previous_constants):
            np.testing.assert_array_equal(actual, expected)

    def test_generator_controls_work_in_perf_mode(self):
        """The preflight and setup passes share materialized controls."""
        _bconv = bconv.BConvMontgomery([17, 19, 23, 29])
        controls = (
            control
            for control in (([0], [2]), ([1], [3]))
        )

        _bconv.control_gen(controls, perf_test=True)

        self.assertEqual(_bconv._use_bat, [True, True])
        self.assertLen(_bconv.QHatInvModq, 2)
        self.assertLen(_bconv.QHatModp, 2)
        self.assertLen(_bconv.QHatModpBAT, 2)
        self.assertLen(_bconv.crns_rho, 2)
        self.assertLen(_bconv.crns_g, 2)
        self.assertLen(_bconv.crns_lambda_bctx, 2)
        self.assertLen(_bconv.crns_corr_bctx, 2)

if __name__ == "__main__":
    absltest.main()
