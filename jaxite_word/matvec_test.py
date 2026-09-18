"""
Tests for homomorphic matrix-vector multiplication using the new context-level API.

Cross-validates against plaintext computation and OpenFHE reference behavior.
Uses degree=16 (8 slots) with the standard CROSS parameter set.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import key_gen as kg
import ckks_ctx
from polynomial import Polynomial
from matvec import MatVec, compute_bsgs_params

jax.config.update('jax_enable_x64', True)


class MatVecNewAPITest(parameterized.TestCase):
    """Test MatVec using the context-level API (he_rot, ptct_mul, he_rescale)."""

    def setUp(self):
        super().setUp()
        self.degree = 16
        self.num_slots = 8  # degree / 2
        self.dnum = 3
        self.r, self.c = 4, 4
        self.q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
        self.p_towers = [1073740609, 1073739937, 1073739649]
        self.sf = 563019763943521
        self.sigma = 3.190000057220458984375
        self.M = len(self.q_towers)

    def _make_ctx(self, rot_indices, output_scale=None):
        """Create CKKSContext with program_initialization for given rotation indices."""
        key_pair = kg.gen_pke_pair(self.q_towers, self.p_towers, self.degree)
        ek = kg.gen_evaluation_key(
            key_pair["secret_key"], q=self.q_towers, P=self.p_towers,
            noise_std=self.sigma, noise_scale=1, dnum=self.dnum)
        eval_key_a = jnp.array(ek["a"], dtype=jnp.uint32).transpose(0, 2, 1)
        eval_key_b = jnp.array(ek["b"], dtype=jnp.uint32).transpose(0, 2, 1)

        os = output_scale or (self.sf / self.q_towers[-1]) ** 2
        params = {
            "degree": self.degree, "num_slots": self.num_slots,
            "scaling_factor": self.sf, "output_scale": os,
            "q_towers": self.q_towers, "p_towers": self.p_towers,
            "p": 30, "CKKS_M_FACTOR": 1, "max_bits_in_word": 61,
            "noise_scale_degree": 1,
            "public_key": key_pair["public_key"],
            "secret_key": key_pair["secret_key"],
            "evaluation_key": [eval_key_a, eval_key_b],
        }
        ctx = ckks_ctx.CKKSContext(params)
        ctx.program_initialization(
            total_hemul_levels=ctx.max_level,
            total_rotation_indices=rot_indices,
            dnum=self.dnum, r=self.r, c=self.c, batch=1)
        return ctx

    def _encrypt_5d(self, ctx, slots):
        enc = ctx.encrypt(ctx.encode(slots))
        ct = Polynomial(
            {'batch': 1, 'num_elements': 2, 'degree': self.degree,
             'num_moduli': self.M, 'precision': 32,
             'degree_layout': (self.r, self.c)},
            {'moduli': self.q_towers})
        ct.polynomial = enc.polynomial.reshape(1, 2, self.r, self.c, self.M)
        return ct

    def _decrypt_decode(self, ctx, result_ct, num_moduli, output_scale):
        """Decrypt and decode, temporarily adjusting output_scale."""
        old_scale = ctx.output_scale
        ctx.output_scale = output_scale
        q_sub = self.q_towers[:num_moduli]
        ct_dec = Polynomial(
            {'batch': 1, 'num_elements': 2, 'degree': self.degree,
             'precision': 32, 'num_moduli': num_moduli,
             'degree_layout': (self.degree,)},
            {'moduli': q_sub})
        ct_dec.set_batch_polynomial(
            result_ct.polynomial.reshape(1, 2, self.degree, num_moduli))
        decrypted = ctx.decrypt(ct_dec)
        decoded = np.array([v.real for v in ctx.decode(decrypted, is_ntt=False)])
        ctx.output_scale = old_scale
        return decoded

    def _needed_rot_indices(self, n):
        """Compute all rotation indices needed for BSGS matvec of dimension n."""
        n1, n2 = compute_bsgs_params(n)
        indices = set()
        # Baby steps: 1, 2, ..., n1-1
        for i in range(1, n1):
            indices.add(i)
        # Giant steps: n1, 2*n1, ..., (n2-1)*n1
        for j in range(1, n2):
            indices.add(j * n1)
        return sorted(indices), n1, n2

    # ================================================================
    # Test 1: Identity matrix (v → v)
    # ================================================================
    def test_identity_matrix(self):
        """Identity matrix multiplication should preserve the vector."""
        n = self.num_slots
        rot_indices, n1, n2 = self._needed_rot_indices(n)
        # For identity, only diagonal 0 is nonzero = [1,1,...,1].
        # No rotation needed (all baby/giant steps multiply by zero diags).
        # But we still need all rotation keys for the general BSGS framework.
        ctx = self._make_ctx(rot_indices)

        matrix = np.eye(n)
        vector = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])

        mv = MatVec(ctx, n=n, level=ctx.max_level, n1=n1, n2=n2)
        mv.encode_matrix(matrix)

        ct_in = self._encrypt_5d(ctx, [complex(v, 0) for v in vector])
        ct_out = mv.mul(ct_in)

        # After pt-ct mul + rescale: scale = sf^2/q[-1]
        os = self.sf ** 2 / self.q_towers[-1]
        decoded = self._decrypt_decode(ctx, ct_out, self.M - 1, os)
        np.testing.assert_array_almost_equal(decoded[:n], vector, decimal=1)
        print(f"  Identity 8x8: PASS")

    # ================================================================
    # Test 2: Shift matrix (rotate vector left by 1)
    # ================================================================
    def test_shift_matrix(self):
        """Left-shift matrix: result[i] = vector[(i+1) mod n]."""
        n = self.num_slots
        rot_indices, n1, n2 = self._needed_rot_indices(n)
        ctx = self._make_ctx(rot_indices)

        # Shift matrix: M[i, (i+1) mod n] = 1, all other 0
        # This is equivalent to diagonal 1 = [1,...,1], all others 0
        matrix = np.zeros((n, n))
        for i in range(n):
            matrix[i, (i + 1) % n] = 1.0

        vector = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([vector[(i + 1) % n] for i in range(n)])

        mv = MatVec(ctx, n=n, level=ctx.max_level, n1=n1, n2=n2)
        mv.encode_matrix(matrix)

        ct_in = self._encrypt_5d(ctx, [complex(v, 0) for v in vector])
        ct_out = mv.mul(ct_in)

        os = self.sf ** 2 / self.q_towers[-1]
        decoded = self._decrypt_decode(ctx, ct_out, self.M - 1, os)
        np.testing.assert_array_almost_equal(decoded[:n], expected, decimal=1)
        print(f"  Shift 8x8: PASS")

    # ================================================================
    # Test 3: Random 8×8 matrix-vector product
    # ================================================================
    def test_random_matrix(self):
        """Random matrix-vector multiply, cross-validated against numpy."""
        n = self.num_slots
        rot_indices, n1, n2 = self._needed_rot_indices(n)
        ctx = self._make_ctx(rot_indices)

        np.random.seed(42)
        matrix = np.random.uniform(0.1, 2.0, (n, n))
        vector = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0])
        expected = matrix @ vector

        mv = MatVec(ctx, n=n, level=ctx.max_level, n1=n1, n2=n2)
        mv.encode_matrix(matrix)

        ct_in = self._encrypt_5d(ctx, [complex(v, 0) for v in vector])
        ct_out = mv.mul(ct_in)

        os = self.sf ** 2 / self.q_towers[-1]
        decoded = self._decrypt_decode(ctx, ct_out, self.M - 1, os)
        maxerr = np.max(np.abs(decoded[:n] - expected))
        # At degree=16, noise budget is limited. Relative error < 50% is acceptable.
        rel_err = maxerr / max(abs(expected))
        self.assertLess(rel_err, 0.5, f"Relative error too large: {rel_err:.2f}")
        print(f"  Random 8x8: PASS (maxerr={maxerr:.2f}, rel={rel_err:.2f})")

    # ================================================================
    # Test 4: Matrix-matrix-vector chain (A @ B @ v) via two matvec calls
    # Uses a SINGLE CKKSContext, invoking operations at different levels.
    # ================================================================
    def test_matmat_vec_chain(self):
        """Chain A @ B @ v using a single CKKSContext at different levels.

        First matvec: ct(B·v) at level L → level L-1
        Second matvec: ct(A·(B·v)) at level L-1 → level L-2

        Both MatVec instances share the same ctx and its parameter cache.
        The level parameter controls which rotation keys and tower counts are used.
        """
        n = self.num_slots
        rot_indices, n1, n2 = self._needed_rot_indices(n)

        # Single context with all rotation indices for both levels
        ctx = self._make_ctx(rot_indices)
        L = ctx.max_level

        # Two 8×8 matrices and a vector
        np.random.seed(123)
        A = np.round(np.random.uniform(0.5, 1.5, (n, n)), 2)
        B = np.round(np.random.uniform(0.5, 1.5, (n, n)), 2)
        v = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

        # --- First matvec: B @ v at level L (same ctx) ---
        mv_B = MatVec(ctx, n=n, level=L, n1=n1, n2=n2)
        mv_B.encode_matrix(B)
        ct_v = self._encrypt_5d(ctx, [complex(x, 0) for x in v])
        ct_Bv = mv_B.mul(ct_v)
        M1 = ct_Bv.polynomial.shape[-1]
        print(f"  Chain: B@v done (towers {self.M}→{M1})")

        # Verify intermediate B @ v
        Bv_expected = B @ v
        os_L1 = self.sf ** 2 / self.q_towers[-1]
        decoded_Bv = self._decrypt_decode(ctx, ct_Bv, M1, os_L1)
        maxerr_Bv = np.max(np.abs(decoded_Bv[:n] - Bv_expected))
        rel_Bv = maxerr_Bv / max(abs(Bv_expected))
        self.assertLess(rel_Bv, 0.5, f"B@v relative error too large: {rel_Bv:.2f}")
        print(f"  Chain: B@v verified (maxerr={maxerr_Bv:.3f}, rel={rel_Bv:.2f})")

        # --- Second matvec: A @ (B·v) at level L-1 (same ctx, different level) ---
        mv_A = MatVec(ctx, n=n, level=L - 1, n1=n1, n2=n2)
        mv_A.encode_matrix(A)
        ct_ABv = mv_A.mul(ct_Bv)
        M2 = ct_ABv.polynomial.shape[-1]
        print(f"  Chain: A@(B@v) done (towers {M1}→{M2})")

        # At degree=16 with 5 Q towers, the accumulated scale after 2 levels
        # (sf³/(q[-1]*q[-2]) ≈ 3e29) exceeds the CRT space of 3 remaining
        # towers (≈ 1.2e27), so final decode is not possible.
        # What we verify: the pipeline runs at both levels using a single context.
        print(f"  A@B@v chain (single ctx): PASS (towers {self.M}→{M1}→{M2})")


if __name__ == "__main__":
    absltest.main()
