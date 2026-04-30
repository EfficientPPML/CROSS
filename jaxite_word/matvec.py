"""
Homomorphic Matrix-Vector multiplication using the diagonal method with BSGS.

Uses the context-level API (CKKSContext.he_rot, CKKSContext.ptct_mul,
CKKSContext.he_rescale) for all HE operations.

Algorithm (diagonal method):
  Given an n×n matrix A and encrypted vector ct(v), the product A·v is:
    ct(A·v) = sum_{k=0}^{n-1} Rot_k(ct(v)) * Encode(diag_k(A))
  where diag_k(A)[i] = A[i, (i+k) mod n].

  With BSGS optimization (n1 * n2 = n):
    ct(A·v) = sum_{j=0}^{n2-1} Rot_{j*n1}(
                sum_{i=0}^{n1-1} Rot_i(ct(v)) * Encode(diag'_{j*n1+i}(A))
              )
  where diag'_k is the diagonal pre-rotated by -j*n1 to compensate
  for the giant-step rotation.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional
import ckks_ctx as ctx_module
from polynomial import Polynomial


def compute_bsgs_params(n: int) -> Tuple[int, int]:
    """Compute baby-step n1 and giant-step n2 such that n1 * n2 = n."""
    n1 = int(math.sqrt(n))
    while n % n1 != 0:
        n1 -= 1
    return n1, n // n1


def extract_diagonals(matrix: np.ndarray) -> List[np.ndarray]:
    """Extract n diagonals: diag_k[i] = matrix[i, (i+k) mod n]."""
    n = matrix.shape[0]
    return [np.array([matrix[i, (i + k) % n] for i in range(n)]) for k in range(n)]


def pre_rotate_diagonal(diagonal: np.ndarray, j: int, n1: int) -> np.ndarray:
    """Pre-rotate diagonal by -j*n1 for BSGS giant-step compensation."""
    return np.roll(diagonal, -(j * n1) % len(diagonal))


class MatVec:
    """Homomorphic matrix-vector multiplication using the new context-level API.

    Usage:
        ctx = CKKSContext(params)
        ctx.program_initialization(...)

        mv = MatVec(ctx, n=8, level=ctx.max_level)
        mv.encode_matrix(matrix)
        result_ct = mv.mul(ct_input)
    """

    def __init__(self, ctx: 'ctx_module.CKKSContext', n: int, level: int,
                 n1: Optional[int] = None, n2: Optional[int] = None):
        """
        Args:
            ctx: CKKSContext with program_initialization() already called.
            n: Matrix/vector dimension (= num_slots for CKKS packing).
            level: Polynomial level of the input vector.
            n1, n2: BSGS parameters. If None, computed as sqrt(n).
        """
        self.ctx = ctx
        self.n = n
        self.level = level

        if n1 is None or n2 is None:
            self.n1, self.n2 = compute_bsgs_params(n)
        else:
            self.n1, self.n2 = n1, n2
            assert n1 * n2 == n

        self.encoded_diagonals = None  # Set by encode_matrix

    def encode_matrix(self, matrix: np.ndarray):
        """Encode a matrix as BSGS-preprocessed diagonal plaintexts.

        Encodes diagonals at the tower count matching self.level, so the
        same MatVec instance can be used at any level within a single context.

        Args:
            matrix: Square numpy array of shape (n, n), real or complex.
        """
        n, n1, n2 = self.n, self.n1, self.n2
        assert matrix.shape == (n, n)

        diagonals = extract_diagonals(matrix)

        # Encode at the tower count for this level
        num_q = self.ctx._param_cache.num_q_at_level(self.level)
        q_at_level = self.ctx.q_towers[:num_q]

        # Create a temporary CKKSContext for encoding at this level's towers
        import ckks_ctx as ctx_mod
        encode_params = dict(self.ctx.parameters)
        encode_params["q_towers"] = q_at_level
        encode_ctx = ctx_mod.CKKSContext(encode_params)

        self.encoded_diagonals = []
        for k in range(n):
            j = k // n1  # giant-step index
            rotated = pre_rotate_diagonal(diagonals[k], j, n1)
            complex_diag = [complex(x) for x in rotated]
            encoded = encode_ctx.encode(complex_diag)
            self.encoded_diagonals.append(encoded)

    def mul(self, ct_input: Polynomial) -> Polynomial:
        """Perform homomorphic matrix-vector multiplication.

        The input ciphertext is at self.level. The output is at self.level - 1
        (one rescale consumed by the pt-ct multiplication).

        Args:
            ct_input: Encrypted vector at the configured level.

        Returns:
            Encrypted result vector (at level - 1).
        """
        if self.encoded_diagonals is None:
            raise ValueError("Matrix not encoded. Call encode_matrix() first.")

        n1, n2, L = self.n1, self.n2, self.level
        r, c = self.ctx._param_cache.r, self.ctx._param_cache.c
        degree = self.ctx.degree
        M = ct_input.polynomial.shape[-1]

        # --- Baby-step rotations: cache Rot_i(ct_input) for i = 0..n1-1 ---
        # All cached data is in (batch, 2, r, c, M) layout.
        baby_rotations = {}
        for i in range(n1):
            if i == 0:
                baby_rotations[0] = ct_input.polynomial.copy()
            else:
                ct_copy = Polynomial(
                    {'batch': 1, 'num_elements': 2, 'degree': degree,
                     'num_moduli': M, 'precision': 32,
                     'degree_layout': (r, c)}, {'moduli': ct_input.moduli})
                ct_copy.polynomial = ct_input.polynomial.copy()
                rot_ct = self.ctx.he_rot[L, i].rotate(ct_copy)
                # HERot outputs (batch, 2, ring_dim, M) — reshape to (batch, 2, r, c, M)
                baby_rotations[i] = rot_ct.polynomial.reshape(1, 2, r, c, M).copy()

        # --- Process giant-step groups ---
        M_out = M - 1  # after rescale
        q_out = self.ctx.q_towers[:M_out]
        giant_step_results = []

        for j in range(n2):
            inner_sum = None

            for i in range(n1):
                k = j * n1 + i
                rotated_data = baby_rotations[i]
                pt_encoded = self.encoded_diagonals[k]

                # pt-ct multiply: rotated_ct * diag_plaintext
                pt_ntt = pt_encoded.polynomial[0, 0].reshape(r, c, M)
                op = self.ctx.ptct_mul[L]
                op.set_plaintext(pt_ntt)

                ct_work = Polynomial(
                    {'batch': 1, 'num_elements': 2, 'degree': degree,
                     'num_moduli': M, 'precision': 32,
                     'degree_layout': (r, c)}, {'moduli': ct_input.moduli})
                ct_work.polynomial = rotated_data.copy()
                result = op.mul(ct_work, use_bat=False)

                # Rescale after pt-ct multiply
                rescaled = self.ctx.he_rescale[L, L - 1](result)
                product_data = rescaled.polynomial

                # Accumulate (modular addition)
                if inner_sum is None:
                    inner_sum = product_data.copy()
                else:
                    moduli_arr = jnp.array(q_out, dtype=jnp.uint32)
                    s = inner_sum + product_data
                    inner_sum = jnp.where(s >= moduli_arr, s - moduli_arr, s)

            # Giant-step rotation (if j > 0)
            if j > 0 and inner_sum is not None:
                gs_rot_idx = j * n1
                ct_gs = Polynomial(
                    {'batch': 1, 'num_elements': 2, 'degree': degree,
                     'num_moduli': M_out, 'precision': 32,
                     'degree_layout': (r, c)}, {'moduli': q_out})
                ct_gs.polynomial = inner_sum
                gs_result = self.ctx.he_rot[L - 1, gs_rot_idx].rotate(ct_gs)
                # Reshape back to (batch, 2, r, c, M_out)
                inner_sum = gs_result.polynomial.reshape(1, 2, r, c, M_out)

            if inner_sum is not None:
                giant_step_results.append(inner_sum)

        # Sum all giant-step results
        result = giant_step_results[0]
        moduli_arr = jnp.array(q_out, dtype=jnp.uint32)
        for gs_res in giant_step_results[1:]:
            s = result + gs_res
            result = jnp.where(s >= moduli_arr, s - moduli_arr, s)

        out_ct = Polynomial(
            {'batch': 1, 'num_elements': 2, 'degree': degree,
             'num_moduli': M_out, 'precision': 32,
             'degree_layout': (r, c)}, {'moduli': q_out})
        out_ct.polynomial = result
        return out_ct


def make_ptct_no_rescale_fn(ptct_obj):
    """Build a pure ptct_mul function (no rescale) from a configured HE op.

    Used by BSGS's lazy-rescale path: accumulate raw ptct products at
    in_level inside the per-diag inner scan, then rescale ONCE per giant
    step before the giant rotation. Cuts 1024 rescales -> 32 in FC1 BSGS.

    Returns a closure (ct_data, pt_ntt) -> mulled_data with the same Barrett
    reduction as `make_ptct_rescale_fn` but WITHOUT the modulus drop.
    """
    barrett_mul = ptct_obj.barrett_ctx

    def ptct_no_rescale_fn(ct_data, pt_ntt):
        """Pure ptct_mul (VPU), no rescale.

        Args:
            ct_data: (batch, elems, r, c, M) uint32 input ciphertext (in_level).
            pt_ntt:  (r, c, M) uint32 encoded plaintext diagonal.

        Returns:
            (batch, elems, r, c, M) uint32 product (still at in_level).
        """
        product = ct_data.astype(jnp.uint64) * pt_ntt.astype(jnp.uint64)
        return barrett_mul.modular_reduction(product).astype(jnp.uint32)

    return ptct_no_rescale_fn


def make_rescale_only_fn(rescale_obj):
    """Build a pure rescale function from a configured HERescale op.

    Used together with `make_ptct_no_rescale_fn` for BSGS's lazy-rescale
    path. Drops the last modulus from a polynomial that's already in NTT
    eval form (composite_degree=1). Same math as the rescale half of
    `make_ptct_rescale_fn`, factored out so we can call it once per giant
    step instead of once per diagonal.
    """
    r, c = rescale_obj.r, rescale_obj.c

    # The historical post-INTT inv_psi multiply is a no-op now that the
    # NTT itself is negacyclic (psi/psi^{-1} are baked into the NTT
    # parameter generation — see commits 62bab5a / 410ee19 and the note
    # in rescale.py::HERescale.control_gen). Skipped here.
    gammas = rescale_obj.gammas_stacked[0]      # (1, r, c, M-1)
    betas = rescale_obj.betas_stacked[0]        # (M-1,)
    threshold = rescale_obj.moduli_threshold_all[-1]
    drop_moduli = rescale_obj.full_moduli_arr[:-1].astype(jnp.uint32)
    last_modulus = rescale_obj.full_moduli_arr[-1]

    ct_last_ntt = rescale_obj.ct_last_list[0].ntt_ctx
    ct_last_ff = ct_last_ntt.ff_ctx
    ct_work_ntt = rescale_obj.ct_work_list[0].ntt_ctx
    ct_work_ff = ct_work_ntt.ff_ctx

    def rescale_fn(mulled):
        """Drop the last modulus from `mulled` (already NTT eval, in_level).

        Args:
            mulled: (batch, elems, r, c, M) uint32, e.g. an accumulated
                sum of ptct products at in_level.

        Returns:
            (batch, elems, r, c, M-1) uint32 at out_level (in_level - 1).
        """
        # Extract last tower and INTT (negacyclic — already includes
        # the psi^{-i} post-twist that used to be a separate step).
        last_tower = mulled[..., -1:]
        shape_last = last_tower.shape
        last_reshaped = last_tower.reshape(-1, r, c, 1)
        last_coeffs = ct_last_ntt.intt(last_reshaped).reshape(shape_last)
        # (No inv_psi modmul — redundant under negacyclic NTT.)

        switched = jnp.where(
            last_coeffs < threshold,
            last_coeffs,
            drop_moduli.astype(jnp.uint64) - last_modulus + last_coeffs)
        twisted = switched.astype(jnp.uint64) * gammas

        shape_work = mulled[..., :-1].shape
        twisted_reduced = ct_work_ff.modular_reduction(
            twisted.astype(jnp.uint64)
        ).astype(jnp.uint32)
        ntt_result = ct_work_ntt.ntt(
            twisted_reduced.reshape(-1, r, c, shape_work[-1])
        ).reshape(shape_work)

        main_part = mulled[..., :-1].astype(jnp.uint64) * betas.astype(jnp.uint64)
        combined = ct_work_ff.modular_reduction(main_part)
        combined = combined.astype(jnp.uint64) + ntt_result.astype(jnp.uint64)
        result = ct_work_ff.modular_reduction(combined)
        return result.astype(jnp.uint32)

    return rescale_fn


def make_ptct_rescale_fn(ptct_obj, rescale_obj):
    """Build a pure ptct_mul+rescale function from configured HE operators.

    Returns a closure (ct_data, pt_ntt) -> rescaled_data that performs
    plaintext-ciphertext multiplication followed by single-modulus rescale.

    Args:
        ptct_obj: A configured HEPtCtMul instance (from ptct_mul.py).
        rescale_obj: A configured HERescale instance (from rescale.py)
            with composite_degree=1.

    Returns:
        Pure function: (ct_data, pt_ntt) -> jnp.ndarray
    """
    # Capture Barrett context for ptct multiply
    barrett_mul = ptct_obj.barrett_ctx
    r, c = ptct_obj.r, ptct_obj.c

    # Capture rescale constants (composite_degree=1, iter_idx=0).
    # The historical post-INTT inv_psi multiply is a no-op now that the
    # NTT itself is negacyclic (psi/psi^{-1} are baked into the NTT
    # parameter generation — see commits 62bab5a / 410ee19 and the note
    # in rescale.py::HERescale.control_gen). HERescale no longer creates
    # `power_of_inv_psi_all`; we simply skip the inv_psi modmul.
    gammas = rescale_obj.gammas_stacked[0]  # (1, r, c, M-1)
    betas = rescale_obj.betas_stacked[0]    # (M-1,)
    threshold = rescale_obj.moduli_threshold_all[-1]
    drop_moduli = rescale_obj.full_moduli_arr[:-1].astype(jnp.uint32)
    last_modulus = rescale_obj.full_moduli_arr[-1]

    # NTT/FF contexts from pre-allocated Polynomial objects
    ct_last_ntt = rescale_obj.ct_last_list[0].ntt_ctx
    ct_last_ff = ct_last_ntt.ff_ctx
    ct_work_ntt = rescale_obj.ct_work_list[0].ntt_ctx
    ct_work_ff = ct_work_ntt.ff_ctx

    def ptct_rescale_fn(ct_data, pt_ntt):
        """Pure ptct_mul (VPU) + rescale (1 modulus drop).

        Args:
            ct_data: (batch, elems, r, c, M) uint32 input ciphertext
            pt_ntt: (r, c, M) uint32 encoded plaintext diagonal

        Returns:
            (batch, elems, r, c, M-1) uint32 rescaled result
        """
        # --- ptct_mul_vpu ---
        product = ct_data.astype(jnp.uint64) * pt_ntt.astype(jnp.uint64)
        mulled = barrett_mul.modular_reduction(product).astype(jnp.uint32)

        # --- rescale (composite_degree=1) ---
        # Extract last tower and INTT (negacyclic — already includes
        # the psi^{-i} post-twist that used to be a separate step).
        last_tower = mulled[..., -1:]
        shape_last = last_tower.shape
        last_reshaped = last_tower.reshape(-1, r, c, 1)
        last_coeffs = ct_last_ntt.intt(last_reshaped).reshape(shape_last)
        # (No inv_psi modmul — redundant under negacyclic NTT.)

        # Modulus switch
        switched = jnp.where(
            last_coeffs < threshold,
            last_coeffs,
            drop_moduli.astype(jnp.uint64) - last_modulus + last_coeffs)
        twisted = switched.astype(jnp.uint64) * gammas

        # mod_reduce + NTT
        shape_work = mulled[..., :-1].shape
        twisted_reduced = ct_work_ff.modular_reduction(
            twisted.astype(jnp.uint64)
        ).astype(jnp.uint32)
        ntt_result = ct_work_ntt.ntt(
            twisted_reduced.reshape(-1, r, c, shape_work[-1])
        ).reshape(shape_work)

        # beta multiply + add + mod_reduce
        main_part = mulled[..., :-1].astype(jnp.uint64) * betas.astype(jnp.uint64)
        combined = ct_work_ff.modular_reduction(main_part)
        combined = combined.astype(jnp.uint64) + ntt_result.astype(jnp.uint64)
        result = ct_work_ff.modular_reduction(combined)

        return result.astype(jnp.uint32)

    return ptct_rescale_fn


class MatVecScannable:
    """Scannable MatVec using jax.lax.scan for the BSGS loops.

    Unlike MatVec which uses Python loops (causing JIT trace unrolling),
    this class uses jax.lax.scan so the entire operation compiles into
    a single XLA kernel with O(1) HLO body size.

    Usage:
        # Build pure-function closures from configured HE operators
        mv = MatVecScannable(herot_baby, herot_giant, ptct_obj, rescale_obj,
                             n, n1, n2, q_out, batch)
        # Run with stacked per-rotation/per-diagonal data
        result = mv.mul_scan(ct_data, baby_eval_a, baby_eval_b, baby_coefmaps,
                             giant_eval_a, giant_eval_b, giant_coefmaps, diag_pts)
    """

    def __init__(self, herot_baby, herot_giant, ptct_obj, rescale_obj,
                 n, n1, n2, q_out, batch):
        """Initialize MatVecScannable.

        Args:
            herot_baby: Configured HERot for baby-step rotations (input level).
            herot_giant: Configured HERot for giant-step rotations (output level).
            ptct_obj: Configured HEPtCtMul for the input level.
            rescale_obj: Configured HERescale for input -> output level.
            n: Matrix/vector dimension (= num_slots).
            n1, n2: BSGS parameters (n1 * n2 = n).
            q_out: List of output-level moduli (len = M-1).
            batch: Batch size.
        """
        self.n, self.n1, self.n2 = n, n1, n2
        self.batch = batch
        self.q_out = q_out
        self.r = herot_baby.r
        self.c = herot_baby.c
        self.sizeQl = herot_baby.sizeQl

        # Build pure-function closures
        from herot import HERot
        self.rotate_baby_fn = HERot.make_rotate_fn(herot_baby)
        self.rotate_giant_fn = HERot.make_rotate_fn(herot_giant)
        self.ptct_rescale_fn = make_ptct_rescale_fn(ptct_obj, rescale_obj)

    def mul_scan(self, ct_data, baby_eval_a, baby_eval_b, baby_coefmaps,
                 giant_eval_a, giant_eval_b, giant_coefmaps, diag_pts):
        """Perform MatVec using jax.lax.scan.

        Args:
            ct_data: (batch, 2, r, c, M) uint32 input ciphertext
            baby_eval_a: (n1-1, dnum, r, c, sizeQlP) uint64 baby rotation keys A
            baby_eval_b: (n1-1, dnum, r, c, sizeQlP) uint64 baby rotation keys B
            baby_coefmaps: (n1-1, ring_dim) int32 baby coef permutations
            giant_eval_a: (n2, dnum, r, c, sizeQlP_out) uint64 giant keys A
            giant_eval_b: (n2, dnum, r, c, sizeQlP_out) uint64 giant keys B
            giant_coefmaps: (n2, ring_dim) int32 giant coef permutations
            diag_pts: (n2, n1, r, c, M) uint32 diagonal plaintexts

        Returns:
            (batch, 2, r, c, M-1) uint32 result ciphertext
        """
        n1, n2, batch = self.n1, self.n2, self.batch
        r, c = self.r, self.c
        M = ct_data.shape[-1]
        M_out = M - 1
        moduli_out = jnp.array(self.q_out, dtype=jnp.uint32)
        rotate_baby = self.rotate_baby_fn
        rotate_giant = self.rotate_giant_fn
        ptct_rescale = self.ptct_rescale_fn

        # ---- Phase 1: Baby-step rotations via scan ----
        def baby_body(carry, xs):
            ct_input = carry  # original CT, unchanged across iterations
            ea, eb, cmap = xs
            rotated = rotate_baby(ct_input, ea, eb, cmap)
            # rotate_fn returns (batch, 2, ring_dim, M) — reshape to 5D
            rotated_5d = rotated.reshape(batch, 2, r, c, M)
            return ct_input, rotated_5d

        _, baby_rotated = jax.lax.scan(
            baby_body, ct_data,
            (baby_eval_a, baby_eval_b, baby_coefmaps))
        # baby_rotated: (n1-1, batch, 2, r, c, M)
        # Prepend the unrotated ct_data as index 0
        all_baby = jnp.concatenate(
            [ct_data[jnp.newaxis], baby_rotated], axis=0)
        # all_baby: (n1, batch, 2, r, c, M)

        # ---- Phase 2: Giant-step scan with inner ptct+rescale scan ----
        zero_result = jnp.zeros(
            (batch, 2, r, c, M_out), dtype=jnp.uint32)

        def _modadd(a, b):
            s = a.astype(jnp.uint64) + b.astype(jnp.uint64)
            return jnp.where(s >= moduli_out, s - moduli_out, s).astype(
                jnp.uint32)

        def inner_body(inner_sum, xs):
            baby_ct, pt_ntt = xs
            product = ptct_rescale(baby_ct, pt_ntt)
            result = _modadd(inner_sum, product)
            # Ensure carry type matches (strip any sharding annotation)
            return result, None

        def giant_body(accumulated, xs):
            j, gs_ea, gs_eb, gs_cmap, diag_group = xs
            # diag_group: (n1, r, c, M)

            # Inner scan: ptct_mul+rescale over n1 diagonals
            zero_inner = jnp.zeros(
                (batch, 2, r, c, M_out), dtype=jnp.uint32)
            inner_sum, _ = jax.lax.scan(
                inner_body, zero_inner, (all_baby, diag_group))

            # Giant-step rotation (skip for j=0)
            rotated = rotate_giant(inner_sum, gs_ea, gs_eb, gs_cmap)
            rotated_5d = rotated.reshape(batch, 2, r, c, M_out)
            # Use j==0 to bypass rotation
            inner_result = jnp.where(j > 0, rotated_5d, inner_sum)

            result = _modadd(accumulated, inner_result)
            return result, None

        j_indices = jnp.arange(n2, dtype=jnp.int32)
        result, _ = jax.lax.scan(
            giant_body, zero_result,
            (j_indices, giant_eval_a, giant_eval_b, giant_coefmaps, diag_pts))

        return result
