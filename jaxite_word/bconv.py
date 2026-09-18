"""
BConv: Basis Conversion class for JAX-based homomorphic encryption.

This module provides the BConv class and subclasses which handle basis extension and modulus switching
using efficient modular reduction. It is designed to work with
vectorized operations on JAX arrays.
"""
import math

import jax
import jax.numpy as jnp
import util
import finite_field as ff_context

# Maintain 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)
_is_nvidia = "NVIDIA" in jax.devices()[0].device_kind


def _barrett_exact_quotient(z: jnp.ndarray, bctx) -> jnp.ndarray:
    """Exact floor(z/q) via Barrett multiply-high, division-free.

    The estimate t never overestimates and undershoots by <= 2 for z < 2^s
    (s = 2*ceil(log2 q)), so two conditional corrections recover the exact
    quotient. Unlike BarrettContext.modular_reduction, the low split uses the
    exact mask (1 << w) - 1 so it also holds for moduli with w < 32. Requires
    z < 2^s per modulus.
    """
    m, w, s_w, q = bctx.m, bctx.w, bctx.s_w, bctx.moduli_reduction
    mask = (jnp.uint64(1) << w.astype(jnp.uint64)) - jnp.uint64(1)
    z1 = z & mask
    z2 = z >> w
    t = (((z1 * m) >> w) + z2 * m) >> s_w
    r = z - t * q
    return t + (r >= q).astype(jnp.uint64) + (r >= 2 * q).astype(jnp.uint64)


def _barrett_exact_mod(z: jnp.ndarray, bctx) -> jnp.ndarray:
    """Exact z mod q via _barrett_exact_quotient (no integer division)."""
    return z - _barrett_exact_quotient(z, bctx) * bctx.moduli_reduction

class BConv:
    def __init__(self, overall_moduli):
        """
        Initialize the BConv object.

        Args:
            overall_moduli: A list or tuple of integers representing the all available moduli.
        """
        self.overall_moduli = overall_moduli
        # Lists to store configurations for each control index
        self.original_moduli = []
        self.target_moduli = []
        self.ff_ctx_origin = []
        self.ff_ctx_target = []

        # Lists to store precomputed constants for each control index
        self.QHatInvModq = []
        self.QHatModp = []
        self.QHatModpBAT = []
        # True selects BAT; False selects the dense fallback.
        self._use_bat = []

    def _safe_guardrail(
        self, original_moduli, target_moduli, control_index
    ):
        """Validate common inputs and return whether BAT passes its shared bound.

        Subclasses add backend-specific dense and BAT checks and raise when
        neither path is safe.
        """
        if not original_moduli:
            raise ValueError(
                f"BConv control {control_index}: original basis is empty"
            )
        if not target_moduli:
            raise ValueError(
                f"BConv control {control_index}: target basis is empty"
            )

        invalid_moduli = [
            modulus
            for modulus in (*original_moduli, *target_moduli)
            if modulus <= 1
        ]
        if invalid_moduli:
            raise ValueError(
                f"BConv control {control_index}: moduli must be greater than "
                f"one; got {invalid_moduli}"
            )

        return 4 * len(original_moduli) * 255 * 255 < 1 << 32

    def _reset_control_state(self):
        """Clear generated state after every control passes validation."""
        self.original_moduli = []
        self.target_moduli = []
        self.ff_ctx_origin = []
        self.ff_ctx_target = []
        self.QHatInvModq = []
        self.QHatModp = []
        self.QHatModpBAT = []
        self._use_bat = []

    def basis_change(
        self, in_tower: jnp.ndarray, control_index: int = 0
    ) -> jnp.ndarray:
        """Convert bases, preferring BAT and falling back to dense safely."""
        if self._use_bat[control_index]:
            return self.basis_change_bat(in_tower, control_index)
        return self._basis_change_dense(in_tower, control_index)

    def basis_change_bat(
        self, in_tower: jnp.ndarray, control_index: int = 0
    ) -> jnp.ndarray:
        """Run the explicit BAT path retained for tests and benchmarks."""
        if not self._use_bat[control_index]:
            raise ValueError(
                f"{type(self).__name__} control {control_index}: BAT basis "
                "conversion is outside its safe arithmetic envelope"
            )
        return self._basis_change_bat(in_tower, control_index)

    def _create_contexts(self, original_moduli, target_moduli):
        """
        Initialize the finite field contexts. Must be implemented by subclasses.
        Returns: (ff_ctx_origin, ff_ctx_target)
        """
        raise NotImplementedError

    def _generate_constants_single(self, original_moduli, target_moduli, ff_ctx_origin, ff_ctx_target):
        """
        Generates constants for a single configuration.
        """
        # compute_QHatInvModq_QHatModp returns lists, we convert them to JAX arrays
        # with appropriate shapes for broadcasting.
        QHatInvModq_list, QHatModp_list = util.compute_QHatInvModq_QHatModp(
            original_moduli, target_moduli
        )

        # QHatInvModq: Inverse of (Q/q_i) mod q_i
        # Shape: (sizeQ,) -> JAX array
        QHatInvModq = jnp.array(QHatInvModq_list, dtype=jnp.uint64)
        QHatInvModq = ff_ctx_origin.to_computation_format(QHatInvModq)

        # QHatModp: (Q/q_i) mod p_j
        # Shape: (sizeQ, sizeP) -> JAX array
        QHatModp = jnp.array(QHatModp_list, dtype=jnp.uint64)
        QHatModp = ff_ctx_target.to_computation_format(QHatModp)

        # BAT Preprocessing
        # QHatModpBAT
        # Input QHatModp: (sizeQ, sizeP)
        # _basis_aligned_transformation -> (4, sizeQ, sizeP, 4) (dims: a, q, p, b)
        # We want to match input (..., d, q, a) -> output (..., d, p, b)
        # Transpose to (q, a, p, b) -> (q*a, p, b) for einsum "...dq, qpb -> ...dpb"
        QHatModpBAT_raw = self._basis_aligned_transformation(
            QHatModp,
            target_moduli
        )
        QHatModpBAT = QHatModpBAT_raw.transpose(1, 0, 2, 3).reshape(-1, QHatModpBAT_raw.shape[2], 4)

        return QHatInvModq, QHatModp, QHatModpBAT

    def control_gen(self, control_indices_list, perf_test=False):
        """
        Generates and stores precomputed constants QHatInvModq and QHatModp necessary for
        the basis change operation.

        Args:
            control_indices_list: A sequence of (original_index, target_index) tuples/lists.
                                  original_index: Indices of original_moduli in overall_moduli.
                                  target_index: Indices of target_moduli in overall_moduli.
        """
        # Resolve and preflight every control before replacing generated state.
        resolved_controls = []
        use_bat = []
        for control_index, (original_index, target_index) in enumerate(
            control_indices_list
        ):
            omParams = [int(self.overall_moduli[i]) for i in original_index]
            tmParams = [int(self.overall_moduli[i]) for i in target_index]
            resolved_controls.append((omParams, tmParams))
            use_bat.append(
                self._safe_guardrail(
                    omParams, tmParams, control_index
                )
            )

        # Replace generated state only after every control passes preflight.
        self._reset_control_state()
        self._use_bat = use_bat

        for omParams, tmParams in resolved_controls:
            self.original_moduli.append(omParams)
            self.target_moduli.append(tmParams)

            ctx_origin, ctx_target = self._create_contexts(omParams, tmParams)
            self.ff_ctx_origin.append(ctx_origin)
            self.ff_ctx_target.append(ctx_target)

            if perf_test:
                sizeQ = len(omParams)
                sizeP = len(tmParams)
                QHatInvModq = util.random_parameters(
                    (sizeQ,), omParams, dtype=jnp.uint64
                )
                QHatModp = util.random_parameters(
                    (sizeQ, sizeP), tmParams, dtype=jnp.uint64
                )
                QHatModpBAT = jnp.zeros(
                    (sizeQ * 4, sizeP, 4), dtype=jnp.uint8
                )
            else:
                QHatInvModq, QHatModp, QHatModpBAT = (
                    self._generate_constants_single(
                        omParams, tmParams, ctx_origin, ctx_target
                    )
                )

            self.QHatInvModq.append(QHatInvModq)
            self.QHatModp.append(QHatModp)
            self.QHatModpBAT.append(QHatModpBAT)

    def _basis_aligned_transformation(self, matrix: jnp.ndarray, moduli):
        """
        Prepares a matrix for Basis Aligned Transformation (BAT).
        Adapted from ntt_mm.py.
        Assumes matrix last dimension corresponds to 'moduli'.
        """
        return util.shifted_mod_bytes_host(matrix, moduli)

    # @functools.partial(jax.jit, static_argnames=("self",))
    def _basis_change_dense(self, in_tower: jnp.ndarray, control_index: int = 0) -> jnp.ndarray:
        """
        Performs the approximate basis change from original_moduli to target_moduli.

        Input:
            in_tower: Coefficients in original basis.
                      Shape: (..., ring_dim, sizeQ)
            control_index: Index of the control set to use.

        Output:
            out_tower: Coefficients in new basis.
                       Shape: (..., ring_dim, sizeP)
        """
        # Ensure inputs are correctly typed
        in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

        # Retrieve constants and contexts for this control index
        QHatInvModq = self.QHatInvModq[control_index]
        QHatModp = self.QHatModp[control_index]
        ff_ctx_origin = self.ff_ctx_origin[control_index]
        ff_ctx_target = self.ff_ctx_target[control_index]

        # Step 1: Compute c_unreduced = in_tower * QHatInvModq
        c_unreduced = in_tower * QHatInvModq

        # Step 2: Modular Reduction on c_unreduced using original moduli context
        c = ff_ctx_origin.modular_reduction(c_unreduced)

        # Base Term: c * QHatModp
        # Shape: (..., d, p)
        if _is_nvidia:
            summed_terms = jnp.einsum("...dq,qp->...dp", c.astype(jnp.uint32), QHatModp.astype(jnp.uint32), preferred_element_type=jnp.uint64)
        else:
            products = c[..., None].astype(jnp.uint64) * QHatModp[None, ...] # Need to convert it into BAT based implementation
            summed_terms = jnp.sum(products, axis=-2)

        # Step 4: Final Modular Reduction using target moduli context
        out_tower = ff_ctx_target.modular_reduction(summed_terms)

        return out_tower

    # @functools.partial(jax.jit, static_argnames=("self",))
    def _basis_change_bat(self, in_tower: jnp.ndarray, control_index: int = 0) -> jnp.ndarray:
        """
        Performs the approximate basis change using BAT optimization.
        Currently does not support modulus switching.

        Input:
            in_tower: Coefficients in original basis.
                      Shape: (..., ring_dim, sizeQ)
            control_index: Index of the control set to use.

        Output:
            out_tower: Coefficients in new basis.
                       Shape: (..., ring_dim, sizeP)
        """

        # Ensure inputs are u64 for BAT
        # Note: We assume inputs fit in u64 (< 2^64)
        in_tower_u64 = jnp.asarray(in_tower, dtype=jnp.uint64)

        # Retrieve constants and contexts
        QHatInvModq = self.QHatInvModq[control_index]
        QHatModpBAT = self.QHatModpBAT[control_index]
        ff_ctx_origin = self.ff_ctx_origin[control_index]
        ff_ctx_target = self.ff_ctx_target[control_index]

        # Step 1: Compute c_unreduced = in_tower * QHatInvModqBAT
        c_unreduced = in_tower_u64 * QHatInvModq

        # Step 2: Modular Reduction
        c = ff_ctx_origin.modular_reduction(c_unreduced).astype(jnp.uint32)

        # QHatModpBAT: (q*a, p, b)
        summed_terms = util.matmul(
            c, QHatModpBAT, "...q,qpb->...pb", flatten_lhs_bytes=True
        )

        # Step 4: Final Modular Reduction
        out_tower = ff_ctx_target.modular_reduction(summed_terms)

        return out_tower


class BConvBarrett(BConv):
    def _safe_guardrail(
        self, original_moduli, target_moduli, control_index
    ):
        """Validate uint64 accumulation and Barrett's exact input range."""
        bat_ok = super()._safe_guardrail(
            original_moduli, target_moduli, control_index
        )
        original_moduli = tuple(int(q) for q in original_moduli)
        target_moduli = tuple(int(p) for p in target_moduli)
        oversized_moduli = [
            q
            for q in original_moduli + target_moduli
            if q >= 1 << 32
        ]
        if oversized_moduli:
            raise ValueError(
                "Barrett basis conversion requires moduli < 2**32; got "
                f"{oversized_moduli}"
            )

        dense_factor = sum(q - 1 for q in original_moduli)
        bat_factor = sum(min(q - 1, 4 * 255) for q in original_moduli)
        dense_ok = all(
            dense_factor * (p - 1)
            < min(1 << 64, 1 << (2 * (p - 1).bit_length()))
            for p in target_moduli
        )
        bat_ok = bat_ok and all(
            bat_factor * (p - 1)
            < min(1 << 64, 1 << (2 * (p - 1).bit_length()))
            for p in target_moduli
        )
        if not (dense_ok or bat_ok):
            raise ValueError(
                f"BConvBarrett control {control_index}: neither dense nor BAT "
                "basis conversion is safe for this configuration"
            )
        return bat_ok

    def _create_contexts(self, original_moduli, target_moduli):
        return (ff_context.BarrettContext(moduli=original_moduli),
                ff_context.BarrettContext(moduli=target_moduli))


class BConvMontgomery(BConv):
    """CRNS basis conversion for Montgomery-format towers.

    For c_i = y_i*R mod q_i, conversion uses

      MontRed_p(sum_i(c_i*E_i) + (Lambda mod p)*g),

    where Lambda = sum_i floor(c_i*R^-1/q_i). The result remains in Montgomery
    form. ``basis_change`` prefers the BAT contraction and uses dense only when
    BAT is outside its checked accumulator range.
    """

    def __init__(self, overall_moduli):
        super().__init__(overall_moduli)
        # CRNS constants, one entry per control index (parallel to
        # QHatInvModq / QHatModp in the base class).
        self.crns_rho = []  # (sizeQ,) u64: R^-1 mod q_i
        self.crns_g = []    # (sizeP,) u64: (-Q * R^2) mod p_j
        self.crns_lambda_bctx = []  # BarrettContext(q): exact lambda quotients
        self.crns_corr_bctx = []    # BarrettContext(p): exact Lambda mod p_j

    def _create_contexts(self, original_moduli, target_moduli):
        return (ff_context.MontgomeryContext(moduli=original_moduli),
                ff_context.MontgomeryContext(moduli=target_moduli))

    def _safe_guardrail(
        self, original_moduli, target_moduli, control_index
    ):
        """Validate the CRNS and Montgomery accumulator bounds."""
        bat_ok = super()._safe_guardrail(
            original_moduli, target_moduli, control_index
        )

        original_moduli = tuple(int(q) for q in original_moduli)
        target_moduli = tuple(int(p) for p in target_moduli)
        all_moduli = original_moduli + target_moduli
        oversized_moduli = [q for q in all_moduli if q >= 1 << 31]
        if oversized_moduli:
            raise ValueError(
                "Montgomery reduction requires moduli < 2**31; got "
                f"{oversized_moduli}"
            )

        sizeQ = len(original_moduli)
        q_max = max(original_moduli)

        # Lambda <= sizeQ*(max(q)-1) must stay below each target's exact
        # Barrett range 2^(2*ceil(log2(p))). For positive integer p,
        # (p-1).bit_length() computes ceil(log2(p)) exactly without floats.
        lambda_max = sizeQ * (q_max - 1)
        bad_correction_moduli = [
            p
            for p in target_moduli
            if lambda_max >= 1 << (2 * (p - 1).bit_length())
        ]
        if bad_correction_moduli:
            raise ValueError(
                "CRNS quotient correction exceeds the exact Barrett range of "
                f"target moduli {bad_correction_moduli}: need sizeQ * "
                "max(q) < 2^(2*ceil(log2 p_j))"
            )

        dense_ok = all(
            sizeQ * (q_max - 1) * (p - 1) + (p - 1) ** 2
            < (1 << 32) * ((1 << 32) - p)
            for p in target_moduli
        )

        bat_ok = bat_ok and all(
            4 * sizeQ * 255 * (p - 1) + (p - 1) ** 2
            < (1 << 32) * ((1 << 32) - p)
            for p in target_moduli
        )
        if not (dense_ok or bat_ok):
            raise ValueError(
                f"BConvMontgomery control {control_index}: neither dense nor "
                "BAT basis conversion is safe for this configuration"
            )
        return bat_ok

    def _reset_control_state(self):
        super()._reset_control_state()
        self.crns_rho = []
        self.crns_g = []
        self.crns_lambda_bctx = []
        self.crns_corr_bctx = []

    def control_gen(self, control_indices_list, perf_test=False):
        super().control_gen(control_indices_list, perf_test)
        if perf_test:
            # Normal constant generation populates these CRNS-only fields.
            for om, tm in zip(
                self.original_moduli, self.target_moduli, strict=True
            ):
                self.crns_rho.append(
                    util.random_parameters((len(om),), om, dtype=jnp.uint64))
                self.crns_g.append(
                    util.random_parameters((len(tm),), tm, dtype=jnp.uint64))
                self.crns_lambda_bctx.append(
                    ff_context.BarrettContext(moduli=om))
                self.crns_corr_bctx.append(
                    ff_context.BarrettContext(moduli=tm))

    def _generate_constants_single(self, original_moduli, target_moduli, ff_ctx_origin, ff_ctx_target):
        # _safe_guardrail preflighted every control before constant generation.
        QHatInvModq_list, _ = util.compute_QHatInvModq_QHatModp(
            original_moduli, target_moduli
        )
        # Montgomery computation format: QHatInv_i * R mod q_i.
        QHatInvModq = jnp.array(QHatInvModq_list, dtype=jnp.uint64)
        QHatInvModq = ff_ctx_origin.to_computation_format(QHatInvModq)

        # CRNS precomputation (Python big ints, exact).
        R = 1 << ff_ctx_origin.w
        R2 = R * R
        Q = math.prod(original_moduli)
        rho_list = ff_ctx_origin.w_inv  # R^-1 mod q_i
        I_list = [rho_i * (Q // q_i)
                  for rho_i, q_i in zip(
                      rho_list, original_moduli, strict=True
                  )]

        E = jnp.array(
            [[(I_i * R2) % p_j for p_j in target_moduli] for I_i in I_list],
            dtype=jnp.uint64,
        )
        g = jnp.array([(-Q * R2) % p_j for p_j in target_moduli],
                      dtype=jnp.uint64)

        self.crns_rho.append(jnp.array(rho_list, dtype=jnp.uint64))
        self.crns_g.append(g)

        # Barrett reciprocal contexts for the division-free exact quotients;
        # exactness ranges are enforced by _safe_guardrail.
        lambda_bctx = ff_context.BarrettContext(moduli=original_moduli)
        corr_bctx = ff_context.BarrettContext(moduli=target_moduli)
        self.crns_lambda_bctx.append(lambda_bctx)
        self.crns_corr_bctx.append(corr_bctx)

        # BAT (MXU) version of E for basis_change_bat: byte-position weights
        # 2^(8a) folded into the constants mod p_j, exactly like QHatModpBAT.
        E_BAT_raw = self._basis_aligned_transformation(E, target_moduli)
        E_BAT = E_BAT_raw.transpose(1, 0, 2, 3).reshape(
            -1, E_BAT_raw.shape[2], 4)

        # E takes the QHatModp slot, its BAT form the QHatModpBAT slot.
        return QHatInvModq, E, E_BAT

    def _basis_change_dense(self, in_tower: jnp.ndarray, control_index: int = 0) -> jnp.ndarray:
        """Approximate basis change on Montgomery-format towers.

        Input:
            in_tower: Montgomery-format coefficients (x_i * R mod q_i) in the
                      original basis. Shape: (..., ring_dim, sizeQ), u32 range.
            control_index: Index of the control set to use.

        Output:
            out_tower: Montgomery-format coefficients in the target basis,
                       lazy (congruent to S * R mod p_j).
                       Shape: (..., ring_dim, sizeP)
        """
        in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

        QHatInvModq = self.QHatInvModq[control_index]
        E = self.QHatModp[control_index]
        ff_ctx_origin = self.ff_ctx_origin[control_index]
        ff_ctx_target = self.ff_ctx_target[control_index]

        # Step 1+2: Montgomery digits c_i = y_i * R mod q_i, made strict.
        c = self._montgomery_digits(in_tower, QHatInvModq, ff_ctx_origin)

        # CRNS quotient (paper steps 10-11, exact per-digit variant).
        c64 = c.astype(jnp.uint64)
        lam_sum = self._lambda_sum(c64, control_index)

        # CRNS conversion (paper step 12): x_N = x_M * E + k * g.
        if _is_nvidia:
            summed_terms = jnp.einsum("...dq,qp->...dp", c.astype(jnp.uint32), E.astype(jnp.uint32), preferred_element_type=jnp.uint64)
        else:
            products = c64[..., None] * E[None, ...]
            summed_terms = jnp.sum(products, axis=-2)
        summed_terms = summed_terms + self._lambda_correction(
            lam_sum, control_index)

        # Montgomery reduction folds away one R factor of R^2, leaving the
        # target-basis result in Montgomery form (lazy).
        out_tower = ff_ctx_target.modular_reduction(summed_terms)

        return out_tower

    def _montgomery_digits(self, in_tower, QHatInvModq, ff_ctx_origin):
        """Strict digits c_i = y_i*R mod q_i in [0, q_i): strictness is what the envelope digit bound (c_i <= q_i-1) and the exact-quotient range (c_i*rho_i < 2^s) are proven against; the CRNS result itself is invariant to q_i-shifted digits (q_i*I_i = rho_i*Q cancels in Lambda)."""
        c_unreduced = in_tower * QHatInvModq
        c = ff_ctx_origin.modular_reduction(c_unreduced)
        return jnp.where(c >= ff_ctx_origin.q, c - ff_ctx_origin.q, c)

    def _lambda_sum(self, c64, control_index):
        """Exact CRNS quotient Lambda = sum_i floor(c_i*rho_i/q_i), division-free; exact because c_i*rho_i < q_i^2 <= 2^s."""
        rho = self.crns_rho[control_index]
        return jnp.sum(
            _barrett_exact_quotient(
                c64 * rho, self.crns_lambda_bctx[control_index]),
            axis=-1,
        )

    def _lambda_correction(self, lam_sum, control_index):
        """(Lambda mod p_j) * g_j, division-free (< p_j^2, fits u64)."""
        g = self.crns_g[control_index]
        lam_mod_p = _barrett_exact_mod(
            lam_sum[..., None], self.crns_corr_bctx[control_index])
        return lam_mod_p * g

    def _basis_change_bat(self, in_tower: jnp.ndarray, control_index: int = 0) -> jnp.ndarray:
        """Approximate basis change on Montgomery towers, contraction on MXU.

        Same math and identical canonical residues as basis_change; only the
        sizeQ x sizeP contraction is replaced by the uint8 BAT einsum. The
        correction (Lambda mod p_j)*g_j is additive after the matmul (never
        rides the u8 matmul); the BAT representative < 4*sizeQ*255*p_j plus
        correction < p_j^2 stays inside the envelope checked at control_gen.
        I/O contracts match basis_change (Montgomery form; output lazy).
        """
        in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

        QHatInvModq = self.QHatInvModq[control_index]
        E_BAT = self.QHatModpBAT[control_index]
        ff_ctx_origin = self.ff_ctx_origin[control_index]
        ff_ctx_target = self.ff_ctx_target[control_index]

        c = self._montgomery_digits(in_tower, QHatInvModq, ff_ctx_origin)
        lam_sum = self._lambda_sum(c.astype(jnp.uint64), control_index)

        summed_terms = util.matmul(
            c.astype(jnp.uint32), E_BAT, "...q,qpb->...pb",
            flatten_lhs_bytes=True,
        )
        summed_terms = summed_terms + self._lambda_correction(
            lam_sum, control_index)

        return ff_ctx_target.modular_reduction(summed_terms)


class BConvBATLazy(BConv):
    def _safe_guardrail(
        self, original_moduli, target_moduli, control_index
    ):
        """Validate the u64 inputs consumed by BAT-lazy target reduction."""
        bat_ok = super()._safe_guardrail(
            original_moduli, target_moduli, control_index
        )
        original_moduli = tuple(int(q) for q in original_moduli)
        target_moduli = tuple(int(p) for p in target_moduli)
        oversized_moduli = [
            q
            for q in original_moduli + target_moduli
            if q >= 1 << 32
        ]
        if oversized_moduli:
            raise ValueError(
                "BAT-lazy basis conversion requires moduli < 2**32; got "
                f"{oversized_moduli}"
            )

        dense_factor = sum(q - 1 for q in original_moduli)
        bat_factor = sum(min(q - 1, 4 * 255) for q in original_moduli)
        dense_ok = all(
            dense_factor * (p - 1) < 1 << 64
            for p in target_moduli
        )
        bat_ok = bat_ok and all(
            bat_factor * (p - 1) < 1 << 64
            for p in target_moduli
        )
        if not (dense_ok or bat_ok):
            raise ValueError(
                f"BConvBATLazy control {control_index}: neither dense nor BAT "
                "basis conversion is safe for this configuration"
            )
        return bat_ok

    def _create_contexts(self, original_moduli, target_moduli):
        return (ff_context.BATLazyContext(moduli=original_moduli),
                ff_context.BATLazyContext(moduli=target_moduli))

    def _basis_change_dense(self, in_tower: jnp.ndarray, control_index: int = 0) -> jnp.ndarray:
        in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

        QHatInvModq = self.QHatInvModq[control_index]
        QHatModp = self.QHatModp[control_index]
        ff_ctx_origin = self.ff_ctx_origin[control_index]
        ff_ctx_target = self.ff_ctx_target[control_index]

        c_unreduced = in_tower * QHatInvModq
        c = ff_ctx_origin.modular_reduction(c_unreduced)

        # Force strict reduction for BATLazy correctness
        c = ff_ctx_origin.to_original_format(c)

        if _is_nvidia:
            summed_terms = jnp.einsum("...dq,qp->...dp", c.astype(jnp.uint32), QHatModp.astype(jnp.uint32), preferred_element_type=jnp.uint64)
        else:
            products = c[..., None].astype(jnp.uint64) * QHatModp[None, ...] # Need to convert it into BAT based implementation
            summed_terms = jnp.sum(products, axis=-2)

        out_tower = ff_ctx_target.modular_reduction(summed_terms)
        return out_tower

    def _basis_change_bat(self, in_tower: jnp.ndarray, control_index: int = 0) -> jnp.ndarray:
        in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

        QHatInvModq = self.QHatInvModq[control_index]
        QHatModpBAT = self.QHatModpBAT[control_index]
        ff_ctx_origin = self.ff_ctx_origin[control_index]
        ff_ctx_target = self.ff_ctx_target[control_index]

        c_unreduced = in_tower * QHatInvModq
        # Strictify the lazy wide result before narrowing it. Casting first can
        # discard high bits that BATLazyContext.to_original_format still needs.
        c = ff_ctx_origin.to_original_format(
            ff_ctx_origin.modular_reduction(c_unreduced)
        ).astype(jnp.uint32)

        summed_terms = util.matmul(
            c, QHatModpBAT, "...q,qpb->...pb", flatten_lhs_bytes=True
        )
        out_tower = ff_ctx_target.modular_reduction(summed_terms)
        return out_tower


def make_bconv(finite_field_context, overall_moduli):
  """Return the BConv subclass named by the reduction context's bconv_cls hook; raise TypeError if it declares none (e.g. ShoupContext)."""
  # getattr reads the class attr through either a class or an instance.
  name = getattr(finite_field_context, "bconv_cls", None)
  if name is None:
    ctx_name = getattr(
        finite_field_context, "__name__", type(finite_field_context).__name__)
    raise TypeError(f"{ctx_name} declares no bconv_cls hook and cannot back a BConv")
  return globals()[name](overall_moduli)
