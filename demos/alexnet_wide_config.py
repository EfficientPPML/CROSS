"""Path A config + cleartext model for the reference-width AlexNet demo.

Path A restores the ORIGINAL AlexNet channel widths ([64, 192, 384, 256, 256])
— rather than the narrow surrogate the shipped `AlexNetHE` uses (8, 16, 32, 32,
32) — while keeping the 16×16 downsampled input. The central claim of Path A is
that every layer's materialized activations still fit in ONE CKKS ciphertext at
a cryptographically secure ring: the demo runs each layer as a single
`num_slots`-slot ciphertext (`num_slots = degree/2`), and the widest layer
(conv3, 384·4·4 = 6144 slots) fits under `num_slots` for both a ~100-bit ring
(degree 16384 → 8192 slots) and a 128-bit ring (degree 32768 → 16384 slots).

This module ships the config + the reference-width CLEARTEXT forward + the
one-ciphertext-fit proof. The HE build (wiring these constants into a
parameterized `AlexNetHE.precompute_plaintexts`/`infer`/`from_cache`) is the
next increment and is intentionally out of scope here: a full secure-ring HE
build is a ~8-min-keygen offline job. A single degree-16384 HE BSGS matvec was
verified feasible; keygen (~8 min for the ~190-key balanced rotation set) is a
one-time cost shared across all layers — rotation-key generation dominates, not
multiplicative depth.

The reference-width cleartext forward reuses the EXISTING dim-parameterized
primitives in `alexnet_he.py` (`conv2d_ref`, `add_conv_bias`, `avg_pool_2x2`,
`adaptive_avg_pool_2x2`, `quad_ref`, `matmul_ref`) so there is exactly one
implementation of each op; only the shape constants differ from
`alexnet_full_cleartext`.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from alexnet_he import (
    conv2d_ref,
    add_conv_bias,
    avg_pool_2x2,
    adaptive_avg_pool_2x2,
    matmul_ref,
    quad_ref,
    check_p_tower_coverage,
)

# ===========================================================================
# Reference-width AlexNet architecture constants (16×16 input, ONE ciphertext).
#
# Original AlexNet channel widths — the shipped "full" demo narrows these to
# (8, 16, 32, 32, 32) so it fits NUM_SLOTS=1024 at degree 2048. Path A keeps the
# reference widths and instead widens the ring so the worst layer (6144 slots)
# still fits one ciphertext.
#
# Spatial schedule (square, so W == H everywhere):
#   16 →conv1+AvgPool2→ 8 →conv2+AvgPool2→ 4 →conv3→ 4 →conv4→ 4
#      →conv5+AdaptivePool→ 2
# Every conv is 3×3, stride 1, pad 1 (spatial-preserving); only the pools shrink.
# ===========================================================================
WIDE_CI = 3
WIDE_H_IN = 16        # 32 → 16 via client-side avg-pool 2×2 (as in the full demo)
WIDE_W_IN = 16
WIDE_CO1 = 64
WIDE_CO2 = 192
WIDE_CO3 = 384
WIDE_CO4 = 256
WIDE_CO5 = 256
WIDE_CO = [WIDE_CO1, WIDE_CO2, WIDE_CO3, WIDE_CO4, WIDE_CO5]
WIDE_KH = 3
WIDE_KW = 3
WIDE_PAD = 1
WIDE_POOL_K = 2
# Per-conv pooling: AvgPool2 after conv1 & conv2, none after conv3 & conv4,
# adaptive-2×2 after conv5. Drives the spatial dims below (see _wide_spatial).
WIDE_POOL = ["avg2", "avg2", None, None, "adaptive2"]
# Spatial sizes after each conv (+pool). Derived from WIDE_POOL by _wide_spatial;
# spelled out here for the cleartext forward's explicit dim arguments.
WIDE_H1 = WIDE_W1 = 8         # post-Conv1 + AvgPool2      (64 channels)
WIDE_H2 = WIDE_W2 = 4         # post-Conv2 + AvgPool2      (192 channels)
WIDE_H3 = WIDE_W3 = 4         # post-Conv3 (no pool)       (384 channels)
WIDE_H4 = WIDE_W4 = 4         # post-Conv4 (no pool)       (256 channels)
WIDE_H5 = WIDE_W5 = 2         # post-Conv5 + AdaptivePool  (256 channels)
# FC head: 1024 → 4096 → 4096 → 10 (reference widths, NOT the narrow 128→64→32→10).
WIDE_FC1_IN = WIDE_CO5 * WIDE_H5 * WIDE_W5    # 256·2·2 = 1024
WIDE_FC1_OUT = 4096
WIDE_FC2_OUT = 4096
WIDE_FC3_OUT = 10


def _wide_spatial() -> list[int]:
    """Post-conv spatial size (H == W) for each of the 5 conv layers.

    Convs are spatial-preserving (3×3, s=1, p=1); only `WIDE_POOL` shrinks the
    map. Computed from the pooling schedule so the sizes stay correct if the
    input resolution or pool schedule changes.
    """
    h = WIDE_H_IN
    dims: list[int] = []
    for pool in WIDE_POOL:
        if pool == "avg2":
            h = h // 2
        elif pool == "adaptive2":
            h = 2
        elif pool is not None:
            raise ValueError(f"unknown pool spec {pool!r}")
        dims.append(h)
    return dims


def wide_activation_sizes() -> list[tuple[str, int]]:
    """Ordered (layer_name, materialized_slot_count) for the reference-width net.

    Slots are the post-pool activation counts a single-ciphertext demo must
    hold. Computed from the channel/spatial/FC constants (not hardcoded) so the
    table tracks any change to `WIDE_CO` or the pooling schedule.
    """
    dims = _wide_spatial()  # [8, 4, 4, 4, 2]
    sizes: list[tuple[str, int]] = [
        ("input", WIDE_CI * WIDE_H_IN * WIDE_W_IN),
    ]
    for i, (co, s) in enumerate(zip(WIDE_CO, dims, strict=True), start=1):
        sizes.append((f"conv{i}", co * s * s))
    sizes.append(("fc1", WIDE_FC1_OUT))
    sizes.append(("fc2", WIDE_FC2_OUT))
    sizes.append(("fc3", WIDE_FC3_OUT))
    return sizes


# ===========================================================================
# NTT-friendly modulus chains for the two rings.
#
# Each chain is 17 Q-towers (bit_length 30, ~2^29 — matching the DEEP pool's
# ~536M primes) + 4 P-towers (bit_length 32, ~2^31), targeting the depth-15
# schedule's log2(Q·P) ≈ 617 bits. Every prime is ≡ 1 (mod 2·degree) so it
# admits a negacyclic NTT at that ring degree.
#
# GENERATION (deterministic, reproducible): for a given degree, enumerate the
# candidates k·(2·degree)+1 with k increasing from the smallest value that makes
# the candidate a `bits`-bit number (2^(bits-1) ≤ cand < 2^bits), and keep the
# first 17 (resp. 4) that are prime (sympy.isprime). See the module-adjacent
# comment for the exact script. The literals are pasted below so importing this
# module is fast and reproducible — the primes are NOT regenerated at import.
#
#   from sympy import isprime
#   def first_primes(count, bits, M):
#       lo, hi = 1 << (bits - 1), 1 << bits
#       k = (lo + M - 1) // M
#       out = []
#       while len(out) < count:
#           c = k * M + 1
#           if lo <= c < hi and c.bit_length() == bits and isprime(c):
#               out.append(c)
#           k += 1
#       return out
#   # Q = first_primes(17, 30, 2*degree); P = first_primes(4, 32, 2*degree)
# ===========================================================================

# degree 16384  (M = 2·degree = 32768):  log2(Q·P) ≈ 617.10, dnum=5 coverage OK.
WIDE_16384_Q_TOWERS = [
    536903681, 537133057, 537296897, 537591809, 537722881,
    538116097, 538411009, 538673153, 539394049, 539754497,
    540082177, 540540929, 540639233, 540672001, 540966913,
    541327361,
    541655041,   # 17th tower — headroom (mirrors the DEEP pool's 1 extra level)
]
WIDE_16384_P_TOWERS = [
    2148728833, 2148794369, 2149810177, 2150072321,
]

# degree 32768  (M = 2·degree = 65536):  log2(Q·P) ≈ 617.23, dnum=5 coverage OK.
WIDE_32768_Q_TOWERS = [
    537133057, 537591809, 537722881, 538116097, 539754497,
    540082177, 540540929, 540672001, 541327361, 541655041,
    542310401, 543031297, 543293441, 545062913, 546766849,
    547749889,
    548012033,   # 17th tower — headroom
]
WIDE_32768_P_TOWERS = [
    2148728833, 2148794369, 2150301697, 2150563841,
]

# CKKS scaling factor: product of the first two Q-towers (~2^58), matching the
# full demo's FULL_SF = q[0]·q[1] two-tower scale. sigma / dnum mirror AlexNetHE.
WIDE_SIGMA = 3.190000057220458984375
WIDE_DNUM = 5


RingConfig = namedtuple(
    "RingConfig",
    ["name", "degree", "num_slots", "q_towers", "p_towers",
     "scaling_factor", "sigma", "dnum"],
)


def _make_config(name, degree, q_towers, p_towers) -> RingConfig:
    return RingConfig(
        name=name,
        degree=degree,
        num_slots=degree // 2,
        q_towers=list(q_towers),
        p_towers=list(p_towers),
        scaling_factor=int(q_towers[0]) * int(q_towers[1]),
        sigma=WIDE_SIGMA,
        dnum=WIDE_DNUM,
    )


# ~100-bit ring: degree 16384 → 8192 slots. Holds the 6144-slot worst layer but
# a 617-bit chain here is only ~100-bit secure (HES128 ceiling ~438 < 617).
WIDE_16384 = _make_config(
    "WIDE_16384", 16384, WIDE_16384_Q_TOWERS, WIDE_16384_P_TOWERS)

# 128-bit ring: degree 32768 → 16384 slots. Holds the 6144-slot worst layer AND
# a 617-bit chain is 128-bit secure here (HES128 ceiling ~881 > 617).
WIDE_32768 = _make_config(
    "WIDE_32768", 32768, WIDE_32768_Q_TOWERS, WIDE_32768_P_TOWERS)


def assert_fits_one_ciphertext(config: RingConfig) -> int:
    """Assert every reference-width layer fits one `config.num_slots` ciphertext.

    Returns the worst (largest) layer slot count — 6144 (conv3) for the
    reference widths. Raises AssertionError on the first layer that overflows.
    """
    worst = 0
    for name, slots in wide_activation_sizes():
        if slots > config.num_slots:
            raise AssertionError(
                f"{config.name}: layer {name!r} needs {slots} slots > "
                f"num_slots={config.num_slots} (degree {config.degree})"
            )
        worst = max(worst, slots)
    return worst


# ---------------------------------------------------------------------------
# Build-time invariants: fail loudly at IMPORT if a pasted chain drifted from
# the coverage / fit guarantees (cheap integer checks, no prime regeneration).
# ---------------------------------------------------------------------------
for _cfg in (WIDE_16384, WIDE_32768):
    # Hybrid key-switch P-tower coverage (largest dnum=5 Q-group ≤ product(P)).
    check_p_tower_coverage(_cfg.q_towers, _cfg.p_towers, _cfg.dnum)
    # One-ciphertext fit for the reference-width activations.
    assert_fits_one_ciphertext(_cfg)
del _cfg


# ===========================================================================
# Reference-width cleartext forward. Mirrors `alexnet_full_cleartext`'s
# structure exactly (same primitives, same op order) but with the reference
# channel/dim constants above — so it is the functional ground truth for the
# Path A HE build.
# ===========================================================================
def alexnet_wide_cleartext(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5,
                           Wf1, bf1, Wf2, bf2, Wf3, bf3):
    """Cleartext forward for the reference-width depth-15 AlexNet.

    Input X: flat (3·16·16 = 768,) — already client-side downsampled. All convs
    use stride=1, pad=1, kernel=3. Conv1 / Conv2 each pool 2×2 (s=2). Conv5 uses
    adaptive 2×2 average pool (4→2 on the spatial dim). Each conv/FC is followed
    by a Quad (x²) except the final FC3. Returns the (10,) logit vector.
    """
    # Conv1 + bias + AvgPool2x2 → (64, 8, 8)
    y = conv2d_ref(X, W1, WIDE_CI, WIDE_CO1, WIDE_H_IN, WIDE_W_IN,
                   WIDE_KH, WIDE_KW, 1, WIDE_PAD)
    y = add_conv_bias(y, b1, WIDE_CO1, WIDE_H_IN, WIDE_W_IN)
    y = avg_pool_2x2(y, WIDE_CO1, WIDE_H_IN, WIDE_W_IN)
    y = quad_ref(y)
    # Conv2 + bias + AvgPool2x2 → (192, 4, 4)
    y = conv2d_ref(y, W2, WIDE_CO1, WIDE_CO2, WIDE_H1, WIDE_W1,
                   WIDE_KH, WIDE_KW, 1, WIDE_PAD)
    y = add_conv_bias(y, b2, WIDE_CO2, WIDE_H1, WIDE_W1)
    y = avg_pool_2x2(y, WIDE_CO2, WIDE_H1, WIDE_W1)
    y = quad_ref(y)
    # Conv3 + bias → (384, 4, 4)   ← worst layer: 6144 slots
    y = conv2d_ref(y, W3, WIDE_CO2, WIDE_CO3, WIDE_H2, WIDE_W2,
                   WIDE_KH, WIDE_KW, 1, WIDE_PAD)
    y = add_conv_bias(y, b3, WIDE_CO3, WIDE_H2, WIDE_W2)
    y = quad_ref(y)
    # Conv4 + bias → (256, 4, 4)
    y = conv2d_ref(y, W4, WIDE_CO3, WIDE_CO4, WIDE_H3, WIDE_W3,
                   WIDE_KH, WIDE_KW, 1, WIDE_PAD)
    y = add_conv_bias(y, b4, WIDE_CO4, WIDE_H3, WIDE_W3)
    y = quad_ref(y)
    # Conv5 + bias + AdaptiveAvgPool2x2 → (256, 2, 2)
    y = conv2d_ref(y, W5, WIDE_CO4, WIDE_CO5, WIDE_H4, WIDE_W4,
                   WIDE_KH, WIDE_KW, 1, WIDE_PAD)
    y = add_conv_bias(y, b5, WIDE_CO5, WIDE_H4, WIDE_W4)
    y = adaptive_avg_pool_2x2(y, WIDE_CO5, WIDE_H4, WIDE_W4,
                              out_h=WIDE_H5, out_w=WIDE_W5)
    y = quad_ref(y)
    # FC1 (1024 → 4096)
    y = matmul_ref(y, Wf1, WIDE_FC1_OUT, WIDE_FC1_IN) + bf1[:WIDE_FC1_OUT]
    y = quad_ref(y)
    # FC2 (4096 → 4096)
    y = matmul_ref(y, Wf2, WIDE_FC2_OUT, WIDE_FC1_OUT) + bf2[:WIDE_FC2_OUT]
    y = quad_ref(y)
    # FC3 (4096 → 10)
    y = matmul_ref(y, Wf3, WIDE_FC3_OUT, WIDE_FC2_OUT) + bf3[:WIDE_FC3_OUT]
    return y


def alexnet_wide_random_inputs(seed: int = 42):
    """Random Gaussian inputs + weights matching the reference-width architecture.

    Weight vectors are flat, sized CO·CI·Kh·Kw (conv) / n_out·k_in (FC), exactly
    as `alexnet_full_random_inputs` does — ready for `prepare_wide_args`.
    """
    rng = np.random.default_rng(seed)
    return {
        "X":   rng.normal(0.0, 0.1,
                          size=WIDE_CI * WIDE_H_IN * WIDE_W_IN
                          ).astype(np.float64),
        "W1":  rng.normal(0.0, 0.1, size=WIDE_CO1 * WIDE_CI  * WIDE_KH * WIDE_KW).astype(np.float64),
        "W2":  rng.normal(0.0, 0.1, size=WIDE_CO2 * WIDE_CO1 * WIDE_KH * WIDE_KW).astype(np.float64),
        "W3":  rng.normal(0.0, 0.1, size=WIDE_CO3 * WIDE_CO2 * WIDE_KH * WIDE_KW).astype(np.float64),
        "W4":  rng.normal(0.0, 0.1, size=WIDE_CO4 * WIDE_CO3 * WIDE_KH * WIDE_KW).astype(np.float64),
        "W5":  rng.normal(0.0, 0.1, size=WIDE_CO5 * WIDE_CO4 * WIDE_KH * WIDE_KW).astype(np.float64),
        "Wf1": rng.normal(0.0, 0.1, size=WIDE_FC1_OUT * WIDE_FC1_IN).astype(np.float64),
        "Wf2": rng.normal(0.0, 0.1, size=WIDE_FC2_OUT * WIDE_FC1_OUT).astype(np.float64),
        "Wf3": rng.normal(0.0, 0.1, size=WIDE_FC3_OUT * WIDE_FC2_OUT).astype(np.float64),
        "b1":  np.zeros(WIDE_CO1, dtype=np.float64),
        "b2":  np.zeros(WIDE_CO2, dtype=np.float64),
        "b3":  np.zeros(WIDE_CO3, dtype=np.float64),
        "b4":  np.zeros(WIDE_CO4, dtype=np.float64),
        "b5":  np.zeros(WIDE_CO5, dtype=np.float64),
        "bf1": np.zeros(WIDE_FC1_OUT, dtype=np.float64),
        "bf2": np.zeros(WIDE_FC2_OUT, dtype=np.float64),
        "bf3": np.zeros(WIDE_FC3_OUT, dtype=np.float64),
    }


def prepare_wide_args(d: dict) -> tuple:
    """Pack a weight dict into the positional args for `alexnet_wide_cleartext`."""
    return (d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"],
            d["W4"], d["b4"], d["W5"], d["b5"],
            d["Wf1"], d["bf1"], d["Wf2"], d["bf2"], d["Wf3"], d["bf3"])
