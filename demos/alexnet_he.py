"""CROSS HE-friendly AlexNet for CIFAR-10 — architecture + cleartext + HE.

The two encrypted entrypoints here, `AlexNetTinyHE` and `AlexNetHE`, no
longer build a packing graph. They drive the canonical pipeline::

    torch.nn.Module  ->  nn.vectorize  ->  packing.pack  ->  Mapping

through `encrypted_demos.AlexNetTinyDemo` and `encrypted_demos.AlexNetFullDemo`.
The source of model truth is the torch module in `alexnet_infer`; the ring is
derived from the packed program rather than picked by hand, so both variants
now run on a 128-bit-secure chain instead of the research-grade degree-2048
pool this module used to hard-code.

This module ships TWO variants:

────────────────────────────────────────────────────────────────────────────
1. AlexNetTiny — depth-7.

    Input X: (3, 32, 32) → client-side avg-pool 2x2 → (3, 16, 16)
    Conv1(3→4, 3×3/s=1/p=1) + AvgPool 2×2 → (4, 8, 8)
    Quad
    Conv2(4→8, 3×3/s=1/p=1) + AvgPool 2×2 → (8, 4, 4)
    Quad
    Conv3(8→16,3×3/s=1/p=1) + AdaptivePool 2×2 → (16, 2, 2)
    Quad
    Linear(64→10)

Multiplicative depth: 7 levels (4 linears + 3 quads). Pool layers are linear
and fuse into the preceding conv's matvec, so they cost no depth.

Cleartext: `alexnet_tiny_cleartext`. HE: `AlexNetTinyHE`.

────────────────────────────────────────────────────────────────────────────
2. AlexNet (full) — depth-15, following the OpenFHE reference architecture
   from `openfhe_ref_code/models/alexnet.py` (5 conv blocks + 3 FC blocks).
   Channels are scaled DOWN from the original, which targets ImageNet-class
   inputs:

    Input X: (3, 32, 32) → client-side avg-pool 2x2 → (3, 16, 16)
    Conv1(3→8,  3×3/s=1/p=1) + AvgPool 2×2 → (8,  8, 8)
    Quad
    Conv2(8→16, 3×3/s=1/p=1) + AvgPool 2×2 → (16, 4, 4)
    Quad
    Conv3(16→32, 3×3/s=1/p=1)              → (32, 4, 4)
    Quad
    Conv4(32→32, 3×3/s=1/p=1)              → (32, 4, 4)
    Quad
    Conv5(32→32, 3×3/s=1/p=1) + AdaptivePool 2×2 → (32, 2, 2)
    Quad
    FC1(128→64)  Quad  FC2(64→32)  Quad  FC3(32→10)

Multiplicative depth: 15 levels (8 linears + 7 quads).

Cleartext: `alexnet_full_cleartext`. HE: `AlexNetHE`.

────────────────────────────────────────────────────────────────────────────
The 32→16 downsample stays a CLIENT-side average pool, outside encryption:
the encrypted model starts at (3, 16, 16). Doing it under encryption would
cost a level and a ring four times wider for a step the client does for free.

Both variants are trained via `alexnet_train.py` (Tiny by default, `--full`
for the depth-15 variant). Only BatchNorm-folded weights are loaded — the
inference modules in `alexnet_infer` carry no BatchNorm to reconstruct.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DEMO_DIR)
_JAXITE = os.path.abspath(os.path.join(_DEMO_DIR, "..", "jaxite_word"))
for p in (_REPO_ROOT, _JAXITE, _DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from cleartext_ops import adaptive_avg_pool_2x2, add_conv_bias
from cleartext_ops import avg_pool_2x2, conv2d_ref, matmul_ref, quad_ref
import canonical_demo

_STATIC_MODEL_CACHE_VERSION = 3


# CIFAR-10 normalisation (matches alexnet_train.py).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def normalize_cifar10_image(x: np.ndarray) -> np.ndarray:
    """Per-channel `(x - mean) / std`. Input/output: flat (3*32*32,) in float."""
    x = x.reshape(3, 32, 32).copy()
    for c in range(3):
        x[c] = (x[c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
    return x.ravel()


# ===========================================================================
# Deep modulus pool for AlexNet (full, depth-15 schedule).
#
# HISTORICAL. `AlexNetHE` no longer runs on this pool: `packing.pack` derives
# the ring from the packed program at 128-bit security. These constants and the
# guards below stay because they are the reference parameter set that
# `alexnet_wide_config` / `alexnet_wide_test` study, and because they document
# exactly why hand-picked towers were the wrong place to make this decision.
#
# 17 Q-towers (30-bit, all NTT-friendly: q ≡ 1 mod 4096 at degree=2048) +
# 4 P-towers (31-bit). The first 9 Q-towers are LoLA's existing pool;
# the additional 8 were enumerated by walking 30-bit primes ≡ 1 mod 4096
# upward from the LoLA tail. Smaller primes give better noise budget per
# multiply, so the additions stay close to LoLA's range. 16 towers are
# consumed by the depth-15 schedule; the 17th is headroom (see q[16] below).
#
# Total log2(Q*P) ≈ 17*29 + 4*31 ≈ 617 bits at degree=2048 (the Q primes are
# ~2^29.0, not 2^30). The HE128 ceiling is log2(Q*P) ≤ ~54 bits at degree 2048,
# ≤ 438 at 16384, ≤ 881 at 32768 (HES-2018), so these parameters are
# research-grade only. A 617-bit chain needs ring degree ≥ 32768 for classical
# 128-bit security; degree 16384 (ceiling 438 < 617) gives only ~100-bit.
# ===========================================================================
DEEP_Q_TOWERS_POOL = [
    536903681,   # q[0]   (LoLA's 9-tower pool starts here)
    536924161,   # q[1]
    536952833,   # q[2]
    536973313,   # q[3]
    536977409,   # q[4]
    536989697,   # q[5]
    537026561,   # q[6]
    537047041,   # q[7]
    537071617,   # q[8]   (last of LoLA's 9-tower pool)
    537120769,   # q[9]   (depth-15 extension begins)
    537133057,   # q[10]
    537149441,   # q[11]
    537169921,   # q[12]
    537243649,   # q[13]
    537272321,   # q[14]
    537296897,   # q[15]  (last consumed by the depth-15 schedule)
    537317377,   # q[16]  — HEADROOM tower: makes max_level=16 so the depth-15
                 #          schedule ends fc3 at level 1 (2 moduli), not level 0.
                 #          A level-0 output has a single ~2^30 modulus which
                 #          cannot hold the SF~2^58 scale, so encode/decode there
                 #          fails ("composite_degree must be < number of moduli").
                 #          Mirrors LeNet/AlexNetTiny's 1-level headroom.
]
DEEP_P_TOWERS_POOL = [
    2147565569,
    2147573761,
    2147577857,
    2147721217,
]


def check_p_tower_coverage(q_towers, p_towers, dnum: int) -> None:
    """Validate that the P-towers cover the largest hybrid key-switch group.

    Hybrid (OpenFHE-style) key switching partitions the Q-towers into
    `ceil(len(Q)/dnum)`-sized CONSECUTIVE groups (matching
    `key_gen.gen_evaluation_key`) and raises the key by P inside each group.
    For rotations/relinearization to be correct the product of every group's
    Q-towers must be <= product(P); otherwise the raised modulus overflows and
    the automorphism produces garbage. This is a PRODUCT check, not a count
    check — towers vary in bit-width, so counting towers is not sufficient.

    Raises:
        ValueError: if any group's Q product exceeds product(P).
    """
    import math as _math
    size_q = len(q_towers)
    if dnum < 1:
        raise ValueError(f"dnum must be >= 1, got {dnum}")
    num_per_part_q = (size_q + dnum - 1) // dnum          # ceil(size_q / dnum)
    num_part_q = _math.ceil(size_q / num_per_part_q)
    p_prod = 1
    for p in p_towers:
        p_prod *= int(p)
    worst_group = 0
    worst_prod = 1
    for part in range(num_part_q):
        start = num_per_part_q * part
        end = min(size_q, start + num_per_part_q)
        prod = 1
        for qi in q_towers[start:end]:
            prod *= int(qi)
        if prod > worst_prod:
            worst_prod, worst_group = prod, part
    if worst_prod > p_prod:
        raise ValueError(
            f"dnum={dnum} is too small for {size_q} Q-towers: the largest "
            f"key-switch group (part {worst_group}, {num_per_part_q} towers, "
            f"~2^{worst_prod.bit_length()}) exceeds the {len(p_towers)} "
            f"P-towers (~2^{p_prod.bit_length()}). Rotations would be garbage. "
            f"Increase dnum to >= "
            f"{_min_dnum_for_coverage(q_towers, p_prod)} (so each group's Q "
            f"product stays <= product(P)).")


# HES-2018 (classical, uniform-ternary secret) maximum log2(Q*P) for 128-bit
# security, by ring degree. A CKKS chain is 128-bit iff log2(Q*P) <= ceiling.
HES128_MAX_LOG_PQ = {
    1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881,
}


def assert_he128_or_warn(degree, q_towers, p_towers, require_128bit=False):
    """Check a CKKS chain against the HES-2018 128-bit ceiling for its ring degree.

    Classical 128-bit security needs log2(Q*P) <= HES128_MAX_LOG_PQ[degree].
    Raises if require_128bit and the chain is insecure; otherwise warns (so the
    research-grade degree-2048 demos still run).
    """
    import math as _math
    log_pq = sum(_math.log2(int(x)) for x in list(q_towers) + list(p_towers))
    ceiling = HES128_MAX_LOG_PQ.get(degree)
    if ceiling is not None and log_pq <= ceiling:
        return
    secure = [d for d, c in HES128_MAX_LOG_PQ.items() if c >= log_pq]
    need = str(min(secure)) if secure else "> 32768"
    msg = (f"CKKS chain is NOT 128-bit secure: log2(Q*P)={log_pq:.0f} bits at "
           f"ring degree {degree} exceeds the HES128 ceiling "
           f"{ceiling if ceiling is not None else '(untabulated)'}; "
           f"need ring degree >= {need}")
    if require_128bit:
        raise ValueError(msg)
    import warnings as _warnings
    _warnings.warn(msg)


def _min_dnum_for_coverage(q_towers, p_prod: int) -> int:
    """Smallest dnum whose max consecutive-group Q product is <= p_prod."""
    import math as _math
    size_q = len(q_towers)
    for dnum in range(1, size_q + 1):
        num_per_part_q = (size_q + dnum - 1) // dnum
        num_part_q = _math.ceil(size_q / num_per_part_q)
        ok = True
        for part in range(num_part_q):
            start = num_per_part_q * part
            end = min(size_q, start + num_per_part_q)
            prod = 1
            for qi in q_towers[start:end]:
                prod *= int(qi)
            if prod > p_prod:
                ok = False
                break
        if ok:
            return dnum
    return size_q


def cache_config_fingerprint(q_towers, p_towers, max_level, dnum) -> dict:
    """The CKKS-config fields stored in a cache so a stale reload fails loudly."""
    return {
        "q_towers": [int(q) for q in q_towers],
        "p_towers": [int(p) for p in p_towers],
        "max_level": int(max_level),
        "dnum": int(dnum),
    }


def validate_cache_config(cfg: dict, expected_q, expected_p,
                          class_name: str, path: str) -> None:
    """Reject a cache whose CKKS config no longer matches the class constants.

    Pre-fingerprint caches (built before this field existed) lack `q_towers`
    in their config; they are treated as stale with the same clear error, since
    loading a mismatched key/tower layout produces garbage rather than a clean
    failure.

    Raises:
        ValueError: on a missing or mismatched fingerprint.
    """
    stored_q = cfg.get("q_towers")
    stored_p = cfg.get("p_towers")
    exp_q = [int(q) for q in expected_q]
    exp_p = [int(p) for p in expected_p]
    exp_max_level = len(exp_q) - 1  # composite_degree=1 for these models
    if stored_q is None or stored_p is None:
        raise ValueError(
            f"stale cache at {path}: built by a pre-fingerprint {class_name} "
            f"(no q_towers/p_towers in its config). The current {class_name} "
            f"uses {len(exp_q)}Q+{len(exp_p)}P (max_level={exp_max_level}); a "
            f"cache built against different towers loads as garbage. Delete the "
            f"cache and rebuild.")
    if ([int(q) for q in stored_q] != exp_q
            or [int(p) for p in stored_p] != exp_p
            or int(cfg.get("max_level", -1)) != exp_max_level):
        raise ValueError(
            f"stale cache at {path}: built with {len(stored_q)}Q+"
            f"{len(stored_p)}P (max_level={cfg.get('max_level')}, "
            f"dnum={cfg.get('dnum')}); current {class_name} is {len(exp_q)}Q+"
            f"{len(exp_p)}P (max_level={exp_max_level}). Delete the cache and "
            f"rebuild.")


# ===========================================================================
# AlexNet (full, depth-15) architecture constants.
#
# Channels scaled DOWN from the original AlexNet (which uses [64, 192, 384,
# 256, 256] at ImageNet sizes) so every layer fits in NUM_SLOTS=1024 at
# CIFAR-10 input downsampled to 16×16.
# ===========================================================================
FULL_CI = 3
FULL_H_IN = 16        # 32 → 16 via client-side avg-pool 2×2
FULL_W_IN = 16
FULL_CO1 = 8
FULL_CO2 = 16
FULL_CO3 = 32
FULL_CO4 = 32
FULL_CO5 = 32
FULL_KH = 3
FULL_KW = 3
FULL_PAD = 1
FULL_POOL_K = 2
# Spatial sizes after each conv+pool block.
FULL_H1, FULL_W1 = 8, 8        # post-Conv1+pool   (8 channels)
FULL_H2, FULL_W2 = 4, 4        # post-Conv2+pool   (16 channels)
FULL_H3, FULL_W3 = 4, 4        # post-Conv3        (32 channels, no pool)
FULL_H4, FULL_W4 = 4, 4        # post-Conv4        (32 channels, no pool)
FULL_H5, FULL_W5 = 2, 2        # post-Conv5+adaptive-pool (32 channels)
FULL_FC1_IN  = FULL_CO5 * FULL_H5 * FULL_W5   # 32·2·2 = 128
FULL_FC1_OUT = 64
FULL_FC2_OUT = 32
FULL_FC3_OUT = 10


# ===========================================================================
# AlexNetTiny — depth-7 variant that runs on CROSS's existing infrastructure.
# Identical CKKS schedule as LeNet (max_level=8, 1 level headroom, dnum=4).
# ===========================================================================

# Architecture (after client-side avg-pool 2x2 to fit NUM_SLOTS=1024).
TINY_CI = 3
TINY_H_IN = 16        # 32 → 16 via client-side avg-pool 2×2
TINY_W_IN = 16
TINY_CO1 = 4          # Conv1 output channels
TINY_CO2 = 8          # Conv2 output channels
TINY_CO3 = 16         # Conv3 output channels
TINY_KH = 3
TINY_KW = 3
TINY_PAD = 1
TINY_POOL_K = 2       # AvgPool kernel
# Spatial sizes after each conv+pool block.
TINY_H1, TINY_W1 = 8, 8     # post-Conv1+pool
TINY_H2, TINY_W2 = 4, 4     # post-Conv2+pool
TINY_H3, TINY_W3 = 2, 2     # post-Conv3+adaptive-pool (4→2)
TINY_FC_IN = TINY_CO3 * TINY_H3 * TINY_W3   # 16·2·2 = 64
TINY_FC_OUT = 10


# Trained-weight loader for the tiny variant.
_TRAINED_DIR = os.path.join(_DEMO_DIR, "maple_data")
_TINY_FILES = {
    "W1":  (TINY_CO1 * TINY_CI * TINY_KH * TINY_KW,    "alexnet_tiny_conv1_W.bin"),
    "b1":  (TINY_CO1,                                   "alexnet_tiny_conv1_b.bin"),
    "W2":  (TINY_CO2 * TINY_CO1 * TINY_KH * TINY_KW,   "alexnet_tiny_conv2_W.bin"),
    "b2":  (TINY_CO2,                                   "alexnet_tiny_conv2_b.bin"),
    "W3":  (TINY_CO3 * TINY_CO2 * TINY_KH * TINY_KW,   "alexnet_tiny_conv3_W.bin"),
    "b3":  (TINY_CO3,                                   "alexnet_tiny_conv3_b.bin"),
    "Wf":  (TINY_FC_OUT * TINY_FC_IN,                   "alexnet_tiny_fc_W.bin"),
    "bf":  (TINY_FC_OUT,                                "alexnet_tiny_fc_b.bin"),
}


def load_trained_alexnet_tiny_weights(data_dir: str | None = None) -> dict | None:
    src = data_dir or _TRAINED_DIR
    out: dict[str, np.ndarray] = {}
    for key, (n_expect, fname) in _TINY_FILES.items():
        path = os.path.join(src, fname)
        if not os.path.isfile(path):
            return None
        arr = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
        if arr.size != n_expect:
            raise ValueError(
                f"{path}: expected {n_expect} float64s, got {arr.size}")
        out[key] = arr.copy()
    return out


def downsample_cifar10_2x(x_3072: np.ndarray) -> np.ndarray:
    """Client-side 2×2 avg-pool from (3, 32, 32) → (3, 16, 16) = 768 elements."""
    return avg_pool_2x2(x_3072, TINY_CI, 32, 32)


def alexnet_tiny_cleartext(X, W1, b1, W2, b2, W3, b3, Wf, bf):
    """Tiny AlexNet cleartext forward.  Input X is (3, 16, 16) flat (768)."""
    # Conv1 + bias + AvgPool + Quad
    y = conv2d_ref(X, W1, TINY_CI, TINY_CO1, TINY_H_IN, TINY_W_IN,
                    TINY_KH, TINY_KW, 1, TINY_PAD)
    y = add_conv_bias(y, b1, TINY_CO1, TINY_H_IN, TINY_W_IN)
    y = avg_pool_2x2(y, TINY_CO1, TINY_H_IN, TINY_W_IN)
    y = quad_ref(y)
    # Conv2 + bias + AvgPool + Quad
    y = conv2d_ref(y, W2, TINY_CO1, TINY_CO2, TINY_H1, TINY_W1,
                    TINY_KH, TINY_KW, 1, TINY_PAD)
    y = add_conv_bias(y, b2, TINY_CO2, TINY_H1, TINY_W1)
    y = avg_pool_2x2(y, TINY_CO2, TINY_H1, TINY_W1)
    y = quad_ref(y)
    # Conv3 + bias + AdaptivePool + Quad
    y = conv2d_ref(y, W3, TINY_CO2, TINY_CO3, TINY_H2, TINY_W2,
                    TINY_KH, TINY_KW, 1, TINY_PAD)
    y = add_conv_bias(y, b3, TINY_CO3, TINY_H2, TINY_W2)
    y = adaptive_avg_pool_2x2(y, TINY_CO3, TINY_H2, TINY_W2,
                               out_h=TINY_H3, out_w=TINY_W3)
    y = quad_ref(y)
    # FC
    y = matmul_ref(y, Wf, TINY_FC_OUT, TINY_FC_IN) + bf[:TINY_FC_OUT]
    return y


def alexnet_tiny_random_inputs(seed: int = 42):
    rng = np.random.default_rng(seed)
    return {
        "X":  rng.normal(0, 0.1, size=TINY_CI * TINY_H_IN * TINY_W_IN
                          ).astype(np.float64),
        "W1": rng.normal(0, 0.1, size=TINY_CO1 * TINY_CI * TINY_KH * TINY_KW
                          ).astype(np.float64),
        "b1": np.zeros(TINY_CO1, dtype=np.float64),
        "W2": rng.normal(0, 0.1, size=TINY_CO2 * TINY_CO1 * TINY_KH * TINY_KW
                          ).astype(np.float64),
        "b2": np.zeros(TINY_CO2, dtype=np.float64),
        "W3": rng.normal(0, 0.1, size=TINY_CO3 * TINY_CO2 * TINY_KH * TINY_KW
                          ).astype(np.float64),
        "b3": np.zeros(TINY_CO3, dtype=np.float64),
        "Wf": rng.normal(0, 0.1, size=TINY_FC_OUT * TINY_FC_IN
                          ).astype(np.float64),
        "bf": np.zeros(TINY_FC_OUT, dtype=np.float64),
    }


def prepare_tiny_args(d: dict) -> tuple:
    return (d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"],
            d["Wf"], d["bf"])


# ---------------------------------------------------------------------------
# AlexNet (full, depth-15) cleartext reference + helpers.
# ---------------------------------------------------------------------------
_FULL_FILES = {
    "W1":  (FULL_CO1 * FULL_CI  * FULL_KH * FULL_KW,  "alexnet_full_conv1_W.bin"),
    "b1":  (FULL_CO1,                                  "alexnet_full_conv1_b.bin"),
    "W2":  (FULL_CO2 * FULL_CO1 * FULL_KH * FULL_KW,  "alexnet_full_conv2_W.bin"),
    "b2":  (FULL_CO2,                                  "alexnet_full_conv2_b.bin"),
    "W3":  (FULL_CO3 * FULL_CO2 * FULL_KH * FULL_KW,  "alexnet_full_conv3_W.bin"),
    "b3":  (FULL_CO3,                                  "alexnet_full_conv3_b.bin"),
    "W4":  (FULL_CO4 * FULL_CO3 * FULL_KH * FULL_KW,  "alexnet_full_conv4_W.bin"),
    "b4":  (FULL_CO4,                                  "alexnet_full_conv4_b.bin"),
    "W5":  (FULL_CO5 * FULL_CO4 * FULL_KH * FULL_KW,  "alexnet_full_conv5_W.bin"),
    "b5":  (FULL_CO5,                                  "alexnet_full_conv5_b.bin"),
    "Wf1": (FULL_FC1_OUT * FULL_FC1_IN,                "alexnet_full_fc1_W.bin"),
    "bf1": (FULL_FC1_OUT,                              "alexnet_full_fc1_b.bin"),
    "Wf2": (FULL_FC2_OUT * FULL_FC1_OUT,               "alexnet_full_fc2_W.bin"),
    "bf2": (FULL_FC2_OUT,                              "alexnet_full_fc2_b.bin"),
    "Wf3": (FULL_FC3_OUT * FULL_FC2_OUT,               "alexnet_full_fc3_W.bin"),
    "bf3": (FULL_FC3_OUT,                              "alexnet_full_fc3_b.bin"),
}


def load_trained_alexnet_full_weights(data_dir: str | None = None
                                        ) -> dict | None:
    src = data_dir or _TRAINED_DIR
    out: dict[str, np.ndarray] = {}
    for key, (n_expect, fname) in _FULL_FILES.items():
        path = os.path.join(src, fname)
        if not os.path.isfile(path):
            return None
        arr = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
        if arr.size != n_expect:
            raise ValueError(
                f"{path}: expected {n_expect} float64s, got {arr.size}")
        out[key] = arr.copy()
    return out


def alexnet_full_cleartext(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5,
                            Wf1, bf1, Wf2, bf2, Wf3, bf3):
    """Cleartext forward for the full depth-15 AlexNet.

    Input X: flat (3*16*16=768,) — already client-side downsampled. All
    convs use stride=1, pad=1, kernel=3. Conv1 / Conv2 each pool 2×2 (s=2).
    Conv5 uses adaptive 2×2 average pool (4→2 on the spatial dim).
    """
    # Conv1 + bias + AvgPool2x2 → (8, 8, 8)
    y = conv2d_ref(X, W1, FULL_CI, FULL_CO1, FULL_H_IN, FULL_W_IN,
                    FULL_KH, FULL_KW, 1, FULL_PAD)
    y = add_conv_bias(y, b1, FULL_CO1, FULL_H_IN, FULL_W_IN)
    y = avg_pool_2x2(y, FULL_CO1, FULL_H_IN, FULL_W_IN)
    y = quad_ref(y)
    # Conv2 + bias + AvgPool2x2 → (16, 4, 4)
    y = conv2d_ref(y, W2, FULL_CO1, FULL_CO2, FULL_H1, FULL_W1,
                    FULL_KH, FULL_KW, 1, FULL_PAD)
    y = add_conv_bias(y, b2, FULL_CO2, FULL_H1, FULL_W1)
    y = avg_pool_2x2(y, FULL_CO2, FULL_H1, FULL_W1)
    y = quad_ref(y)
    # Conv3 + bias → (32, 4, 4)
    y = conv2d_ref(y, W3, FULL_CO2, FULL_CO3, FULL_H2, FULL_W2,
                    FULL_KH, FULL_KW, 1, FULL_PAD)
    y = add_conv_bias(y, b3, FULL_CO3, FULL_H2, FULL_W2)
    y = quad_ref(y)
    # Conv4 + bias → (32, 4, 4)
    y = conv2d_ref(y, W4, FULL_CO3, FULL_CO4, FULL_H3, FULL_W3,
                    FULL_KH, FULL_KW, 1, FULL_PAD)
    y = add_conv_bias(y, b4, FULL_CO4, FULL_H3, FULL_W3)
    y = quad_ref(y)
    # Conv5 + bias + AdaptiveAvgPool2x2 → (32, 2, 2)
    y = conv2d_ref(y, W5, FULL_CO4, FULL_CO5, FULL_H4, FULL_W4,
                    FULL_KH, FULL_KW, 1, FULL_PAD)
    y = add_conv_bias(y, b5, FULL_CO5, FULL_H4, FULL_W4)
    y = adaptive_avg_pool_2x2(y, FULL_CO5, FULL_H4, FULL_W4,
                                out_h=FULL_H5, out_w=FULL_W5)
    y = quad_ref(y)
    # FC1
    y = matmul_ref(y, Wf1, FULL_FC1_OUT, FULL_FC1_IN) + bf1[:FULL_FC1_OUT]
    y = quad_ref(y)
    # FC2
    y = matmul_ref(y, Wf2, FULL_FC2_OUT, FULL_FC1_OUT) + bf2[:FULL_FC2_OUT]
    y = quad_ref(y)
    # FC3
    y = matmul_ref(y, Wf3, FULL_FC3_OUT, FULL_FC2_OUT) + bf3[:FULL_FC3_OUT]
    return y


def alexnet_full_random_inputs(seed: int = 42):
    """Random Gaussian inputs + weights matching the full architecture."""
    rng = np.random.default_rng(seed)
    return {
        "X":   rng.normal(0.0, 0.1,
                            size=FULL_CI * FULL_H_IN * FULL_W_IN
                            ).astype(np.float64),
        "W1":  rng.normal(0.0, 0.1, size=FULL_CO1 * FULL_CI  * FULL_KH * FULL_KW).astype(np.float64),
        "W2":  rng.normal(0.0, 0.1, size=FULL_CO2 * FULL_CO1 * FULL_KH * FULL_KW).astype(np.float64),
        "W3":  rng.normal(0.0, 0.1, size=FULL_CO3 * FULL_CO2 * FULL_KH * FULL_KW).astype(np.float64),
        "W4":  rng.normal(0.0, 0.1, size=FULL_CO4 * FULL_CO3 * FULL_KH * FULL_KW).astype(np.float64),
        "W5":  rng.normal(0.0, 0.1, size=FULL_CO5 * FULL_CO4 * FULL_KH * FULL_KW).astype(np.float64),
        "Wf1": rng.normal(0.0, 0.1, size=FULL_FC1_OUT * FULL_FC1_IN).astype(np.float64),
        "Wf2": rng.normal(0.0, 0.1, size=FULL_FC2_OUT * FULL_FC1_OUT).astype(np.float64),
        "Wf3": rng.normal(0.0, 0.1, size=FULL_FC3_OUT * FULL_FC2_OUT).astype(np.float64),
        "b1":  np.zeros(FULL_CO1, dtype=np.float64),
        "b2":  np.zeros(FULL_CO2, dtype=np.float64),
        "b3":  np.zeros(FULL_CO3, dtype=np.float64),
        "b4":  np.zeros(FULL_CO4, dtype=np.float64),
        "b5":  np.zeros(FULL_CO5, dtype=np.float64),
        "bf1": np.zeros(FULL_FC1_OUT, dtype=np.float64),
        "bf2": np.zeros(FULL_FC2_OUT, dtype=np.float64),
        "bf3": np.zeros(FULL_FC3_OUT, dtype=np.float64),
    }


def prepare_full_args(d: dict) -> tuple:
    """Pack a weight dict into the positional args expected by `alexnet_full_cleartext`."""
    return (d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"],
            d["W4"], d["b4"], d["W5"], d["b5"],
            d["Wf1"], d["bf1"], d["Wf2"], d["bf2"], d["Wf3"], d["bf3"])


# The dense/sparse matrix builders that used to live here were the demo's
# own lowering of conv+pool and fc into slot matrices. packing.pack now
# derives those matrices from the traced torch model, so keeping a second
# implementation would mean two things to keep in agreement and only one
# of them in use. Their equivalence coverage moved to
# jaxite_word/packing_test.py, which checks the packed constants against a
# cleartext evaluation of the same program.


class _AlexNetHEEntrypoint(canonical_demo.CanonicalDemoAdapter):
    """Shared delegation for `AlexNetTinyHE` and `AlexNetHE`.

    The legacy surface is preserved deliberately: `precompute_plaintexts`
    still takes the same positional, already-BatchNorm-folded arrays, and
    `infer` still takes one flat client image. What changed is where those
    arrays go. They are written into the torch module that `nn.vectorize`
    traces, so the compiled program is the caller's model rather than a
    hand-built graph that happens to resemble it.
    """

    #: Attribute on `encrypted_demos` naming this entrypoint's demo class.
    _DEMO_NAME = ""

    def __init__(self, batch, dnum, devices, _cached_state):
        # Imported here, not at module scope: `encrypted_demos` pulls in torch
        # and the whole demo stack, and callers that only want the cleartext
        # references or the sparse builders above should not pay for it.
        import encrypted_demos

        if _cached_state is not None:
            raise ValueError(
                f"{type(self).__name__} no longer accepts a pickled HE "
                f"network. The cache format (version "
                f"{_STATIC_MODEL_CACHE_VERSION}) stored a hand-built packing "
                f"graph together with its keys; this class builds neither. "
                f"Construct it normally and call precompute_plaintexts(...) "
                f"— planning the model is cheap."
            )

        super().__init__(
            getattr(encrypted_demos, self._DEMO_NAME),
            batch=batch,
            devices=devices,
            dnum=dnum,
        )

    def _client_preprocess(self, image):
        """Client-side, outside encryption: 3x32x32 → 3x16x16 if needed.

        The encrypted model starts at (3, 16, 16). A caller that already ran
        `downsample_cifar10_2x` passes 768 values and this is a reshape; a
        caller holding a raw CIFAR-10 image passes 3072 and the average pool
        runs here, in the clear, exactly where it belongs.
        """
        import encrypted_demos

        array = np.asarray(image, dtype=np.float64)
        client = int(np.prod(encrypted_demos.ALEXNET_CLIENT_INPUT))
        encrypted = int(np.prod(encrypted_demos.ALEXNET_HE_INPUT))
        if array.size == encrypted:
            return array.reshape(encrypted_demos.ALEXNET_HE_INPUT)
        if array.size == client:
            return encrypted_demos.downsample_for_client(array)
        raise ValueError(
            f"expected {encrypted} values ({encrypted_demos.ALEXNET_HE_INPUT}, "
            f"already client-downsampled) or {client} "
            f"({encrypted_demos.ALEXNET_CLIENT_INPUT}, downsampled here); got "
            f"{array.size}."
        )

    # -- retired cache path ---------------------------------------------------

    def save_cache(self, path: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} no longer writes an HE cache. The cache "
            f"existed to avoid rebuilding a hand-lowered packing graph and its "
            f"keys; the canonical pipeline re-derives the plan from the torch "
            f"model cheaply, and key generation is well under a second. "
            f"Call precompute_plaintexts(...) instead of loading {path!r}."
        )

    @classmethod
    def from_cache(cls, path: str, *, batch: int | None = None, devices=None):
        raise NotImplementedError(
            f"{cls.__name__}.from_cache is retired: {path!r} holds a hand-built "
            f"packing graph and a key set for a ring this class no longer "
            f"uses. Construct {cls.__name__}(batch=..., devices=...) and call "
            f"precompute_plaintexts(...)."
        )


class AlexNetTinyHE(_AlexNetHEEntrypoint):
    """Encrypted AlexNetTiny inference (depth-7 schedule, CIFAR-10).

    Drives `encrypted_demos.AlexNetTinyDemo`: the torch module
    `alexnet_infer.QuadAlexNetTinyInfer`, vectorized, packed onto the ring
    that packing derives for it, and scheduled by one Mapping.
    """

    _DEMO_NAME = "AlexNetTinyDemo"
    def __init__(self, batch: int = 1, dnum: int | None = None,
                 use_multiplexed_conv1: bool = True,
                 devices=None,
                 _cached_state: dict | None = None):
        """`use_multiplexed_conv1` no longer selects a lowering.

        It used to choose between two hand-written conv1 matrices, row-major
        and stride-multiplexed. The packer now derives the input layout from
        the program it is packing, so there is one lowering and the flag
        cannot change it. It is accepted and recorded so existing call sites
        keep working.
        """
        super().__init__(batch, dnum, devices, _cached_state)
        self._use_multiplexed_conv1 = bool(use_multiplexed_conv1)

    def precompute_plaintexts(self, W1, b1, W2, b2, W3, b3, Wf, bf,
                              bsgs_ratio: float = 2.0):
        """Load the BatchNorm-folded weights and compile the HE program.

        The arrays arrive in the order `alexnet_train.export_weights` writes
        them and go straight into `QuadAlexNetTinyInfer`, which carries no
        BatchNorm — these have already been folded.
        """
        import time as _t
        print("[precompute] loading weights and compiling the HE program ...")
        t0 = _t.perf_counter()
        self._prepare(
            (W1, b1, W2, b2, W3, b3, Wf, bf),
            legacy_bsgs_ratios={'bsgs_ratio': bsgs_ratio},
        )
        print(f"[precompute] prepared and compiled in "
              f"{_t.perf_counter() - t0:.1f}s")


# ===========================================================================
# AlexNet (full, depth-15) HE inference — 5 conv + 3 FC + 7 quads.
# ===========================================================================
# Scale factor and noise width of the retired depth-15 chain. Kept because
# `alexnet_wide_config` builds its own parameter study on top of them.
FULL_SF = DEEP_Q_TOWERS_POOL[0] * DEEP_Q_TOWERS_POOL[1]
FULL_SIGMA = 3.190000057220458984375


class AlexNetHE(_AlexNetHEEntrypoint):
    """Encrypted depth-15 AlexNet inference (CIFAR-10).

    Drives `encrypted_demos.AlexNetFullDemo`: the torch module
    `alexnet_infer.QuadAlexNetFullInfer`, vectorized, packed and mapped.
    5 conv blocks + 3 FC blocks, each followed by a squaring except the last
    — 8 matvecs and 7 multiplications, so 15 multiplicative levels.

    SECURITY: this class used to hard-code a 17Q+4P chain at ring degree 2048,
    roughly 617 bits against an HES-2018 128-bit ceiling of 54 — research-grade
    only, and it warned about it at construction. It no longer picks a chain:
    `packing.pack` sizes the ring from the program's own slot demand and depth
    at `packing.DEFAULT_SECURITY_BITS`. `require_128bit` is still accepted and
    is now checked against the ring that packing actually derived.
    """

    _DEMO_NAME = "AlexNetFullDemo"
    def __init__(self, batch: int = 1, dnum: int | None = None,
                 require_128bit: bool = False,
                 devices=None,
                 _cached_state: dict | None = None):
        super().__init__(batch, dnum, devices, _cached_state)
        self._require_128bit = bool(require_128bit)

    def _validate_ring(self, ring_config):
        if not self._require_128bit:
            return
        bits = int(ring_config.security_bits)
        if bits < 128:
            raise ValueError(
                f"require_128bit was set, but packing derived a ring at "
                f"{bits}-bit security (degree {int(ring_config.degree)}, "
                f"{int(ring_config.logical_num_q)} moduli)."
            )

    def precompute_plaintexts(self, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5,
                              Wf1, bf1, Wf2, bf2, Wf3, bf3,
                              bsgs_ratio: float = 2.0):
        """Load the BatchNorm-folded weights and compile the HE program.

        `QuadAlexNetFullInfer` folds BatchNorm into conv1..conv5 and fc1/fc2,
        which is exactly what `alexnet_train.export_weights` already did to
        these arrays, so they load directly.
        """
        import time as _t
        print("[precompute] loading weights and compiling the depth-15 "
              "HE program ...")
        t0 = _t.perf_counter()
        self._prepare(
            (W1, b1, W2, b2, W3, b3, W4, b4, W5, b5,
             Wf1, bf1, Wf2, bf2, Wf3, bf3),
            legacy_bsgs_ratios={'bsgs_ratio': bsgs_ratio},
        )
        print(f"[precompute] prepared and compiled in "
              f"{_t.perf_counter() - t0:.1f}s")
