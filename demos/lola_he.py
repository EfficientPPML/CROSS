"""Core LoLA HE model and shared utilities.

This module provides a single encrypted LoLA inference class (`LoLAHE`) using
ORION-style packing and scheduling: every linear layer (conv1, fc1, fc2) is
expressed as one BSGS matrix-vector product, and the two square activations
are the only ct*ct multiplications. Total multiplicative depth = 5.

Level budget on LoLA-MNIST (degree=2048, slots=1024, 9 Q-towers, dnum=3):

    L8 input -> conv1 BSGS -> L7 -> square -> L6
             -> fc1   BSGS -> L5 -> square -> L4
             -> fc2   BSGS -> L3 -> bias add -> decrypt

The conv1 matvec uses a Toeplitz construction over the multiplexed input
packing (4 stride-channels x 196 spatial = 784 slots input; 5 output channels
x 196 spatial = 980 slots output). Sparse BSGS skips the entirely-zero
diagonals automatically.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import os
import sys
import time
from typing import Any

import numpy as np

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_JAXITE = os.path.abspath(os.path.join(_DEMO_DIR, "..", "jaxite_word"))
for p in (_JAXITE, _DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import bsgs as _bsgs
import ckks_ctx
import decrypt_fast
import encrypt_fast
from herot import HERot
import key_gen as kg
from matvec import make_ptct_rescale_fn
from polynomial import Polynomial
from profiler import KernelWrapper, Profiler, collect_logs


# ---------------------------------------------------------------------------
# Data loading (inlined from former lola_data_loader.py).  load_trained()
# reads BN-folded weights + the 200-image MNIST test slice from the
# `maple_data/` directory next to this file (override with the
# LOLA_WEIGHTS_DIR env var). generate() prefers loading the trained set
# but falls back to a deterministic random tensor if it isn't present.
# ---------------------------------------------------------------------------
_DATA_SEED = 42
_N_TEST_IMAGES = 200
_TRAINED_DIR = os.environ.get(
    "LOLA_WEIGHTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "maple_data"),
)


def _load_floats(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float64)


def _load_ints(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.int32)


def _save_floats(path, arr):
    with open(path, "wb") as f:
        f.write(np.asarray(arr, dtype=np.float64).ravel().tobytes())


def _save_ints(path, arr):
    with open(path, "wb") as f:
        f.write(np.asarray(arr, dtype=np.int32).ravel().tobytes())


def load_trained(data_dir=None):
    """Load the BN-folded LoLA weights + 200 MNIST test images.

    Reads from `LOLA_WEIGHTS_DIR` (default: `demos/maple_data/`). Optionally
    copies the same files into `data_dir` for callers that want a local
    snapshot. Raises `FileNotFoundError` if the weights aren't on disk.
    """
    src = _TRAINED_DIR
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Trained weights not found at {src}. "
            "Run train_lola_mnist.py at the repo root to populate it."
        )
    W1 = _load_floats(os.path.join(src, "lola_conv1_W.bin"))
    b1 = _load_floats(os.path.join(src, "lola_conv1_b.bin"))
    W2 = _load_floats(os.path.join(src, "lola_fc1_W.bin")).reshape(100, 980)
    b2 = _load_floats(os.path.join(src, "lola_fc1_b.bin"))
    W3 = _load_floats(os.path.join(src, "lola_fc2_W.bin")).reshape(10, 100)
    b3 = _load_floats(os.path.join(src, "lola_fc2_b.bin"))
    imgs = _load_floats(
        os.path.join(src, "mnist_test_200.bin")
    ).reshape(_N_TEST_IMAGES, 784)
    labels = _load_ints(os.path.join(src, "mnist_test_200_labels.bin"))
    pt_pred = _load_ints(os.path.join(src, "lola_plaintext_pred_200.bin"))

    if data_dir is not None and os.path.abspath(data_dir) != os.path.abspath(src):
        os.makedirs(data_dir, exist_ok=True)
        for fname in os.listdir(src):
            if fname.endswith(".bin"):
                src_p = os.path.join(src, fname)
                dst_p = os.path.join(data_dir, fname)
                if not os.path.exists(dst_p):
                    import shutil
                    shutil.copy2(src_p, dst_p)

    return {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3,
        "imgs": imgs, "labels": labels, "pt_pred": pt_pred,
    }


def generate(out_dir):
    """Prefer trained weights at `LOLA_WEIGHTS_DIR`; fall back to a
    deterministic random set if they're absent."""
    try:
        return load_trained(data_dir=out_dir)
    except FileNotFoundError:
        print("WARNING: trained weights missing; generating deterministic "
              "random fallback (use only for plumbing-correctness checks).")
        return _generate_random_fallback(out_dir)


def _generate_random_fallback(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(_DATA_SEED)
    W1 = (rng.randn(20).astype(np.float64) * 0.5)
    b1 = (rng.randn(5).astype(np.float64) * 0.1)
    W2 = (rng.randn(100, 980).astype(np.float64) * 0.05)
    b2 = (rng.randn(100).astype(np.float64) * 0.1)
    W3 = (rng.randn(10, 100).astype(np.float64) * 0.1)
    b3 = (rng.randn(10).astype(np.float64) * 0.1)
    imgs = rng.rand(_N_TEST_IMAGES, 784).astype(np.float64)
    labels = rng.randint(0, 10, _N_TEST_IMAGES).astype(np.int32)
    # Compute a plaintext prediction so callers always have pt_pred to consult.
    pt_pred = np.zeros(_N_TEST_IMAGES, dtype=np.int32)
    for i in range(_N_TEST_IMAGES):
        scores = lola_cleartext_inference(
            imgs[i], W1, b1, W2, b2, W3, b3)
        pt_pred[i] = int(np.argmax(scores))
    _save_floats(os.path.join(out_dir, "lola_conv1_W.bin"), W1)
    _save_floats(os.path.join(out_dir, "lola_conv1_b.bin"), b1)
    _save_floats(os.path.join(out_dir, "lola_fc1_W.bin"), W2)
    _save_floats(os.path.join(out_dir, "lola_fc1_b.bin"), b2)
    _save_floats(os.path.join(out_dir, "lola_fc2_W.bin"), W3)
    _save_floats(os.path.join(out_dir, "lola_fc2_b.bin"), b3)
    _save_floats(os.path.join(out_dir, "mnist_test_200.bin"), imgs)
    _save_ints(os.path.join(out_dir, "mnist_test_200_labels.bin"), labels)
    _save_ints(os.path.join(out_dir, "lola_plaintext_pred_200.bin"), pt_pred)
    return {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3,
        "imgs": imgs, "labels": labels, "pt_pred": pt_pred,
    }


# Architecture
CI, H_IN, W_IN = 1, 28, 28
CO, KH, KW = 5, 2, 2
H_OUT, W_OUT = 14, 14
CONV_AREA = H_OUT * W_OUT
CONV_OUT_SIZE = CO * CONV_AREA
FC1_IN, FC1_OUT = 980, 100
FC2_IN, FC2_OUT = 100, 10

# CKKS — 7Q+3P (canonical baseline as of 2026-04-27).  Earlier 9Q+4P was
# the historical baseline (~1140 ms wall, ≤1.9e-02 logits err); 7Q+3P with
# the depth-5 LoLA schedule has max_level=6 (2 levels of headroom over the
# minimum required 5), wall ~263 ms (warm bench), and ≤4.8e-02 max logits
# err over 100 MNIST images (still ≤ 5e-2 CKKS noise tolerance, 100/100
# pt==he agreement). The Q/P limb sweep history that established this
# choice is captured in demos/TRIALS.md "Configuration trials".
DEGREE = 2048
NUM_SLOTS = 1024
R, C = 32, 64
DNUM = 3
# The full 9-tower / 4-P pool retained for sweep scripts and overrides; the
# active 7Q + 3P slice below is what every demo / test / cache uses by default.
Q_TOWERS_POOL = [
    536903681,
    536924161,
    536952833,
    536973313,
    536977409,
    536989697,
    537026561,
    537047041,
    537071617,
]
P_TOWERS_POOL = [2147565569, 2147573761, 2147577857, 2147721217]
Q_TOWERS = Q_TOWERS_POOL[:7]
P_TOWERS = P_TOWERS_POOL[:3]
SF = Q_TOWERS[0] * Q_TOWERS[1]
SIGMA = 3.190000057220458984375
M = len(Q_TOWERS)


@dataclass
class RunResult:
    scores: np.ndarray
    wall_s: float
    intermediates: dict[str, np.ndarray] | None = None


def pack_lola_input(img_flat: np.ndarray, num_slots: int = NUM_SLOTS) -> np.ndarray:
    img = img_flat.reshape(H_IN, W_IN)
    out = np.zeros(num_slots)
    for r in range(KH):
        for s in range(KW):
            for h in range(H_OUT):
                for w in range(W_OUT):
                    out[(r * 2 + s) * CONV_AREA + h * W_OUT + w] = img[2 * h + r, 2 * w + s]
    return out


def conv1_cleartext(img_flat, W1, b1):
    w1_flat = W1.ravel() if W1.ndim > 1 else W1
    img = img_flat.reshape(H_IN, W_IN)
    out = np.zeros((CO, H_OUT, W_OUT))
    for co in range(CO):
        for h in range(H_OUT):
            for w in range(W_OUT):
                acc = 0.0
                for r_ in range(KH):
                    for c_ in range(KW):
                        acc += img[2 * h + r_, 2 * w + c_] * w1_flat[co * KH * KW + r_ * KW + c_]
                out[co, h, w] = acc + b1[co]
    return out


def lola_cleartext_inference(img_flat, W1, b1, W2, b2, W3, b3):
    conv = conv1_cleartext(img_flat, W1, b1)
    q1 = conv ** 2
    fc1 = W2.reshape(FC1_OUT, FC1_IN) @ q1.ravel() + b2
    q2 = fc1 ** 2
    return W3.reshape(FC2_OUT, FC2_IN) @ q2 + b3


def load_or_generate_data():
    try:
        data = load_trained()
        source = "trained"
    except FileNotFoundError:
        fallback_dir = os.path.join(_DEMO_DIR, "maple_data")
        data = generate(fallback_dir)
        source = fallback_dir
    return data, source


def prepare_weights(data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    w1 = data["W1"].ravel() if data["W1"].ndim > 1 else data["W1"]
    return (w1, data["b1"], data["W2"], data["b2"], data["W3"], data["b3"])


def resolve_indices(n_images: int, n: int, indices: str) -> list[int]:
    if indices:
        out = []
        for token in indices.split(","):
            token = token.strip()
            if not token:
                continue
            idx = int(token)
            if not 0 <= idx < n_images:
                raise ValueError(f"Index {idx} out of range [0, {n_images - 1}]")
            out.append(idx)
        if not out:
            raise ValueError("No valid indices were parsed.")
        return out
    count = min(n, n_images)
    if count < 1:
        raise ValueError("At least one image must be selected.")
    return list(range(count))


def build_conv1_toeplitz(W1: np.ndarray) -> np.ndarray:
    """Build the (CO*CONV_AREA, KH*KW*CONV_AREA) Toeplitz matrix for conv1.

    The packed input vector laid out by `pack_lola_input` consists of 4 stride-
    channels (rs in [0, 4)) of CONV_AREA=196 spatial cells each, total 784
    slots. Conv1 (1->5, 2x2, stride 2) maps this to 5 output channels of 196
    cells, total 980 slots. Each output cell is the dot product of the trained
    2x2 kernel with the four corresponding input cells from the four stride
    channels. The whole conv1 thus reduces to a single (980, 784) sparse
    matrix-vector product whose nonzero diagonals are at indices
    `(rs - co) * CONV_AREA mod NUM_SLOTS` for `(co, rs) in [0, CO) x
    [0, KH*KW)`. Identical packing convention to ORION's stride-2 multiplexing
    (Gazelle / `models/lola.py` in the orion repo).
    """
    W1_flat = W1.ravel()  # (5*4=20)
    M = np.zeros((CO * CONV_AREA, KH * KW * CONV_AREA), dtype=np.float64)
    eye_block = np.eye(CONV_AREA, dtype=np.float64)
    for co in range(CO):
        for r_ in range(KH):
            for s_ in range(KW):
                rs = r_ * 2 + s_
                w = W1_flat[co * KH * KW + r_ * KW + s_]
                M[co * CONV_AREA: (co + 1) * CONV_AREA,
                  rs * CONV_AREA: (rs + 1) * CONV_AREA] = w * eye_block
    return M


def build_conv1_bias_slots(b1: np.ndarray) -> np.ndarray:
    """Pack conv1 bias into a NUM_SLOTS-vector matching the conv1 output layout."""
    slots = np.zeros(NUM_SLOTS, dtype=np.float64)
    for co in range(CO):
        slots[co * CONV_AREA: (co + 1) * CONV_AREA] = b1[co]
    return slots


def run_model(model: Any, img: np.ndarray, weights: tuple[np.ndarray, ...]) -> RunResult:
    if hasattr(model, "infer_with_intermediates"):
        t0 = time.perf_counter()
        scores, intermediates = model.infer_with_intermediates(img, *weights)
        wall_s = time.perf_counter() - t0
        return RunResult(
            scores=np.asarray(scores),
            wall_s=wall_s,
            intermediates={k: np.asarray(v) for k, v in intermediates.items()},
        )
    t0 = time.perf_counter()
    scores = model.infer(img, *weights)
    wall_s = time.perf_counter() - t0
    return RunResult(scores=np.asarray(scores), wall_s=wall_s, intermediates=None)


# ---------------------------------------------------------------------------
# Polynomial cache + fast he_mul path
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _PolynomialTemplate:
    num_q: int
    shapes: dict[str, object]
    parameters: dict[str, object]
    moduli_array: jnp.ndarray
    batch: int = 1

    def make(self, data: jnp.ndarray) -> Polynomial:
        poly = Polynomial(self.shapes, parameters=self.parameters)
        poly.polynomial = data.reshape(self.batch, 2, R, C, self.num_q)
        return poly


@dataclass
class _FastHEMulPath:
    op: object
    input_shapes: dict[str, object]
    input_parameters: dict[str, object]
    batch: int = 1

    def square(self, ct: Polynomial) -> Polynomial:
        """Square via duplicated 4-element shell + standard HEMul.mul.

        We measured a polynomial_square shortcut (2 modmul + shift instead of
        3 modmul + add for the cross term `a0*b1 + a1*b0 = 2*a0*a1`) but it
        produced **no measurable wall improvement** at 7Q+3P — XLA likely
        fuses both formulations into similar compiled kernels, and the cost
        of the tensor stage is dominated by the modular_reduction not the
        raw multiply count. The helper is kept in `polynomial.py` /
        `hemul.py` (`polynomial_square`, `hemul_no_relin_square`) for
        future use and as documentation, but the live path stays on the
        generic mul. The real lever for Quad time is per-level dnum tuning
        (relinearization key-switch is ~65 % of square cost — see
        demos/TRIALS.md "Quad / he_mul squaring").
        """
        input_shell = Polynomial(self.input_shapes,
                                  parameters=self.input_parameters)
        input_shell.polynomial = jnp.concatenate(
            [ct.polynomial, ct.polynomial], axis=1
        ).reshape(self.batch, 4, R, C, -1)
        out = self.op._hemul.mul(input_shell)
        if out.polynomial.ndim != 5:
            out.polynomial = out.polynomial.reshape(
                self.batch, 2, R, C, out.num_moduli)
        return out


def _ptct_rescale_kernel(ct_data, pt_ntt, parameters):
    return parameters["ptct_rescale_fn"](ct_data, pt_ntt)


def _jit_modadd_impl(a: jnp.ndarray, b: jnp.ndarray,
                      moduli: jnp.ndarray) -> jnp.ndarray:
    """JIT body for ciphertext mod-add (used 20× in conv1)."""
    total = a.astype(jnp.uint64) + b.astype(jnp.uint64)
    return jnp.where(total >= moduli, total - moduli, total).astype(jnp.uint32)


def _jit_add_encoded_pt_impl(ct_data: jnp.ndarray, pt_ntt: jnp.ndarray,
                              moduli: jnp.ndarray) -> jnp.ndarray:
    """JIT body for `_add_encoded_plaintext`: c0 += pt mod q, c1 unchanged.

    Folded into one jit closure so the bias-add does not pay a fresh dispatch.
    """
    c0 = ct_data[:, 0:1]
    c1 = ct_data[:, 1:2]
    total = c0.astype(jnp.uint64) + pt_ntt[None, None].astype(jnp.uint64)
    c0_out = jnp.where(total >= moduli, total - moduli, total).astype(jnp.uint32)
    return jnp.concatenate([c0_out, c1], axis=1)


# ---------------------------------------------------------------------------
# LoLAHE: single end-to-end encrypted inference class.
# ---------------------------------------------------------------------------
class LoLAHE:
    """Encrypted LoLA-MNIST inference (JIT + BSGS-FC1 + naive-FC2 hybrid)."""

    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------
    def _rotation_indices(self):
        """Rotation indices to register at program_initialization time.

        Registers candidates for several BSGS factorizations so we can swap
        n1 at decision time (e.g. for FC2 variant comparison) without
        regenerating rotation keys.  The set is the union of:

          * fc1 BSGS at n1=32     -> baby [1..31], giant {32k}
          * fc2 BSGS at n1=16     -> baby [1..15] (subset of above),
                                     giant {16k}
          * fc2 BSGS at n1=8      -> baby [1..7]  (subset of above),
                                     giant {8k}  (covers the 8/24/40/... not
                                     already in {16k})
          * conv1 7 JIT'd rotation offsets

        Adding the n1=8 set adds ~64 extra rotation keys (~5 MB each, so
        roughly 320 MB extra in the cache) but is necessary to bench n1=8.
        """
        rot_indices = set()
        # fc1 BSGS at n1=32: baby + giant
        for i in range(1, 32):
            rot_indices.add(i)
        for j in range(1, 32):
            rot_indices.add(j * 32)
        # fc2 BSGS at n1=16: extra giant indices (baby subset already covered)
        for j in range(1, NUM_SLOTS // 16):
            rot_indices.add(j * 16)
        # fc2 BSGS at n1=8 (variant) — adds the 8/24/40/... giant indices
        # not already present in the n1=16 set.  Baby [1..7] is already a
        # subset of fc1's [1..31].
        for j in range(1, NUM_SLOTS // 8):
            rot_indices.add(j * 8)
        # conv1 (rs - co) * CONV_AREA — 7 distinct non-zero offsets.
        for rs in range(KH * KW):
            for co in range(CO):
                rot = (rs - co) * CONV_AREA
                if rot != 0:
                    rot_indices.add(rot % NUM_SLOTS)
        return rot_indices

    def __init__(
        self,
        _cached_state: dict | None = None,
        q_towers: list[int] | None = None,
        p_towers: list[int] | None = None,
        dnum: int | None = None,
        sf: float | None = None,
        batch: int = 1,
    ):
        """Construct a LoLAHE instance.

        Defaults: 7 Q-towers + 3 P-towers, dnum=3, batch=1.

        `batch` controls the leading axis of every ciphertext (the "image"
        dim). Setting `batch=N` lets a single LoLAHE process N independent
        images per HE-eval (see demos/TRIALS.md "Multi-image batching").
        All cached HE-parameters (NTT contexts, BConv, evaluation/rotation
        keys) are sized at construction time, so batch is FIXED for the
        lifetime of the LoLAHE.
        """
        # Resolve per-instance config, defaulting to module-level constants.
        self._q_towers = list(q_towers) if q_towers is not None else list(Q_TOWERS)
        self._p_towers = list(p_towers) if p_towers is not None else list(P_TOWERS)
        self._dnum = int(dnum) if dnum is not None else DNUM
        self._sf = float(sf) if sf is not None else float(SF)
        self._M = len(self._q_towers)
        self._batch = int(batch)

        # When `_cached_state` is provided (via `from_cache`), reuse the saved
        # encryption keys + rotation keys instead of regenerating them.
        if _cached_state is None:
            print("[setup] generating keys ...")
            kp = kg.gen_pke_pair(self._q_towers, self._p_towers, DEGREE)
            ek = kg.gen_evaluation_key(
                kp["secret_key"],
                q=self._q_towers,
                P=self._p_towers,
                noise_std=SIGMA,
                noise_scale=1,
                dnum=self._dnum,
            )
            pregenerated_rotation_keys = None
        else:
            print("[setup] reusing cached keys ...")
            kp = _cached_state["kp"]
            ek = _cached_state["ek"]
            pregenerated_rotation_keys = _cached_state["rotation_keys"]

        ea = jnp.array(ek["a"], dtype=jnp.uint32).transpose(0, 2, 1)
        eb = jnp.array(ek["b"], dtype=jnp.uint32).transpose(0, 2, 1)
        # Stash raw keys so `save_cache` can serialize them later.
        self._cached_kp = kp
        self._cached_ek = ek

        rot_indices = self._rotation_indices()

        params = {
            "degree": DEGREE,
            "num_slots": NUM_SLOTS,
            "scaling_factor": self._sf,
            "output_scale": self._sf,
            "q_towers": self._q_towers,
            "p_towers": self._p_towers,
            "p": 60,
            "CKKS_M_FACTOR": 1,
            "max_bits_in_word": 61,
            "noise_scale_degree": 1,
            "composite_degree": 1,
            "public_key": kp["public_key"],
            "secret_key": kp["secret_key"],
            "evaluation_key": [ea, eb],
        }
        print("[setup] program_initialization ...")
        self.ctx = ckks_ctx.CKKSContext(params)
        self.ctx.program_initialization(
            total_hemul_levels=self.ctx.max_level,
            total_rotation_indices=sorted(rot_indices),
            dnum=self._dnum,
            r=R,
            c=C,
            batch=self._batch,
            pregenerated_rotation_keys=pregenerated_rotation_keys,
        )
        print(f"[setup] max_level = {self.ctx.max_level}")

        self._build_jit_closures()
        self._set_level_constants()
        self._init_polynomial_templates()
        self._init_fast_he_mul_paths()
        self._warm_rotation_ops()

        # Encoded bias placeholders — populated in precompute_plaintexts.
        self._conv1_bias_pt = None
        self._fc1_bias_pt = None
        self._fc2_bias_pt = None
        # Per-num_moduli JIT'd bias-add and mod-add closures.  Built lazily
        # on first use so XLA only compiles the (nq) variants we actually run.
        self._jit_add_encoded_pt: dict[int, Any] = {}
        self._jit_modadd: dict[int, Any] = {}

        # JIT rotate + scan sum_reduce.
        self._HERot_make = HERot.make_rotate_fn
        self._jit_rotate_fns: dict[tuple[int, int], Any] = {}
        self._rotate_keys: dict[tuple[int, int], tuple] = {}
        self._sum_reduce_scan: dict[int, tuple] = {}
        self._prepare_jit_rotations()

    # --------------------------------------------------------------
    # Level map
    # --------------------------------------------------------------
    def _set_level_constants(self):
        # ORION-style scheduling: every linear layer (conv1, fc1, fc2) is one
        # BSGS matrix-vector product (1 level); each Quad activation is one
        # ct*ct mul (1 level). Total multiplicative depth = 5:
        #   L8 -> conv1 BSGS -> L7 -> square -> L6 -> fc1 BSGS -> L5
        #      -> square -> L4 -> fc2 BSGS -> L3
        self._input_level = self.ctx.max_level
        self._conv_out_level = self._input_level - 1
        self._square1_out_level = self._input_level - 2
        self._fc1_out_level = self._square1_out_level - 1
        self._square2_out_level = self._fc1_out_level - 1
        self._logits_level = self._square2_out_level - 1

    # --------------------------------------------------------------
    # Offline helpers
    # --------------------------------------------------------------
    def _build_jit_closures(self):
        # Note: matvec.py used to read `rescale_op.power_of_inv_psi_all` but
        # that attribute is now vestigial (negacyclic NTT bakes psi/psi^{-1}
        # into NTT params; the explicit modmul was a no-op). HERescale no
        # longer creates it, and matvec.py no longer references it.
        print("[jit] building ptct_rescale closures ...")
        cache = self.ctx._param_cache
        self._jit_ptctr: dict[int, Any] = {}
        self._ptctr_pure: dict[int, Any] = {}
        for src in range(1, self._M):
            resc = self.ctx.he_rescale[src, src - 1]._he_rescale
            pure_fn = make_ptct_rescale_fn(self.ctx.ptct_mul[src]._ptct, resc)
            self._ptctr_pure[src] = pure_fn
            self._jit_ptctr[src] = jax.jit(pure_fn)
        print("[jit] done")

    def _init_polynomial_templates(self):
        cache = self.ctx._param_cache
        self._poly_templates: dict[int, _PolynomialTemplate] = {}
        for level in range(self.ctx.max_level + 1):
            nq = cache.num_q_at_level(level)
            if nq in self._poly_templates:
                continue
            self._poly_templates[nq] = _PolynomialTemplate(
                num_q=nq,
                shapes={
                    "batch": self._batch,
                    "num_elements": 2,
                    "degree": DEGREE,
                    "num_moduli": nq,
                    "precision": 32,
                    "degree_layout": (R, C),
                },
                parameters={
                    "moduli": self._q_towers[:nq],
                    "ntt_ctx": cache.level_params[level].sliced_ntt_q,
                },
                moduli_array=jnp.asarray(self._q_towers[:nq], dtype=jnp.uint32),
                batch=self._batch,
            )

    def _init_fast_he_mul_paths(self):
        cache = self.ctx._param_cache
        self._fast_he_mul_paths: dict[int, _FastHEMulPath] = {}
        for out_level in (self._square1_out_level, self._square2_out_level):
            op = self.ctx.he_mul[out_level]
            input_level = min(out_level + 1, self.ctx.max_level)
            self._fast_he_mul_paths[out_level] = _FastHEMulPath(
                op=op,
                input_shapes=op._ct_in_shapes,
                input_parameters={
                    "moduli": op._q_input,
                    "ntt_ctx": cache.level_params[input_level].sliced_ntt_q,
                },
                batch=self._batch,
            )

    def _warm_rotation_ops(self):
        # Conv1 uses 7 distinct (rs - co)*CONV_AREA rotations at conv_out_level.
        # FC1, FC2 each use baby/giant at their respective input levels (BSGS
        # also warms these lazily on encode_matrix; we warm them here for
        # symmetry with the conv1 JIT path).
        for rs in range(KH * KW):
            for co in range(CO):
                rot = (rs - co) * CONV_AREA
                if rot != 0:
                    self.ctx.he_rot[self._conv_out_level, rot % NUM_SLOTS]
        n1, n2 = _bsgs.compute_bsgs_params(NUM_SLOTS)
        for input_level in (self._square1_out_level, self._square2_out_level):
            for i in range(1, n1):
                self.ctx.he_rot[input_level, i]
            for j in range(1, n2):
                self.ctx.he_rot[input_level - 1, j * n1]

    def _prepare_jit_rotations(self):
        """JIT-compile the 7 distinct conv1 rotations at conv_out_level.

        FC1 and FC2 rotations are handled inside `BSGSMatVec`'s scan kernel
        and don't need entries in `_jit_rotate_fns`. Conv1 stays on the
        per-call JIT'd rotate path (one compiled XLA kernel per distinct
        rotation index, dispatched 16 times per inference) — that path is
        markedly faster on this size of rotation than running conv1 through
        BSGS, where 31 baby rotations are computed even though only 8 are
        actually used.
        """
        print("[jit-rot] preparing conv1 JIT rotations ...")
        t0 = time.perf_counter()
        for rs in range(KH * KW):
            for co in range(CO):
                idx = (rs - co) * CONV_AREA
                if idx == 0:
                    continue
                idx_mod = idx % NUM_SLOTS
                key = (self._conv_out_level, idx_mod)
                if key in self._jit_rotate_fns:
                    continue
                inst = self.ctx.he_rot[self._conv_out_level, idx_mod]._herot
                pure_fn = self._HERot_make(inst)
                self._jit_rotate_fns[key] = jax.jit(pure_fn)
                self._rotate_keys[key] = (
                    inst.evalkey_a_vector.astype(jnp.uint64),
                    inst.evalkey_b_vector.astype(jnp.uint64),
                    jnp.asarray(inst.coef_map, dtype=jnp.int32),
                )
        print(
            f"[jit-rot] done in {time.perf_counter() - t0:.1f}s  "
            f"({len(self._jit_rotate_fns)} conv1 rot kernels)"
        )

    # --------------------------------------------------------------
    # Encryption / encoding / decryption
    # --------------------------------------------------------------
    def _mk(self, data, nq):
        return self._poly_templates[nq].make(data)

    def encrypt(self, slots):
        """Encrypt a real-valued slot vector to a fresh batch=1 ciphertext.

        For batch>1 use `encrypt_batch(list_of_slots)`.
        """
        if self._batch != 1:
            raise ValueError(
                f"encrypt() only valid at batch=1; current batch={self._batch}. "
                f"Use encrypt_batch(...).")
        ct_np = encrypt_fast.fast_encode_encrypt(
            [complex(float(v)) for v in slots[:NUM_SLOTS]],
            self.ctx, scale=self._sf,
        )                                                # (2, N, M) uint64
        return self._mk(jnp.asarray(ct_np, dtype=jnp.uint32), self._M)

    def encrypt_batch(self, slots_list):
        """Encrypt B independent slot vectors and stack along the batch dim.

        Args:
          slots_list: list of length B, each element a length-NUM_SLOTS array.

        Returns:
          A Polynomial whose `.polynomial` has shape (B, 2, R, C, M).
        """
        B = len(slots_list)
        if B != self._batch:
            raise ValueError(
                f"encrypt_batch got {B} slot vectors but model batch={self._batch}.")
        cts = []
        for slots in slots_list:
            ct_np = encrypt_fast.fast_encode_encrypt(
                [complex(float(v)) for v in slots[:NUM_SLOTS]],
                self.ctx, scale=self._sf,
            )
            cts.append(ct_np)                            # (2, N, M)
        stacked = np.stack(cts, axis=0)                  # (B, 2, N, M)
        return self._mk(jnp.asarray(stacked, dtype=jnp.uint32), self._M)

    def decrypt(self, ct, scale=None, slots_to_decode=None):
        """Decrypt a batch=1 ciphertext to a real-valued NumPy slot vector.

        For batch>1, use `decrypt_batch(ct)` which returns a list of B arrays.
        """
        if scale is None:
            scale = self._sf
        if self._batch != 1:
            raise ValueError(
                f"decrypt() only valid at batch=1; current batch={self._batch}. "
                f"Use decrypt_batch(...).")
        import ckks_ctx as cc
        nq = ct.num_moduli
        ct_np = np.asarray(
            ct.polynomial.reshape(2, DEGREE, nq), dtype=np.uint64
        )
        old_bypass = getattr(cc, "BYPASS_DECODE_STDDEV_CHECK", False)
        cc.BYPASS_DECODE_STDDEV_CHECK = True
        try:
            return decrypt_fast.fast_decrypt_decode(
                ct_np, self.ctx, scale, slots_to_decode=slots_to_decode
            )
        finally:
            cc.BYPASS_DECODE_STDDEV_CHECK = old_bypass

    def decrypt_batch(self, ct, scale=None, slots_to_decode=None):
        """Decrypt a (B, 2, R, C, M) ciphertext into a list of B slot arrays.

        Each batch index decrypts independently on CPU (the cached
        decrypt_fast tables are shared, only the per-call SK*INTT runs
        per item).
        """
        if scale is None:
            scale = self._sf
        import ckks_ctx as cc
        B = self._batch
        nq = ct.num_moduli
        ct_np = np.asarray(
            ct.polynomial.reshape(B, 2, DEGREE, nq), dtype=np.uint64
        )                                                # (B, 2, N, M)
        old_bypass = getattr(cc, "BYPASS_DECODE_STDDEV_CHECK", False)
        cc.BYPASS_DECODE_STDDEV_CHECK = True
        try:
            return [
                decrypt_fast.fast_decrypt_decode(
                    ct_np[b], self.ctx, scale, slots_to_decode=slots_to_decode)
                for b in range(B)
            ]
        finally:
            cc.BYPASS_DECODE_STDDEV_CHECK = old_bypass

    def encode_plaintext_at_level(self, slots, level, scale=None):
        nq = self.ctx._param_cache.num_q_at_level(level)
        params = dict(self.ctx.parameters)
        params["q_towers"] = self._q_towers[:nq]
        if scale is not None:
            params["scaling_factor"] = scale
        ectx = ckks_ctx.CKKSContext(params)
        pt = ectx.encode([complex(float(v)) for v in slots[:NUM_SLOTS]])
        return pt.polynomial[0, 0].reshape(R, C, nq).astype(jnp.uint32)

    # --------------------------------------------------------------
    # Core HE ops
    # --------------------------------------------------------------
    def _modadd_num_q(self, a: jnp.ndarray, b: jnp.ndarray, nq: int) -> jnp.ndarray:
        moduli = self._poly_templates[nq].moduli_array
        fn = self._jit_modadd.get(nq)
        if fn is None:
            fn = jax.jit(_jit_modadd_impl)
            self._jit_modadd[nq] = fn
        return fn(a, b, moduli)

    def _modadd(self, a, b, level):
        return self._modadd_num_q(
            a, b, self.ctx._param_cache.num_q_at_level(level))

    def _add_encoded_plaintext(self, ct: Polynomial, pt_ntt: jnp.ndarray) -> Polynomial:
        nq = ct.num_moduli
        moduli = self._poly_templates[nq].moduli_array
        fn = self._jit_add_encoded_pt.get(nq)
        if fn is None:
            fn = jax.jit(_jit_add_encoded_pt_impl)
            self._jit_add_encoded_pt[nq] = fn
        out_data = fn(ct.polynomial, pt_ntt, moduli)
        return self._mk(out_data, nq)

    def add_ct(self, a, b):
        return self._mk(
            self._modadd_num_q(a.polynomial, b.polynomial, a.num_moduli),
            a.num_moduli,
        )

    def add_pt(self, ct, slots, level, scale=None):
        pt = self.encode_plaintext_at_level(slots, level, scale=scale)
        return self._add_encoded_plaintext(ct, pt)

    def he_mul(self, ct, lev_out):
        fast_path = self._fast_he_mul_paths.get(lev_out)
        if fast_path is not None:
            return fast_path.square(ct)
        # Fallback: full HEMul.mul.
        res = self.ctx.he_mul[lev_out].mul(ct, ct)
        if res.polynomial.ndim != 5:
            res.polynomial = res.polynomial.reshape(self._batch, 2, R, C, res.num_moduli)
        return res

    def rotate(self, ct, idx, level):
        # CKKS slot rotation is periodic mod num_slots — accept signed idx and
        # normalize so callers don't have to care.
        idx_mod = idx % NUM_SLOTS
        if idx_mod == 0:
            return ct
        jit_fn = self._jit_rotate_fns.get((level, idx_mod))
        if jit_fn is not None:
            eval_a, eval_b, coef_map = self._rotate_keys[(level, idx_mod)]
            rotated = jit_fn(ct.polynomial, eval_a, eval_b, coef_map)
            return self._mk(
                rotated.reshape(self._batch, 2, R, C, ct.num_moduli), ct.num_moduli)
        # Fallback: eager rotate.
        res = self.ctx.he_rot[level, idx_mod].rotate(ct)
        if res.polynomial.ndim != 5:
            res.polynomial = res.polynomial.reshape(self._batch, 2, R, C, res.num_moduli)
        return res

    def sum_reduce_all(self, ct, level):
        scan_entry = self._sum_reduce_scan.get(level)
        if scan_entry is not None:
            fn, ea, eb, cm = scan_entry
            ct_data = ct.polynomial
            if ct_data.ndim != 5:
                ct_data = ct_data.reshape(self._batch, 2, R, C, ct.num_moduli)
            result = fn(ct_data, ea, eb, cm)
            return self._mk(
                result.reshape(self._batch, 2, R, C, ct.num_moduli), ct.num_moduli)
        # Fallback: eager log2-stride loop.
        stride = 1
        while stride < NUM_SLOTS:
            ct = self.add_ct(ct, self.rotate(ct, stride, level))
            stride *= 2
        return ct

    # --------------------------------------------------------------
    # Network stages.  Conv1 stays on the eager 20-iteration loop with JIT'd
    # rotations (cheaper than BSGS for the conv1 sparsity pattern at this
    # scale); fc1, fc2 are single BSGS matrix-vector products.
    # --------------------------------------------------------------
    def conv1_lola(self, ct, W1, b1, level):
        del W1, b1
        # Use the pre-built fused JIT graph if available (covers the entire
        # 8 ptct + 7 rotates + 7 modadds + 1 bias-add as one XLA computation,
        # eliminating Python-level dispatch overhead between iterations).
        # Falls back to the per-iter Python loop if the fused graph hasn't
        # been compiled (e.g. on a level / batch combo we haven't seen).
        fused = getattr(self, "_jit_conv1_fused", {}).get(level)
        if fused is not None:
            out_data = fused(ct.polynomial)
            return self._mk(out_data, out_data.shape[-1])
        acc = None
        for pt_ntt, rot in zip(self._conv1_pts, self._conv1_rots):
            partial = self._jit_ptctr[level](ct.polynomial, pt_ntt)
            if rot != 0:
                rotated = self.rotate(
                    self._mk(partial, partial.shape[-1]), rot, level - 1)
                partial = rotated.polynomial.reshape(partial.shape)
            acc = partial if acc is None else self._modadd(
                acc, partial, level - 1)
        return self._add_encoded_plaintext(
            self._mk(acc, acc.shape[-1]), self._conv1_bias_pt)

    def build_conv1_fused_jit(self, level: int):
        """Pre-compile a single XLA graph for the entire conv1 pipeline.

        Returns a jax.jit'd function `(ct_data) -> out_data` covering
        8 ptct_rescale + 7 rotate + 7 modadd + 1 bias-add in ONE compiled
        kernel. Stored on self._jit_conv1_fused[level] so subsequent
        `conv1_lola(ct, ..., level)` calls take the fused fast-path.

        Returns the closure (caller may also discard it; setting
        `_jit_conv1_fused` on the instance is the side effect the
        fast-path checks).
        """
        if not hasattr(self, "_jit_conv1_fused"):
            self._jit_conv1_fused = {}

        out_level = level - 1
        out_nq = self.ctx._param_cache.num_q_at_level(out_level)
        moduli_out = self._poly_templates[out_nq].moduli_array

        # Pure (non-JIT'd) helpers for the fused graph.
        pure_ptctr = self._ptctr_pure[level]            # ct_data, pt -> partial
        pure_rotates = {}                                # rot_offset -> pure fn
        rot_keys = {}                                    # rot_offset -> keys
        for rot in self._conv1_rots:
            if rot == 0:
                continue
            rot_mod = rot % NUM_SLOTS
            inst = self.ctx.he_rot[out_level, rot_mod]._herot
            pure_rotates[rot] = self._HERot_make(inst)
            rot_keys[rot] = self._rotate_keys[(out_level, rot_mod)]

        pts = list(self._conv1_pts)
        rots = list(self._conv1_rots)
        bias_pt = self._conv1_bias_pt
        batch = self._batch
        # Capture R, C in module-scope; use the constants directly.

        @jax.jit
        def conv1_fused(ct_data: jnp.ndarray) -> jnp.ndarray:
            acc = None
            for pt, rot in zip(pts, rots):
                partial = pure_ptctr(ct_data, pt)
                if rot != 0:
                    ea, eb, cm = rot_keys[rot]
                    partial = pure_rotates[rot](partial, ea, eb, cm)
                # Reshape to canonical (batch, 2, R, C, out_nq) layout for
                # the mod-add step (rotate output is (batch, 2, ring_dim, M);
                # ptct_rescale output already has the (batch, 2, R, C, M)
                # layout, but reshape unifies the two paths).
                partial = partial.reshape(batch, 2, R, C, out_nq)
                if acc is None:
                    acc = partial
                else:
                    s = (acc.astype(jnp.uint64)
                         + partial.astype(jnp.uint64))
                    acc = jnp.where(
                        s >= moduli_out, s - moduli_out, s
                    ).astype(jnp.uint32)
            # Bias add: c0 += bias_pt mod q
            c0 = acc[:, 0:1]
            c1 = acc[:, 1:2]
            s = (c0.astype(jnp.uint64)
                 + bias_pt[None, None].astype(jnp.uint64))
            c0_out = jnp.where(
                s >= moduli_out, s - moduli_out, s
            ).astype(jnp.uint32)
            return jnp.concatenate([c0_out, c1], axis=1)

        self._jit_conv1_fused[level] = conv1_fused
        return conv1_fused

    def matmul_he(self, ct, W, n_out, k_in, level):
        del W, k_in, n_out
        if level == self._square1_out_level:
            return self._fc1_bsgs.mul(ct)
        if level == self._square2_out_level:
            return self._fc2_bsgs.mul(ct)
        raise ValueError(
            f"matmul_he received unexpected level={level}; "
            f"expected {self._square1_out_level} or {self._square2_out_level}."
        )

    # --------------------------------------------------------------
    # Precompute
    # --------------------------------------------------------------
    def precompute_plaintexts(self, W1, b1, W2, b2, W3, b3,
                              fc1_bsgs_ratio: float = 2.0,
                              fc2_bsgs_ratio: float = 2.0):
        """Encode plaintexts for the depth-5 ORION-style schedule.

        - conv1 -> 20 weight plaintexts (one per (co, rs) pair) at L_max.
        - fc1   -> (100, 980) BSGSMatVec; `fc1_bsgs_ratio` controls n1.
        - fc2   -> (10, 100) BSGSMatVec; `fc2_bsgs_ratio` controls n1.
        Bias plaintexts are encoded at each layer's output level (level-1)
        so they can be added in-place after the rescale.

        `fc{1,2}_bsgs_ratio` is the `bsgs_ratio` arg passed through to
        `BSGSMatVec.encode_matrix`. The default 2.0 picks
        n1 ≈ sqrt(num_diagonals × 2). Smaller ratios (e.g. 0.5) prefer fewer
        baby pre-rotations + more giants; larger ratios go the other way.
        """
        print("[precompute] encoding plaintexts ...")
        t0 = time.perf_counter()
        cache = self.ctx._param_cache
        top_level = self.ctx.max_level
        qd = self._q_towers[-1]
        n1, n2 = _bsgs.compute_bsgs_params(NUM_SLOTS)

        # ---------- conv1: 20 (co, rs) row plaintexts, GROUPED by rs-co. ----
        # Within a `rs-co` group every (co, rs) pair has the same rotation
        # offset (rs-co)*CONV_AREA AND non-overlapping slot regions
        # (different rs values => different slot sub-blocks). So summing the
        # encoded plaintexts within a group, ptct-multiplying once, and
        # rotating once is mathematically equivalent to processing each
        # (co, rs) individually — but uses fewer rotations (8 distinct
        # rs-co values instead of 20 pairs => 7 non-zero rotations
        # instead of 16). Saves ~9 expensive key-switching rotations per
        # conv1 call.
        nq_top = cache.num_q_at_level(top_level)
        moduli_top = self._poly_templates[nq_top].moduli_array
        grouped: dict[int, jnp.ndarray] = {}      # {rot_offset: summed_pt}
        for co in range(CO):
            for r_ in range(KH):
                for s_ in range(KW):
                    rs = r_ * 2 + s_
                    rot_offset = (rs - co) * CONV_AREA
                    wv = np.zeros(NUM_SLOTS)
                    wv[rs * CONV_AREA:(rs + 1) * CONV_AREA] = \
                        W1[co * KH * KW + r_ * KW + s_]
                    pt = self.encode_plaintext_at_level(
                        wv, top_level, scale=qd)
                    if rot_offset in grouped:
                        # Mod-add the plaintext entries (NTT distributes over
                        # addition, so summing in eval form is correct).
                        s = grouped[rot_offset].astype(jnp.uint64) + \
                            pt.astype(jnp.uint64)
                        grouped[rot_offset] = jnp.where(
                            s >= moduli_top, s - moduli_top, s
                        ).astype(jnp.uint32)
                    else:
                        grouped[rot_offset] = pt
        # Order by rotation offset for stable iteration.
        ordered = sorted(grouped.items(), key=lambda kv: kv[0])
        self._conv1_pts: list[jnp.ndarray] = [pt for _, pt in ordered]
        self._conv1_rots: list[int] = [rot for rot, _ in ordered]
        print(
            f"[conv1] grouped 20 (co,rs) plaintexts into "
            f"{len(self._conv1_pts)} rs-co groups "
            f"(rot offsets = {self._conv1_rots})"
        )

        # ---------- fc1 BSGS at L6 -> L5 (adaptive via fc1_bsgs_ratio) ----------
        self._fc1_bsgs = self.ctx.bsgs_matvec[
            self._square1_out_level, NUM_SLOTS, n1, n2]
        self._fc1_bsgs.encode_matrix(
            W2.reshape(FC1_OUT, FC1_IN), bsgs_ratio=fc1_bsgs_ratio)

        # ---------- fc2 BSGS at L4 -> L3 (adaptive via fc2_bsgs_ratio) ----------
        self._fc2_bsgs = self.ctx.bsgs_matvec[
            self._square2_out_level, NUM_SLOTS, n1, n2]
        self._fc2_bsgs.encode_matrix(
            W3.reshape(FC2_OUT, FC2_IN), bsgs_ratio=fc2_bsgs_ratio)

        # ---------- Bias plaintexts ----------
        self._conv1_bias_pt = self.encode_plaintext_at_level(
            build_conv1_bias_slots(b1),
            self._conv_out_level,
            scale=self._sf,
        )
        self._fc1_bias_pt = self.encode_plaintext_at_level(
            self._make_dense_bias_slots(b2, FC1_OUT),
            self._fc1_out_level,
            scale=self._sf,
        )
        self._fc2_bias_pt = self.encode_plaintext_at_level(
            self._make_dense_bias_slots(b3, FC2_OUT),
            self._logits_level,
            scale=self._sf,
        )
        # Pre-compile the fused XLA graph for conv1 (8 ptct + 7 rotate +
        # 7 modadd + 1 bias-add) at the input level. ~13 % wall reduction
        # (loop vs fused: 331 -> 288 ms median at 7Q+3P B=1; full A/B
        # writeup in demos/TRIALS.md "Conv1 XLA scan-fusion").
        self.build_conv1_fused_jit(self.ctx.max_level)
        print(f"[precompute] done in {time.perf_counter() - t0:.1f}s")

    def _make_dense_bias_slots(self, bias, width):
        slots = np.zeros(NUM_SLOTS)
        slots[:width] = bias[:width]
        return slots

    # --------------------------------------------------------------
    # End-to-end inference
    # --------------------------------------------------------------
    def infer(self, img, W1, b1, W2, b2, W3, b3, trace_dir=None):
        # Multiplicative depth = 5 (3 BSGS matvecs + 2 squarings).
        # img: single image array OR (B, ...) for batch>1.
        if self._batch == 1:
            ct = self.encrypt(pack_lola_input(img))
        else:
            # img is expected to be iterable of B images.
            packed = [pack_lola_input(im) for im in img]
            ct = self.encrypt_batch(packed)
        level = self.ctx.max_level
        with (jax.profiler.trace(trace_dir) if trace_dir else nullcontext()):
            ct = self.conv1_lola(ct, W1, b1, level)
            level -= 1                                    # L_max -> L_max-1
            ct = self.he_mul(ct, level - 1)
            level -= 1
            ct = self.matmul_he(
                ct, W2.reshape(FC1_OUT, FC1_IN), FC1_OUT, FC1_IN, level)
            level -= 1
            ct = self._add_encoded_plaintext(ct, self._fc1_bias_pt)
            ct = self.he_mul(ct, level - 1)
            level -= 1
            ct = self.matmul_he(
                ct, W3.reshape(FC2_OUT, FC2_IN), FC2_OUT, FC2_IN, level)
            level -= 1
            ct = self._add_encoded_plaintext(ct, self._fc2_bias_pt)
        if self._batch == 1:
            return self.decrypt(ct, scale=self._sf)[:FC2_OUT]
        else:
            return [logits[:FC2_OUT]
                    for logits in self.decrypt_batch(ct, scale=self._sf)]

    # --------------------------------------------------------------
    # Cache: save / load the heavy state for fast re-launch
    # --------------------------------------------------------------
    def save_cache(self, path: str) -> None:
        """Save keys + encoded plaintexts + BSGS state to disk.

        Skipping these on reload avoids ~85-95% of total setup wall time
        (rotation key generation + 1024 BSGS diagonal encodings). JIT closures
        are NOT saved — they recompile on the first inference, but XLA's
        compilation cache makes that fast.
        """
        if not hasattr(self, "_fc1_bsgs"):
            raise RuntimeError(
                "Call precompute_plaintexts(...) before save_cache(...)."
            )
        import pickle
        state = {
            "config": {
                "q_towers": list(self._q_towers),
                "p_towers": list(self._p_towers),
                "dnum": self._dnum,
                "sf": self._sf,
                "batch": self._batch,
            },
            "kp": self._cached_kp,
            "ek": self._cached_ek,
            "rotation_keys": self.ctx._raw_rotation_keys,
            "conv1_pts": [np.asarray(p) for p in self._conv1_pts],
            "conv1_rots": list(self._conv1_rots),
            "conv1_bias_pt": np.asarray(self._conv1_bias_pt),
            "fc1_bias_pt": np.asarray(self._fc1_bias_pt),
            "fc2_bias_pt": np.asarray(self._fc2_bias_pt),
            "fc1_bsgs_state": self._fc1_bsgs.serializable_state(),
            "fc2_bsgs_state": self._fc2_bsgs.serializable_state(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_cache(cls, path: str) -> "LoLAHE":
        """Reconstruct a LoLAHE from a previously saved cache file.

        If the cache predates the per-instance config keys (`q_towers`,
        `p_towers`, `dnum`, `sf` in `state["config"]`), default module-level
        constants are used (legacy 9Q+4P).
        """
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        cfg = state.get("config", {})
        inst = cls(
            _cached_state=state,
            q_towers=cfg.get("q_towers"),
            p_towers=cfg.get("p_towers"),
            dnum=cfg.get("dnum"),
            sf=cfg.get("sf"),
            batch=cfg.get("batch", 1),
        )
        inst._load_cached_precompute(state)
        return inst

    def _load_cached_precompute(self, state: dict) -> None:
        self._conv1_pts = [
            jnp.asarray(p, dtype=jnp.uint32) for p in state["conv1_pts"]
        ]
        self._conv1_rots = list(state["conv1_rots"])
        self._conv1_bias_pt = jnp.asarray(state["conv1_bias_pt"], dtype=jnp.uint32)
        self._fc1_bias_pt = jnp.asarray(state["fc1_bias_pt"], dtype=jnp.uint32)
        self._fc2_bias_pt = jnp.asarray(state["fc2_bias_pt"], dtype=jnp.uint32)

        n1, n2 = _bsgs.compute_bsgs_params(NUM_SLOTS)
        self._fc1_bsgs = self.ctx.bsgs_matvec[
            self._square1_out_level, NUM_SLOTS, n1, n2]
        self._fc1_bsgs.load_serialized_state(state["fc1_bsgs_state"])
        self._fc2_bsgs = self.ctx.bsgs_matvec[
            self._square2_out_level, NUM_SLOTS, n1, n2]
        self._fc2_bsgs.load_serialized_state(state["fc2_bsgs_state"])

        # Pre-compile the fused XLA graph for conv1 (default fast path).
        self.build_conv1_fused_jit(self.ctx.max_level)

    # --------------------------------------------------------------
    # Profiler entry point (for perf tests)
    # --------------------------------------------------------------
    def build_profiler(self, trace_dir):
        os.makedirs(trace_dir, exist_ok=True)
        profiler = Profiler(
            output_trace_path=trace_dir,
            profile_naming="lola_he_kernel_profile",
            configuration={"iterations": 1, "save_to_file": True},
        )
        cache = self.ctx._param_cache
        wrappers: dict[int, KernelWrapper] = {}
        for src in range(1, self._M):
            nq_in = cache.num_q_at_level(src)
            nq_out = cache.num_q_at_level(src - 1)
            kw = KernelWrapper(
                kernel_name=f"ptct_rescale_L{src}_L{src - 1}",
                function_to_wrap=_ptct_rescale_kernel,
                input_structs=[
                    ((1, 2, R, C, nq_in), jnp.uint32),
                    ((R, C, nq_in), jnp.uint32),
                ],
                parameters={"ptct_rescale_fn": self._ptctr_pure[src]},
            )
            profiler.add_profile(
                name=f"ptct_rescale_L{src}_L{src - 1}",
                kernel_wrapper=kw,
                kernel_setting_cols={
                    "degree": DEGREE,
                    "r": R,
                    "c": C,
                    "nq_in": nq_in,
                    "nq_out": nq_out,
                    "src_level": src,
                },
            )
            wrappers[src] = kw
        return profiler, wrappers


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def instantiate_model(spec: str | None = None) -> "LoLAHE":
    """Return a fresh LoLAHE instance. `spec` is accepted for backwards
    compatibility but is ignored since there is only one variant."""
    return LoLAHE()


def maybe_precompute(model: Any, weights: tuple[np.ndarray, ...]) -> None:
    if hasattr(model, "precompute_plaintexts"):
        model.precompute_plaintexts(*weights)


def run_demo(
    model,
    title: str,
    data: dict[str, np.ndarray],
    n: int,
    trace_last_image: bool = False,
):
    weights = prepare_weights(data)
    imgs = data["imgs"]
    labels = data["labels"]
    pt_pred = data["pt_pred"]
    num_images = min(n, len(imgs))

    print("=" * 70)
    print(title)
    print(f"  images={num_images}")
    print("=" * 70)

    wall_times = []
    he_correct = 0
    match_pt = 0
    for i in range(num_images):
        trace_dir = None
        if trace_last_image and i == num_images - 1:
            trace_dir = os.path.join(_DEMO_DIR, "log", f"inference_img{i}")
            os.makedirs(trace_dir, exist_ok=True)
        t0 = time.perf_counter()
        # `infer` returns a NumPy array (decrypted on CPU via fast_decrypt_decode),
        # which already implies a full sync — no extra block_until_ready needed.
        he_scores = model.infer(*((imgs[i],) + weights), trace_dir=trace_dir)
        wall = time.perf_counter() - t0
        wall_times.append(wall)

        he_argmax = int(np.argmax(he_scores))
        if he_argmax == labels[i]:
            he_correct += 1
        if he_argmax == pt_pred[i]:
            match_pt += 1
        pt_scores = lola_cleartext_inference(imgs[i], *weights)
        diff = float(np.max(np.abs(he_scores - pt_scores)))
        print(
            f"  [{i}] label={labels[i]} pt={pt_pred[i]} he={he_argmax} "
            f"|he-pt|={diff:.2e} wall={wall:.1f}s"
        )

    print(f"\n{'=' * 70}")
    print(
        f"HE accuracy: {he_correct}/{num_images} "
        f"({100 * he_correct / num_images:.1f}%)"
    )
    print(f"HE == PT:    {match_pt}/{num_images}")
    print(f"Avg wall:    {np.mean(wall_times):.1f}s/image")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="Number of images.")
    ap.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Profile ptct_rescale kernels at each level and trace the last "
            "full inference. Outputs under demos/log/."
        ),
    )
    args = ap.parse_args(argv)

    data, data_source = load_or_generate_data()
    weights = prepare_weights(data)
    print(f"[data] source={data_source}")

    model = LoLAHE()
    model.precompute_plaintexts(*weights)

    if args.profile:
        trace_dir = os.path.join(_DEMO_DIR, "log")
        print("\n[profile] Profiling ptct_rescale kernels ...")
        profiler, _ = model.build_profiler(trace_dir)
        profiler.profile_all_profilers()
        profiler.post_process_all_profilers()
        collect_logs(trace_dir, output_csv_name="lola_he_kernel_profiling")
        print(f"[profile] Results in {trace_dir}/\n")

    run_demo(
        model,
        f"CROSS HE LoLA (BSGS-FC1 + naive-FC2) — degree={DEGREE}, slots={NUM_SLOTS}",
        data,
        args.n,
        trace_last_image=args.profile,
    )
    if args.profile:
        print(f"Logs:        {os.path.join(_DEMO_DIR, 'log')}/")


if __name__ == "__main__":
    main()
