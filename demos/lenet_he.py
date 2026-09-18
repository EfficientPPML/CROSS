"""CROSS port of the OpenFHE MAPLE-LeNet demo, on the canonical pipeline.

Mirrors `openfhe_ref_code/models_openfhe/maple-lenet.cpp` — a SCALED-DOWN LeNet
under CKKS, end-to-end encrypted forward pass with Quad activations:

    Input X: 1x28x28
    Conv1(1->4,  5x5, s=2, p=2)  -> 4x14x14
    Quad1
    Conv2(4->8,  5x5, s=2, p=2)  -> 8x7x7
    Quad2
    FC1(392 -> 32)
    Quad3
    FC2(32  -> 10)

Nothing in this file builds that network out of packing IR any more. The model
of record is `lenet_train.QuadLeNet`, and it reaches CKKS through the one
pipeline every demo uses:

    torch.nn.Module  ->  nn.vectorize  ->  packing.pack  ->  Mapping

`LeNetHE` is a thin compatibility shell over `encrypted_demos.LeNetDemo`: it
keeps the legacy positional-weight entrypoint, the encrypt/infer surface and
the CLI, and forwards everything else. Multiplicative depth is still 7 (one
BSGS matvec per linear layer plus one ct*ct per Quad), but neither the slot
layout nor the ring is chosen here:

  * conv1's stride-multiplexed layout is now one of the layout templates the
    packer scores while packing, not a demo flag;
  * degree, tower pool, scaling factor and dnum are derived by `packing.pack`
    from this program's own slot demand and depth. `.ring_config` reports what
    it picked; the module-level pool constants below are legacy exports.

Historical performance measurements predate the canonical pipeline and must be
remeasured on it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DEMO_DIR)
_JAXITE = os.path.abspath(os.path.join(_DEMO_DIR, "..", "jaxite_word"))
for p in (_REPO_ROOT, _JAXITE, _DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


import canonical_demo
import encrypted_demos

from cleartext_ops import conv2d_ref, conv_bias
from cleartext_ops import matmul_ref, quad_ref
import lola_he as _lola  # legacy 9-tower/4-P pool + DEGREE / R / C / NUM_SLOTS


# ---------------------------------------------------------------------------
# Legacy CKKS constants. The executed ring is derived by `packing.pack` and
# read back from `LeNetHE.ring_config`; these names survive because
# `alexnet_he.py` and the LeNet perf gate still import them.
# ---------------------------------------------------------------------------
DEGREE = _lola.DEGREE
NUM_SLOTS = _lola.NUM_SLOTS
R = _lola.R
C = _lola.C
Q_TOWERS = list(_lola.Q_TOWERS_POOL)             # 9 towers -> max_level=8
P_TOWERS = list(_lola.P_TOWERS_POOL)             # 4 P-towers
DNUM = 4
SF = Q_TOWERS[0] * Q_TOWERS[1]
SIGMA = _lola.SIGMA
_LENET_CACHE_FORMAT = "lenet-packing-mapping-v3"


# ---------------------------------------------------------------------------
# LeNet (scaled) architecture — matches maple-lenet.cpp and QuadLeNet.
# ---------------------------------------------------------------------------
CI = 1
H_IN = 28
W_IN = 28
CO1 = 4
H1 = 14
W1 = 14
CO2 = 8
H2 = 7
W2 = 7
KH = 5
KW = 5
STRIDE = 2
PAD = 2
FC1_IN = CO2 * H2 * W2     # 392
FC1_OUT = 32
FC2_OUT = 10


_broadcast_conv_bias = conv_bias


def lenet_cleartext(X, W1w, W2w, W3w, W4w,
                    b1=None, b2=None, b3=None, b4=None):
    """Run the full LeNet cleartext reference (matches OpenFHE layout).

    Biases are optional — pass `None` (or zero arrays) for the original
    no-bias C++ reference; pass real per-output-channel/per-output-neuron
    biases to match a trained model.
    """
    y = conv2d_ref(X, W1w, CI, CO1, H_IN, W_IN, KH, KW, STRIDE, PAD)
    if b1 is not None:
        y = y + _broadcast_conv_bias(b1, CO1, H1, W1)
    y = quad_ref(y)
    y = conv2d_ref(y, W2w, CO1, CO2, H1, W1, KH, KW, STRIDE, PAD)
    if b2 is not None:
        y = y + _broadcast_conv_bias(b2, CO2, H2, W2)
    y = quad_ref(y)
    y = matmul_ref(y, W3w, FC1_OUT, FC1_IN)
    if b3 is not None:
        y = y + b3[:FC1_OUT]
    y = quad_ref(y)
    y = matmul_ref(y, W4w, FC2_OUT, FC1_OUT)
    if b4 is not None:
        y = y + b4[:FC2_OUT]
    return y


# ---------------------------------------------------------------------------
# Legacy slot packers. The layout a ciphertext actually uses is now chosen by
# the packer and applied by `Packing.pack` (reachable as
# `LeNetHE.packed_program.pack`), so serving code needs neither of these.
# They are kept because they document the two hand-lowered layouts the demo
# used before the migration, and are cheap, self-contained numpy.
# ---------------------------------------------------------------------------
def pack_input(X, num_slots=NUM_SLOTS):
    """Pack a (Ci*Hi*Wi)-flat real input into a row-major slot vector."""
    out = np.zeros(num_slots, dtype=np.float64)
    out[: X.size] = X
    return out


def pack_input_multiplexed_conv1(X, num_slots=NUM_SLOTS):
    """Stride-multiplexed input packing for a conv1 with stride S.

    Slot rs * H_out*W_out + h * W_out + w  (rs ∈ [0, S²); h, w ∈ [0, H_out))
    holds X[(S*h + r%S) * W_in + (S*w + s%S)] for the unique r, s with that
    stride sub-position.  H_out = H_in / S = 14 for the 28×28 / s=2 case.

    The packer emits this family of layouts itself when it scores cheaper than
    row-major; the copy here only describes it.
    """
    H_out = H_IN // STRIDE
    W_out = W_IN // STRIDE
    out = np.zeros(num_slots, dtype=np.float64)
    for h in range(H_out):
        for w in range(W_out):
            for r_mod in range(STRIDE):
                for s_mod in range(STRIDE):
                    rs = r_mod * STRIDE + s_mod
                    src = (STRIDE * h + r_mod) * W_IN + (STRIDE * w + s_mod)
                    out[rs * H_out * W_out + h * W_out + w] = X[src]
    return out


# ---------------------------------------------------------------------------
# Random-weight generator. The C++ uses std::mt19937(42) + N(0, 0.1); numpy's
# default_rng(42) is a different stream, which is fine — what matters for the
# test is that the cleartext reference and HE forward use the SAME weights.
# ---------------------------------------------------------------------------
def lenet_random_inputs(seed: int = 42):
    """Random Gaussian inputs + weights matching the C++ reference exactly.

    Biases are returned as zeros so both cleartext evaluation and the
    encrypted forward match the bias-free C++ demo.
    """
    rng = np.random.default_rng(seed)
    return {
        "X":   rng.normal(0.0, 0.1, size=CI * H_IN * W_IN).astype(np.float64),
        "W1":  rng.normal(0.0, 0.1, size=CO1 * CI * KH * KW).astype(np.float64),
        "W2":  rng.normal(0.0, 0.1, size=CO2 * CO1 * KH * KW).astype(np.float64),
        "W3":  rng.normal(0.0, 0.1, size=FC1_OUT * FC1_IN).astype(np.float64),
        "W4":  rng.normal(0.0, 0.1, size=FC2_OUT * FC1_OUT).astype(np.float64),
        "b1":  np.zeros(CO1, dtype=np.float64),
        "b2":  np.zeros(CO2, dtype=np.float64),
        "b3":  np.zeros(FC1_OUT, dtype=np.float64),
        "b4":  np.zeros(FC2_OUT, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Trained-weight loader. Looks for binaries written by `lenet_train.py`.
# Returns `None` if any file is missing.
# ---------------------------------------------------------------------------
_TRAINED_DIR = os.path.join(_DEMO_DIR, "maple_data")
_TRAINED_FILES = {
    "W1": (CO1 * CI * KH * KW,        "lenet_conv1_W.bin"),
    "b1": (CO1,                       "lenet_conv1_b.bin"),
    "W2": (CO2 * CO1 * KH * KW,       "lenet_conv2_W.bin"),
    "b2": (CO2,                       "lenet_conv2_b.bin"),
    "W3": (FC1_OUT * FC1_IN,          "lenet_fc1_W.bin"),
    "b3": (FC1_OUT,                   "lenet_fc1_b.bin"),
    "W4": (FC2_OUT * FC1_OUT,         "lenet_fc2_W.bin"),
    "b4": (FC2_OUT,                   "lenet_fc2_b.bin"),
}
# MNIST normalisation matching torchvision's standard MNIST transform.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_trained_lenet_weights(data_dir: str | None = None) -> dict | None:
    src = data_dir or _TRAINED_DIR
    out: dict[str, np.ndarray] = {}
    for key, (n_expect, fname) in _TRAINED_FILES.items():
        path = os.path.join(src, fname)
        if not os.path.isfile(path):
            return None
        arr = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
        if arr.size != n_expect:
            raise ValueError(
                f"{path}: expected {n_expect} float64s, got {arr.size}")
        out[key] = arr.copy()
    return out


def normalize_mnist_image(x: np.ndarray) -> np.ndarray:
    """Apply (x - mean) / std with the MNIST canonical statistics.

    Operates on a flat (Ci*H*W,) or unraveled image. Use BEFORE encrypting
    when the model was trained with torchvision's `Normalize` transform.
    """
    return (x - MNIST_MEAN) / MNIST_STD


# ---------------------------------------------------------------------------
# LeNetHE — the legacy entrypoint, over the canonical pipeline.
# ---------------------------------------------------------------------------
class LeNetHE(canonical_demo.CanonicalDemoAdapter):
    """Encrypted LeNet-MNIST inference, delegated to `LeNetDemo`.

    This class owns no compilation of its own. It translates the legacy
    positional-weight call into the demo's torch model, then lets
    vectorize/pack/Mapping do the rest.
    """

    def __init__(self, batch: int = 1,
                 dnum: int | None = None,
                 use_multiplexed_conv1: bool = True,
                 devices=None,
                 _cached_state: dict | None = None):
        """Construct a LeNetHE instance.

        Args:
          batch: leading axis of every ciphertext; the demo's `global_batch`.
          dnum: key-switch decomposition factor. Defaults to module-level
            `DNUM=4`, which is also what the derived ring asks for at this
            depth.
          use_multiplexed_conv1: retained for signature compatibility only.
            Layout selection is the packer's now — it scores stride-
            multiplexed against row-major layout templates while packing —
            so this flag no longer selects a conv1 lowering.
          devices: runtime JAX devices used by Mapping. Device handles are
            deployment state and never travel with the model.
          _cached_state: legacy pickled key/constant state. Rejected; see
            `from_cache`.
        """
        if _cached_state is not None:
            raise ValueError(
                "LeNetHE no longer accepts a pickled cache: its securely "
                "derived ring comes from packing.pack and its constants from "
                "the packed program, so a legacy cache "
                f"({_LENET_CACHE_FORMAT} and older) pins neither the ring nor "
                "the graph that runs. Construct LeNetHE() and call "
                "precompute_plaintexts(...) instead."
            )
        super().__init__(
            encrypted_demos.LeNetDemo,
            batch=batch,
            devices=devices,
            dnum=dnum,
        )
        # Stored so callers that still pass it keep working. Layout selection
        # belongs to the packer now -- it scores conv1 against its layout
        # templates -- so this flag no longer picks a lowering.
        self._use_multiplexed_conv1 = bool(use_multiplexed_conv1)
        self._bsgs_ratio = None

    def _on_plan_ready(self, packed) -> None:
        ring = packed.ring_config
        print(
            f"[setup] degree={int(ring.degree)}, "
            f"num_slots={packed.num_slots}, depth={packed.depth}, "
            f"dnum={int(ring.dnum)}, "
            f"security={int(ring.security_bits)}-bit"
        )

    # --------------------------------------------------------------
    # Offline: bind this caller's weights, then compile them, once.
    # --------------------------------------------------------------
    def precompute_plaintexts(self, W1w, W2w, W3w, W4w,
                              b1=None, b2=None, b3=None, b4=None,
                              bsgs_ratio: float = 2.0):
        """Bind model weights and compile the one static HE network.

        The arrays are written into the torch QuadLeNet that `nn.vectorize`
        traces, so what gets compiled is the caller's model — never the demo's
        seeded fallback. A `None` bias leaves that layer's bias at zero, which
        is what the old lowering produced explicitly.

        `bsgs_ratio` is retained only for compatibility: the historic default
        2.0 is accepted, while any other global ratio is rejected because
        Mapping chooses the split independently for each matvec.
        """
        print("[precompute] binding weights and compiling ...")
        self._bsgs_ratio = float(bsgs_ratio)
        self._prepare(
            (W1w, W2w, W3w, W4w, b1, b2, b3, b4),
            legacy_bsgs_ratios={'bsgs_ratio': bsgs_ratio},
        )

    # --------------------------------------------------------------
    # Caches. A cache pinned keys, a hand-picked modulus pool and hand-lowered
    # constants — none of which describe a program whose ring is derived.
    # --------------------------------------------------------------
    def save_cache(self, path: str) -> None:
        raise NotImplementedError(
            "LeNetHE caches were removed with the hand-built graph: the ring "
            "is now derived from the packed program, and the constants are "
            "the packer's, so there is nothing left that a cache could pin. "
            "Rebuild with LeNetHE().precompute_plaintexts(...) — planning is "
            "cheap and key generation is well under a second."
        )

    @classmethod
    def from_cache(
        cls,
        path: str,
        *,
        batch: int | None = None,
        devices=None,
    ) -> "LeNetHE":
        raise ValueError(
            f"{path} is a legacy LeNet cache. Its keys belong to a hand-picked "
            "modulus pool and its constants to the hand-lowered packing graph; "
            "the securely derived ring and the Packing format make both "
            "invalid. Construct LeNetHE() and call precompute_plaintexts(...)."
        )


# ---------------------------------------------------------------------------
# Demo entry point — generates random inputs (matching the C++ shape),
# runs the cleartext reference + HE forward, prints per-stage timing.
# ---------------------------------------------------------------------------
def run_demo(seed: int = 42, trace: bool = False):
    print("=" * 64)
    print("CROSS HE LeNet (scaled) — depth-7, canonical pipeline")
    print(f"  Conv1({CI}->{CO1}, {KH}x{KW}/s{STRIDE}/p{PAD}) -> Quad")
    print(f"  Conv2({CO1}->{CO2}, {KH}x{KW}/s{STRIDE}/p{PAD}) -> Quad")
    print(f"  FC1({FC1_IN}->{FC1_OUT}) -> Quad")
    print(f"  FC2({FC1_OUT}->{FC2_OUT})")
    print("=" * 64)

    inputs = lenet_random_inputs(seed=seed)
    X = inputs["X"]
    Wargs = (inputs["W1"], inputs["W2"], inputs["W3"], inputs["W4"])
    Bargs = (inputs["b1"], inputs["b2"], inputs["b3"], inputs["b4"])

    print(f"\n[ref] cleartext forward (seed={seed}) ...")
    Y_ref = lenet_cleartext(X, *Wargs, *Bargs)
    print(f"  Ref logits: {np.array2string(Y_ref, precision=6, max_line_width=200)}")

    print("\n[setup] building LeNetHE ...")
    model = LeNetHE()
    model.precompute_plaintexts(*Wargs, *Bargs)

    print("\n[infer] HE forward pass ...")
    trace_dir = (os.path.join(_DEMO_DIR, "log", "lenet_inference")
                 if trace else None)
    t0 = time.perf_counter()
    Y_he = np.asarray(model.infer(X, trace_dir=trace_dir)).reshape(-1)
    wall_s = time.perf_counter() - t0
    print(f"  HE  logits: {np.array2string(Y_he, precision=6, max_line_width=200)}")
    print(f"  wall = {wall_s:.2f} s")

    diff = float(np.max(np.abs(Y_he - Y_ref)))
    print(f"\n[verify] max |HE - ref| = {diff:.4e}")
    return Y_he, Y_ref, diff, wall_s


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for X and weights.")
    ap.add_argument("--profile", action="store_true",
                    help="Emit a jax.profiler trace under demos/log/.")
    args = ap.parse_args(argv)
    run_demo(seed=args.seed, trace=args.profile)


if __name__ == "__main__":
    main()
