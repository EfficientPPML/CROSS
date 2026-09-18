"""Core LoLA HE model and shared utilities.

`LoLAHE` is the encrypted LoLA-MNIST entry point. It no longer builds a
packing graph: it is a thin, signature-preserving adapter over the one
canonical pipeline every demo goes through::

    torch.nn.Module  ->  nn.vectorize  ->  packing.pack  ->  Mapping

The torch module is `lola_train.QuadLoLA`, reached through
`encrypted_demos.LoLADemo`. `precompute_plaintexts` writes the caller's
arrays into that module and then materializes exactly one Mapping;
`infer` runs it.

What that replaced. The old class lowered LoLA by hand: eight grouped
`MulPlain`/`Rescale`/`Rotate` branches under a `ParallelSum` for conv1, and
`SparseDiagonals` built from each dense layer's non-zero diagonals for fc1
and fc2 -- 37 PP-ops in all. The packer emits the same computation as three
`matvec` ops (conv1, fc1, fc2), three `add_plain` biases and two `square`
activations, at depth 5. The eight-branch decomposition was a hand-chosen
lowering of one linear map; the packer chooses it now, so it is gone.

The ring is a *result* of packing, not an input to it. `packing.pack` sizes
the modulus chain from the program's own slot demand and emitted depth and
validates it at 128-bit security, which is why `q_towers`, `p_towers` and
`sf` can no longer be supplied. For LoLA it derives degree 32768 / 16384
slots, num_q 6, dnum 3. The module-level `DEGREE`, `NUM_SLOTS`, `Q_TOWERS`,
`P_TOWERS`, `SF` and friends below are the retired hand-tuned ring; they are
kept only because `lenet_he.py` and `jaxite_word/ckks_ctx_test.py` still
import them, and they no longer describe anything `LoLAHE` runs. Ask the
instance instead: `.ring_config`, `.packed_program`, `.metadata()`.

Everything cleartext -- `load_trained`, `generate`, `load_or_generate_data`,
`prepare_weights`, `pack_lola_input`, `conv1_cleartext` and
`lola_cleartext_inference` -- is unchanged and remains the reference the
encrypted path is checked against.
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


from cleartext_ops import add_conv_bias, conv2d_ref, matmul_ref, quad_ref
import canonical_demo

# ---------------------------------------------------------------------------
# Data loading (inlined from former lola_data_loader.py).  load_trained()
# reads BN-folded weights + the 200-image MNIST test slice from the
# `$CROSS_DATA_ROOT/pretrained_weights/lola` (override with the
# LOLA_WEIGHTS_DIR env var). generate() prefers loading the trained set
# but falls back to a deterministic random tensor if it isn't present.
# ---------------------------------------------------------------------------
_DATA_SEED = 42
_N_TEST_IMAGES = 200
_DATA_ROOT = os.environ.get("CROSS_DATA_ROOT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
_TRAINED_DIR = os.environ.get(
    "LOLA_WEIGHTS_DIR",
    os.path.join(_DATA_ROOT, "pretrained_weights", "lola"),
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

    Reads from `LOLA_WEIGHTS_DIR` (default: `$CROSS_DATA_ROOT/pretrained_weights/lola`). Optionally
    copies the same files into `data_dir` for callers that want a local
    snapshot. Raises `FileNotFoundError` if the weights aren't on disk.
    """
    src = _TRAINED_DIR
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Trained weights not found at {src}. Populate it with "
            "`python3 demos/lola_train.py --train` (or "
            "`python3 demos/prepare_demo_assets.py` to fetch data and train "
            "every demo model)."
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
FC1_IN, FC1_OUT = 980, 100
FC2_IN, FC2_OUT = 100, 10

# ---------------------------------------------------------------------------
# RETIRED RING. These constants were the hand-tuned CKKS parameters the old
# hand-built LoLA graph ran on (7Q + 3P, degree 2048, 1024 slots, dnum 3).
# `LoLAHE` does not read any of them any more: `packing.pack` derives the
# modulus chain from the packed program and validates it at 128-bit security,
# and passing q_towers / p_towers / sf to LoLAHE is now an error.
#
# They stay exported because `lenet_he.py` reuses the tower pools, DEGREE,
# NUM_SLOTS, R and C for its own hand-built graph.
# `pack_lola_input` also still defaults to the 1024-slot stride-multiplexed
# layout, which is what the cleartext reference below is written against.
# ---------------------------------------------------------------------------
DEGREE = 2048
NUM_SLOTS = 1024
R, C = 32, 64
DNUM = 3
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
    out = conv2d_ref(
        img_flat, w1_flat, 1, CO, H_IN, W_IN, KH, KW, 2, 0
    )
    return add_conv_bias(out, b1, CO, H_OUT, W_OUT).reshape(
        CO, H_OUT, W_OUT
    )


def lola_cleartext_inference(img_flat, W1, b1, W2, b2, W3, b3):
    conv = conv1_cleartext(img_flat, W1, b1)
    q1 = quad_ref(conv)
    fc1 = matmul_ref(q1.ravel(), W2, FC1_OUT, FC1_IN) + b2
    q2 = quad_ref(fc1)
    return matmul_ref(q2, W3, FC2_OUT, FC2_IN) + b3


def load_or_generate_data():
    try:
        data = load_trained()
        source = "trained"
    except FileNotFoundError:
        fallback_dir = os.path.join(_DATA_ROOT, "pretrained_weights", "lola")
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


_DERIVED_RING_MESSAGE = (
    "The ring is derived from the model, not supplied: packing.pack sizes the "
    "modulus chain from the packed program's own slot demand and emitted "
    "depth, then validates it at 128-bit security. Passing {names} would run "
    "LoLA on parameters that were never validated for it. Drop the "
    "argument(s) and read the ring back from .ring_config / .metadata(); "
    "batch, devices and dnum still choose how the derived program runs."
)

_LEGACY_CACHE_MESSAGE = (
    "LoLA caches are retired. A cache pickled a hand-lowered network plus the "
    "hand-tuned 7Q+3P ring; both are now derived -- the program by "
    "nn.vectorize/packing.pack from the torch model, the ring by packing.pack "
    "from that program -- so a saved one describes neither what runs nor what "
    "it runs on. Build the model directly: LoLAHE() then "
    "precompute_plaintexts(W1, b1, W2, b2, W3, b3)."
)


# ---------------------------------------------------------------------------
# LoLAHE: single end-to-end encrypted inference class.
# ---------------------------------------------------------------------------
class LoLAHE(canonical_demo.CanonicalDemoAdapter):
    """Encrypted LoLA-MNIST inference, delegated to the canonical pipeline.

    Holds one `encrypted_demos.LoLADemo`, which owns the torch module, the
    vectorized program, the packed program and the single Mapping. This class
    contributes only the legacy call sequence -- construct,
    `precompute_plaintexts(...)`, `infer(...)` -- and the cleartext-facing
    conveniences the demos and tests around it still use.
    """

    def __init__(
        self,
        _cached_state: dict | None = None,
        q_towers: list[int] | None = None,
        p_towers: list[int] | None = None,
        dnum: int | None = None,
        sf: float | None = None,
        batch: int = 1,
        devices=None,
    ):
        """Construct a LoLAHE instance.

        `batch` controls the leading axis of every ciphertext (the "image"
        dim): `batch=N` lets a single LoLAHE process N independent images per
        HE-eval. `devices` selects the runtime Mapping placement. `dnum` is
        the HYBRID key-switching digit count; leave it `None` to take the
        derived ring's own, which is the only value Mapping accepts -- one
        that contradicts the plan is refused there rather than silently run.
        These three are scheduling: they change how the program runs, never
        what it computes or which ring it runs on.

        `q_towers`, `p_towers` and `sf` are rejected -- they are ring
        configuration, and the ring is now derived. `_cached_state` is
        rejected too; see `from_cache`.
        """
        if _cached_state is not None:
            raise ValueError(_LEGACY_CACHE_MESSAGE)
        supplied = [
            name
            for name, value in (
                ("q_towers", q_towers),
                ("p_towers", p_towers),
                ("sf", sf),
            )
            if value is not None
        ]
        if supplied:
            raise ValueError(
                _DERIVED_RING_MESSAGE.format(names=", ".join(supplied))
            )

        # Deferred: `encrypted_demos` pulls in torch and the whole demo model
        # zoo, and `lenet_he.py` imports this module only for constants.
        import encrypted_demos

        super().__init__(
            encrypted_demos.LoLADemo,
            batch=batch,
            devices=devices,
            dnum=dnum,
        )

    def _on_plan_ready(self, packed) -> None:
        ring = packed.ring_config
        print(
            f"[precompute] depth {packed.depth} over {packed.num_slots} slots "
            f"-> degree {int(ring.degree)} "
            f"(num_q {int(ring.logical_num_q)}, dnum {int(ring.dnum)}) at "
            f"{int(ring.security_bits)}-bit security"
        )

    # --------------------------------------------------------------
    # Offline: bind the caller's weights, then materialize one Mapping.
    # --------------------------------------------------------------
    def precompute_plaintexts(self, W1, b1, W2, b2, W3, b3,
                              fc1_bsgs_ratio: float = 2.0,
                              fc2_bsgs_ratio: float = 2.0):
        """Bind the caller's constants and build this instance's Mapping.

        The arrays are the model, so they are written into the torch module
        `nn.vectorize` traces -- not layered beside it -- and the program and
        ring are then derived from that module.
        """
        t0 = time.perf_counter()
        print("[precompute] vectorizing, packing and mapping LoLA ...")
        self._prepare(
            (W1, b1, W2, b2, W3, b3),
            legacy_bsgs_ratios={
                'fc1_bsgs_ratio': fc1_bsgs_ratio,
                'fc2_bsgs_ratio': fc2_bsgs_ratio,
            },
        )
        print(f"[precompute] done in {time.perf_counter() - t0:.1f}s")

    # --------------------------------------------------------------
    # Cache: retired, because the program and its ring are derived.
    # --------------------------------------------------------------
    def save_cache(self, path: str) -> None:
        """Retired. See `_LEGACY_CACHE_MESSAGE`."""
        raise RuntimeError(_LEGACY_CACHE_MESSAGE)

    @classmethod
    def from_cache(
        cls,
        path: str,
        *,
        batch: int | None = None,
        devices=None,
    ) -> "LoLAHE":
        """Retired. Rejected without unpickling the file."""
        raise ValueError(f"{path!r}: {_LEGACY_CACHE_MESSAGE}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
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
        # `infer` returns a NumPy array (decrypted via the Mapping's context),
        # which already implies a full sync — no extra block_until_ready needed.
        he_scores = model.infer(imgs[i], trace_dir=trace_dir)
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1, help="Number of images.")
    ap.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Trace the final full Mapping execution under demos/log/."
        ),
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Vectorize and pack, print the derived ring, and stop. Skips the "
            "expensive half: no keys, no constants, no Mapping."
        ),
    )
    args = ap.parse_args(argv)

    data, data_source = load_or_generate_data()
    weights = prepare_weights(data)
    print(f"[data] source={data_source}")

    model = LoLAHE()

    if args.plan_only:
        model.demo.load_positional_weights(*weights)
        for key, value in sorted(model.metadata().items()):
            print(f"{key}: {value}")
        return

    model.precompute_plaintexts(*weights)
    ring = model.ring_config
    run_demo(
        model,
        "CROSS HE LoLA (canonical pipeline) — "
        f"degree={int(ring.degree)}, slots={model.packed_program.num_slots}",
        data,
        args.n,
        trace_last_image=args.profile,
    )
    if args.profile:
        print(f"Logs:        {os.path.join(_DEMO_DIR, 'log')}/")


if __name__ == "__main__":
    main()
