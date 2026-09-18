"""Performance gates for the LeNet HE pipeline.

Canonical throughput mode is **B=64 with fused-forward**. The current
TPUv6e-8 baseline is 30.855 s per global batch, or 482.1 ms/image, with eight
local ciphertexts assigned to each of the eight devices.
The perf gate measures the Mapping-owned fused executable end to end. The old
per-stage diagnostic path depended on a second interpreted execution mode and
was removed with that compiler lifecycle.

Two entry points:

1. **absltest gate** (no subcommand): with `LENET_PERF=1`, builds the B=1
   model (~16 min), runs
   the fused end-to-end warm-wall benchmark, and renders
   `demos/log/lenet_he_perf_test.png`:

       LENET_PERF=1 python3 lenet_he_perf_test.py

2. **CLI subcommands**:

       python3 lenet_he_perf_test.py build   [--batch 64]
       python3 lenet_he_perf_test.py forward [--n 8]

   - `build`: materialize a Mapping for `--batch` images and run the fused
     e2e benchmark over exactly that many.
   - `forward`: materialize a B=1 Mapping, then predict N MNIST images one
     at a time. It takes no `--batch`: a Mapping serves one exact global
     batch, and this command feeds `infer` a single image per call.

   Neither writes or reads a cache: the plan is re-derived from the torch
   model, and the constants follow from the weights.

Numeric correctness checks live in `lenet_he_test.py`. The
`LeNetRegressionTest` class in this file checks B=1 latency, B=32
throughput, label/cleartext-ref agreement, rebuild equivalence, and memory
footprint at B=1, B=32 and B=64. It builds one Mapping per batch and keeps
only one resident at a time, so it is opt-in: set `LENET_REGRESSION=1`. An
opt-in 200-image deep validation gate is enabled with `LENET_DEEP_VAL=1`.

Set `LENET_TEST_NO_PNG=1` to skip the PNG render (CI without matplotlib).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from unittest import mock

import numpy as np

from absl.testing import absltest
import jax

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_JAXITE = os.path.abspath(os.path.join(_DEMO_DIR, "..", "jaxite_word"))
for p in (_JAXITE, _DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

jax.config.update("jax_enable_x64", True)

from lenet_he import (                                       # noqa: E402
    CI, CO1, CO2, FC1_IN, FC1_OUT, FC2_OUT,
    H_IN, W_IN,
    LeNetHE, lenet_cleartext, lenet_random_inputs,
    load_trained_lenet_weights, normalize_mnist_image,
)
from demo_test_utils import RecordingPerfModel as _RecordingModel  # noqa: E402
_LOG_DIR = os.path.join(_DEMO_DIR, "log")
_DATA_DIR = os.path.join(_DEMO_DIR, "maple_data")
_PNG_OUT = os.path.join(_LOG_DIR, "lenet_he_perf_test.png")

_N_WARM = 2
_N_SAMPLE = 5


def benchmark_end_to_end(model: LeNetHE, X) -> tuple[float, float]:
    """Median + min wall over `_N_SAMPLE` warm full inferences."""
    for _ in range(_N_WARM):
        _ = model.infer(X)
    samples = []
    for _ in range(_N_SAMPLE):
        t0 = time.perf_counter()
        _ = model.infer(X)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)), float(np.min(samples))


def render_perf_png(image_2d: np.ndarray, label: int,
                    he_arg: int, ref_arg: int, max_err: float,
                    e2e_wall_ms: float,
                    timings: list[tuple[str, float]],
                    out_path: str, *, mode_label: str = "trained") -> None:
    """Render the prediction and Mapping-owned serving timing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 2.4]})
    ax_im = axes[0]
    ax_im.imshow(image_2d, cmap="Greys", vmin=0.0, vmax=1.0)
    ax_im.set_xticks([]); ax_im.set_yticks([])
    is_correct = (he_arg == label) if label >= 0 else (he_arg == ref_arg)
    color = "green" if is_correct else "red"
    for s in ax_im.spines.values():
        s.set_edgecolor(color); s.set_linewidth(3)
    badge = ("[correct]" if (he_arg == label and label >= 0)
             else "[match]" if (he_arg == ref_arg) else "[MISS]")
    title = (f"label={label if label >= 0 else 'rand'}\n"
             f"ref={ref_arg} he={he_arg}  {badge}\n"
             f"|HE-ref|={max_err:.1e}\n"
             f"e2e wall = {e2e_wall_ms:.0f} ms")
    ax_im.set_title(title, color=color, fontsize=10)

    ax_bar = axes[1]
    names = [name for name, _ in timings][::-1]
    walls = [wall for _, wall in timings][::-1]
    y = np.arange(len(names))
    ax_bar.barh(y, walls, color="#3a82f6")
    s_total = sum(walls)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=10)
    ax_bar.set_xlabel("wall time (ms)")
    ax_bar.set_title("Mapping-owned fused serving path", fontsize=10)
    for k, w in enumerate(walls):
        pct = 100 * w / s_total if s_total else 0
        ax_bar.text(w + max(walls) * 0.01, k,
                    f" {w:.1f} ms ({pct:.1f}%)",
                    va="center", fontsize=9)
    ax_bar.set_xlim(0, max(walls) * 1.25 if walls else 1.0)

    fig.suptitle(f"CROSS HE LeNet — performance gate ({mode_label})",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Shared input resolver (used by both absltest gate and CLI subcommands)
# ==========================================================================
def _resolve_inputs(image_idx: int = 0, n_imgs: int = 1):
    """Return (Wargs, Bargs, prepare_x, weight_src, sample_X, sample_label).

    Trained weights take precedence; falls back to random Gaussian.  When
    using random, the "image" is the seeded Gaussian X array and label=-1.
    """
    trained = load_trained_lenet_weights()
    if trained is not None:
        Wargs = (trained["W1"], trained["W2"], trained["W3"], trained["W4"])
        Bargs = (trained["b1"], trained["b2"], trained["b3"], trained["b4"])
        prepare_x = normalize_mnist_image
        if _mnist_data_available():
            imgs, labels = _load_mnist_slice(start=image_idx, n=n_imgs)
            sample_X = imgs[0]
            sample_label = int(labels[0])
        else:
            # The checkpoint is useful for a real encrypted performance run
            # even when the optional validation corpus was not downloaded.
            # A zero image preserves the graph and weights; label accuracy is
            # intentionally unavailable, while HE-vs-cleartext remains valid.
            sample_X = np.zeros(H_IN * W_IN, dtype=np.float64)
            sample_label = -1
        return Wargs, Bargs, prepare_x, "trained", sample_X, sample_label
    rng = lenet_random_inputs(seed=42)
    Wargs = (rng["W1"], rng["W2"], rng["W3"], rng["W4"])
    Bargs = (rng["b1"], rng["b2"], rng["b3"], rng["b4"])
    return Wargs, Bargs, (lambda x: x), "random", rng["X"], -1


def _mnist_data_available() -> bool:
    return all(os.path.isfile(path) for path in (
        os.path.join(_DATA_DIR, "mnist_test_200.bin"),
        os.path.join(_DATA_DIR, "mnist_test_200_labels.bin"),
    ))


def _load_mnist_slice(start: int = 0, n: int = 1):
    imgs_path = os.path.join(_DATA_DIR, "mnist_test_200.bin")
    lbls_path = os.path.join(_DATA_DIR, "mnist_test_200_labels.bin")
    imgs = np.frombuffer(open(imgs_path, "rb").read(),
                         dtype=np.float64).reshape(-1, H_IN * W_IN)
    labels = np.frombuffer(open(lbls_path, "rb").read(), dtype=np.int32)
    take = [(start + i) % imgs.shape[0] for i in range(n)]
    return imgs[take], labels[take]


def _build_model(batch: int = 1):
    """Bind weights and materialize one Mapping, from the torch model.

    This is what the pickled cache used to stand in for. The cache is gone:
    the plan is re-derived from `lenet_train.QuadLeNet` cheaply, and key
    generation is well under a second. What it cost -- constant
    materialization -- a cache could not have skipped safely anyway, because
    the constants follow from the weights.
    """
    Wargs, Bargs, _, _, _, _ = _resolve_inputs()
    model = LeNetHE(batch=batch)
    model.precompute_plaintexts(*Wargs, *Bargs)
    return model


# ==========================================================================
# Memory profiling — coarse footprint of the live model state.
# ==========================================================================
def memory_footprint(model: LeNetHE) -> dict[str, float]:
    """Report the shared Mapping live-state estimate in MiB.

    Reports:
      ciphertext_in_MB      — single fresh ciphertext at L_max
      ciphertext_out_MB     — single ciphertext at L_min (after depth-7)
      bsgs_state_MB         — sum of all encoded BSGS diagonal plaintexts
      bias_pts_MB           — sum of the 4 bias plaintexts
      eval_keys_MB          — relinearization key (Q+P) × 2 polynomials
      rotation_keys_MB      — sum of all (Q+P) × 2 rotation keys
      total_live_MB
    """
    estimate = model.mapping.estimate_live_memory()
    MB = 1024 * 1024
    return {
        "ciphertext_in_MB": estimate["ciphertext_in_bytes"] / MB,
        "ciphertext_out_MB": estimate["ciphertext_out_bytes"] / MB,
        "bsgs_state_MB": estimate["matvec_constants_bytes"] / MB,
        "bias_pts_MB": estimate["plaintext_constants_bytes"] / MB,
        "eval_keys_MB": estimate["evaluation_key_bytes"] / MB,
        "rotation_keys_MB": estimate["rotation_keys_bytes"] / MB,
        "n_rotation_keys": estimate["rotation_key_count"],
        "total_live_MB": estimate["total_bytes"] / MB,
    }


def print_memory(model: LeNetHE) -> None:
    mem = memory_footprint(model)
    print("\n[memory] approximate live model state:")
    for k in ("ciphertext_in_MB", "ciphertext_out_MB",
              "bsgs_state_MB", "bias_pts_MB",
              "eval_keys_MB", "rotation_keys_MB"):
        print(f"  {k:<22} {mem[k]:>10.2f} MB")
    print(f"  {'n_rotation_keys':<22} {int(mem['n_rotation_keys']):>10d}")
    print(f"  {'total_live_MB':<22} {mem['total_live_MB']:>10.2f} MB "
          f"(B={model.mapping.global_batch})")


# ==========================================================================
# absltest gate
# ==========================================================================
class LeNetHEPerformanceTest(absltest.TestCase):
    """Fused end-to-end performance gate for the LeNet HE pipeline.

    Builds one B=1 model through the canonical pipeline (~16 min) and
    measures single-image latency against it.

    Mapping owns the only serving executable; this gate deliberately does
    not reconstruct an interpreted per-operation path for diagnostics.
    """

    def test_fused_end_to_end_perf(self):
        if not os.environ.get("LENET_PERF", ""):
            self.skipTest("LENET_PERF not set; skipping LeNet HE benchmark.")
        Wargs, Bargs, prepare_x, weight_src, X_raw, label = _resolve_inputs()
        X = prepare_x(X_raw)
        mode_label = (
            "trained on MNIST" if weight_src == "trained"
            else "random Gaussian (C++ ref)")

        print("=" * 78)
        print("CROSS HE LeNet performance test (fused end-to-end wall)")
        print(f"  arch: Conv1({CI}->{CO1}) Quad Conv2({CO1}->{CO2}) Quad "
              f"FC1({FC1_IN}->{FC1_OUT}) Quad FC2({FC1_OUT}->{FC2_OUT})")
        print(f"  warmup={_N_WARM}, samples={_N_SAMPLE}, depth=7, "
              f"weights={weight_src}")
        print("=" * 78)

        # B=1 is the canonical latency gate. Throughput is gated separately
        # by the batched regression tests below.
        print("\n[perf-gate] building B=1 model ...")
        model = LeNetHE()
        model.precompute_plaintexts(*Wargs, *Bargs)
        # Free the ~7 GiB Mapping when this test ends, so the gate can share
        # a process with LeNetRegressionTest without two models resident.
        self.addCleanup(model.release)

        print("\n[bench] end-to-end wall ...")
        wall_e2e_med, wall_e2e_min = benchmark_end_to_end(model, X)
        print(f"  e2e median wall = {wall_e2e_med*1000:.2f} ms "
              f"(min {wall_e2e_min*1000:.2f} ms)")

        # Single forward for the rendered prediction.
        Y_ref = lenet_cleartext(X, *Wargs, *Bargs)
        Y_he = np.asarray(model.infer(X))
        ref_arg = int(np.argmax(Y_ref))
        he_arg = int(np.argmax(Y_he))
        max_err = float(np.max(np.abs(Y_he - Y_ref[:FC2_OUT])))
        print(f"\n[forward] label={label if label >= 0 else 'rand'} "
              f"ref={ref_arg} he={he_arg}  max|HE-ref|={max_err:.2e}")

        mapping = model.mapping
        self.assertIsNotNone(mapping)
        self.assertIs(model.packing, mapping.packing)
        kinds = [kind for _, kind, _, _ in mapping.packing.operations]
        self.assertEqual(kinds.count("matvec"), 4)
        self.assertEqual(kinds.count("add_plain"), 4)
        self.assertEqual(kinds.count("square"), 3)
        self.assertEqual(mapping.regions[0][0], True)

        # Memory footprint (Phase A — coarse profiling).
        print_memory(model)

        # ---- Render PNG (default ON) ----
        if not os.environ.get("LENET_TEST_NO_PNG", ""):
            disp = X_raw.reshape(H_IN, W_IN)
            if weight_src != "trained":
                lo, hi = float(disp.min()), float(disp.max())
                disp = (disp - lo) / max(hi - lo, 1e-9)
            render_perf_png(disp, label, he_arg, ref_arg, max_err,
                            wall_e2e_med * 1000.0,
                            [("fused e2e", wall_e2e_med * 1000.0)],
                            _PNG_OUT, mode_label=mode_label)
            print(f"[lenet_he_perf_test] wrote {_PNG_OUT}")
        else:
            print("[lenet_he_perf_test] LENET_TEST_NO_PNG set; skipping PNG render.")


# ==========================================================================
# Regression guards (Phase A) — frozen baselines for the LeNet HE pipeline.
# ==========================================================================

# TPUv6e-8 baselines for the canonical Mapping path, including host encryption
# and decryption. The older 55/46/43 ms gates described a retired hand-lowered
# executor and made every current secure-model performance run fail after its
# 15-minute build. These ceilings leave roughly 20--25% run-to-run headroom.
_REG_B1_WALL_MS_MAX  = 850        # measured 692.7 ms
_REG_B32_PERIMG_MAX  = 650        # intermediate placement: 8 devices x B4
_REG_B64_PERIMG_MAX  = 600        # measured 482.1 ms
_REG_LABEL_AGREE_MIN = 30         # of 32 images: HE prediction == true label
_REG_HE_REF_TOL      = 1.5e-1     # max|HE - cleartext_ref| across batch
_REG_REF_AGREE_MIN   = 32         # of 32 images: HE argmax == cleartext-ref argmax

# Deep-validation gates (200-image sweep; gated behind LENET_DEEP_VAL=1
# because the test takes ~3 minutes on top of the suite's ~17 min).
_REG_DEEP_N                  = 200
_REG_DEEP_REF_AGREE_FRAC_MIN = 1.00       # HE-vs-cleartext: must be 100%
_REG_DEEP_LABEL_AGREE_FRAC_MIN = 0.96     # HE-vs-true-label ≥ 96% (model is 99%)
_REG_DEEP_MAX_HE_REF         = 5.0e-1     # max|HE-ref| over 200 images
                                           # (measured 2.94e-1 with p99=1.96e-1)
_REG_DEEP_MEAN_HE_REF        = 1.5e-1     # mean|HE-ref| over 200 images
                                           # (measured 7.92e-2)











class LeNetRegressionTest(absltest.TestCase):
    """Frozen-baseline regression guards for the LeNet HE pipeline.

    Models are built one at a time. Each test asks for the batch it needs
    through ``_model_for``; switching batch releases the previous Mapping
    first, because two resident Mappings (about 7 GiB of keys and encoded
    constants each) plus a third build exceed the 32 GiB HBM of one TPU v6e
    chip. Tests are named so absltest's alphabetical order groups them by
    batch (b1, then b32, then b64), so each batch is built once; the B=32
    rebuild test releases the shared model before building its second one,
    so at most one Mapping is ever resident.

    Set `LENET_REGRESSION=1` to run them; the class skips otherwise, because
    each build costs minutes and gigabytes. Set `LENET_DEEP_VAL=1` to enable
    the 200-image correctness sweep.
    """

    # The one resident model and the batch it was built for.
    _model: "LeNetHE | None" = None
    _model_batch: int | None = None
    weights_args: tuple | None = None
    biases_args: tuple | None = None
    prepare_x: "callable | None" = None
    weight_src: str = "unknown"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Check the opt-in before touching optional data files. Ordinary test
        # discovery must not fail just because the MNIST validation corpus is
        # absent from this checkout.
        if not os.environ.get("LENET_REGRESSION", ""):
            return
        if not _mnist_data_available():
            raise absltest.SkipTest(
                "LENET_REGRESSION requires demos/maple_data MNIST binaries."
            )
        Wargs, Bargs, prepare_x, weight_src, _, _ = _resolve_inputs()
        cls.weights_args = Wargs
        cls.biases_args = Bargs
        # Wrap with staticmethod so `self.prepare_x(im)` doesn't bind `self`
        # as the first argument under Python's descriptor protocol.
        cls.prepare_x = staticmethod(prepare_x)
        cls.weight_src = weight_src

    @classmethod
    def tearDownClass(cls):
        cls._drop_model()
        super().tearDownClass()

    @classmethod
    def _drop_model(cls):
        """Release the resident Mapping so the next build has the whole chip."""
        if cls._model is not None:
            cls._model.release()
            cls._model = None
            cls._model_batch = None

    @classmethod
    def _model_for(cls, batch: int) -> "LeNetHE":
        """The resident model for `batch`, building (and evicting) as needed."""
        if not os.environ.get("LENET_REGRESSION", ""):
            raise absltest.SkipTest(
                f"LENET_REGRESSION not set; skipping B={batch}.")
        if cls._model_batch != batch:
            cls._drop_model()
            print(f"\n[regress setUp] building B={batch} model ...")
            cls._model = _build_model(batch=batch)
            cls._model_batch = batch
        return cls._model

    # ----------------------------------------------------------------------
    def test_b1_latency_under_baseline(self):
        """B=1 single-image wall ≤ frozen baseline (850 ms)."""
        model_b1 = self._model_for(1)
        self.assertEqual(model_b1.batch, 1, "B=1 batch mismatch")
        X = self.prepare_x(_load_mnist_slice(start=0, n=1)[0][0])
        e2e_med, _ = benchmark_end_to_end(model_b1, X)
        wall_ms = e2e_med * 1000
        print(f"\n[regress B=1] e2e median = {wall_ms:.1f} ms  "
              f"(baseline ≤ {_REG_B1_WALL_MS_MAX} ms)")
        self.assertLess(
            wall_ms, _REG_B1_WALL_MS_MAX,
            f"B=1 latency regressed: {wall_ms:.1f} ms > {_REG_B1_WALL_MS_MAX} ms.")

    # ----------------------------------------------------------------------
    def test_b32_throughput_under_baseline(self):
        """B=32 per-image wall stays within the current throughput gate."""
        model_b32 = self._model_for(32)
        self.assertEqual(model_b32.batch, 32, "B=32 batch mismatch")
        imgs, _ = _load_mnist_slice(start=0, n=32)
        X_in = [self.prepare_x(im) for im in imgs]
        e2e_med, _ = benchmark_end_to_end(model_b32, X_in)
        per_img_ms = e2e_med * 1000 / 32
        print(f"\n[regress B=32] per-image = {per_img_ms:.1f} ms  "
              f"(baseline ≤ {_REG_B32_PERIMG_MAX} ms;  total {e2e_med*1000:.0f} ms)")
        self.assertLess(
            per_img_ms, _REG_B32_PERIMG_MAX,
            f"B=32 per-image regressed: {per_img_ms:.1f} ms > "
            f"{_REG_B32_PERIMG_MAX} ms.")

    # ----------------------------------------------------------------------
    def test_b64_throughput_under_baseline(self):
        """B=64 per-image wall stays within the current TPUv6e-8 baseline.

        B=64 is the canonical throughput mode; B=32 is kept as the
        intermediate-batch reference.
        """
        model_b64 = self._model_for(64)
        self.assertEqual(model_b64.batch, 64, "B=64 batch mismatch")
        imgs, _ = _load_mnist_slice(start=0, n=64)
        X_in = [self.prepare_x(im) for im in imgs]
        e2e_med, _ = benchmark_end_to_end(model_b64, X_in)
        per_img_ms = e2e_med * 1000 / 64
        print(f"\n[regress B=64] per-image = {per_img_ms:.1f} ms  "
              f"(baseline ≤ {_REG_B64_PERIMG_MAX} ms;  total {e2e_med*1000:.0f} ms)")
        self.assertLess(
            per_img_ms, _REG_B64_PERIMG_MAX,
            f"B=64 per-image regressed: {per_img_ms:.1f} ms > "
            f"{_REG_B64_PERIMG_MAX} ms.")

    # ----------------------------------------------------------------------
    def test_b32_correctness_label_agreement(self):
        """≥ 30/32 HE predictions match true labels at B=32 on MNIST."""
        if self.weight_src != "trained":
            self.skipTest("trained weights required for label-agreement test.")
        model_b32 = self._model_for(32)
        imgs, labels = _load_mnist_slice(start=0, n=32)
        X_in = [self.prepare_x(im) for im in imgs]
        he_logits_list = model_b32.infer(X_in)
        n_label = 0
        n_ref = 0
        max_err = 0.0
        for i, im in enumerate(imgs):
            he = np.asarray(he_logits_list[i])
            ref = lenet_cleartext(self.prepare_x(im),
                                   *self.weights_args, *self.biases_args)
            err = float(np.max(np.abs(he - ref[:FC2_OUT])))
            max_err = max(max_err, err)
            if int(np.argmax(he)) == int(labels[i]):
                n_label += 1
            if int(np.argmax(he)) == int(np.argmax(ref)):
                n_ref += 1
        print(f"\n[regress B=32 correctness] label-agree {n_label}/32, "
              f"ref-agree {n_ref}/32, max|HE-ref|={max_err:.2e} "
              f"(tol {_REG_HE_REF_TOL:.2e})")
        self.assertGreaterEqual(
            n_label, _REG_LABEL_AGREE_MIN,
            f"B=32 label agreement regressed: {n_label}/32 < "
            f"{_REG_LABEL_AGREE_MIN}/32.")
        self.assertGreaterEqual(
            n_ref, _REG_REF_AGREE_MIN,
            f"B=32 HE-vs-cleartext-ref agreement regressed: {n_ref}/32 < "
            f"{_REG_REF_AGREE_MIN}/32 (numerical drift).")
        self.assertLess(
            max_err, _REG_HE_REF_TOL,
            f"B=32 max|HE-ref| regressed: {max_err:.2e} > "
            f"{_REG_HE_REF_TOL:.2e}.")

    # ----------------------------------------------------------------------
    def test_b32_rebuilding_gives_an_equivalent_model(self):
        """Two independent builds produce the same plan and the same labels.

        This replaces a cache round-trip test. The cache is gone, but what it
        protected is not: a model rebuilt from the same torch weights must
        behave identically, so the plan and its constants are a function of
        the model rather than of build order. Decryption noise is randomized,
        so logits differ at ~1e-4; the argmax must not.
        """
        m1 = self._model_for(32)
        batch = m1.batch
        fingerprint1 = m1.packed_program.fingerprint
        imgs, _ = _load_mnist_slice(start=0, n=batch)
        X_in = ([self.prepare_x(im) for im in imgs] if batch > 1
                else self.prepare_x(imgs[0]))
        y1 = np.asarray(m1.infer(X_in)).reshape(batch, -1)

        # Two B=32 Mappings do not fit one chip's HBM beside a build, so the
        # first model is measured and released before the second is built.
        # The rebuilt model then becomes the resident one for later tests.
        self._drop_model()
        m2 = _build_model(batch=batch)
        type(self)._model = m2
        type(self)._model_batch = batch

        self.assertEqual(
            fingerprint1, m2.packed_program.fingerprint,
            "rebuilding produced a different packed plan")
        y2 = np.asarray(m2.infer(X_in)).reshape(batch, -1)
        np.testing.assert_array_equal(
            y1.argmax(axis=1), y2.argmax(axis=1),
            "rebuilt model classified differently")
        self.assertLess(float(np.max(np.abs(y1 - y2))), 1e-2)

    # ----------------------------------------------------------------------
    def test_b32_memory_footprint_under_ceiling(self):
        """Secure-plan live state ≤ 8 GiB (catches accidental key bloat).

        The retired hand-lowered demo used degree 2,048 and a 1 GiB gate.
        Packing now derives degree 32,768 from the model's slot demand, so
        retaining that limit would compare two different cryptosystems.
        """
        model_b32 = self._model_for(32)
        mem = memory_footprint(model_b32)
        total_MB = mem["total_live_MB"]
        print(f"\n[regress memory] total_live={total_MB:.1f} MB  "
              f"(ceiling 8192 MiB)")
        for k, v in mem.items():
            print(f"  {k}: {v}")
        self.assertLess(total_MB, 8192.0,
            f"live state regressed: {total_MB:.1f} MiB > 8192 MiB.")

    # ----------------------------------------------------------------------
    def test_b32_deep_validation_200_images(self):
        """200-image HE-vs-cleartext correctness sweep at B=32 fused-forward.

        Stronger than the 32-image regression because:
          - Larger sample → tighter confidence on HE-vs-cleartext rate
          - Catches outlier-image noise patterns that 32 images miss
          - Verifies HE-vs-true-label tracks cleartext model accuracy

        Skipped by default (adds ~3 minutes wall on top of the 17-min
        regression suite). Enable with `LENET_DEEP_VAL=1`.

        Pads the last batch with copies of the last real image; we discard
        the padded predictions. Adds 24 padded slots (B=32 × 7 batches = 224
        slots, minus 200 real images = 24 pads).
        """
        if not os.environ.get("LENET_DEEP_VAL", ""):
            self.skipTest("LENET_DEEP_VAL not set; skipping 200-image sweep.")
        if self.weight_src != "trained":
            self.skipTest("trained weights required for deep validation.")
        model_b32 = self._model_for(32)

        N = _REG_DEEP_N
        B = model_b32.batch
        n_batches = (N + B - 1) // B

        imgs, labels = _load_mnist_slice(start=0, n=N)

        he_argmax = np.zeros(N, dtype=np.int32)
        ref_argmax = np.zeros(N, dtype=np.int32)
        abs_err_max_per = np.zeros(N, dtype=np.float64)

        for bi in range(n_batches):
            start = bi * B
            end = min(start + B, N)
            take = list(range(start, end))
            while len(take) < B:
                take.append(end - 1)            # pad with last real image
            batch = [self.prepare_x(imgs[i]) for i in take]
            he_logits_list = model_b32.infer(batch)
            for k, i_global in enumerate(range(start, end)):
                he = np.asarray(he_logits_list[k])
                ref = np.asarray(lenet_cleartext(
                    batch[k], *self.weights_args, *self.biases_args))[:FC2_OUT]
                he_argmax[i_global] = int(np.argmax(he))
                ref_argmax[i_global] = int(np.argmax(ref))
                abs_err_max_per[i_global] = float(np.max(np.abs(he - ref)))

        labels_n = labels[:N]
        n_he_eq_ref = int(np.sum(he_argmax == ref_argmax))
        n_he_eq_lbl = int(np.sum(he_argmax == labels_n))
        max_err = float(abs_err_max_per.max())
        mean_err = float(abs_err_max_per.mean())
        p99_err = float(np.percentile(abs_err_max_per, 99))

        ref_frac = n_he_eq_ref / N
        lbl_frac = n_he_eq_lbl / N

        print(f"\n[regress deep] HE-vs-cleartext: {n_he_eq_ref}/{N} "
              f"({100*ref_frac:.2f}%)")
        print(f"[regress deep] HE-vs-label:     {n_he_eq_lbl}/{N} "
              f"({100*lbl_frac:.2f}%)")
        print(f"[regress deep] max|HE-ref|: {max_err:.3e} (gate ≤ "
              f"{_REG_DEEP_MAX_HE_REF:.0e})")
        print(f"[regress deep] mean|HE-ref|: {mean_err:.3e} (gate ≤ "
              f"{_REG_DEEP_MEAN_HE_REF:.0e})")
        print(f"[regress deep] p99|HE-ref|: {p99_err:.3e}")

        self.assertGreaterEqual(
            ref_frac, _REG_DEEP_REF_AGREE_FRAC_MIN,
            f"HE-vs-cleartext agreement regressed: {n_he_eq_ref}/{N} = "
            f"{100*ref_frac:.2f}% < {100*_REG_DEEP_REF_AGREE_FRAC_MIN:.2f}%. "
            f"This indicates the HE pipeline is no longer bit-faithful.")
        self.assertGreaterEqual(
            lbl_frac, _REG_DEEP_LABEL_AGREE_FRAC_MIN,
            f"HE-vs-label agreement regressed: {n_he_eq_lbl}/{N} = "
            f"{100*lbl_frac:.2f}% < {100*_REG_DEEP_LABEL_AGREE_FRAC_MIN:.2f}%.")
        self.assertLess(
            max_err, _REG_DEEP_MAX_HE_REF,
            f"max|HE-ref| over {N} images regressed: {max_err:.3e} > "
            f"{_REG_DEEP_MAX_HE_REF:.0e}. Argmax may still match — but "
            f"noise drift indicates a numerical regression.")
        self.assertLess(
            mean_err, _REG_DEEP_MEAN_HE_REF,
            f"mean|HE-ref| over {N} images regressed: {mean_err:.3e} > "
            f"{_REG_DEEP_MEAN_HE_REF:.0e}.")


# ==========================================================================
# CLI subcommands (folded in from former lenet_bench.py)
# ==========================================================================
def _cli_make_input(prepare_x, batch: int, image_idx: int):
    """Single image (batch=1) or list of B images (batch>1)."""
    imgs, _ = _load_mnist_slice(start=image_idx, n=max(batch, 1))
    return ([prepare_x(im) for im in imgs] if batch > 1
            else prepare_x(imgs[0]))


def cmd_build(args):
    """Build the model and benchmark it. There is no cache to write."""
    Wargs, Bargs, prepare_x, weight_src, _, _ = _resolve_inputs()
    print(f"[bench] weights = {weight_src}, batch = {args.batch}")

    t0 = time.perf_counter()
    print("\n[setup] building from the torch model ...")
    model = LeNetHE(batch=args.batch)
    model.precompute_plaintexts(*Wargs, *Bargs)
    setup_s = time.perf_counter() - t0
    ring = model.ring_config
    print(f"[setup] done in {setup_s:.1f}s; ring degree={int(ring.degree)} "
          f"num_q={int(ring.logical_num_q)} dnum={int(ring.dnum)}")

    print("\n[bench] e2e wall ...")
    X_in = _cli_make_input(prepare_x, args.batch, args.image_idx)
    e2e_med, e2e_min = benchmark_end_to_end(model, X_in)
    per_image = e2e_med / max(args.batch, 1)
    print(f"  e2e median = {e2e_med*1000:.2f} ms "
          f"(min {e2e_min*1000:.2f} ms, per-image {per_image*1000:.2f} ms)")



def cmd_forward(args):
    """Predict N images one at a time, so the Mapping is always B=1.

    A Mapping is built for one exact global batch: `Mapping.encrypt_input`
    requires exactly `global_batch` logical inputs. This loop hands `infer`
    a single image per call, so anything but B=1 raises after the whole
    build cost has been paid. Batch throughput is what `build` measures.
    """
    Wargs, Bargs, prepare_x, weight_src, _, _ = _resolve_inputs()

    # Load the images first: on a checkout without the MNIST binaries this
    # fails in a second rather than after a multi-minute build.
    imgs, labels = _load_mnist_slice(start=args.image_idx, n=args.n)

    print(f"[setup] building a B=1 model from the torch model "
          f"({weight_src} weights) ...")
    model = LeNetHE(batch=1)
    model.precompute_plaintexts(*Wargs, *Bargs)

    print(f"\n[forward] HE inference on {args.n} MNIST images:")
    correct = 0
    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        x_in = prepare_x(img)
        ref = lenet_cleartext(x_in, *Wargs, *Bargs)
        ref_arg = int(np.argmax(ref))
        t0 = time.perf_counter()
        he = np.asarray(model.infer(x_in))
        wall = time.perf_counter() - t0
        he_arg = int(np.argmax(he))
        err = float(np.max(np.abs(he - ref[:FC2_OUT])))
        ok_label = (he_arg == int(lbl))
        if ok_label:
            correct += 1
        print(f"  [{i}] label={int(lbl)} ref={ref_arg} he={he_arg} "
              f"|HE-ref|={err:.2e}  label-match={'Y' if ok_label else 'N'} "
              f"wall={wall*1000:.0f}ms")
    print(f"\n[forward] {correct}/{len(imgs)} correct vs true label")


def _build_cli_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lenet_he_perf_test.py",
        description="LeNet HE performance gate. Run with no subcommand to "
                    "invoke the absltest gate.")
    sub = ap.add_subparsers(dest="cmd")
    for name, help_ in (
        ("build",   "materialize the Mapping + bench"),
        ("forward", "materialize a B=1 Mapping + predict N images"),
    ):
        sp = sub.add_parser(name, help=help_)
        if name == "build":
            # Only `build` is batch-parametric: it feeds the Mapping exactly
            # `--batch` images. `forward` predicts one image per call.
            sp.add_argument("--batch", type=int, default=64,
                            help="Batch size (DEFAULT: 64 — canonical "
                                 "throughput mode; current TPUv6e-8 wall is "
                                 "~482 ms/image, vs ~693 ms at B=1). Use "
                                 "--batch 32 for the intermediate throughput "
                                 "mode, or --batch 1 for single-image "
                                 "latency measurement.")
        sp.add_argument("--image-idx", type=int, default=0,
                        help="Starting MNIST image index (default 0).")
        if name == "forward":
            sp.add_argument("--n", type=int, default=8,
                            help="Number of images to run (default 8).")
    return ap



# =============================================================================
# Lightweight CLI/setup contracts (no HE materialization)
# =============================================================================

class LeNetForwardBatchContractTest(absltest.TestCase):
    """`forward` feeds one image per call, so its Mapping must be B=1."""

    def setUp(self):
        super().setUp()
        _RecordingModel.reset()
        self.images = np.arange(3 * 784, dtype=np.float64).reshape(3, 784)
        self.labels = np.array([1, 2, 3], dtype=np.int32)

    def _run_forward(self, n=3, image_idx=0):
        arguments = _build_cli_parser().parse_args(
            ["forward", "--n", str(n), "--image-idx", str(image_idx)]
        )
        with mock.patch.object(
            sys.modules[__name__], "LeNetHE", _RecordingModel
        ), mock.patch.object(
            sys.modules[__name__], "_load_mnist_slice",
            return_value=(self.images[:n], self.labels[:n]),
        ):
            cmd_forward(arguments)
        self.assertLen(_RecordingModel.instances, 1)
        return _RecordingModel.instances[0]

    def test_forward_builds_a_batch_one_mapping(self):
        model = self._run_forward()
        self.assertEqual(
            model.batch, 1,
            "forward hands infer() a single image, so its Mapping must be B=1",
        )

    def test_forward_infers_one_image_at_a_time(self):
        model = self._run_forward(n=3)
        self.assertLen(model.inferred, 3)
        for value in model.inferred:
            array = np.asarray(value)
            self.assertEqual(
                array.shape, (784,),
                "a B=1 Mapping takes one logical input, not a list of them",
            )

    def test_forward_takes_no_batch_flag(self):
        """A `--batch` on forward could only ever contradict the loop."""
        parser = _build_cli_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["forward", "--batch", "64"])

    def test_build_is_still_batch_parametric(self):
        parsed = _build_cli_parser().parse_args(
            ["build", "--batch", "32"]
        )
        self.assertEqual(parsed.batch, 32)

    def test_build_pairs_the_model_batch_with_the_input_count(self):
        """`build`'s input length must equal the batch it built for."""
        arguments = _build_cli_parser().parse_args(
            ["build", "--batch", "3"]
        )
        captured = {}

        def _benchmark(model, value):
            captured["model"] = model
            captured["value"] = value
            return 0.5, 0.4

        with mock.patch.object(
            sys.modules[__name__], "LeNetHE", _RecordingModel
        ), mock.patch.object(
            sys.modules[__name__], "_load_mnist_slice",
            return_value=(self.images, self.labels),
        ), mock.patch.object(
            sys.modules[__name__], "benchmark_end_to_end", _benchmark
        ):
            cmd_build(arguments)

        self.assertEqual(captured["model"].batch, 3)
        self.assertLen(captured["value"], 3)

    def test_build_at_batch_one_passes_a_bare_image(self):
        arguments = _build_cli_parser().parse_args(
            ["build", "--batch", "1"]
        )
        captured = {}

        def _benchmark(model, value):
            captured["value"] = value
            return 0.5, 0.4

        with mock.patch.object(
            sys.modules[__name__], "LeNetHE", _RecordingModel
        ), mock.patch.object(
            sys.modules[__name__], "_load_mnist_slice",
            return_value=(self.images[:1], self.labels[:1]),
        ), mock.patch.object(
            sys.modules[__name__], "benchmark_end_to_end", _benchmark
        ):
            cmd_build(arguments)

        self.assertEqual(np.asarray(captured["value"]).shape, (784,))

    def test_forward_reads_its_images_before_building(self):
        """A missing dataset must not cost a multi-minute build first."""
        arguments = _build_cli_parser().parse_args(
            ["forward", "--n", "2"]
        )
        with mock.patch.object(
            sys.modules[__name__], "LeNetHE", _RecordingModel
        ), mock.patch.object(
            sys.modules[__name__], "_load_mnist_slice",
            side_effect=FileNotFoundError("mnist_test_200.bin"),
        ):
            with self.assertRaises(FileNotFoundError):
                cmd_forward(arguments)
        self.assertEmpty(
            _RecordingModel.instances,
            "forward built a model before discovering it had no images",
        )


class DocumentedCommandsParseTest(absltest.TestCase):
    """Every LeNet command in the README must survive the real parser.

    The general documentation checks in ``nn_test.py`` compare documented
    flags with each driver's ``add_argument`` calls, which cannot tell one
    subparser from another. This runs the parser itself, which can.
    """

    def test_readme_lenet_commands_parse(self):
        readme = os.path.join(_DEMO_DIR, "README.md")
        with open(readme, encoding="utf-8") as handle:
            text = handle.read()
        parser = _build_cli_parser()
        commands = re.findall(
            r"python3?\s+lenet_he_perf_test\.py\s*([^\n#]*)", text
        )
        self.assertNotEmpty(commands)
        checked = 0
        for raw in commands:
            tokens = raw.split()
            # `... lenet_he_perf_test.py LeNetRegressionTest` is the absltest
            # form, which argparse never sees.
            if tokens and tokens[0][0].isupper():
                continue
            parser.parse_args(tokens)
            checked += 1
        self.assertGreater(checked, 0, "no CLI commands found in README.md")


if __name__ == "__main__":
    # Dispatch on first positional arg: known subcommand → CLI mode;
    # otherwise hand off to absltest.main() so the absltest gate runs.
    if len(sys.argv) > 1 and sys.argv[1] in ("build", "forward"):
        parser = _build_cli_parser()
        args = parser.parse_args()
        if args.cmd == "build":
            cmd_build(args)
        elif args.cmd == "forward":
            cmd_forward(args)
    else:
        absltest.main()
