"""Performance gates for AlexNetTinyHE on CIFAR-10.

Two test classes:

* `AlexNetTinyHEPerfTest` — warm latency at B=1 and automatically sharded
  B=32.
* `AlexNetTinyMultiDevicePerfTest` — the same end-to-end model API with a
  global batch equal to the visible device count, so every device receives
  one image. Reports CPU encrypt + Mapping execution + CPU decrypt wall time
  and checks every result against `alexnet_tiny_cleartext`.

No cache file is involved. Each test builds its model from
`alexnet_infer.QuadAlexNetTinyInfer` through the canonical pipeline, which
materializes constants and per-level controls -- minutes of work and
gigabytes of memory. Set `ALEXNET_PERF=1` to run them; they skip otherwise.
"""
from __future__ import annotations

import os
import sys
import time
from unittest import mock

import numpy as np

from absl.testing import absltest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from alexnet_he import (                                       # noqa: E402
    TINY_FC_OUT,
    AlexNetTinyHE,
    alexnet_tiny_cleartext,
    alexnet_tiny_random_inputs,
    prepare_tiny_args,
    downsample_cifar10_2x,
)
from demo_test_utils import RecordingPerfModel as _RecordingModel  # noqa: E402

_LOG_DIR = os.path.join(_HERE, "log")


def _make_random_input_and_weights():
    """Reusable: a single CIFAR-10-shaped random input + downsampled-to-tiny."""
    rng = alexnet_tiny_random_inputs(seed=42)
    Wargs = prepare_tiny_args(rng)
    big = np.random.RandomState(0).randn(3 * 32 * 32)
    x_tiny = downsample_cifar10_2x(big)
    return x_tiny, Wargs


def _require_perf(test_self, label: str):
    """Building a Mapping is minutes and gigabytes, so it is opt-in."""
    if not os.environ.get("ALEXNET_PERF", ""):
        test_self.skipTest(f"ALEXNET_PERF not set; skipping {label}.")


def _build_tiny(batch: int = 1, devices=None, Wargs=None):
    """Bind weights and materialize one Mapping, from the torch model.

    Replaces the pickled cache: the plan is re-derived from
    `alexnet_infer.QuadAlexNetTinyInfer` cheaply, and the constants follow
    from the weights, so nothing survives a weight change to be cached.

    `Wargs` is the same deterministic 8-tuple `_make_random_input_and_weights`
    returns. A benchmark that also evaluates `alexnet_tiny_cleartext` must
    pass the tuple it used there: binding one weight set into the ciphertext
    and comparing against another turns the argmax check into a coin flip.
    """
    if Wargs is None:
        _, Wargs = _make_random_input_and_weights()
    model = AlexNetTinyHE(batch=batch, devices=devices)
    model.precompute_plaintexts(*Wargs)
    return model


class AlexNetTinyHEPerfTest(absltest.TestCase):
    """Single-image and automatically sharded warm-latency gates."""

    def test_tiny_he_b1_perf(self):
        _require_perf(self, "AlexNetTiny B=1")
        x_tiny, Wargs = _make_random_input_and_weights()
        model = _build_tiny(batch=1, Wargs=Wargs)
        self.addCleanup(model.release)   # free its HBM before the next gate builds
        for _ in range(2):
            _ = model.infer(x_tiny)
        samples = [time.perf_counter() for _ in range(0)]
        for _ in range(5):
            t0 = time.perf_counter()
            _ = model.infer(x_tiny)
            samples.append(time.perf_counter() - t0)
        wall_ms = float(np.median(samples)) * 1000
        print(f"\n[tiny HE B=1 single-chip] median wall = {wall_ms:.1f} ms (n=5)")
        # TPUv6e-8 canonical Mapping baseline: 640.8 ms.
        self.assertLess(wall_ms, 800.0,
                          f"AlexNetTiny B=1 wall = {wall_ms:.1f} ms > 800 ms")

    def test_tiny_he_b32_perf(self):
        _require_perf(self, "AlexNetTiny B=32")
        x_tiny, Wargs = _make_random_input_and_weights()
        model = _build_tiny(batch=32, Wargs=Wargs)
        self.addCleanup(model.release)   # free its HBM before the next gate builds
        batch = [x_tiny] * 32
        for _ in range(2):
            _ = model.infer(batch)
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = model.infer(batch)
            samples.append(time.perf_counter() - t0)
        wall_ms = float(np.median(samples)) * 1000
        per_img = wall_ms / 32
        print(f"\n[tiny HE B=32 data-parallel] median wall = {wall_ms:.1f} ms "
              f"({per_img:.2f} ms / image, n=5)")
        # TPUv6e-8 canonical Mapping baseline: 475.5 ms/image on 8 devices.
        self.assertLess(per_img, 600.0,
                          f"AlexNetTiny B=32 per-image = {per_img:.2f} ms > 600 ms")


class AlexNetTinyMultiDevicePerfTest(absltest.TestCase):
    """One ordinary global batch with one image assigned to each device."""

    @classmethod
    def setUpClass(cls):
        if not os.environ.get("ALEXNET_PERF", ""):
            raise absltest.SkipTest(
                "ALEXNET_PERF not set; skipping multi-device benchmark.")

    def test_one_image_per_device_end_to_end(self):
        import jax
        devices = tuple(jax.devices())
        global_batch = len(devices)
        if global_batch < 2:
            self.skipTest(
                f"only {global_batch} JAX device(s); need at least two"
            )

        # One weight set for both sides of the comparison below.
        x_tiny, Wargs = _make_random_input_and_weights()

        # The plan is placement-independent; the Mapping is not, so it is
        # built for this exact device tuple.
        model = _build_tiny(batch=global_batch, devices=devices, Wargs=Wargs)
        self.addCleanup(model.release)   # free its HBM before the next gate builds
        self.assertEqual(model.mapping.global_batch, global_batch)
        self.assertEqual(model.mapping.device_count, global_batch)
        self.assertEqual(model.mapping.ctx.batch, 1)

        batch = [x_tiny] * global_batch
        ref_argmax = int(np.argmax(alexnet_tiny_cleartext(x_tiny, *Wargs)))

        # This is exactly the production surface: CPU encryption, one
        # Polynomial-to-Polynomial Mapping execution, then CPU decryption.
        for _ in range(2):
            out = model.infer(batch)
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            out = model.infer(batch)
            samples.append(time.perf_counter() - t0)

        wall_ms = float(np.median(samples)) * 1000
        per_image_ms = wall_ms / global_batch
        print(
            f"\n[data-parallel B={global_batch} on {global_batch} devices] "
            f"median e2e wall = {wall_ms:.1f} ms "
            f"({per_image_ms:.2f} ms / image)"
        )

        self.assertLen(out, global_batch)
        self.assertTrue(np.isfinite(per_image_ms))
        self.assertGreater(per_image_ms, 0.0)
        self.assertSequenceEqual(
            [int(np.argmax(logits[:TINY_FC_OUT])) for logits in out],
            [ref_argmax] * global_batch,
        )



# =============================================================================
# Lightweight setup contracts (no HE materialization)
# =============================================================================

class BuildTinyTest(absltest.TestCase):
    """`_build_tiny` must bind the weights its caller compares against."""

    def setUp(self):
        super().setUp()
        _RecordingModel.reset()

    def test_it_binds_the_weights_it_was_given(self):
        _, Wargs = _make_random_input_and_weights()
        with mock.patch.object(
            sys.modules[__name__], "AlexNetTinyHE", _RecordingModel
        ):
            model = _build_tiny(batch=4, Wargs=Wargs)
        self.assertEqual(model.batch, 4)
        self.assertLen(model.weights, len(Wargs))
        for bound, given in zip(model.weights, Wargs):
            np.testing.assert_array_equal(bound, given)

    def test_it_defaults_to_the_deterministic_weights(self):
        """No argument still binds real weights -- and names no missing symbol.

        This is the regression: the body called an unimported loader, so any
        call raised NameError before reaching the model.
        """
        _, expected = _make_random_input_and_weights()
        with mock.patch.object(
            sys.modules[__name__], "AlexNetTinyHE", _RecordingModel
        ):
            model = _build_tiny()
        self.assertLen(model.weights, len(expected))
        for bound, given in zip(model.weights, expected):
            np.testing.assert_array_equal(bound, given)

    def test_the_bound_weights_are_the_reference_weights(self):
        """The HE model and `alexnet_tiny_cleartext` must see one weight set.

        Binding one set and comparing against another leaves the argmax
        assertion passing only by luck.
        """
        x_tiny, Wargs = _make_random_input_and_weights()
        with mock.patch.object(
            sys.modules[__name__], "AlexNetTinyHE", _RecordingModel
        ):
            model = _build_tiny(batch=2, Wargs=Wargs)
        reference = alexnet_tiny_cleartext(x_tiny, *Wargs)
        self.assertLen(reference, TINY_FC_OUT)
        # The splat order `precompute_plaintexts` received is the order the
        # cleartext reference consumes after its leading input argument.
        for bound, given in zip(model.weights, Wargs):
            np.testing.assert_array_equal(bound, given)

    def test_the_weights_are_deterministic_across_calls(self):
        _, first = _make_random_input_and_weights()
        _, second = _make_random_input_and_weights()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_devices_reach_the_model(self):
        sentinel = ("device-0", "device-1")
        with mock.patch.object(
            sys.modules[__name__], "AlexNetTinyHE", _RecordingModel
        ):
            model = _build_tiny(batch=2, devices=sentinel)
        self.assertEqual(model.devices, sentinel)


if __name__ == "__main__":
    absltest.main()
