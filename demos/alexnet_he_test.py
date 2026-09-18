"""Gates for the AlexNet HE models on CIFAR-10.

* `AlexNetCanonicalDelegationTest` — the two encrypted entrypoints run the
  canonical pipeline (`nn.vectorize -> packing.pack -> Mapping`) once each and
  call none of the retired hand-built builders.
* `AlexNetTinyCleartextTest` / `AlexNetFullCleartextTest` — cleartext sanity.
* `AlexNetTinyAccuracyTest` / the full accuracy gate — real CIFAR-10 accuracy
  is materially above chance, so a collapsed model cannot pass by agreeing
  with itself.
* `SecurityAssertTest` — the HES-2018 ceiling helper this module still exports.

Mapping is patched in every delegation test. Key generation is not what makes
a real Mapping expensive; materializing BSGS diagonals over tens of thousands
of slots and building the per-level operator controls is, and at these widths
that runs to gigabytes. The gates here are plan-level on purpose.
"""
from __future__ import annotations

import os
import pickle
import sys
from unittest import mock

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import alexnet_he                                               # noqa: E402
from alexnet_he import (                                        # noqa: E402
    assert_he128_or_warn, HES128_MAX_LOG_PQ,
    DEEP_Q_TOWERS_POOL, DEEP_P_TOWERS_POOL,
    normalize_cifar10_image,
    TINY_FC_OUT, TINY_CI, TINY_H_IN, TINY_W_IN,
    alexnet_tiny_cleartext, alexnet_tiny_random_inputs, prepare_tiny_args,
    load_trained_alexnet_tiny_weights, downsample_cifar10_2x,
    AlexNetTinyHE,
    # Full depth-15 variant
    FULL_FC3_OUT, FULL_CI, FULL_H_IN, FULL_W_IN,
    alexnet_full_cleartext, alexnet_full_random_inputs, prepare_full_args,
    load_trained_alexnet_full_weights,
    AlexNetHE,
)
import canonical_demo
import demo_test_utils                                           # noqa: E402
import encrypted_demos                                          # noqa: E402
import nn                                                       # noqa: E402
import packing                                                  # noqa: E402

# Imported for its side effect, here rather than lazily, and it has to stay.
# `demo_test_utils.fake_mapping` swaps sys.modules with mock.patch.dict, which
# restores the dict wholesale on exit and so EVICTS anything imported while it
# was active. `packing.pack` imports he_params -- and through it JAX -- the
# first time it runs, which would land inside that window: JAX would be
# dropped and re-imported once per test, and re-initializing its backend in a
# single process aborts. Importing it now puts it in the baseline snapshot
# that patch.dict restores to, where it survives every test.
import he_params                                    # noqa: F401,E402

# Cleartext CIFAR-10 accuracy gate config (shared by the Tiny + Full gates).
# Chance is 10%; the threshold is well above it but below the ~44-55% these
# capacity-limited nets reach, so it catches a collapse-to-chance without being
# brittle. Overridable via env for quick local sweeps.
_ACC_MIN_DEFAULT = 0.35
_ACC_N_TINY_DEFAULT = 256
_ACC_N_FULL_DEFAULT = 128
# Set ALEXNET_ACC_REQUIRE=1 to FAIL (not skip) when weights or the CIFAR-10
# test batch are missing — mirrors the LENET_REGRESSION_REQUIRE convention the
# HE gates use, so the accuracy gates are non-vacuous in CI.
_ACC_REQUIRE_ENV = "ALEXNET_ACC_REQUIRE"


def _acc_require() -> bool:
    val = os.environ.get(_ACC_REQUIRE_ENV, "").strip().lower()
    return val not in ("", "0", "false", "no")


def _skip_or_fail(test, msg: str) -> None:
    """Fail if ALEXNET_ACC_REQUIRE is set, otherwise skip (default)."""
    if _acc_require():
        test.fail(msg)
    else:
        test.skipTest(msg)


def _run_cleartext_accuracy_gate(test, *, load_weights, prepare_args,
                                 cleartext_fn, n_env: str, n_default: int,
                                 name: str) -> None:
    """Shared cleartext CIFAR-10 accuracy gate.

    Loads trained weights + real CIFAR-10 test images, runs the cleartext
    forward, and asserts accuracy is materially above chance (10%). Skips (or
    fails, under ALEXNET_ACC_REQUIRE) if weights or data are missing.
    """
    trained = load_weights()
    if trained is None:
        _skip_or_fail(test, f"Trained {name} weights missing; run "
                            f"alexnet_train.py first.")
        return
    n = int(os.environ.get(n_env, str(n_default)))
    min_acc = float(os.environ.get("ALEXNET_ACC_MIN", str(_ACC_MIN_DEFAULT)))
    try:
        imgs_u8, labels = _load_cifar10_test(n)
    except FileNotFoundError as e:
        _skip_or_fail(test, str(e))
        return

    flat = _normalize_for_inference(imgs_u8)
    Wargs = prepare_args(trained)
    correct = 0
    for i in range(len(labels)):
        x = downsample_cifar10_2x(flat[i])
        y = cleartext_fn(x, *Wargs)
        correct += int(np.argmax(y) == int(labels[i]))
    acc = correct / len(labels)
    print(f"\n[{name} cleartext accuracy] {correct}/{len(labels)} = "
          f"{acc*100:.1f}%  (threshold {min_acc*100:.0f}%, chance 10%)")
    test.assertGreater(
        acc, min_acc,
        msg=f"{name} cleartext CIFAR-10 accuracy {acc*100:.1f}% "
            f"<= {min_acc*100:.0f}% — model is near chance (~10%). "
            f"Retrain with alexnet_train.py.")


# ---------------------------------------------------------------------------
# CIFAR-10 loader — direct from torchvision-format pickle (avoids torchvision
# at import time so the test file is light).
# ---------------------------------------------------------------------------
def _load_cifar10_test(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels). Images are uint8 [0, 255] in (N, 3, 32, 32)
    layout, labels are int.
    """
    cifar_root = os.path.join(_HERE, "cifar10_data", "cifar-10-batches-py")
    test_batch = os.path.join(cifar_root, "test_batch")
    if not os.path.isfile(test_batch):
        raise FileNotFoundError(
            f"CIFAR-10 test batch not found at {test_batch}. "
            f"Run `python3 demos/alexnet_train.py --epochs 1` once to "
            f"trigger torchvision's auto-download.")
    with open(test_batch, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    imgs = d["data"][:n].reshape(n, 3, 32, 32).astype(np.uint8)
    labels = np.array(d["labels"][:n], dtype=np.int32)
    return imgs, labels


def _normalize_for_inference(imgs_u8: np.ndarray) -> np.ndarray:
    """uint8 (N, 3, 32, 32) → normalized float64 (N, Ci*H*W,) flat."""
    out = imgs_u8.astype(np.float64) / 255.0
    out = out.reshape(out.shape[0], -1)
    return np.stack([normalize_cifar10_image(x) for x in out], axis=0)


# ===========================================================================
# Canonical-pipeline delegation.
# ===========================================================================
# No hand-built lowering survives in this module. These names are asserted
# absent rather than spied: a test is not a consumer, so keeping the builders
# alive to be tested would have preserved a second implementation of what
# packing.pack now derives from the traced model. Absence is the stronger
# statement anyway -- a spy proves a call did not happen on one path, an
# absent name proves no path can make it.
_DELETED_BUILDERS = (
    "_alternating_he_model",
    "_sparse_layer_constant",
    "_map_network",
    "_network_from_cache",
    "_conv_then_pool_to_matrix",
    "_conv1_pool_multiplexed_to_matrix",
    "_conv_then_adaptive_pool_to_matrix",
    "_fc_to_matrix",
    "_DiagMap",
)

_CASES = {
    "tiny": dict(
        cls=AlexNetTinyHE,
        demo_name="AlexNetTinyDemo",
        depth=7, matvec=4, square=3,
        cleartext=alexnet_tiny_cleartext,
        random_inputs=alexnet_tiny_random_inputs,
        prepare=prepare_tiny_args,
        logits=TINY_FC_OUT,
    ),
    "full": dict(
        cls=AlexNetHE,
        demo_name="AlexNetFullDemo",
        depth=15, matvec=8, square=7,
        cleartext=alexnet_full_cleartext,
        random_inputs=alexnet_full_random_inputs,
        prepare=prepare_full_args,
        logits=FULL_FC3_OUT,
    ),
}


def _weights(name):
    case = _CASES[name]
    draw = case["random_inputs"](seed=17)
    return draw["X"], case["prepare"](draw)


class _Spies:
    """The canonical stages, wrapped, without importing the mapping module.

    ``vectorize`` and ``pack`` are wrapped rather than replaced, so the plan
    the entrypoint builds is the real one and the call counts still mean
    something. Patching ``mapping.Mapping`` would import the real module
    first, pulling in JAX and XLA before the mock exists -- hundreds of
    megabytes to assert that a plan was built -- so ``fake_mapping``
    intercepts materialization itself.

    There is no spy for the hand-built builders. They no longer exist, and
    ``test_the_hand_built_graph_builders_are_gone`` asserts exactly that: an
    absent name proves no path can reach it, which a call count cannot.
    """

    def __init__(self):
        self._patches = [
            mock.patch.object(canonical_demo.nn, "vectorize",
                              wraps=nn.vectorize),
            mock.patch.object(canonical_demo.packing, "pack",
                              wraps=packing.pack),
        ]
        self._mapping = demo_test_utils.fake_mapping()

    def __enter__(self):
        self.vectorize, self.pack = [
            patch.start() for patch in self._patches
        ]
        self.Mapping = self._mapping.__enter__()
        return self

    def __exit__(self, *exception):
        self._mapping.__exit__(*exception)
        for patch in reversed(self._patches):
            patch.stop()
        return False


class AlexNetCanonicalDelegationTest(parameterized.TestCase):
    """Both entrypoints are adapters over `nn.vectorize -> pack -> Mapping`."""

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_precompute_runs_each_canonical_stage_exactly_once(self, name):
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            self.assertEqual(spies.vectorize.call_count, 1)
            self.assertEqual(spies.pack.call_count, 1)
            self.assertEqual(spies.Mapping.call_count, 1)

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_the_mapping_is_built_from_the_packed_program(self, name):
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            (positional, keyword) = spies.Mapping.call_args
            self.assertIs(positional[0], model.packed_program)
            self.assertEqual(keyword["global_batch"], 1)

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_infer_reuses_the_one_mapping_and_adds_no_stages(self, name):
        model = _CASES[name]["cls"]()
        sample, arrays = _weights(name)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            model.infer(sample)
            model.infer(sample)
            self.assertEqual(spies.vectorize.call_count, 1)
            self.assertEqual(spies.pack.call_count, 1)
            self.assertEqual(spies.Mapping.call_count, 1)
            self.assertEqual(spies.Mapping.return_value.infer.call_count, 2)

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_a_32x32_image_is_downsampled_client_side(self, name):
        """The 32->16 average pool runs in the clear, before encryption."""
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        raw = np.random.RandomState(0).randn(3 * 32 * 32)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            model.infer(raw)
            (positional, _) = spies.Mapping.return_value.infer.call_args
        handed = np.asarray(positional[0])
        self.assertEqual(handed.shape, encrypted_demos.ALEXNET_HE_INPUT)
        np.testing.assert_allclose(
            handed, encrypted_demos.downsample_for_client(raw), atol=1e-12
        )

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_an_already_downsampled_image_reaches_the_demo_unchanged(
            self, name):
        model = _CASES[name]["cls"]()
        sample, arrays = _weights(name)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            model.infer(sample)
            (positional, _) = spies.Mapping.return_value.infer.call_args
        np.testing.assert_allclose(
            np.asarray(positional[0]).reshape(-1), sample, atol=0.0
        )

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_a_batched_model_hands_the_demo_one_sample_per_entry(self, name):
        model = _CASES[name]["cls"](batch=3)
        sample, arrays = _weights(name)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            self.assertEqual(
                spies.Mapping.call_args[1]["global_batch"], 3
            )
            model.infer([sample, sample, sample])
            (positional, _) = spies.Mapping.return_value.infer.call_args
        self.assertLen(positional[0], 3)
        for item in positional[0]:
            self.assertEqual(
                np.asarray(item).shape, encrypted_demos.ALEXNET_HE_INPUT
            )

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_an_image_of_the_wrong_size_is_refused(self, name):
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        with _Spies():
            model.precompute_plaintexts(*arrays)
            with self.assertRaisesRegex(ValueError, "already client-down"):
                model.infer(np.zeros(1000))

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_infer_before_precompute_is_refused(self, name):
        model = _CASES[name]["cls"]()
        with self.assertRaisesRegex(RuntimeError, "precompute_plaintexts"):
            model.infer(np.zeros(TINY_CI * TINY_H_IN * TINY_W_IN))

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_the_plan_matches_the_architecture(self, name):
        case = _CASES[name]
        model = case["cls"]()
        packed = model.packed_program
        kinds = packed.kinds()
        self.assertEqual(packed.depth, case["depth"])
        self.assertEqual(kinds.count("matvec"), case["matvec"])
        self.assertEqual(kinds.count("square"), case["square"])
        self.assertEqual(packed.logical_input_shape,
                         encrypted_demos.ALEXNET_HE_INPUT)
        self.assertEqual(packed.logical_output_shape, (case["logits"],))
        # The ring is derived from the program, not hand-picked, so it clears
        # the 128-bit bar the retired degree-2048 pool never did.
        self.assertGreaterEqual(int(packed.ring_config.security_bits), 128)

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_the_model_exposes_the_canonical_artifacts(self, name):
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        self.assertIsInstance(model.vectorized_program, nn.VectorizedProgram)
        self.assertIsInstance(model.packed_program, packing.Packing)
        self.assertIs(model.ring_config, model.packed_program.ring_config)
        self.assertIs(model.packing, model.packed_program)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            self.assertIs(model.mapping, spies.Mapping.return_value)
            self.assertIs(model.ctx, spies.Mapping.return_value.ctx)

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_mapping_backed_access_before_precompute_is_refused(self, name):
        """Reaching the Mapping early would compile the seeded fallback.

        The demo fills its module with a deterministic draw when no weights
        have been bound. Materializing a Mapping over that would cost the
        full constant build and produce an answer that looks real and means
        nothing, so every Mapping-backed accessor refuses instead.
        """
        model = _CASES[name]["cls"]()
        with demo_test_utils.fake_mapping() as mapping_class:
            for reach in (lambda: model.mapping, lambda: model.ctx,
                          lambda: model.encrypt(np.zeros(768)),
                          lambda: model.decrypt(object())):
                with self.assertRaises(RuntimeError):
                    reach()
            mapping_class.assert_not_called()

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_reading_the_plan_never_builds_a_mapping(self, name):
        model = _CASES[name]["cls"]()
        with demo_test_utils.fake_mapping() as mapping_class:
            _ = model.vectorized_program
            _ = model.packed_program
            _ = model.ring_config
            _ = model.metadata()
            mapping_class.assert_not_called()

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_precompute_routes_the_caller_arrays_into_the_torch_model(
            self, name):
        """The caller's arrays must be what gets compiled.

        The demo seeds its module with a deterministic draw when it has no
        checkpoint. If `precompute_plaintexts` layered the caller's weights
        beside that instead of into it, the compiled program would be the
        seeded model — so compare the torch module against this file's own
        numpy reference for the same arrays.
        """
        case = _CASES[name]
        model = case["cls"]()
        sample, arrays = _weights(name)
        with _Spies():
            model.precompute_plaintexts(*arrays)
        reference = case["cleartext"](sample, *arrays)
        traced = model.demo.cleartext(sample)
        self.assertEqual(traced.shape, reference.shape)
        scale = max(float(np.max(np.abs(reference))), 1e-300)
        self.assertLess(float(np.max(np.abs(traced - reference))) / scale,
                        1e-9)

    @parameterized.named_parameters(("tiny", "tiny"))
    def test_a_packed_matvec_leaves_the_bsgs_split_open(self, name):
        """Packing records no ratio: Mapping chooses per operation."""
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        with _Spies():
            model.precompute_plaintexts(*arrays)
        for _, kind, _, argument in model.packed_program.operations:
            if kind == "matvec":
                self.assertEqual(argument[1:], (None, None, None, None))

    @parameterized.named_parameters(("tiny", "tiny"))
    def test_the_historic_default_ratio_is_accepted(self, name):
        """2.0 was the default; old calls that pass it keep working."""
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays, bsgs_ratio=2.0)
            self.assertEqual(spies.pack.call_count, 1)

    @parameterized.named_parameters(("tiny", "tiny"))
    def test_a_nonsensical_bsgs_ratio_is_refused(self, name):
        """Any non-default value is refused; nonsense is not a special case."""
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        for ratio in (0.0, -1.0, 1.5):
            with self.subTest(bsgs_ratio=ratio):
                with self.assertRaisesRegex(ValueError, "per matvec"):
                    _CASES[name]["cls"]().precompute_plaintexts(
                        *arrays, bsgs_ratio=ratio
                    )
        del model

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_preparing_one_instance_twice_is_refused(self, name):
        """Rebinding constants would leave the built Mapping describing the
        previous weights, so the second call is refused rather than silently
        reusing a stale program."""
        model = _CASES[name]["cls"]()
        _, arrays = _weights(name)
        with _Spies():
            model.precompute_plaintexts(*arrays)
            with self.assertRaisesRegex(RuntimeError, "already prepared"):
                model.precompute_plaintexts(*arrays)

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_a_legacy_cache_is_rejected_rather_than_loaded(self, name):
        cls = _CASES[name]["cls"]
        with self.assertRaisesRegex(ValueError, "pickled HE network"):
            cls(_cached_state={"network": object(), "kp": object()})
        with self.assertRaises(NotImplementedError):
            cls.from_cache(os.path.join(_HERE, "log", "whatever.pkl"))
        with self.assertRaises(NotImplementedError):
            cls().save_cache("/tmp/unused.pkl")

    @parameterized.named_parameters(("tiny", "tiny"), ("full", "full"))
    def test_scheduling_knobs_reach_the_mapping(self, name):
        model = _CASES[name]["cls"](batch=2, dnum=3, devices=("d0", "d1"))
        _, arrays = _weights(name)
        with _Spies() as spies:
            model.precompute_plaintexts(*arrays)
            keyword = spies.Mapping.call_args[1]
        self.assertEqual(keyword["global_batch"], 2)
        self.assertEqual(keyword["dnum"], 3)
        self.assertEqual(keyword["devices"], ("d0", "d1"))

    def test_use_multiplexed_conv1_no_longer_selects_a_lowering(self):
        """The packer derives the input layout; the flag cannot change it."""
        multiplexed = AlexNetTinyHE(use_multiplexed_conv1=True)
        row_major = AlexNetTinyHE(use_multiplexed_conv1=False)
        self.assertEqual(
            multiplexed.packed_program.fingerprint,
            row_major.packed_program.fingerprint,
        )

    def test_require_128bit_is_checked_against_the_derived_ring(self):
        model = AlexNetHE(require_128bit=True)
        _, arrays = _weights("full")
        with _Spies():
            model.precompute_plaintexts(*arrays)
        self.assertGreaterEqual(int(model.ring_config.security_bits), 128)

    def test_the_hand_built_graph_builders_are_gone(self):
        """No second lowering survives beside the one packing.pack derives.

        Their equivalence proof moved to jaxite_word/packing_test.py, which
        checks the packed constants against a cleartext evaluation of the
        same program -- a test of the code that actually runs.
        """
        for name in _DELETED_BUILDERS:
            self.assertFalse(
                hasattr(alexnet_he, name),
                msg=f"{name} built the retired packing graph by hand and has "
                    f"no caller left; it must not linger.")


# ===========================================================================
# AlexNetTiny — cleartext gates.
# ===========================================================================
class AlexNetTinyCleartextTest(absltest.TestCase):

    def test_tiny_cleartext_random_inputs_runs(self):
        rng = alexnet_tiny_random_inputs(seed=42)
        Wargs = prepare_tiny_args(rng)
        Y = alexnet_tiny_cleartext(rng["X"], *Wargs)
        self.assertEqual(Y.shape, (TINY_FC_OUT,))
        self.assertTrue(np.all(np.isfinite(Y)))

    def test_tiny_downsample_helper(self):
        big = np.random.RandomState(0).randn(3 * 32 * 32)
        small = downsample_cifar10_2x(big)
        self.assertEqual(small.shape, (TINY_CI * TINY_H_IN * TINY_W_IN,))

    def test_the_downsample_helper_matches_the_client_step(self):
        """`downsample_cifar10_2x` and the demo's client step are one pool."""
        big = np.random.RandomState(1).randn(3 * 32 * 32)
        np.testing.assert_allclose(
            downsample_cifar10_2x(big),
            encrypted_demos.downsample_for_client(big).reshape(-1),
            atol=1e-12,
        )


class AlexNetTinyAccuracyTest(absltest.TestCase):
    """Cleartext CIFAR-10 accuracy gate.

    The delegation gates only check the pipeline, so a model that classifies
    at chance (~10%) would pass every other test. This gate runs the shipped
    cleartext forward with the trained weights over real CIFAR-10 test images
    and fails if accuracy is not materially above chance.

    Skips if trained weights or the CIFAR-10 test batch are absent (set
    ALEXNET_ACC_REQUIRE=1 to fail instead), matching the rest of the suite.
    Tunable via env:
        ALEXNET_ACC_N    number of test images   (default 256)
        ALEXNET_ACC_MIN  minimum accuracy [0,1]  (default 0.35)
    """

    def test_tiny_cleartext_accuracy_beats_chance(self):
        _run_cleartext_accuracy_gate(
            self,
            load_weights=load_trained_alexnet_tiny_weights,
            prepare_args=prepare_tiny_args,
            cleartext_fn=alexnet_tiny_cleartext,
            n_env="ALEXNET_ACC_N",
            n_default=_ACC_N_TINY_DEFAULT,
            name="AlexNetTiny")


# ===========================================================================
# AlexNet (full depth-15) — cleartext gates.
# ===========================================================================
class AlexNetFullCleartextTest(absltest.TestCase):
    """Cleartext sanity for the depth-15 architecture."""

    def test_full_cleartext_random_inputs_runs(self):
        rng = alexnet_full_random_inputs(seed=42)
        Wargs = prepare_full_args(rng)
        Y = alexnet_full_cleartext(rng["X"], *Wargs)
        self.assertEqual(Y.shape, (FULL_FC3_OUT,))
        self.assertTrue(np.all(np.isfinite(Y)))

    def test_full_input_shape(self):
        # The full path uses the same client-side downsample (3, 16, 16) as Tiny.
        rng = alexnet_full_random_inputs(seed=0)
        self.assertEqual(rng["X"].shape, (FULL_CI * FULL_H_IN * FULL_W_IN,))

    def test_full_cleartext_accuracy_beats_chance(self):
        """Cleartext CIFAR-10 accuracy gate for the depth-15 variant.

        The depth-15 net trains fine (~50%) with the shipped recipe when run
        fresh (`alexnet_train.py --full`), but its 7 stacked squarings make it
        LR-sensitive and prone to collapsing to chance under a bad init/LR.
        This gate catches such a collapse. Skips if weights or CIFAR are absent
        (ALEXNET_ACC_REQUIRE=1 to fail instead).
        Env: ALEXNET_ACC_N_FULL (default 128), ALEXNET_ACC_MIN (default 0.35).
        """
        _run_cleartext_accuracy_gate(
            self,
            load_weights=load_trained_alexnet_full_weights,
            prepare_args=prepare_full_args,
            cleartext_fn=alexnet_full_cleartext,
            n_env="ALEXNET_ACC_N_FULL",
            n_default=_ACC_N_FULL_DEFAULT,
            name="AlexNetFull")


class SecurityAssertTest(absltest.TestCase):
    """assert_he128_or_warn against the HES-2018 128-bit ceilings.

    The shipped classes no longer choose a chain — `packing.pack` derives one
    at 128 bits. These constants and this helper stay because
    `alexnet_wide_config` / `alexnet_wide_test` build their parameter study on
    them, and they record why the old hand-picked pool was unacceptable.
    """

    def test_deep_chain_is_617_bits(self):
        import math
        log_pq = sum(math.log2(x)
                     for x in list(DEEP_Q_TOWERS_POOL) + list(DEEP_P_TOWERS_POOL))
        self.assertAlmostEqual(log_pq, 617, delta=1)

    def test_the_retired_pool_still_has_its_documented_shape(self):
        self.assertLen(DEEP_Q_TOWERS_POOL, 17)  # 16 consumed + 1 headroom
        self.assertLen(DEEP_P_TOWERS_POOL, 4)

    def test_degree_2048_warns_but_does_not_raise(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert_he128_or_warn(2048, DEEP_Q_TOWERS_POOL, DEEP_P_TOWERS_POOL)
        self.assertTrue(any("NOT 128-bit" in str(x.message) for x in w))

    def test_require_128bit_raises_at_2048_and_16384(self):
        # 617-bit chain: insecure at both 2048 (ceil 54) and 16384 (ceil 438).
        for degree in (2048, 16384):
            with self.assertRaises(ValueError):
                assert_he128_or_warn(degree, DEEP_Q_TOWERS_POOL,
                                     DEEP_P_TOWERS_POOL, require_128bit=True)

    def test_degree_32768_is_secure(self):
        # ceiling 881 >= 617: no raise even under require_128bit.
        assert_he128_or_warn(32768, DEEP_Q_TOWERS_POOL, DEEP_P_TOWERS_POOL,
                             require_128bit=True)
        self.assertEqual(HES128_MAX_LOG_PQ[32768], 881)


if __name__ == "__main__":
    absltest.main()
