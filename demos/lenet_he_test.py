"""Structural gates for the LeNet HE entrypoint.

`LeNetHE` builds nothing itself any more. It binds the caller's positional
weight arrays into `LeNetDemo`'s torch model and lets the one pipeline every
demo uses do the rest::

    torch.nn.Module  ->  nn.vectorize  ->  packing.pack  ->  Mapping

So the gates here are about delegation: each stage runs exactly once per
entrypoint call, no hand-built packing graph is constructed on the way, and
the arrays the caller passed are the ones that get compiled.

Every gate is plan-level. Constructing a real Mapping over LeNet's own ring
costs tens of gigabytes and minutes -- the cost is Mapping initialization and
the physical constants, not key generation -- so every test that reaches
materialization patches `Mapping` out. The end-to-end accuracy gate that used
to live in this file cannot run under that rule and is gone; run the demo CLI
(`python3 lenet_he.py`) for a real encrypted forward pass.
"""
from __future__ import annotations

import ast
import contextlib
import os
import pickle
import sys
import tempfile
from unittest import mock

import numpy as np
from absl.testing import absltest

_HERE = os.path.dirname(os.path.abspath(__file__))
_JAXITE = os.path.abspath(os.path.join(_HERE, "..", "jaxite_word"))
for _path in (_JAXITE, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                   # noqa: E402
jax.config.update("jax_enable_x64", True)

import lenet_he                                              # noqa: E402
from lenet_he import (                                       # noqa: E402
    FC2_OUT, H_IN, W_IN,
    LeNetHE, lenet_cleartext, lenet_random_inputs,
    load_trained_lenet_weights, normalize_mnist_image,
)
import canonical_demo                                        # noqa: E402
import encrypted_demos                                       # noqa: E402
import mapping as mapping_module                             # noqa: E402
import nn                                                    # noqa: E402
import packing                                               # noqa: E402


_VALUES = lenet_random_inputs(seed=42)
_WEIGHTS = (_VALUES["W1"], _VALUES["W2"], _VALUES["W3"], _VALUES["W4"])
_BIASES = (_VALUES["b1"], _VALUES["b2"], _VALUES["b3"], _VALUES["b4"])

# The packing-IR constructors the demo used to call by hand. None of them may
# be reached from this entrypoint any more.
_HAND_BUILT = (
    "Sequential", "Rotate", "MulPlain", "ParallelSum", "AddPlain", "Linear",
)

# Removed with the hand-built graph. Named here so a re-import cannot bring
# them back unnoticed.
_REMOVED = (
    "conv_to_matrix", "conv1_to_matrix_multiplexed",
    "_matrix_to_sparse_diagonals", "_build_lenet_network", "_bias_slots",
    "_lower_lenet_constants", "fc_to_matrix",
)


def _prepare(model, *, weights=_WEIGHTS, biases=_BIASES):
    """Run the offline entrypoint with Mapping stubbed out."""
    with mock.patch.object(mapping_module, "Mapping") as mapping_class:
        model.precompute_plaintexts(*weights, *biases)
    return mapping_class


class CanonicalDelegationTest(absltest.TestCase):
    """The entrypoint is vectorize -> pack -> Mapping, once each."""

    def test_precompute_runs_each_stage_exactly_once(self):
        model = LeNetHE()
        with (
            mock.patch.object(
                canonical_demo.nn, "vectorize", wraps=nn.vectorize
            ) as vectorize,
            mock.patch.object(
                canonical_demo.packing, "pack", wraps=packing.pack
            ) as pack,
            mock.patch.object(mapping_module, "Mapping") as mapping_class,
        ):
            model.precompute_plaintexts(*_WEIGHTS, *_BIASES)
            # Re-reading the artifacts must not re-run anything.
            _ = model.vectorized_program
            _ = model.packed_program
            _ = model.ring_config
            _ = model.mapping

        self.assertEqual(vectorize.call_count, 1)
        self.assertEqual(pack.call_count, 1)
        self.assertEqual(mapping_class.call_count, 1)

    def test_vectorize_receives_the_torch_model_and_pack_its_program(self):
        model = LeNetHE()
        with (
            mock.patch.object(
                canonical_demo.nn, "vectorize", wraps=nn.vectorize
            ) as vectorize,
            mock.patch.object(
                canonical_demo.packing, "pack", wraps=packing.pack
            ) as pack,
            mock.patch.object(mapping_module, "Mapping"),
        ):
            model.precompute_plaintexts(*_WEIGHTS, *_BIASES)

        traced, shape = vectorize.call_args.args
        self.assertIs(traced, model.demo.model)
        self.assertEqual(shape, (1, H_IN, W_IN))
        self.assertIs(pack.call_args.args[0], model.vectorized_program)

    def test_the_mapping_gets_the_packed_program_and_scheduling_only(self):
        model = LeNetHE(batch=2, dnum=3)
        mapping_class = _prepare(model)

        positional, keywords = mapping_class.call_args
        self.assertLen(positional, 1, "Mapping derives its ring from the program")
        self.assertIs(positional[0], model.packed_program)
        self.assertEqual(
            keywords, {"global_batch": 2, "devices": None, "dnum": 3}
        )

    def test_planning_properties_never_build_a_mapping(self):
        model = LeNetHE()
        with mock.patch.object(mapping_module, "Mapping") as mapping_class:
            self.assertIsInstance(
                model.vectorized_program, nn.VectorizedProgram
            )
            self.assertIsInstance(model.packed_program, packing.Packing)
            self.assertIs(model.ring_config, model.packed_program.ring_config)
            self.assertIs(model.packing, model.packed_program)
            mapping_class.assert_not_called()

    def test_the_plan_is_the_canonical_depth_seven_lenet(self):
        model = LeNetHE()
        packed = model.packed_program
        self.assertSequenceEqual(
            packed.kinds(),
            (
                "matvec", "add_plain", "square",
                "matvec", "add_plain", "square",
                "matvec", "add_plain", "square",
                "matvec", "add_plain",
            ),
        )
        self.assertEqual(packed.depth, 7)
        self.assertEqual(packed.logical_input_shape, (1, H_IN, W_IN))
        self.assertEqual(packed.logical_output_shape, (FC2_OUT,))
        self.assertEqual(int(model.ring_config.degree), 32768)
        self.assertEqual(int(model.ring_config.security_bits), 128)

    def test_the_shell_delegates_to_a_lenet_demo(self):
        model = LeNetHE(batch=3)
        self.assertIsInstance(model.demo, encrypted_demos.LeNetDemo)
        self.assertIsInstance(model.demo, canonical_demo.CanonicalDemo)
        self.assertFalse(model.demo.is_mapped())

    def test_infer_delegates_to_the_demo(self):
        model = LeNetHE()
        sample = object()
        with self.assertRaisesRegex(RuntimeError, "precompute_plaintexts"):
            model.infer(sample)

        _prepare(model)
        logits = np.arange(FC2_OUT, dtype=np.float64)
        with mock.patch.object(
            model.demo, "infer", return_value=logits
        ) as infer:
            actual = model.infer(sample)
        infer.assert_called_once_with(sample, trace_dir=None)
        np.testing.assert_array_equal(actual, logits)

    def test_batch_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "batch must be >= 1"):
            LeNetHE(batch=0)


class NoHandBuiltGraphTest(absltest.TestCase):
    """The packing-IR constructors are the packer's business, not the demo's."""

    def test_the_entrypoint_calls_no_hand_built_graph_builder(self):
        """The retired builders are gone, so no path can construct one."""
        for builder in ('Sequential', 'Rotate', 'MulPlain', 'ParallelSum', 'AddPlain',
                    'Linear', 'Identity', 'Square', 'Rescale', 'Bootstrap'):
            self.assertFalse(
                hasattr(packing, builder), f"packing.{builder} came back")

    def test_the_module_never_names_a_hand_built_graph_builder(self):
        tree = ast.parse(open(lenet_he.__file__).read())
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        names |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for builder in _HAND_BUILT + ("SparseDiagonals", "PlainSlots", "Square"):
            self.assertNotIn(builder, names, f"lenet_he names {builder}")

    def test_the_hand_built_lowering_helpers_are_gone(self):
        for name in _REMOVED:
            self.assertFalse(
                hasattr(lenet_he, name), f"lenet_he still defines {name}"
            )


class CallerWeightsTest(absltest.TestCase):
    """The compiled model is the caller's, not the demo's seeded fallback."""

    def test_the_supplied_arrays_land_in_the_traced_model(self):
        model = LeNetHE()
        _prepare(model)
        state = model.demo.model.state_dict()
        for name, expected in (
            ("conv1.weight", _WEIGHTS[0]),
            ("conv2.weight", _WEIGHTS[1]),
            ("fc1.weight", _WEIGHTS[2]),
            ("fc2.weight", _WEIGHTS[3]),
        ):
            np.testing.assert_allclose(
                state[name].numpy().reshape(-1), expected, rtol=0, atol=0
            )

    def test_the_bound_model_computes_the_cleartext_reference(self):
        model = LeNetHE()
        _prepare(model)
        expected = lenet_cleartext(_VALUES["X"], *_WEIGHTS, *_BIASES)
        np.testing.assert_allclose(
            model.demo.cleartext(_VALUES["X"]), expected, atol=1e-9
        )
        # The seeded fallback is a different model; this must not silently be
        # what got compiled.
        fallback = encrypted_demos.LeNetDemo().cleartext(_VALUES["X"])
        self.assertGreater(
            float(np.max(np.abs(fallback - expected))), 1e-9
        )

    def test_none_biases_leave_the_model_biases_at_zero(self):
        model = LeNetHE()
        with mock.patch.object(mapping_module, "Mapping"):
            model.precompute_plaintexts(*_WEIGHTS)
        state = model.demo.model.state_dict()
        for name in ("conv1.bias", "conv2.bias", "fc1.bias", "fc2.bias"):
            self.assertFalse(np.any(state[name].numpy()))
        np.testing.assert_allclose(
            model.demo.cleartext(_VALUES["X"]),
            lenet_cleartext(_VALUES["X"], *_WEIGHTS),
            atol=1e-9,
        )

    def test_a_mis_sized_weight_array_is_refused(self):
        model = LeNetHE()
        broken = (np.zeros(7),) + _WEIGHTS[1:]
        with self.assertRaisesRegex(ValueError, "conv1.weight"):
            model.precompute_plaintexts(*broken, *_BIASES)

    def test_precompute_refuses_to_rebind_a_prepared_model(self):
        model = LeNetHE()
        _prepare(model)
        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            model.precompute_plaintexts(*_WEIGHTS, *_BIASES)

    def test_the_layout_flag_no_longer_changes_the_program(self):
        """Layout selection moved into the packer's templates."""
        multiplexed = LeNetHE(use_multiplexed_conv1=True)
        row_major = LeNetHE(use_multiplexed_conv1=False)
        _prepare(multiplexed)
        _prepare(row_major)
        self.assertFalse(row_major._use_multiplexed_conv1)
        self.assertEqual(
            multiplexed.packed_program.fingerprint,
            row_major.packed_program.fingerprint,
        )

    def test_trained_binaries_drive_the_same_model(self):
        trained = load_trained_lenet_weights()
        if trained is None:
            self.skipTest("no trained LeNet binaries under demos/maple_data")
        weights = tuple(trained[key] for key in ("W1", "W2", "W3", "W4"))
        biases = tuple(trained[key] for key in ("b1", "b2", "b3", "b4"))
        model = LeNetHE()
        _prepare(model, weights=weights, biases=biases)
        image = normalize_mnist_image(
            np.random.default_rng(0).random(H_IN * W_IN)
        )
        np.testing.assert_allclose(
            model.demo.cleartext(image),
            lenet_cleartext(image, *weights, *biases),
            atol=1e-9,
        )


class LegacyCacheTest(absltest.TestCase):
    """Old caches describe a ring and a graph that no longer exist."""

    def test_a_cached_state_is_rejected_by_the_constructor(self):
        with self.assertRaisesRegex(ValueError, "securely derived ring"):
            LeNetHE(_cached_state={"cache_format": lenet_he._LENET_CACHE_FORMAT})

    def test_from_cache_refuses_without_reading_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy.pkl")
            with open(path, "wb") as stream:
                pickle.dump({"config": {}}, stream)
            with self.assertRaisesRegex(ValueError, "securely derived ring"):
                LeNetHE.from_cache(path)
            # A missing file raises the same error, so nothing was unpickled.
            with self.assertRaisesRegex(ValueError, "securely derived ring"):
                LeNetHE.from_cache(os.path.join(directory, "absent.pkl"))

    def test_save_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(NotImplementedError):
                LeNetHE().save_cache(os.path.join(directory, "static.pkl"))


class CleartextHelpersTest(absltest.TestCase):
    """The reference helpers and loaders this demo still owns."""

    def test_random_inputs_have_the_reference_shapes(self):
        values = lenet_random_inputs(seed=7)
        self.assertEqual(values["X"].shape, (H_IN * W_IN,))
        self.assertEqual(values["W1"].shape, (4 * 1 * 5 * 5,))
        self.assertEqual(values["W2"].shape, (8 * 4 * 5 * 5,))
        self.assertEqual(values["W3"].shape, (32 * 392,))
        self.assertEqual(values["W4"].shape, (10 * 32,))
        for key in ("b1", "b2", "b3", "b4"):
            self.assertFalse(np.any(values[key]))

    def test_the_cleartext_reference_returns_ten_logits(self):
        logits = lenet_cleartext(_VALUES["X"], *_WEIGHTS, *_BIASES)
        self.assertEqual(np.asarray(logits).shape, (FC2_OUT,))

    def test_mnist_normalisation_is_the_torchvision_transform(self):
        sample = np.random.default_rng(1).random(16)
        np.testing.assert_allclose(
            normalize_mnist_image(sample),
            (sample - lenet_he.MNIST_MEAN) / lenet_he.MNIST_STD,
        )

    def test_the_trained_loader_reports_missing_binaries(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_trained_lenet_weights(directory))

    def test_the_legacy_slot_packers_still_describe_their_layouts(self):
        image = np.arange(H_IN * W_IN, dtype=np.float64)
        row_major = lenet_he.pack_input(image)
        self.assertEqual(row_major.shape, (lenet_he.NUM_SLOTS,))
        np.testing.assert_array_equal(row_major[: image.size], image)

        multiplexed = lenet_he.pack_input_multiplexed_conv1(image)
        height = H_IN // lenet_he.STRIDE
        width = W_IN // lenet_he.STRIDE
        for h in range(height):
            for w in range(width):
                for r_mod in range(lenet_he.STRIDE):
                    for s_mod in range(lenet_he.STRIDE):
                        rs = r_mod * lenet_he.STRIDE + s_mod
                        self.assertEqual(
                            multiplexed[rs * height * width + h * width + w],
                            image[
                                (lenet_he.STRIDE * h + r_mod) * W_IN
                                + (lenet_he.STRIDE * w + s_mod)
                            ],
                        )


if __name__ == "__main__":
    absltest.main()
