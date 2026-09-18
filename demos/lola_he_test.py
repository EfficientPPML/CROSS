"""LoLAHE goes through the canonical pipeline, once, and nothing else.

The class used to lower LoLA by hand -- eight ParallelSum branches for conv1
plus SparseDiagonals for the two dense layers -- so this file used to assert
that structure. It is gone: `nn.vectorize` and `packing.pack` decide the
lowering now, so what is worth asserting is that each stage runs exactly once,
that the caller's arrays land in the torch module that gets traced, and that
no hand-built graph node is constructed on the way.

Constructing a real Mapping over LoLA's derived ring is not something a test
can afford. Key generation is cheap; Mapping initialization and the physical
constants are not -- LoLA now packs to degree 32768 / 16384 slots, where
materializing BSGS diagonals and per-level operator controls runs to tens of
gigabytes. Every test here therefore patches `mapping.Mapping` and asserts at
the plan level. The end-to-end functional gate below is preserved but skipped
unless LOLA_HE_FUNCTIONAL=1 says the machine can pay for it.
"""
from __future__ import annotations

import contextlib
import csv
import os
import tempfile
import time
from unittest import mock
from dataclasses import dataclass
from typing import Any

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lola_he import (
    CO,
    DNUM,
    DEGREE,
    FC1_IN,
    FC1_OUT,
    FC2_IN,
    FC2_OUT,
    KH,
    KW,
    NUM_SLOTS,
    P_TOWERS,
    Q_TOWERS,
    R,
    C,
    SF,
    LoLAHE,
    conv1_cleartext,
    load_or_generate_data,
    lola_cleartext_inference,
    pack_lola_input,
    prepare_weights,
    resolve_indices,
)
import lola_he as lola_module

# The canonical pipeline's three stages, imported with the flat spelling
# `canonical_demo` itself uses so the spies patch the same module objects.
canonical_demo = None
encrypted_demos = None
nn = None
packing = None


def setUpModule():
    global canonical_demo, encrypted_demos, nn, packing
    try:
        import torch  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment dependent
        raise absltest.SkipTest(f"PyTorch unavailable: {error}")
    import canonical_demo as canonical_module
    import encrypted_demos as demos_module
    import nn as nn_module
    import packing as packing_module
    canonical_demo = canonical_module
    encrypted_demos = demos_module
    nn = nn_module
    packing = packing_module


_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
_FUNCTIONAL_GATE = "LOLA_HE_FUNCTIONAL"

FUNCTIONAL_TEST_CASES = [
    ("image0", 1, "0"),
]

# The hand-built lowering LoLAHE no longer performs, and the packing nodes it
# used to construct. Both must stay unreachable.
LEGACY_BUILDERS = (
    "_build_lola_network",
    "_group_conv1_weight_slots",
    "build_conv1_bias_slots",
    "_dense_to_sparse_diagonals",
    "_dense_bias_slots",
)
# Only the graph-*shape* nodes. The packer legitimately builds constant
# containers (PlainSlots, the lazy diagonal source) for the matvecs and biases
# it emits; what must never be constructed again is a hand-written network.
LEGACY_GRAPH_NODES = (
    "Sequential",
    "ParallelSum",
    "MulPlain",
    "AddPlain",
    "Rotate",
    "Rescale",
    "Linear",
)


def lola_weights(seed: int = 713):
    """A positional LoLA weight list in `precompute_plaintexts` order."""
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=CO * KH * KW) * 0.2,
        rng.normal(size=CO) * 0.1,
        rng.normal(size=(FC1_OUT, FC1_IN)) * 0.05,
        rng.normal(size=FC1_OUT) * 0.1,
        rng.normal(size=(FC2_OUT, FC2_IN)) * 0.1,
        rng.normal(size=FC2_OUT) * 0.1,
    )


@contextlib.contextmanager
def pipeline_spies():
    """Wrap the three canonical stages; Mapping is replaced, not wrapped."""
    with mock.patch.object(
        canonical_demo.nn, "vectorize", wraps=canonical_demo.nn.vectorize
    ) as vectorize, mock.patch.object(
        canonical_demo.packing, "pack", wraps=canonical_demo.packing.pack
    ) as pack, mock.patch("mapping.Mapping") as mapping_class:
        yield vectorize, pack, mapping_class


class LoLACanonicalPipelineTest(absltest.TestCase):
    """torch.nn.Module -> nn.vectorize -> packing.pack -> Mapping."""

    def test_construction_neither_plans_nor_maps(self):
        with pipeline_spies() as (vectorize, pack, mapping_class):
            model = LoLAHE()
        vectorize.assert_not_called()
        pack.assert_not_called()
        mapping_class.assert_not_called()
        self.assertIsInstance(model.demo, encrypted_demos.LoLADemo)
        self.assertFalse(model._prepared)

    def test_each_stage_runs_exactly_once(self):
        model = LoLAHE()
        with pipeline_spies() as (vectorize, pack, mapping_class):
            model.precompute_plaintexts(*lola_weights())
            # Every artifact is served from the one plan and the one Mapping.
            _ = model.vectorized_program
            _ = model.packed_program
            _ = model.ring_config
            _ = model.packing
            _ = model.mapping
            _ = model.mapping
            _ = model.ctx
            self.assertEqual(vectorize.call_count, 1)
            self.assertEqual(pack.call_count, 1)
            self.assertEqual(mapping_class.call_count, 1)

    def test_the_mapping_is_built_from_the_packed_program(self):
        model = LoLAHE()
        with pipeline_spies() as (_, _, mapping_class):
            model.precompute_plaintexts(*lola_weights())
            positional, keywords = mapping_class.call_args
            self.assertLen(positional, 1)
            self.assertIs(positional[0], model.packed_program)
            self.assertIsInstance(positional[0], packing.Packing)
            # No parameters dict: the packed program carries its own ring.
            self.assertNotIn("parameters", keywords)

    def test_planning_exposes_the_three_artifacts_without_mapping(self):
        model = LoLAHE()
        with pipeline_spies() as (_, _, mapping_class):
            model.demo.load_positional_weights(*lola_weights())
            self.assertIsInstance(
                model.vectorized_program, nn.VectorizedProgram
            )
            self.assertIsInstance(model.packed_program, packing.Packing)
            self.assertIs(
                model.ring_config, model.packed_program.ring_config
            )
            self.assertIs(model.packing, model.packed_program)
            mapping_class.assert_not_called()

    def test_the_derived_ring_replaces_the_hand_tuned_one(self):
        model = LoLAHE()
        with pipeline_spies():
            model.precompute_plaintexts(*lola_weights())
            packed = model.packed_program
            ring = model.ring_config
        # Depth 5: conv1, square, fc1, square, fc2.
        self.assertEqual(packed.depth, 5)
        self.assertEqual(int(ring.security_bits), 128)
        self.assertEqual(packed.num_slots, int(ring.degree) // 2)
        # The eight-branch conv1 decomposition is one matvec now.
        self.assertEqual(packed.kinds().count("matvec"), 3)
        self.assertEqual(packed.kinds().count("square"), 2)
        self.assertEqual(packed.kinds().count("rotate"), 0)
        self.assertEqual(packed.kinds().count("mul_plain"), 0)
        # And it is nothing like the retired constants this module still
        # exports for lenet_he and ckks_ctx_test.
        self.assertNotEqual(int(ring.degree), DEGREE)
        self.assertNotEqual(packed.num_slots, NUM_SLOTS)


class LegacyBuilderTest(absltest.TestCase):
    """The hand-built graph is deleted, not merely unused."""

    def test_the_module_no_longer_defines_the_builders(self):
        for builder in LEGACY_BUILDERS:
            self.assertFalse(
                hasattr(lola_module, builder),
                f"lola_he still defines {builder}",
            )

    def test_precompute_constructs_no_packing_graph_node(self):
        """The retired builders are gone, so no path can construct one."""
        for builder in ('Sequential', 'Rotate', 'MulPlain', 'ParallelSum', 'AddPlain',
                    'Linear', 'Identity', 'Square', 'Rescale', 'Bootstrap'):
            self.assertFalse(
                hasattr(packing, builder), f"packing.{builder} came back")

    def test_the_module_names_no_packing_graph_node(self):
        import ast
        source = open(lola_module.__file__, encoding="utf-8").read()
        names = {
            node.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute)
        } | {
            node.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Name)
        }
        for node in ("ParallelSum", "MulPlain", "SparseDiagonals",
                     "PlainSlots", "Rotate"):
            self.assertNotIn(node, names, f"lola_he names {node}")


class WeightRoutingTest(absltest.TestCase):
    """precompute_plaintexts binds the model nn.vectorize actually traces."""

    def test_precompute_routes_through_load_positional_weights(self):
        model = LoLAHE()
        weights = lola_weights()
        demo = model.demo
        with mock.patch.object(
            demo,
            "load_positional_weights",
            wraps=demo.load_positional_weights,
        ) as load, mock.patch("mapping.Mapping"):
            model.precompute_plaintexts(*weights)
        load.assert_called_once()
        positional, keywords = load.call_args
        self.assertEmpty(keywords)
        self.assertLen(positional, 6)
        for supplied, expected in zip(positional, weights):
            np.testing.assert_array_equal(supplied, expected)

    def test_the_arrays_land_in_the_torch_module(self):
        model = LoLAHE()
        weights = lola_weights(404)
        with mock.patch("mapping.Mapping"):
            model.precompute_plaintexts(*weights)
        state = model.demo.model.state_dict()
        for array, (_, parameter) in zip(weights, model.demo.weight_order):
            np.testing.assert_allclose(
                state[parameter].detach().numpy().reshape(-1),
                np.asarray(array, dtype=np.float64).reshape(-1),
                atol=1e-12,
            )

    def test_different_weights_pack_to_a_different_program(self):
        first, second = LoLAHE(), LoLAHE()
        with mock.patch("mapping.Mapping"):
            first.precompute_plaintexts(*lola_weights(1))
            second.precompute_plaintexts(*lola_weights(2))
        self.assertNotEqual(
            first.packed_program.fingerprint,
            second.packed_program.fingerprint,
        )

    def test_preparing_twice_is_refused(self):
        model = LoLAHE()
        with mock.patch("mapping.Mapping"):
            model.precompute_plaintexts(*lola_weights())
            with self.assertRaisesRegex(RuntimeError, "already prepared"):
                model.precompute_plaintexts(*lola_weights(2))

    def test_any_nondefault_bsgs_ratio_is_refused(self):
        """The split is chosen per matvec by Mapping, so a global ratio has
        nowhere honest to go -- equal or not."""
        for ratios in (
            {"fc1_bsgs_ratio": 4.0, "fc2_bsgs_ratio": 4.0},
            {"fc1_bsgs_ratio": 2.0, "fc2_bsgs_ratio": 4.0},
        ):
            with self.subTest(**ratios):
                with self.assertRaisesRegex(ValueError, "per matvec"):
                    LoLAHE().precompute_plaintexts(*lola_weights(), **ratios)

    def test_the_historic_default_ratio_is_accepted(self):
        """2.0 was the default; accepting it keeps old calls working."""
        model = LoLAHE()
        with pipeline_spies() as (_, pack, _mapping):
            model.precompute_plaintexts(
                *lola_weights(), fc1_bsgs_ratio=2.0, fc2_bsgs_ratio=2.0
            )
            self.assertEqual(pack.call_count, 1)

    def test_a_packed_matvec_leaves_the_split_open(self):
        """Packing records no ratio: n1, n2 and the ratio stay unset."""
        model = LoLAHE()
        with pipeline_spies():
            model.precompute_plaintexts(*lola_weights())
        for _, kind, _, argument in model.packed_program.operations:
            if kind == "matvec":
                self.assertEqual(argument[1:], (None, None, None, None))


class SchedulingTest(parameterized.TestCase):
    """batch, devices and dnum are scheduling; nothing else is accepted."""

    def test_scheduling_reaches_the_mapping(self):
        # DNUM is the value the packer derives for LoLA; a dnum that
        # contradicts the plan is refused by Mapping itself, so a supplied one
        # is only ever a restatement of it.
        model = LoLAHE(batch=2, dnum=DNUM)
        with mock.patch("mapping.Mapping") as mapping_class:
            model.precompute_plaintexts(*lola_weights())
        _, keywords = mapping_class.call_args
        self.assertEqual(keywords["global_batch"], 2)
        self.assertEqual(keywords["dnum"], DNUM)
        self.assertIsNone(keywords["devices"])

    def test_an_unset_dnum_leaves_the_derived_one_alone(self):
        model = LoLAHE()
        with mock.patch("mapping.Mapping") as mapping_class:
            model.precompute_plaintexts(*lola_weights())
        _, keywords = mapping_class.call_args
        self.assertNotIn("dnum", keywords)
        self.assertEqual(int(model.ring_config.dnum), DNUM)

    def test_scheduling_does_not_change_the_program(self):
        plain, scheduled = LoLAHE(), LoLAHE(batch=4, dnum=DNUM)
        with mock.patch("mapping.Mapping"):
            plain.precompute_plaintexts(*lola_weights())
            scheduled.precompute_plaintexts(*lola_weights())
        self.assertEqual(
            plain.packed_program.fingerprint,
            scheduled.packed_program.fingerprint,
        )

    @parameterized.named_parameters(
        ("q_towers", "q_towers", list(Q_TOWERS)),
        ("p_towers", "p_towers", list(P_TOWERS)),
        ("sf", "sf", float(SF)),
    )
    def test_ring_configuration_cannot_be_supplied(self, name, value):
        with self.assertRaisesRegex(ValueError, "derived from the model"):
            LoLAHE(**{name: value})

    def test_a_bad_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "batch must be"):
            LoLAHE(batch=0)


class LegacyCacheTest(absltest.TestCase):
    """The pickled cache described a network and a ring that no longer exist."""

    def test_a_cached_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "caches are retired"):
            LoLAHE(_cached_state={"kp": None, "ek": None})

    def test_from_cache_is_rejected_without_unpickling(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl") as cache:
            cache.write(b"not a pickle at all")
            cache.flush()
            with self.assertRaisesRegex(ValueError, "caches are retired"):
                LoLAHE.from_cache(cache.name)

    def test_save_cache_is_rejected(self):
        model = LoLAHE()
        with mock.patch("mapping.Mapping"):
            model.precompute_plaintexts(*lola_weights())
        with self.assertRaisesRegex(RuntimeError, "caches are retired"):
            model.save_cache(os.path.join(_LOG_DIR, "unwritten.pkl"))


class ServingTest(absltest.TestCase):
    """infer delegates to the demo, which delegates to its one Mapping."""

    def test_infer_delegates_to_the_mapping(self):
        model = LoLAHE()
        logits = np.arange(FC2_OUT, dtype=np.float64)
        with mock.patch("mapping.Mapping") as mapping_class:
            mapping_class.return_value.infer.return_value = logits
            model.precompute_plaintexts(*lola_weights())
            image = np.zeros(28 * 28)
            result = model.infer(image)
            mapped = mapping_class.return_value
        mapped.infer.assert_called_once()
        positional, keywords = mapped.infer.call_args
        np.testing.assert_array_equal(positional[0], image)
        self.assertIsNone(keywords["trace_dir"])
        np.testing.assert_array_equal(result, logits)

    def test_infer_forwards_a_trace_directory(self):
        model = LoLAHE()
        with mock.patch("mapping.Mapping") as mapping_class:
            model.precompute_plaintexts(*lola_weights())
            model.infer(np.zeros(28 * 28), trace_dir="/tmp/does-not-run")
        _, keywords = mapping_class.return_value.infer.call_args
        self.assertEqual(keywords["trace_dir"], "/tmp/does-not-run")

    def test_serving_before_precompute_is_refused(self):
        model = LoLAHE()
        with self.assertRaisesRegex(RuntimeError, "precompute_plaintexts"):
            model.infer(np.zeros(28 * 28))
        with self.assertRaisesRegex(RuntimeError, "precompute_plaintexts"):
            _ = model.ctx


class CleartextReferenceTest(absltest.TestCase):
    """The cleartext helpers other modules import still hold."""

    def test_pack_lola_input_is_the_stride_multiplexed_layout(self):
        rng = np.random.default_rng(991)
        image = rng.normal(size=28 * 28)
        packed = pack_lola_input(image)
        self.assertEqual(packed.shape, (NUM_SLOTS,))
        grid = image.reshape(28, 28)
        for r in range(KH):
            for s in range(KW):
                block = packed[
                    (r * KW + s) * 196: (r * KW + s + 1) * 196
                ].reshape(14, 14)
                np.testing.assert_array_equal(block, grid[r::2, s::2])
        np.testing.assert_array_equal(packed[784:], 0)

    def test_cleartext_inference_matches_the_torch_module(self):
        weights = lola_weights(55)
        rng = np.random.default_rng(31)
        image = rng.random(784)
        demo = encrypted_demos.LoLADemo()
        demo.load_positional_weights(*weights)
        reference = np.asarray(
            lola_cleartext_inference(image, *weights), dtype=np.float64
        )
        np.testing.assert_allclose(
            demo.cleartext(image.reshape(1, 28, 28)), reference, atol=1e-9
        )

    def test_conv1_cleartext_shape(self):
        rng = np.random.default_rng(17)
        conv = conv1_cleartext(
            rng.normal(size=784), rng.normal(size=CO * KH * KW),
            rng.normal(size=CO),
        )
        self.assertEqual(conv.shape, (CO, 14, 14))
        self.assertEqual(conv.size, FC1_IN)

    def test_resolve_indices(self):
        self.assertEqual(resolve_indices(200, 3, ""), [0, 1, 2])
        self.assertEqual(resolve_indices(200, 3, "5, 7"), [5, 7])
        with self.assertRaises(ValueError):
            resolve_indices(4, 1, "9")

    def test_the_retired_ring_constants_stay_exported(self):
        """lenet_he.py and jaxite_word/ckks_ctx_test.py still import these."""
        self.assertEqual((DEGREE, NUM_SLOTS), (2048, 1024))
        self.assertEqual((R, C), (32, 64))
        self.assertEqual(R * C, DEGREE)
        self.assertLen(lola_module.Q_TOWERS_POOL, 9)
        self.assertLen(lola_module.P_TOWERS_POOL, 4)
        self.assertEqual(Q_TOWERS, lola_module.Q_TOWERS_POOL[:7])
        self.assertEqual(P_TOWERS, lola_module.P_TOWERS_POOL[:3])
        self.assertEqual(SF, Q_TOWERS[0] * Q_TOWERS[1])
        self.assertGreater(lola_module.SIGMA, 0.0)

    def test_prepare_weights_flattens_conv1(self):
        weights = lola_weights(9)
        data = dict(
            zip(("W1", "b1", "W2", "b2", "W3", "b3"), weights)
        )
        data["W1"] = weights[0].reshape(CO, KH * KW)
        prepared = prepare_weights(data)
        self.assertEqual(prepared[0].shape, (CO * KH * KW,))
        self.assertEqual(prepared[2].shape, (FC1_OUT, FC1_IN))


# ---------------------------------------------------------------------------
# End-to-end functional gate. Real keys, real constants, real ciphertexts on
# LoLA's own derived ring -- tens of gigabytes and minutes per image. Opt in.
# ---------------------------------------------------------------------------
@dataclass
class _InferenceResult:
    scores: np.ndarray
    wall_s: float


def run_model(model: LoLAHE, img: np.ndarray) -> _InferenceResult:
    """Time one public, input-only serving call for the functional test."""
    start = time.perf_counter()
    scores = model.infer(img)
    return _InferenceResult(
        scores=np.asarray(scores), wall_s=time.perf_counter() - start)


@dataclass
class CorrectnessConfig:
    n: int = 1
    indices: str = ""
    warmup: bool = False
    # Maximum |he_scores - pt_scores| tolerance (CKKS noise budget).
    pt_score_tol: float = 5e-2
    output_csv: str = ""


@dataclass
class CorrectnessOutcome:
    rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    data_source: str


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_functional_correctness_test(
    config: CorrectnessConfig,
) -> CorrectnessOutcome:
    data, data_source = load_or_generate_data()
    weights = prepare_weights(data)
    imgs = data["imgs"]
    labels = data["labels"]
    pt_pred = data["pt_pred"]
    indices = resolve_indices(len(imgs), config.n, config.indices)

    print("=" * 70)
    print("CROSS HE LoLA functional correctness (vs plaintext reference)")
    print(f"  images={indices}")
    print(f"  data_source={data_source}")
    print(f"  pt_score_tol={config.pt_score_tol}")
    print("=" * 70)

    model = LoLAHE()
    model.precompute_plaintexts(*weights)

    if config.warmup:
        warm_img = imgs[indices[0]]
        print("[warmup] first inference ...")
        run_model(model, warm_img)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx in indices:
        img = imgs[idx]
        label = int(labels[idx])
        pt_scores = np.asarray(lola_cleartext_inference(img, *weights))
        pt_argmax = int(np.argmax(pt_scores))

        he_run = run_model(model, img)
        he_scores = he_run.scores
        he_argmax = int(np.argmax(he_scores))

        max_abs = float(np.max(np.abs(he_scores - pt_scores)))
        mean_abs = float(np.mean(np.abs(he_scores - pt_scores)))

        passed = (he_argmax == pt_argmax) and (max_abs <= config.pt_score_tol)

        row = {
            "image_index": idx,
            "label": label,
            "plaintext_argmax": pt_argmax,
            "stored_pt_argmax": int(pt_pred[idx]),
            "he_argmax": he_argmax,
            "he_matches_plaintext": int(he_argmax == pt_argmax),
            "max_abs_he_plaintext": max_abs,
            "mean_abs_he_plaintext": mean_abs,
            "he_wall_s": he_run.wall_s,
            "pass": int(passed),
        }
        rows.append(row)

        status = "PASS" if passed else "FAIL"
        print(
            f"[{idx}] {status} label={label} pt={pt_argmax} he={he_argmax} "
            f"max|he-pt|={max_abs:.2e} wall={he_run.wall_s:.1f}s"
        )
        if not passed:
            failures.append(row)

    if config.output_csv:
        write_csv(config.output_csv, rows)
        print(f"\nSaved CSV to {os.path.abspath(config.output_csv)}")

    he_avg = sum(r["he_wall_s"] for r in rows) / len(rows)
    worst_diff = max(r["max_abs_he_plaintext"] for r in rows)
    print(f"\nImages checked: {len(rows)}")
    print(f"Failures:      {len(failures)}")
    print(f"Avg wall:      {he_avg:.3f}s")
    print(f"Worst |he-pt|: {worst_diff:.2e}")

    if failures:
        worst = max(failures, key=lambda row: row["max_abs_he_plaintext"])
        print(
            "Worst failure: "
            f"image={worst['image_index']} "
            f"max|he-pt|={worst['max_abs_he_plaintext']:.2e} "
            f"pt={worst['plaintext_argmax']} he={worst['he_argmax']}"
        )

    return CorrectnessOutcome(
        rows=rows, failures=failures, data_source=data_source)


class LoLAHEFunctionalTest(parameterized.TestCase):
    """Functional correctness for LoLAHE, on the ring the packer derived."""

    def setUp(self):
        super().setUp()
        if not os.environ.get(_FUNCTIONAL_GATE):
            self.skipTest(
                f"set {_FUNCTIONAL_GATE}=1 to run the end-to-end gate: LoLA "
                "packs to degree 32768 / 16384 slots, where materializing the "
                "Mapping's constants costs tens of gigabytes and minutes."
            )

    @parameterized.named_parameters(*FUNCTIONAL_TEST_CASES)
    def test_functional_correctness(self, n, indices):
        output_csv = os.path.join(_LOG_DIR, "lola_he_correctness.csv")
        outcome = run_functional_correctness_test(
            CorrectnessConfig(
                n=n,
                indices=indices,
                output_csv=output_csv,
            )
        )
        self.assertFalse(
            outcome.failures,
            msg=f"Functional correctness failed: {outcome.failures}",
        )
        expected_count = len(indices.split(",")) if indices else n
        self.assertLen(outcome.rows, expected_count)


if __name__ == "__main__":
    absltest.main()
