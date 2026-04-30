"""Functional correctness test for LoLAHE vs the plaintext reference."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from lola_he import (
    LoLAHE,
    load_or_generate_data,
    lola_cleartext_inference,
    prepare_weights,
    resolve_indices,
    run_model,
)


_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")

FUNCTIONAL_TEST_CASES = [
    ("image0", 1, "0"),
]


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
        run_model(model, warm_img, weights)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx in indices:
        img = imgs[idx]
        label = int(labels[idx])
        pt_scores = np.asarray(lola_cleartext_inference(img, *weights))
        pt_argmax = int(np.argmax(pt_scores))

        he_run = run_model(model, img, weights)
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
    """Functional correctness tests for LoLAHE."""

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
