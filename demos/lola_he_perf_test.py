"""Performance test for LoLAHE — records B=1 and B=32 per-image latency.

The two officially-supported modes, both built here from the torch model:

  * **B=1** (interactive single-image): 540.8 ms warm median on TPUv6e-8.
    Used for end-to-end correctness gating.
  * **B=32** (recommended throughput): 13.027 s per global batch, or
    407.1 ms/image, automatically split over eight TPUv6e devices.

Output: `demos/log/lola_he_perf.csv` with one row per image, columns:
  batch, image_index, label, pt_argmax, he_argmax, max_err, wall_s_per_image

No cache is involved: each mode materializes its own Mapping, which is
minutes of work and gigabytes of memory, so the whole benchmark is opt-in
behind `LOLA_PERF=1`. The LoLA section of ``demos/README.md`` has the
complete workflow.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any

import jax
import numpy as np
from absl.testing import absltest

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_JAXITE = os.path.abspath(os.path.join(_DEMO_DIR, "..", "jaxite_word"))
for p in (_JAXITE, _DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

jax.config.update("jax_enable_x64", True)

from lola_he import (LoLAHE, FC2_OUT, load_or_generate_data,
                     lola_cleartext_inference, prepare_weights)


OUTPUT_CSV = os.path.join(_DEMO_DIR, "log", "lola_he_perf.csv")
N_IMAGES_B1 = 5     # quick interactive cycle
N_IMAGES_B32 = 32   # one batch (32 images) for the throughput row
TOL = 5e-2


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bench_b1(model, data, weights, indices) -> list[dict[str, Any]]:
    """Run one inference per index, record per-image wall."""
    # Warm
    _ = model.infer(data["imgs"][indices[0]])
    _ = model.infer(data["imgs"][indices[0]])

    rows: list[dict[str, Any]] = []
    pt_match = 0
    for idx in indices:
        img = data["imgs"][idx]
        pt = np.asarray(lola_cleartext_inference(img, *weights))
        pt_lbl = int(np.argmax(pt))
        t0 = time.perf_counter()
        he = model.infer(img)
        wall_s = time.perf_counter() - t0
        he_lbl = int(np.argmax(he))
        err = float(np.max(np.abs(he[:FC2_OUT] - pt[:FC2_OUT])))
        rows.append({
            "batch": 1,
            "image_index": int(idx),
            "label": int(data["labels"][idx]),
            "pt_argmax": pt_lbl,
            "he_argmax": he_lbl,
            "max_err": err,
            "wall_s_per_image": float(wall_s),
        })
        if pt_lbl == he_lbl:
            pt_match += 1
        print(f"  [B=1 idx={idx:>3d}] true={int(data['labels'][idx])} "
              f"pt={pt_lbl} he={he_lbl} max_err={err:.2e} "
              f"wall={wall_s*1000.0:.1f}ms")
    print(f"  [B=1] pt==he {pt_match}/{len(indices)}")
    return rows


def _bench_b32(model, data, weights, indices) -> list[dict[str, Any]]:
    """Run one batched inference of B=32 images, record per-image
    amortized wall."""
    assert len(indices) == 32, (
        f"B=32 perf row expects exactly 32 indices, got {len(indices)}")
    imgs = [data["imgs"][i] for i in indices]
    pt_logits = [np.asarray(lola_cleartext_inference(im, *weights))
                 for im in imgs]
    pt_labels = [int(np.argmax(p)) for p in pt_logits]

    # Warm
    _ = model.infer(imgs)
    _ = model.infer(imgs)

    t0 = time.perf_counter()
    he_logits_list = model.infer(imgs)
    wall_s = time.perf_counter() - t0
    per_image_s = wall_s / 32.0

    rows: list[dict[str, Any]] = []
    pt_match = 0
    for k, idx in enumerate(indices):
        he_lbl = int(np.argmax(he_logits_list[k]))
        err = float(np.max(np.abs(
            he_logits_list[k] - pt_logits[k][:FC2_OUT])))
        rows.append({
            "batch": 32,
            "image_index": int(idx),
            "label": int(data["labels"][idx]),
            "pt_argmax": pt_labels[k],
            "he_argmax": he_lbl,
            "max_err": err,
            "wall_s_per_image": per_image_s,
        })
        if pt_labels[k] == he_lbl:
            pt_match += 1
    print(f"  [B=32] one-batch wall {wall_s*1000.0:.1f}ms "
          f"(per-image {per_image_s*1000.0:.1f}ms), "
          f"pt==he {pt_match}/32")
    return rows



def _build_model(batch: int):
    """Bind the shipped weights and materialize one Mapping.

    Replaces the pickled cache: the plan is re-derived from the torch module
    `lola_train.QuadLoLA`, which is cheap, and key generation is well under a
    second. What the cache saved was constant materialization, which follows
    from the weights and so could not be reused across a change anyway.
    """
    data, _ = load_or_generate_data()
    model = LoLAHE(batch=batch)
    model.precompute_plaintexts(*prepare_weights(data))
    return model


class LoLAHEPerformanceTest(absltest.TestCase):
    """Records per-image latency at B=1 (interactive) and B=32 (throughput).

    Both modes build their own Mapping, so the whole test skips (not fails)
    unless `LOLA_PERF=1` — that lets the suite run cleanly on a fresh
    checkout without spending minutes and gigabytes."""

    def test_b1_and_b32_perf(self):
        data, data_source = load_or_generate_data()
        weights = prepare_weights(data)
        n_imgs = len(data["imgs"])
        print("=" * 70)
        print(f"CROSS HE LoLA performance test (B=1 + B=32)")
        print(f"  data_source={data_source}, total images={n_imgs}")
        print("=" * 70)

        all_rows: list[dict[str, Any]] = []

        # Each build materializes constants and per-level controls, which is
        # minutes and gigabytes. Opt in rather than run it by default.
        if not os.environ.get("LOLA_PERF", ""):
            self.skipTest("LOLA_PERF not set; skipping LoLA HE benchmark.")

        # --------- B=1 -----------
        print("\n[B=1] building model ...")
        model_b1 = _build_model(batch=1)
        self.assertEqual(model_b1.batch, 1,
                         f"expected batch=1, got {model_b1.batch}")
        indices = list(range(min(N_IMAGES_B1, n_imgs)))
        rows = _bench_b1(model_b1, data, weights, indices)
        for r in rows:
            self.assertEqual(
                r["pt_argmax"], r["he_argmax"],
                f"B=1 idx={r['image_index']}: he disagrees with pt")
            self.assertLessEqual(
                r["max_err"], TOL,
                f"B=1 idx={r['image_index']}: err {r['max_err']} > tol")
        all_rows.extend(rows)
        model_b1.release()   # free its HBM before the B=32 build
        del model_b1

        # --------- B=32 -----------
        print("\n[B=32] building model ...")
        model_b32 = _build_model(batch=32)
        self.assertEqual(model_b32.batch, 32,
                         f"expected batch=32, got {model_b32.batch}")
        indices = list(range(min(N_IMAGES_B32, n_imgs)))
        assert len(indices) == 32, (
            f"need at least 32 images for B=32 test; have {len(indices)}")
        rows = _bench_b32(model_b32, data, weights, indices)
        for r in rows:
            self.assertEqual(
                r["pt_argmax"], r["he_argmax"],
                f"B=32 idx={r['image_index']}: he disagrees with pt")
            self.assertLessEqual(
                r["max_err"], TOL,
                f"B=32 idx={r['image_index']}: err {r['max_err']} > tol")
        all_rows.extend(rows)
        model_b32.release()
        del model_b32

        if not all_rows:
            self.skipTest("no benchmark rows produced.")

        _write_csv(OUTPUT_CSV, all_rows)
        print(f"\n[ok] wrote {OUTPUT_CSV} ({len(all_rows)} rows)")

        # Headline summary
        print("\n" + "=" * 70)
        print("Performance summary (per-image wall in ms)")
        print("=" * 70)
        for B in (1, 32):
            sub = [r for r in all_rows if r["batch"] == B]
            if not sub:
                continue
            walls_ms = np.array([r["wall_s_per_image"] for r in sub]) * 1000.0
            errs = np.array([r["max_err"] for r in sub])
            print(f"  B={B:>2d}  n={len(sub):>3d}  "
                  f"min={walls_ms.min():>6.1f}  "
                  f"avg={walls_ms.mean():>6.1f}  "
                  f"max={walls_ms.max():>6.1f}  ms/image  "
                  f"max_err={errs.max():.3e}")
        print("=" * 70)


if __name__ == "__main__":
    absltest.main()
