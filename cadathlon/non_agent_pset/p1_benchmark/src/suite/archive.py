"""Archive one ranked case's measurement files under src/suite/results/<case>.

The suite writes to `measurements/<instance>/results-N.json`, keyed by instance
size only. Two ranked cases can share an instance and differ in chip count --
C1 and C2 are both the single instance -- so a second run would overwrite the
first. This copies each case's files aside under its own case id, which is what
score.py reads.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from p1_evidence import case_identity
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((SUITE_DIR / "suite_config.json").read_text())
INSTANCE_NAMES = ("single", "small", "medium", "large")
RESULTS = SUITE_DIR / "results"


def case_by_id(case_id: str) -> dict:
    for case in CONFIG["cases"]:
        if case["id"] == case_id:
            return case
    raise SystemExit(f"unknown case {case_id!r}; "
                     f"known: {[c['id'] for c in CONFIG['cases']]}")


def case_by_shape(instance: int, devices: int) -> dict:
    """Resolve the ranked case from the instance and chip count.

    The Makefile knows SIZE and P3_DEVICES, not the case id, so let it hand
    those over rather than making the caller keep the mapping in their head.
    """
    for case in CONFIG["cases"]:
        if case["instance"] == instance and case["devices"] == devices:
            return case
    known = ", ".join(f"{c['id']}=(size {c['instance']}, {c['devices']} chip"
                      f"{'s' if c['devices'] > 1 else ''})" for c in CONFIG["cases"])
    raise SystemExit(f"no ranked case for size {instance} on {devices} chip(s); "
                     f"known: {known}")


def _labels(path: Path):
    if not path.is_file():
        return None
    return [line.strip() for line in path.read_text().strip().split("\n") if line.strip()]


def own_cleartext_predictions(root: Path, case: dict):
    """Run *our* model on the harness's pixels, in cleartext, organizer-side.

    The numerics gate is "does the encrypted run agree with the same model in
    cleartext". The suite cannot answer it: `harness/cleartext_impl.py` runs
    `harness/mnist/mnist_ffnn_model.pth`, a 784-128-64-10 network, while this
    problem's fixed model is 784-50-10 with an x**2 activation. Agreement
    between those two measures how differently two networks were trained, not
    HE fidelity. So compute the real reference here, in organizer-owned code
    that the contestant's `bench_opt.py` never touches.
    """
    name = INSTANCE_NAMES[case["instance"]]
    pixels_file = root / "datasets" / name / "intermediate" / "test_pixels.txt"
    if not pixels_file.is_file():
        return None, "harness test_pixels.txt missing"
    try:
        # This runs right after a case, while the resident service may still
        # hold the accelerator; one TPU admits one process. The cleartext
        # reference is pure PyTorch, so keep any transitive JAX import on CPU.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        sys.path.insert(0, str(SUITE_DIR))
        import stage_lib as lib
        import numpy as np
        import torch

        pixels = lib.read_pixels(pixels_file)
        normalized = (pixels - lib.MNIST_MEAN) / lib.MNIST_STD
        model = lib.load_model()
        with torch.no_grad():
            logits = model(torch.from_numpy(np.asarray(normalized, dtype=np.float64)))
        return [str(int(i)) for i in logits.argmax(dim=1).tolist()], None
    except Exception as error:                      # noqa: BLE001 - reported, not raised
        return None, f"{type(error).__name__}: {error}"


def freeze_verdict(root: Path, case: dict) -> dict:
    """Recompute correctness from the harness's own files before they are lost.

    Two gaps in `results-N.json` make this necessary. The single instance has
    no `Quality` block at all -- the harness calls `verify_result` with
    `check=False`, so a FAIL neither stops the run nor is recorded. And
    `calculate_quality` divides by `len(preds)`, so a submission that emits
    fewer predictions than there are images still reports a high accuracy.
    Both the labels and the predictions are written by the harness, so
    reading them here keeps correctness out of the submission's hands.

    This covers the last of the `num_runs` runs: the harness overwrites these
    files each run and only the final one survives to be archived.
    """
    name = INSTANCE_NAMES[case["instance"]]
    truth = _labels(root / "datasets" / name / "intermediate" / "test_labels.txt")
    encrypted = _labels(root / "io" / name / "encrypted_model_predictions.txt")
    harness = _labels(root / "io" / name / "harness_model_predictions.txt")
    if truth is None or encrypted is None:
        return {"scope": "final run", "status": "unavailable",
                "detail": "harness label or prediction file missing"}

    expected = case["batch"]
    verdict = {
        "scope": "final run",
        "expected_images": expected,
        "predictions_written": len(encrypted),
        "labels_written": len(truth),
        "complete": len(encrypted) == expected and len(truth) == expected,
        "correct": sum(1 for a, b in zip(truth, encrypted) if a == b),
    }
    verdict["accuracy_over_expected"] = verdict["correct"] / expected if expected else 0.0

    # Informational only: a different network, so disagreement here is a model
    # difference. Kept because the harness's own accuracy gate uses it.
    if harness is not None and len(harness) == len(encrypted):
        agree = sum(1 for a, b in zip(harness, encrypted) if a == b)
        verdict["top1_agreement_with_harness_model"] = agree / len(encrypted)
        verdict["harness_model_correct"] = sum(
            1 for a, b in zip(truth, harness) if a == b)

    # The real numerics gate: the same model, encrypted against cleartext.
    own, error = own_cleartext_predictions(root, case)
    if own is None:
        verdict["own_cleartext"] = {"status": "unavailable", "detail": error}
    elif len(own) != expected:
        verdict["own_cleartext"] = {"status": "unavailable",
                                    "detail": f"expected {expected} cleartext predictions, got {len(own)}"}
    else:
        n = min(len(own), len(encrypted))
        agree = sum(1 for a, b in zip(own[:n], encrypted[:n]) if a == b)
        verdict["own_cleartext"] = {
            "status": "ok",
            "compared": n,
            "top1_agreement": agree / n if n else 0.0,
            "cleartext_correct": sum(1 for a, b in zip(truth, own) if a == b),
        }

    verdict["status"] = "PASS" if verdict["complete"] and (
        expected > 1 or verdict["correct"] == 1) else "FAIL"
    return verdict


def copy_evidence(root: Path, case: dict, target: Path) -> list:
    """Keep the harness's label and prediction files; the next run overwrites them."""
    name = INSTANCE_NAMES[case["instance"]]
    sources = {
        "test_labels.txt": root / "datasets" / name / "intermediate" / "test_labels.txt",
        "test_pixels.txt": root / "datasets" / name / "intermediate" / "test_pixels.txt",
        "encrypted_model_predictions.txt": root / "io" / name / "encrypted_model_predictions.txt",
        "harness_model_predictions.txt": root / "io" / name / "harness_model_predictions.txt",
    }
    evidence = target / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    kept = []
    for filename, source in sources.items():
        if not source.is_file():
            continue
        # Labels and predictions are the evidence and are tiny. The pixels are
        # only needed to recompute the cleartext reference offline, and they are
        # regenerable from the published seed
        # (`harness/generate_input.py <size> --dataset mnist --seed <derived>`),
        # so keep them only while they stay small.
        if filename == "test_pixels.txt" and source.stat().st_size > 1024 * 1024:
            continue
        shutil.copy2(source, evidence / filename)
        kept.append(filename)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="ranked case id, e.g. C4")
    group.add_argument("--instance", type=int, help="use with --devices")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--suite-root", type=Path, default=None)
    ap.add_argument("--label", default="", help="baseline or candidate")
    args = ap.parse_args()

    case = case_by_id(args.case) if args.case \
        else case_by_shape(args.instance, args.devices)
    root = args.suite_root or Path(os.environ.get("MLINF_ROOT", Path.home() / "ml-inference"))
    source = root / "measurements" / INSTANCE_NAMES[case["instance"]]
    files = sorted(source.glob("results-*.json"))
    if not files:
        raise SystemExit(f"no results-*.json in {source}; run the harness first")

    target = RESULTS / (f"{case['id']}_{args.label}" if args.label else case["id"])
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("results-*.json"):
        stale.unlink()
    for path in files:
        shutil.copy2(path, target / path.name)
    verdict = freeze_verdict(root, case)
    kept = copy_evidence(root, case, target)
    execution = json.loads((root / 'io' / INSTANCE_NAMES[case['instance']] /
                            'public_keys' / 'ring_manifest.json').read_text())
    execution = {name: execution[name] for name in ('devices', 'device_kinds', 'execution_rings', 'batch')}
    (target / "case.json").write_text(json.dumps(
        {**case, "verify": verdict, "evidence_files": kept,
         "suite_commit": CONFIG["suite"]["commit"],
         "seed": CONFIG["determinism"]["seed"],
         "baseline_profile": os.environ.get("P1_BASELINE_PROFILE", "legacy"),
         "evaluation_id": os.environ.get("P1_EVALUATION_ID"),
         "identity": case_identity(SUITE_DIR.parent.parent, root, INSTANCE_NAMES[case['instance']]),
         "execution": execution,
         "devices_env": os.environ.get("P3_DEVICES"),
         "archived_from": str(source),
         "label": args.label or None}, indent=2) + "\n")
    print(f"archived {len(files)} file(s) -> {target}")
    print(f"  harness verdict ({verdict['scope']}): {verdict['status']}"
          + (f" | {verdict['correct']}/{verdict['expected_images']} correct"
             if "correct" in verdict else f" | {verdict.get('detail','')}"))


if __name__ == "__main__":
    main()
