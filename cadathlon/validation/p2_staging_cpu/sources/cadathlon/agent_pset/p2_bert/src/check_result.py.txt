"""Strict post-run output/accuracy check, supplementing the untouched harness.

Passing is necessary but does not certify encryption, timing or security.
Organizers must own and independently run this check and audit the backend.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import BATCH_SIZES, INSTANCE_NAMES
from stages import ordered_logits, read_jsonl, stage_paths


def labels(path, count):
    lines = path.read_text().splitlines()
    if len(lines) != count or any(line.strip() not in ("0", "1") for line in lines):
        raise ValueError(f"{path}: expected exactly {count} binary labels")
    return np.asarray([int(line) for line in lines])


def check(root, size):
    paths = stage_paths(root, size)
    count = BATCH_SIZES[size]
    inputs = read_jsonl(paths["test_input"])
    if len(inputs) != count:
        raise ValueError(f"Expected {count} input records")
    logits = ordered_logits(inputs, read_jsonl(paths["decrypted_results"]))
    actual = labels(paths["predictions"], count)
    if not np.array_equal(actual, logits[:, 1] >= logits[:, 0]):
        raise ValueError("Predictions disagree with decrypted logits in input order")
    truth = labels(paths["test_input"].with_name("test_labels.txt"), count)
    clear = labels(paths["predictions"].with_name("harness_model_predictions.txt"), count)
    accuracy = float(np.mean(actual == truth))
    clear_accuracy = float(np.mean(clear == truth))
    passed = (accuracy == 1.0 if size == 0
              else accuracy >= 0.75 and accuracy + 0.02 + 1e-12 >= clear_accuracy)
    return {"instance": INSTANCE_NAMES[size], "samples": count, "passed": passed,
            "accuracy": accuracy, "cleartext_accuracy": clear_accuracy,
            "prediction_agreement": float(np.mean(actual == clear)),
            "accuracy_drop_pp": 100.0 * (clear_accuracy - accuracy),
            "encryption_audited": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    try:
        result = check(args.benchmark_root.resolve(), args.size)
    except Exception as exc:
        result = {"passed": False, "error": str(exc)}
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
