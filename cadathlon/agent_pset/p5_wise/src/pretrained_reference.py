"""Independent checkpoint accuracy and WISE layout checks; no encrypted claims."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from pretrained_mnist import assets, build_pretrained, native_model
from runner import save
from schedule import execute, pack_inputs, NumpyArithmetic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wise-root", type=Path, default=os.environ.get("WISE_ROOT"))
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--slots", type=int, default=16384)
    p.add_argument("--full-cleartext-test-set", action="store_true")
    p.add_argument("--convolution", nargs="+", choices=("toeplitz", "winograd", "tiled-toeplitz", "hybrid"),
                   default=["tiled-toeplitz", "hybrid"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.wise_root: p.error("set WISE_ROOT or --wise-root")
    if args.output.exists(): p.error("output already exists")
    before = {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(__file__).parent.glob("*.py")}
    started = time.perf_counter()
    report = {"mode": "plaintext-reference", "status": "running", "passed": False,
              "sources_before": before, "plans": []}
    save(args.output, report)
    with redirect_stdout(sys.stderr), torch.no_grad():
        _, provenance, checkpoint, images, labels = assets(args.wise_root)
        model = native_model(args.wise_root, checkpoint)
        report["wise"] = provenance
        if args.full_cleartext_test_set:
            correct = 0
            for start in range(0, len(images), 128):
                x = (images[start:start+128].astype(np.float64)/255-.1307)/.3081
                expected = model(torch.from_numpy(x)).numpy()
                correct += int((expected.argmax(-1) == labels[start:start+128]).sum())
            report["cleartext_test_set"] = {"samples": len(images), "correct": correct,
                                             "accuracy": correct/len(images)}
            save(args.output, report)
        for convolution in args.convolution:
            plan, x, _, model = build_pretrained(args.batch, args.slots, args.wise_root, convolution=convolution)
            record = {"convolution": convolution, "schedule": plan.summary(), "checks": []}
            for name, values in [("original", x), ("perturbed", x+np.random.default_rng(7864).normal(0, .02, x.shape))]:
                expected = model(torch.from_numpy(values)).numpy()
                actual = np.concatenate(execute(plan, pack_inputs(plan, values), NumpyArithmetic()))
                actual = actual[:plan.output_size].reshape(expected.shape)
                error = float(np.max(np.abs(actual-expected)))
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
                record["checks"].append({"input": name, "samples": args.batch, "max_abs_error": error})
            report["plans"].append(record)
            save(args.output, report)
    report["sources_after"] = {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(__file__).parent.glob("*.py")}
    if before != report["sources_after"]: raise RuntimeError("reference sources changed")
    report.update(status="completed", passed=True, elapsed_s=time.perf_counter()-started)
    save(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "passed", "elapsed_s")}))


if __name__ == "__main__":
    main()
