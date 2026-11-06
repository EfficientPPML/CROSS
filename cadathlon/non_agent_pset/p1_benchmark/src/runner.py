"""Local fixture runner. The organizer performs independent ranked reruns.

``setup_s`` measures only the service constructor. Lazy key generation and
compilation are charged to the first request, recorded in ``warmup_s`` when
warmups are enabled. With zero warmups, ``cold_start_included`` marks that the
first measured sample may include that work. ``total_s`` covers this invocation
of run(), including fixture loading, initialization, warmups and release.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from support import cross_root, load_module, metrics, read_json, require_ciphertexts, require_secure_execution, synchronize
from reference import make_case


def _bundle_ciphertexts(bundle):
    """Accept whatever shape the service uses, and require it be ciphertexts."""
    if isinstance(bundle, (tuple, list)):
        for item in bundle:
            _bundle_ciphertexts(item)
        return
    if isinstance(bundle, dict):
        for item in bundle.values():
            _bundle_ciphertexts(item)
        return
    if isinstance(bundle, (int, np.integer)):
        return  # lane counts and similar bookkeeping
    require_ciphertexts(bundle)


def run(args):
    run_started = time.perf_counter()
    case = read_json(args.case)
    model, shape, x, oracle = make_case(case)
    expected = np.asarray(read_json(args.case.with_suffix(".out"))["logits"])
    if not np.allclose(oracle, expected, atol=1e-10, rtol=1e-10):
        raise ValueError("Reference fixture drift")
    if args.reference:
        return {"case": args.case.name, "mode": "plaintext-reference", **metrics(oracle, expected)}
    cross_root()
    import jax
    jax.config.update("jax_enable_x64", True)
    available = jax.devices()
    want = case.get("devices", 1)
    if isinstance(want, bool) or not isinstance(want, int) or want < 1:
        raise ValueError("case devices must be a positive integer")
    if len(available) < want:
        raise RuntimeError(
            f"Case requires {want} devices, but JAX reports {len(available)}. "
            "Use a host with the required devices; cases are never silently downscaled.")
    devices = tuple(available[:want])
    student_baseline = getattr(args, "student_baseline", False)
    implementation = load_module(Path(__file__).with_name(
        "baseline_student.py" if student_baseline else
        "baseline.py" if args.baseline else "bench_opt.py"))
    started = time.perf_counter()
    service = implementation.build_service(model, shape, batch=case["batch"], devices=devices)
    setup = time.perf_counter() - started
    try:
        samples, server_samples, outputs = [], [], []
        warmup_samples, warmup_server_samples = [], []
        for iteration in range(args.warmups + args.repeats):
            started = time.perf_counter()
            bundle = service.encrypt(x)
            _bundle_ciphertexts(bundle)
            require_secure_execution(service, bundle)
            synchronize(bundle)
            server_start = time.perf_counter()
            result = service.execute(bundle)
            synchronize(result)
            server_time = time.perf_counter() - server_start
            _bundle_ciphertexts(result)
            actual_rings = require_secure_execution(service, result)
            output = np.asarray(service.decrypt(result), dtype=np.float64)
            elapsed = time.perf_counter() - started
            if output.shape != expected.shape:
                raise ValueError(f"decrypt returned {output.shape}, expected {expected.shape}")
            if iteration >= args.warmups:
                samples.append(elapsed)
                server_samples.append(server_time)
                outputs.append(output)
            else:
                warmup_samples.append(elapsed)
                warmup_server_samples.append(server_time)
            errors = metrics(output, expected)
            if not np.all(np.abs(output - expected) <= 0.02 + 0.01 * np.abs(expected)):
                raise ValueError(f"Logit gate failed: {errors}")
            if errors["top1_agreement"] < 0.99:
                raise ValueError(f"Argmax gate failed: {errors}")
        median = float(np.median(samples))
        report = {"case": args.case.name, "mode": "HE-synthetic-smoke",
                "devices": [str(d) for d in devices], "batch": case["batch"],
                "profile": "student" if student_baseline else "legacy-or-submission",
                "execution_rings": actual_rings,
                "setup_s": setup, "end_to_end_s": samples, "server_s": server_samples,
                "warmup_s": warmup_samples, "warmup_server_s": warmup_server_samples,
                "cold_start_included": args.warmups == 0,
                "median_s": median,
                "images_per_s": case["batch"] / median if median > 0 else None,
                "ms_per_image": 1000.0 * median / case["batch"],
                "trained_accuracy_evaluated": False, **metrics(outputs[-1], expected)}
    finally:
        service.release()
    report["total_s"] = time.perf_counter() - run_started
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--reference", action="store_true")
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--student-baseline", action="store_true",
                      help="opt-in secure N=8192 baseline; separate from legacy suite scoring")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        p.error("warmups >= 0 and repeats >= 1 required")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run(args)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"case": args.case.name, "passed": False, "error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
