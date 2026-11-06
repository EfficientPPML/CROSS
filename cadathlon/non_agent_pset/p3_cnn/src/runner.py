"""Local fixture runner. The organizer performs independent ranked reruns.

``setup_s`` measures only the mapping constructor. Any first-request work is
recorded in ``warmup_s`` when warmups are enabled. With zero warmups,
``cold_start_included`` marks that the first measured sample may include lazy
initialization or compilation. ``total_s`` covers this invocation of run(),
including fixture loading, initialization, warmups and release.
When a caller supplies a prepared mapping, that caller owns its release; the
per-case total then covers only this invocation, and the suite measures cleanup.
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


def run(args, *, prepared=None):
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
    devices = tuple(jax.devices()[:1])
    module_name = (
        "baseline_student.py" if getattr(args, "student_baseline", False)
        else "baseline.py" if args.baseline else "cnn_opt.py"
    )
    if prepared is None:
        implementation = load_module(Path(__file__).with_name(module_name))
        started = time.perf_counter()
        mapping = implementation.build_mapping(model, shape, batch=case["batch"], devices=devices)
        setup = time.perf_counter() - started
    else:
        mapping, setup = prepared
    primary_error = None
    try:
        samples, server_samples, outputs = [], [], []
        warmup_samples, warmup_server_samples = [], []
        for iteration in range(args.warmups + args.repeats):
            started = time.perf_counter()
            encrypted = mapping.encrypt_input(x)
            require_ciphertexts(encrypted)
            require_secure_execution(mapping, encrypted)
            synchronize(encrypted)
            server_start = time.perf_counter()
            result = mapping.execute(encrypted)
            synchronize(result)
            server_time = time.perf_counter() - server_start
            require_ciphertexts(result)
            execution_rings = require_secure_execution(mapping, result)
            output = np.asarray(mapping.decrypt_output(result))
            if case["batch"] == 1 and output.shape == expected.shape[1:]:
                output = output[None, :]
            elapsed = time.perf_counter() - started
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
        report = {"case": args.case.name, "mode": "HE-synthetic-smoke", "device": str(devices[0]),
                "passed": True, "implementation": module_name,
                "execution_rings": execution_rings,
                "setup_s": setup, "end_to_end_s": samples, "server_s": server_samples,
                "warmup_s": warmup_samples, "warmup_server_s": warmup_server_samples,
                "mapping_release_included": prepared is None,
                "cold_start_included": args.warmups == 0,
                "median_s": float(np.median(samples)), "batch": case["batch"],
                "trained_accuracy_evaluated": False, **metrics(outputs[-1], expected)}
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if prepared is None:
            try:
                mapping.release()
            except BaseException as error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Mapping cleanup also failed: {error}")
    report["total_s"] = time.perf_counter() - run_started
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--reference", action="store_true")
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--student-baseline", action="store_true",
                      help="Use the smaller secure local-development baseline")
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
        print(json.dumps({"case": args.case.name, "passed": False, "error": str(exc),
                          "notes": list(getattr(exc, "__notes__", ()))}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
