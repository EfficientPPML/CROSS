"""Toy decoder fixtures: local development only, not full-model judging."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from support import cross_root, metrics, read_json
from transformer_runner import (add_arguments, benchmark_session,
                                implementation_preflight, load_encrypted_implementation)
from reference import make_case


def check_output(actual, expected, errors):
    if errors["nrmse"] > 0.10 or errors["max_abs_error"] > 0.20:
        raise ValueError(f"Toy numeric gate failed: {errors}")


def run(args):
    total_started = time.perf_counter()
    case = read_json(args.case)
    weights, embeddings, oracle = make_case(case)
    expected = np.asarray(read_json(args.case.with_suffix(".out"))["logits"])
    if not np.allclose(oracle, expected, atol=1e-10, rtol=1e-10):
        raise ValueError("Reference fixture drift")
    result = {"case": args.case.name, "full_qwen_evaluated": False}
    if args.reference:
        return dict(result, mode="plaintext-reference", passed=True, **metrics(oracle, expected))
    cross_root()
    implementation = load_encrypted_implementation(args.implementation)
    import jax
    jax.config.update("jax_enable_x64", True)
    measured = benchmark_session(
        implementation, config=case["config"], weights=weights,
        embeddings=embeddings, expected=expected, devices=tuple(jax.devices()[:1]),
        evaluate_kwargs={}, check_output=check_output,
        warmups=args.warmups, repeats=args.repeats, require_tpu=args.require_tpu)
    return dict(result, mode="HE-toy-smoke", total_s=time.perf_counter() - total_started, **measured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, Path(__file__).with_name("qwen_he.py"))
    args = parser.parse_args()
    if args.case is None and not args.preflight:
        parser.error("--case is required unless --preflight is used")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.preflight:
                _, result = implementation_preflight(args.implementation, bert=False)
                result["passed"] = result["ready_for_execution"]
            else:
                result = run(args)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"case": args.case.name if args.case else None,
                          "passed": False, "error": str(exc),
                          "error_type": type(exc).__name__,
                          "notes": getattr(exc, "__notes__", [])}))
        return 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
