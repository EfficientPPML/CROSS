"""Toy decoder fixtures: local development only, not full-model judging."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from support import cross_root, load_module, metrics, read_json, require_ciphertexts, synchronize
from reference import make_case


def run(args):
    case = read_json(args.case)
    weights, embeddings, oracle = make_case(case)
    expected = np.asarray(read_json(args.case.with_suffix(".out"))["logits"])
    if not np.allclose(oracle, expected, atol=1e-10, rtol=1e-10):
        raise ValueError("Reference fixture drift")
    if args.reference:
        return {"case": args.case.name, "mode": "plaintext-reference",
                "full_qwen_evaluated": False, **metrics(oracle, expected)}
    devices = ()
    if not args.reference:
        cross_root()
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = tuple(jax.devices()[:1])
    impl = load_module(Path(__file__).with_name("qwen_he.py"))
    started = time.perf_counter()
    session = impl.create_session(case["config"], weights, batch=len(embeddings),
                                  sequence_length=embeddings.shape[1], devices=devices)
    setup = time.perf_counter() - started
    try:
        started = time.perf_counter()
        encrypted = session.encrypt(embeddings)
        if not args.reference:
            require_ciphertexts(encrypted)
            synchronize(encrypted)
        server_started = time.perf_counter()
        result = session.evaluate(encrypted)
        if not args.reference:
            require_ciphertexts(result)
            synchronize(result)
        server = time.perf_counter() - server_started
        output = session.decrypt(result)
        elapsed = time.perf_counter() - started
        errors = metrics(output, expected)
        if errors["nrmse"] > 0.10 or errors["max_abs_error"] > 0.20:
            raise ValueError(f"Toy numeric gate failed: {errors}")
        return {"case": args.case.name, "mode": "plaintext-reference" if args.reference else "HE-toy-smoke",
                "full_qwen_evaluated": False, "setup_s": setup, "server_s": server,
                "end_to_end_s": elapsed, **errors}
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--reference", action="store_true")
    args = p.parse_args()
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
