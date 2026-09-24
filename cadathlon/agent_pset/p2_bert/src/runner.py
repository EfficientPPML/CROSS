"""Six synthetic BERT checks, distinct from the upstream MRPC benchmark."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from support import cross_root, metrics, read_json, require_ciphertexts, synchronize
from reference import make_case


def run(args):
    case = read_json(args.case)
    weights, embeddings, mask, oracle = make_case(case)
    expected = np.asarray(read_json(args.case.with_suffix(".out"))["logits"])
    if not np.allclose(oracle, expected, atol=1e-10, rtol=1e-10):
        raise ValueError("Reference fixture drift")
    result = {"case": args.case.name, "full_bert_evaluated": False,
              "mrpc_accuracy_evaluated": False}
    if args.reference:
        return dict(result, mode="plaintext-reference", passed=True, **metrics(oracle, expected))
    cross_root()
    import jax
    import bert_he
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    session = bert_he.create_session(case["config"], weights, batch=len(embeddings),
                                    sequence_length=embeddings.shape[1],
                                    devices=tuple(jax.devices()[:1]))
    setup = time.perf_counter() - started
    try:
        started = time.perf_counter()
        encrypted = session.encrypt(embeddings)
        require_ciphertexts(encrypted)
        synchronize(encrypted)
        server_start = time.perf_counter()
        encrypted_result = session.evaluate(encrypted, attention_mask=mask)
        require_ciphertexts(encrypted_result)
        synchronize(encrypted_result)
        server = time.perf_counter() - server_start
        actual = np.asarray(session.decrypt(encrypted_result), dtype=np.float64)
        elapsed = time.perf_counter() - started
        errors = metrics(actual, expected)
        if not np.all(np.abs(actual - expected) <= 0.02 + 0.01 * np.abs(expected)):
            raise ValueError(f"Synthetic logit gate failed: {errors}")
        if not np.array_equal(actual[:, 1] >= actual[:, 0], expected[:, 1] >= expected[:, 0]):
            raise ValueError("Synthetic binary predictions disagree")
        return dict(result, mode="HE-synthetic-smoke", passed=True, setup_s=setup,
                    server_s=server, end_to_end_s=elapsed, **errors)
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
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
