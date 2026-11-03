"""Check that each scoring gate rejects the failure it was written for.

Pure arithmetic over synthetic summaries: no accelerator, no suite checkout.
Run it after changing `score.py` or the gate values in `suite_config.json`.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

SUITE_DIR = Path(__file__).resolve().parent


def load_score():
    spec = importlib.util.spec_from_file_location("score", SUITE_DIR / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEALTHY = {
    "runs": 3, "median_total_s": 100.0, "one_time_setup_s": 140.0,
    "median_encrypted_compute_s": 2.0, "median_server_execute_s": 1.0,
    "images": 100, "throughput_images_per_s": 50.0, "ms_per_image": 20.0,
    "encrypted_accuracy": 0.98, "harness_model_accuracy": 0.982,
    "samples_scored": 100,
    "harness_verdict": {"status": "PASS", "complete": True,
                        "top1_agreement_with_harness_model": 0.98,
                        "own_cleartext": {"status": "ok", "compared": 100,
                                          "top1_agreement": 1.0,
                                          "cleartext_correct": 98}},
}

# (name, mutation, should_pass, substring expected in the reason)
CHECKS = [
    ("healthy case scores", lambda s: None, True, ""),
    ("short prediction file", lambda s: s.update(samples_scored=87), False,
     "87 predictions"),
    ("missing frozen verdict", lambda s: s.update(harness_verdict=None), False,
     "no frozen harness verdict"),
    ("harness verdict FAIL",
     lambda s: s["harness_verdict"].update(status="FAIL"), False,
     "harness verdict FAIL"),
    ("verdict unavailable",
     lambda s: s.update(harness_verdict={"status": "unavailable",
                                         "detail": "file missing"}), False,
     "(file missing)"),
    ("own-cleartext agreement below gate",
     lambda s: s["harness_verdict"]["own_cleartext"].update(
         top1_agreement=0.95), False, "own cleartext model"),
    # A different network, so this must never decide a case on its own.
    ("harness-model disagreement is ignored",
     lambda s: s["harness_verdict"].update(
         top1_agreement_with_harness_model=0.80), True, ""),
    ("cleartext check unavailable",
     lambda s: s["harness_verdict"].update(
         own_cleartext={"status": "unavailable", "detail": "weights missing"}),
     False, "no cleartext agreement check"),
    # (0.97 - 0.96) * 100 is 1.0000000000000009; exactly the gate must pass.
    ("drop of exactly 1pp against own cleartext passes",
     lambda s: (s.update(encrypted_accuracy=0.96),
                s["harness_verdict"]["own_cleartext"].update(cleartext_correct=97)),
     True, ""),
    ("accuracy drop beyond 1pp vs own cleartext",
     lambda s: s.update(encrypted_accuracy=0.96), False,
     "below the same model in cleartext"),
    # The measured C3 case: our model scores 96/100 both ways, while the
    # harness's larger network scores 97. That is a model difference and must
    # not cost the submission anything.
    ("harness model 1pp better is not a drop",
     lambda s: (s.update(encrypted_accuracy=0.96, harness_model_accuracy=0.97),
                s["harness_verdict"]["own_cleartext"].update(cleartext_correct=96)),
     True, ""),
    ("setup over budget", lambda s: s.update(one_time_setup_s=2000.0), False,
     "setup"),
    ("batch over budget",
     lambda s: s.update(median_encrypted_compute_s=400.0), False,
     "batch exceeded"),
    # The single instance has no Quality block at all; it must still score on
    # the strength of the frozen verdict alone.
    ("single instance, no Quality",
     lambda s: s.update(samples_scored=None, encrypted_accuracy=None,
                        harness_model_accuracy=None, images=1) or
         s["harness_verdict"]["own_cleartext"].update(compared=1, cleartext_correct=1), True, ""),
    ("NaN timing", lambda s: s.update(median_encrypted_compute_s=float("nan")), False, "invalid"),
    ("zero timing", lambda s: s.update(median_encrypted_compute_s=0), False, "invalid"),
    ("negative setup", lambda s: s.update(one_time_setup_s=-1), False, "invalid"),
    ("infinite throughput", lambda s: s.update(throughput_images_per_s=float("inf")), False, "invalid"),
    ("NaN accuracy", lambda s: s.update(encrypted_accuracy=float("nan")), False, "invalid"),
    ("NaN agreement", lambda s: s["harness_verdict"]["own_cleartext"].update(top1_agreement=float("nan")), False, "invalid"),
    ("partial agreement", lambda s: s["harness_verdict"]["own_cleartext"].update(compared=1), False, "incomplete"),
    ("missing batch Quality", lambda s: s.update(samples_scored=None, encrypted_accuracy=None), False, "missing Quality"),
    ("single still requires own reference", lambda s: s.update(images=1, samples_scored=None, encrypted_accuracy=None, harness_verdict={"status": "PASS"}), False, "no cleartext"),
]


def main() -> int:
    score = load_score()
    gates = score.CONFIG["gates"]
    failures = 0
    for name, mutate, should_pass, substring in CHECKS:
        summary = copy.deepcopy(HEALTHY)
        mutate(summary)
        passed, reasons = score.gate(summary, gates)
        ok = passed == (1 if should_pass else 0)
        if ok and substring:
            ok = any(substring in reason for reason in reasons)
        failures += not ok
        print(f"{'ok  ' if ok else 'BAD '} {name:34s} passed={passed} {reasons}")
    total = len(CHECKS)
    print(f"\n{total - failures}/{total} gate behaviors correct")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
