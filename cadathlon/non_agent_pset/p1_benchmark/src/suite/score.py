"""Deterministic scoring from the suite's own measurement files.

Reads `measurements/<instance>/results-N.json`, which the unmodified harness
writes, applies the handout's rubric and prints one JSON verdict. Nothing here
trusts a contestant-reported number: correctness comes from the harness's
`Quality` block (computed from the frozen ground-truth labels) and performance
comes from the harness's `Timing` block plus the `Server Reported` steps the
submission publishes through the harness's documented hook.

This is the organizer's arithmetic, not an official score by itself: the judge
still reruns on exclusive hardware with its own baseline manifest.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

SUITE_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((SUITE_DIR / "suite_config.json").read_text())
INSTANCE_NAMES = ("single", "small", "medium", "large")


def _seconds(text) -> float:
    """Harness timings are strings like '12.1014s', '1.8286h', '10.55m'."""
    if isinstance(text, (int, float)):
        return float(text)
    text = str(text).strip()
    for suffix, factor in (("ms", 1e-3), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)]) * factor
            except ValueError:
                break
    return float(text)


def read_case(root: Path, instance: int, runs: int, case_id: str = "",
              label: str = ""):
    """Prefer the per-case archive; fall back to the live measurements dir.

    Two ranked cases can share an instance and differ only in chip count, so
    the live `measurements/<instance>` directory holds whichever ran last.
    archive.py copies each case aside under its own id.
    """
    folder = archived = None
    if case_id:
        for candidate in ((SUITE_DIR / "results" / f"{case_id}_{label}") if label else None,
                          SUITE_DIR / "results" / case_id):
            if candidate is not None and candidate.is_dir() and \
                    any(candidate.glob("results-*.json")):
                folder = candidate
                meta = folder / "case.json"
                if meta.is_file():
                    archived = json.loads(meta.read_text())
                break
    if folder is None:
        folder = root / "measurements" / INSTANCE_NAMES[instance]
    records = []
    for n in range(1, runs + 1):
        path = folder / f"results-{n}.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run the harness with --num_runs {runs}")
        records.append(json.loads(path.read_text()))
    return records, archived, folder


def summarize(records, batch: int, archived=None):
    totals = [_seconds(r["Timing"]["Total"]) for r in records]
    # One-off cost the client and server pay before any image is served: key
    # generation plus the encrypted-model preprocessing that compiles and
    # warms the circuit. Amortized across a batch, not charged per image.
    setups = [_seconds(r["Timing"]["Key Generation"])
              + _seconds(r["Timing"]["Encrypted model preprocessing"])
              for r in records]
    computes = [_seconds(r["Timing"]["Encrypted computation"]) for r in records]
    server = [_seconds(r["Server Reported"]["server_execute_s"])
              for r in records if "server_execute_s" in r.get("Server Reported", {})]
    quality = [r.get("Quality") for r in records if r.get("Quality")]
    accuracy = harness_accuracy = None
    if quality:
        accuracy = statistics.median(
            q["Encrypted model quality"]["accuracy"] for q in quality)
        harness_accuracy = statistics.median(
            q["Harness model quality"]["accuracy"] for q in quality)
    median_compute = statistics.median(computes)
    out = {
        "runs": len(records),
        "runs_note": "same images every run; spread is machine noise only",
        # Stages 1/2.2/3 run once outside the harness's run loop, so these two
        # are one-time costs that every run's JSON repeats, not per-run figures.
        "total_s_including_one_time_setup": statistics.median(totals),
        "one_time_setup_s": statistics.median(setups),
        "median_encrypted_compute_s": median_compute,
        "median_server_execute_s": statistics.median(server) if server else None,
        "images": batch,
        "throughput_images_per_s": batch / median_compute if median_compute > 0 else None,
        "ms_per_image": 1000.0 * median_compute / batch,
        "encrypted_accuracy": accuracy,
        "harness_model_accuracy": harness_accuracy,   # different network; context only
        "samples_scored": statistics.median(
            q["Encrypted model quality"]["total_samples"] for q in quality) if quality else None,
        "harness_verdict": (archived or {}).get("verify"),
        "own_cleartext_agreement": ((archived or {}).get("verify") or {})
            .get("own_cleartext", {}).get("top1_agreement"),
        # Informational: a different network, never gated on.
        "harness_reference_model_accuracy_for_context": harness_accuracy,
    }
    return out


def gate(summary, gates) -> tuple[int, list]:
    reasons = []
    if summary["median_encrypted_compute_s"] > gates["max_batch_s"]:
        reasons.append(f"batch exceeded {gates['max_batch_s']}s")
    if summary["one_time_setup_s"] > gates["max_setup_s"]:
        reasons.append(f"setup {summary['one_time_setup_s']:.1f}s exceeded "
                       f"{gates['max_setup_s']}s")
    # `calculate_quality` divides by the number of predictions written, so a
    # short output file reports a healthy accuracy over the images it skipped.
    samples = summary.get("samples_scored")
    if samples is not None and samples != summary["images"]:
        reasons.append(f"submission wrote {samples:g} predictions for "
                       f"{summary['images']} images")

    # The handout's gate is a drop "relative to that same model's cleartext
    # predictions on the same images" -- encryption must not cost accuracy.
    # Comparing against `Harness model quality` instead would measure this
    # problem's 784-50-10 network against the harness's 784-128-64-10 one and
    # charge the difference in training to the HE implementation.
    acc = summary["encrypted_accuracy"]
    own = (summary.get("harness_verdict") or {}).get("own_cleartext") or {}
    images = summary["images"]
    if acc is not None and own.get("status") == "ok" and images:
        cleartext_acc = own["cleartext_correct"] / images
        # (0.97 - 0.96) * 100 is 1.0000000000000009 in binary floating point,
        # so a drop of exactly the gate value must not read as exceeding it.
        drop_pp = (cleartext_acc - acc) * 100.0
        if drop_pp - gates["max_accuracy_drop_pp"] > 1e-9:
            reasons.append(f"encrypted accuracy {acc:.4f} is more than "
                           f"{gates['max_accuracy_drop_pp']}pp below the same "
                           f"model in cleartext {cleartext_acc:.4f}")

    # The single instance has no Quality block: the harness runs verify_result
    # with check=False, so neither a FAIL nor the comparison reaches
    # results-N.json. archive.py recomputes it from the harness's own files.
    verdict = summary.get("harness_verdict")
    if verdict is None:
        reasons.append("no frozen harness verdict; re-run through "
                       "`make suite-baseline`/`suite-test` so archive.py records one")
    else:
        if verdict.get("status") != "PASS":
            detail = verdict.get("detail", "")
            reasons.append(f"harness verdict {verdict.get('status')}"
                           + (f" ({detail})" if detail else ""))
        # The numerics gate compares the encrypted run against the SAME model
        # in cleartext. `top1_agreement_with_harness_model` is a different
        # network (784-128-64-10 vs this problem's 784-50-10) and is recorded
        # for information only -- gating on it would penalize a correct
        # implementation for how the harness's own model was trained.
        own = verdict.get("own_cleartext") or {}
        if own.get("status") == "ok":
            agreement = own["top1_agreement"]
            if agreement - gates["min_top1_agreement"] < -1e-9:
                reasons.append(
                    f"top-1 agreement with its own cleartext model "
                    f"{agreement:.4f} below {gates['min_top1_agreement']} "
                    f"over {own['compared']} images")
        elif summary["images"] > 1:
            reasons.append("no cleartext agreement check: "
                           f"{own.get('detail', 'not computed')}")
    return (0 if reasons else 1), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-root", type=Path, default=None)
    ap.add_argument("--baseline", type=Path,
                    default=SUITE_DIR / "baseline_manifest.json")
    ap.add_argument("--cases", default="", help="comma-separated case ids, default all")
    ap.add_argument("--label", default="",
                    help="archive label to read, e.g. baseline or candidate")
    ap.add_argument("--emit-baseline", action="store_true",
                    help="write the measured numbers as a baseline manifest instead of scoring")
    args = ap.parse_args()

    import os
    root = args.suite_root or Path(os.environ.get("MLINF_ROOT", Path.home() / "ml-inference"))
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    cases = [c for c in CONFIG["cases"] if not wanted or c["id"] in wanted]
    runs = CONFIG["determinism"]["num_runs"]
    gates = CONFIG["gates"]

    measured = {}
    for case in cases:
        try:
            records, archived, folder = read_case(root, case["instance"], runs,
                                                  case["id"], args.label)
        except FileNotFoundError as error:
            measured[case["id"]] = {"error": str(error)}
            continue
        summary = summarize(records, case["batch"], archived)
        summary["read_from"] = str(folder)
        # The upstream checkout ships the reference implementation's own
        # measurements under measurements/<instance>/. Reading those instead of
        # an archive would score somebody else's run, so say so loudly.
        if archived is None:
            summary["warning"] = ("no per-case archive; these numbers came from "
                                  "the live measurements directory, which the "
                                  "upstream checkout also populates")
        measured[case["id"]] = summary

    if args.emit_baseline:
        # Keep only what a ratio is computed from, plus enough context to tell
        # whether a case is comparable. Absolute paths and the nested verdict
        # describe one machine's run and have no meaning in a shipped manifest.
        keep = ("runs", "one_time_setup_s", "median_encrypted_compute_s",
                "median_server_execute_s", "images", "throughput_images_per_s",
                "ms_per_image", "encrypted_accuracy", "own_cleartext_agreement")
        cases = {}
        for cid, summary in measured.items():
            if "error" in summary:
                cases[cid] = {"error": summary["error"]}
                continue
            entry = {k: summary[k] for k in keep if k in summary}
            verdict = summary.get("harness_verdict") or {}
            entry["verified"] = verdict.get("status")
            entry["correct"] = verdict.get("correct")
            cases[cid] = entry
        incomplete = sorted(c for c, e in cases.items() if "error" in e)
        payload = {
            "note": "authored baseline measured on the author's 8-chip TPU v6e host; "
                    "indicative only. ORGANIZER.md requires the judge to measure its "
                    "own on contest hardware before scoring.",
            "suite_commit": CONFIG["suite"]["commit"],
            "seed": CONFIG["determinism"]["seed"],
            "runs_per_case": runs,
            "statistic": CONFIG["determinism"]["statistic"],
            "metric": "throughput_images_per_s = images / median "
                      "Timing['Encrypted computation'], the harness's own stage 7",
            "cases": cases,
        }
        if incomplete:
            payload["incomplete_cases"] = incomplete
        args.baseline.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"wrote": str(args.baseline),
                          "cases": sorted(cases),
                          "incomplete": incomplete}, indent=2))
        return

    baseline = {}
    if args.baseline.is_file():
        baseline = json.loads(args.baseline.read_text()).get("cases", {})

    verdict, execution, speed = {}, 0.0, 0.0
    for case in cases:
        cid = case["id"]
        summary = measured.get(cid, {})
        if "error" in summary:
            verdict[cid] = {"passed": 0, "reasons": [summary["error"]], "points": 0.0}
            continue
        passed, reasons = gate(summary, gates)
        base = baseline.get(cid, {})
        ratio = None
        if passed and base.get("throughput_images_per_s") and summary["throughput_images_per_s"]:
            ratio = summary["throughput_images_per_s"] / base["throughput_images_per_s"]
        points_exec = 5.0 * passed
        points_speed = 0.0
        if passed and ratio and ratio > 0:
            points_speed = 10.0 * min(max(math.log2(ratio) / 2.0, 0.0), 1.0)
        execution += points_exec
        speed += points_speed
        verdict[cid] = {"passed": passed, "reasons": reasons,
                        "throughput_ratio": ratio,
                        "points": round(points_exec + points_speed, 3), **summary}

    ratios = [v["throughput_ratio"] for v in verdict.values()
              if v.get("passed") and v.get("throughput_ratio")]
    print(json.dumps({
        "suite_commit": CONFIG["suite"]["commit"],
        "seed": CONFIG["determinism"]["seed"],
        "runs_per_case": runs,
        "statistic": CONFIG["determinism"]["statistic"],
        "cases": verdict,
        "execution_points": round(execution, 3),
        "speed_points": round(speed, 3),
        "evidence_points": "assessed by the judge from REPORT.md (10 max)",
        "subtotal_without_evidence": round(execution + speed, 3),
        "geometric_mean_throughput_ratio":
            round(math.exp(sum(math.log(r) for r in ratios) / len(ratios)), 4) if ratios else None,
        "note": "organizer arithmetic over harness-written measurement files; "
                "not an official score without a judge rerun",
    }, indent=2))


if __name__ == "__main__":
    main()
