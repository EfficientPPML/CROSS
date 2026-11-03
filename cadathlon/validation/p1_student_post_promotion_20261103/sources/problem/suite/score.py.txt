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
from p1_evidence import archive_hashes, model_identity, sha256

SUITE_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((SUITE_DIR / "suite_config.json").read_text())
INSTANCE_NAMES = ("single", "small", "medium", "large")
STUDENT_GATE_VERSION = 'student-v2'


def _seconds(text) -> float:
    """Harness timings are strings like '12.1014s', '1.8286h', '10.55m'."""
    if isinstance(text, bool):
        raise ValueError("boolean is not a timing measurement")
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


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    # Python's JSON reader accepts NaN/Infinity by default; none may leak into
    # a score report, including informational fields that are not scored.
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains invalid/nonfinite JSON data") from error
    return value


def _read_object(path):
    return _object(json.loads(path.read_text()), str(path))


def read_case(root: Path, instance: int, runs: int, case_id: str = "",
              label: str = ""):
    """Prefer the per-case archive; fall back to the live measurements dir.

    Two ranked cases can share an instance and differ only in chip count, so
    the live `measurements/<instance>` directory holds whichever ran last.
    archive.py copies each case aside under its own id.
    """
    folder = archived = None
    if case_id:
        candidates = [SUITE_DIR / "results" / (f"{case_id}_{label}" if label else case_id)]
        for candidate in candidates:
            if candidate is not None and candidate.is_dir() and \
                    any(candidate.glob("results-*.json")):
                folder = candidate
                meta = folder / "case.json"
                if meta.is_file():
                    archived = _read_object(meta)
                break
    if folder is None and label:
        raise FileNotFoundError(f"no archive for {case_id} with label {label!r}; run and archive this case first")
    if folder is None:
        folder = root / "measurements" / INSTANCE_NAMES[instance]
    if archived is not None:
        expected = next((c for c in CONFIG["cases"] if c["id"] == case_id), None)
        if expected is not None:
            for field in ("id", "instance", "batch", "devices"):
                if archived.get(field) != expected[field]:
                    raise ValueError(f"archive {field} does not match {case_id}")
        if archived.get("suite_commit") != CONFIG["suite"]["commit"] or archived.get("seed") != CONFIG["determinism"]["seed"]:
            raise ValueError("archive suite commit or seed differs from configuration")
        if label and archived.get("label") != label:
            raise ValueError("archive label differs from requested label")
    records = []
    for n in range(1, runs + 1):
        path = folder / f"results-{n}.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run the harness with --num_runs {runs}")
        records.append(_read_object(path))
    return records, archived, folder


def summarize(records, batch: int, archived=None):
    if not records:
        raise ValueError("no measurement records")
    if archived is not None:
        _object(archived, "archive")
        verdict = archived.get("verify")
        if verdict is not None:
            _object(verdict, "archive.verify")
            own = verdict.get("own_cleartext")
            if own is not None:
                _object(own, "archive.verify.own_cleartext")
    # Validate every repetition before aggregation: medians must not hide a
    # failed, zero, negative, or non-finite sample.
    for record in records:
        _object(record, "measurement")
        _object(record.get("Timing"), "measurement.Timing")
        if "Server Reported" in record:
            server = _object(record["Server Reported"], "measurement.Server Reported")
            if "server_execute_s" in server:
                value = _seconds(server["server_execute_s"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid server_execute_s timing")
        for field in ("Total", "Key Generation", "Encrypted model preprocessing", "Encrypted computation"):
            value = _seconds(record["Timing"][field])
            if not math.isfinite(value) or value < 0 or (field in ("Total", "Encrypted computation") and value == 0):
                raise ValueError(f"invalid {field} timing: {value}")
        quality = record.get("Quality")
        if quality is not None:
            _object(quality, "measurement.Quality")
        if batch > 1 and not quality:
            raise ValueError("missing Quality for a multi-image case")
        if quality:
            encrypted = _object(quality["Encrypted model quality"], "Encrypted model quality")
            harness = _object(quality["Harness model quality"], "Harness model quality")
            if not _finite_number(harness.get("accuracy")) or not 0 <= harness["accuracy"] <= 1:
                raise ValueError("invalid harness model accuracy")
            accuracy = encrypted["accuracy"]
            if not _finite_number(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("invalid encrypted accuracy")
            if (type(encrypted["total_samples"]) is not int
                    or encrypted["total_samples"] != batch):
                raise ValueError("incomplete predictions in a measured repetition")
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
        "baseline_profile": (archived or {}).get("baseline_profile", "legacy"),
        "evaluation_id": (archived or {}).get("evaluation_id"),
        "identity": (archived or {}).get("identity"),
        "execution": (archived or {}).get("execution"),
        "runs_note": "same images every run; spread is machine noise only",
        # Stages 1/2.2/3 run once outside the harness's run loop, so these two
        # are one-time costs that every run's JSON repeats, not per-run figures.
        "total_s_including_one_time_setup": statistics.median(totals),
        "one_time_setup_s": max(setups),
        "median_encrypted_compute_s": median_compute,
        "max_encrypted_compute_s": max(computes),
        "median_server_execute_s": statistics.median(server) if server else None,
        "images": batch,
        "throughput_images_per_s": batch / median_compute if median_compute > 0 else None,
        "ms_per_image": 1000.0 * median_compute / batch,
        "encrypted_accuracy": accuracy,
        "min_encrypted_accuracy": min(q["Encrypted model quality"]["accuracy"] for q in quality) if quality else None,
        "harness_model_accuracy": harness_accuracy,   # different network; context only
        "samples_scored": statistics.median(
            q["Encrypted model quality"]["total_samples"] for q in quality) if quality else None,
        "harness_verdict": (archived or {}).get("verify"),
        "own_cleartext_agreement": ((archived or {}).get("verify") or {})
            .get("own_cleartext", {}).get("top1_agreement"),
        # Informational: a different network, never gated on.
        "harness_reference_model_accuracy_for_context": harness_accuracy,
    }
    return _object(out, "measurement summary")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def gate(summary, gates) -> tuple[int, list]:
    reasons = []
    for field in ("median_encrypted_compute_s", "one_time_setup_s", "throughput_images_per_s"):
        value = summary.get(field)
        if not _finite_number(value) or value < 0 or (field != "one_time_setup_s" and value == 0):
            reasons.append(f"invalid {field}")
    images = summary.get("images")
    if not isinstance(images, int) or isinstance(images, bool) or images < 1:
        reasons.append("invalid image count")
    accuracy = summary.get("encrypted_accuracy")
    if accuracy is not None and (not _finite_number(accuracy) or not 0 <= accuracy <= 1):
        reasons.append("invalid encrypted accuracy")
    if reasons:
        return 0, reasons
    if summary.get("max_encrypted_compute_s", summary["median_encrypted_compute_s"]) > gates["max_batch_s"]:
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
    if images > 1 and (samples is None or summary["encrypted_accuracy"] is None):
        reasons.append("missing Quality for a multi-image case")
    acc = summary.get("min_encrypted_accuracy", summary["encrypted_accuracy"])
    own = (summary.get("harness_verdict") or {}).get("own_cleartext") or {}
    images = summary["images"]
    if own.get("status") == "ok":
        agreement = own.get("top1_agreement")
        correct = own.get("cleartext_correct")
        if not _finite_number(agreement) or not 0 <= agreement <= 1:
            reasons.append("invalid own cleartext model agreement")
        if own.get("compared") != images:
            reasons.append("incomplete own cleartext comparison")
        if not isinstance(correct, int) or isinstance(correct, bool) or not 0 <= correct <= images:
            reasons.append("invalid cleartext correct count")
        if reasons:
            return 0, reasons
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
        if verdict.get("complete") is not True:
            reasons.append("frozen verdict is incomplete")
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
        else:
            reasons.append("no cleartext agreement check: "
                           f"{own.get('detail', 'not computed')}")
    return (0 if reasons else 1), reasons


def evaluation_path(label):
    return SUITE_DIR / 'run' / ('evaluation-' + label) / 'report.json'


def calibration_report_path(label, evaluation_id):
    # Public aggregate proof ships with the profile calibration and survives
    # `make clean`; private run/evidence files are never copied here.
    return SUITE_DIR / 'calibration' / label / evaluation_id / 'report.json'


def validate_student_evaluation(report, *, label, folders, mode=None):
    """Require one completed TPU evaluation bound to every current archive.

    Reports and archives must come from the protected organizer evaluator;
    this is identity/freshness validation, not a sandbox for hostile Python.
    """
    _object(report, 'student evaluation')
    ranked = CONFIG['evaluation_profiles']['student']['case_ids']
    expected = {'profile': 'student', 'gate_version': STUDENT_GATE_VERSION,
                'label': label, 'suite_commit': CONFIG['suite']['commit'],
                'seed': CONFIG['determinism']['seed'],
                'runs_per_case': CONFIG['determinism']['num_runs'],
                'require_tpu': True}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('student evaluation profile, gate version, label or configuration differs')
    if mode and report.get('mode') != mode:
        raise ValueError('student calibration needs a baseline evaluation')
    if (report.get('passed') is not True or report.get('within_budget') is not True
            or report.get('cleanup_passed') is not True or report.get('error') or report.get('cleanup_error')):
        raise ValueError('student whole evaluation failed, timed out, or did not finish cleanup')
    budget, elapsed = report.get('budget_s'), report.get('elapsed_s')
    if not _finite_number(budget) or not _finite_number(elapsed) or not 0 < elapsed <= budget <= 300:
        raise ValueError('student whole evaluation must finish within a budget of at most 300 seconds')
    identity = report.get('evaluation_id')
    if not isinstance(identity, str) or len(identity) != 32 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('student evaluation identity is missing')
    cases = _object(report.get('cases'), 'student evaluation cases')
    hashes = _object(report.get('archive_sha256'), 'student archive hashes')
    if set(cases) != set(ranked) or set(hashes) != set(ranked) or set(folders) != set(ranked):
        raise ValueError('student whole evaluation must contain every ranked case exactly once')
    current_model = model_identity(SUITE_DIR.parent.parent)
    if report.get('model_identity') != current_model:
        raise ValueError('student evaluation uses a different frozen model')
    versions = _object(report.get('versions'), 'student software versions')
    if any(not isinstance(versions.get(name), str) or not versions[name]
           for name in ('jax', 'jaxlib', 'libtpu', 'numpy', 'torch', 'python')):
        raise ValueError('student software identity is incomplete')
    sources = _object(report.get('source_sha256'), 'student source hashes')
    if (not sources or report.get('source_files_unchanged') is not True
            or sources != report.get('source_sha256_after')):
        raise ValueError('student evaluation sources were not stable')
    if sources.get('problem/suite/suite_config.json') != sha256(SUITE_DIR / 'suite_config.json'):
        raise ValueError('student schedule/gates configuration differs from this evaluation')
    for cid in ranked:
        case = _object(cases[cid], 'student evaluation case')
        if case.get('passed') is not True or case.get('baseline_profile') != 'student' or case.get('evaluation_id') != identity:
            raise ValueError(f'{cid}: case does not belong to the successful student evaluation')
        asset = _object(case.get('identity'), f'{cid} identity')
        if asset.get('model') != current_model:
            raise ValueError(f'{cid}: frozen model identity differs')
        for field in ('pixels_sha256', 'labels_sha256'):
            value = asset.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'{cid}: data identity is missing')
        execution = _object(case.get('execution'), f'{cid} actual execution')
        definition = next(c for c in CONFIG['cases'] if c['id'] == cid)
        devices, kinds = execution.get('devices'), execution.get('device_kinds')
        if (execution.get('batch') != definition['batch'] or not isinstance(devices, list)
                or len(devices) != definition['devices'] or not isinstance(kinds, list) or len(kinds) != len(devices)
                or any(not isinstance(d, str) or not d.startswith('TPU') for d in devices)
                or any(not isinstance(k, str) or 'TPU' not in k for k in kinds)):
            raise ValueError(f'{cid}: actual TPU hardware does not match the schedule')
        rings = execution.get('execution_rings')
        if not isinstance(rings, list) or not rings or any(
                not isinstance(ring, dict) or type(ring.get('security_bits')) is not int or ring['security_bits'] < 128
                or not _finite_number(ring.get('log2_QP')) or ring['log2_QP'] <= 0 for ring in rings):
            raise ValueError(f'{cid}: actual secure-ring evidence is missing')
        if hashes[cid] != archive_hashes(folders[cid], CONFIG['determinism']['num_runs']):
            raise ValueError(f'{cid}: archived measurements changed since the complete evaluation')
    return report


def calibration_payload(measured, profile, evaluation=None):
    """Build a calibration after the owner has validated the complete run."""
    ranked = CONFIG['evaluation_profiles'][profile]['case_ids']
    invalid = {cid: ([summary['error']] if 'error' in summary else gate(summary, CONFIG['gates'])[1])
               for cid, summary in measured.items()}
    for cid, summary in measured.items():
        if 'error' not in summary and summary['baseline_profile'] != profile:
            invalid[cid].append('archive ring profile differs from selected calibration profile')
    if set(measured) != set(ranked):
        invalid['missing_cases'] = sorted(set(ranked)-set(measured)) or ['unexpected cases']
    invalid = {cid: reasons for cid, reasons in invalid.items() if reasons}
    if invalid:
        raise ValueError('refusing to overwrite baseline with failed/unverified cases: ' + json.dumps(invalid))
    keep = ('runs', 'baseline_profile', 'one_time_setup_s', 'median_encrypted_compute_s',
            'median_server_execute_s', 'images', 'throughput_images_per_s', 'ms_per_image',
            'encrypted_accuracy', 'own_cleartext_agreement', 'identity', 'execution')
    entries = {}
    for cid, summary in measured.items():
        verdict = summary.get('harness_verdict') or {}
        entries[cid] = {**{k: summary[k] for k in keep if k in summary},
                        'verified': verdict.get('status'), 'correct': verdict.get('correct')}
    payload = {'note': 'Measured calibration; organizers must reproduce it on contest hardware.',
               'evaluation_profile': profile, 'suite_commit': CONFIG['suite']['commit'],
               'seed': CONFIG['determinism']['seed'], 'runs_per_case': CONFIG['determinism']['num_runs'],
               'statistic': CONFIG['determinism']['statistic'],
               'metric': "throughput_images_per_s = images / median Timing['Encrypted computation']",
               'cases': entries}
    if profile == 'student':
        if not evaluation or evaluation.get('mode') != 'baseline':
            raise ValueError('refusing student calibration without a complete baseline evaluation')
        payload.update(gate_version=STUDENT_GATE_VERSION, evaluation_id=evaluation['evaluation_id'],
                       evaluation_label=evaluation['label'],
                       versions=evaluation['versions'],
                       evaluation_report=str(calibration_report_path(evaluation['label'], evaluation['evaluation_id']).relative_to(SUITE_DIR)))
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-root", type=Path, default=None)
    ap.add_argument("--baseline", type=Path,
                    default=None, help="defaults to the selected evaluation profile's calibration")
    ap.add_argument("--evaluation-profile", choices=CONFIG["evaluation_profiles"], default="legacy")
    ap.add_argument("--cases", default="", help="comma-separated case ids within the selected evaluation profile")
    ap.add_argument("--label", default="",
                    help="archive label to read, e.g. baseline or candidate")
    ap.add_argument('--evaluation-report', type=Path,
                    help='student whole-run report; defaults to run/evaluation-<label>/report.json')
    ap.add_argument('--baseline-evaluation-report', type=Path,
                    help='protected whole-run report associated with the student calibration')
    ap.add_argument("--emit-baseline", action="store_true",
                    help="write the measured numbers as a baseline manifest instead of scoring")
    args = ap.parse_args()
    if args.emit_baseline and args.evaluation_profile == 'student':
        raise SystemExit('refusing student --emit-baseline: use evaluate.py --profile student --baseline; '
                         'only the complete evaluator transaction may publish student calibration')
    if args.evaluation_profile == 'student' and not args.label:
        args.label = 'student-baseline' if args.emit_baseline else 'student-candidate'
    if args.baseline is None:
        args.baseline = SUITE_DIR / ("baseline_manifest_student.json" if args.evaluation_profile == "student"
                                     else "baseline_manifest.json")

    import os
    root = args.suite_root or Path(os.environ.get("MLINF_ROOT", Path.home() / "ml-inference"))
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    ranked = CONFIG["evaluation_profiles"][args.evaluation_profile]["case_ids"]
    unknown = wanted - set(ranked)
    if unknown:
        ap.error(f"unknown cases: {sorted(unknown)}")
    cases = [c for c in CONFIG["cases"] if c["id"] in ranked and (not wanted or c["id"] in wanted)]
    runs = CONFIG["determinism"]["num_runs"]
    gates = CONFIG["gates"]

    measured, folders = {}, {}
    for case in cases:
        try:
            records, archived, folder = read_case(root, case["instance"], runs,
                                                  case["id"], args.label)
            summary = summarize(records, case["batch"], archived)
        except (OSError, ValueError, KeyError, TypeError) as error:
            measured[case["id"]] = {"error": str(error)}
            continue
        summary["read_from"] = str(folder)
        folders[case['id']] = folder
        # The upstream checkout ships the reference implementation's own
        # measurements under measurements/<instance>/. Reading those instead of
        # an archive would score somebody else's run, so say so loudly.
        if archived is None:
            summary["warning"] = ("no per-case archive; these numbers came from "
                                  "the live measurements directory, which the "
                                  "upstream checkout also populates")
        measured[case["id"]] = summary

    evaluation, evaluation_error = None, None
    if args.evaluation_profile == 'student':
        try:
            evaluation = validate_student_evaluation(
                _read_object(args.evaluation_report or evaluation_path(args.label)),
                label=args.label, folders=folders, mode='baseline' if args.emit_baseline else None)
            for cid, summary in measured.items():
                if (summary.get('baseline_profile') != 'student'
                        or summary.get('evaluation_id') != evaluation['evaluation_id']
                        or summary.get('identity') != evaluation['cases'][cid]['identity']):
                    raise ValueError(f'{cid}: archived profile/identity differs from selected student evaluation')
        except (OSError, ValueError, KeyError, TypeError) as error:
            evaluation_error = str(error)

    if args.emit_baseline:
        if evaluation_error:
            raise SystemExit('refusing student calibration: ' + evaluation_error)
        try:
            payload = calibration_payload(measured, args.evaluation_profile, evaluation)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        args.baseline.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"wrote": str(args.baseline),
                          "cases": sorted(payload['cases']),
                          "incomplete": []}, indent=2))
        return 0

    baseline = {}
    baseline_error = None
    if args.baseline.is_file():
        try:
            calibration = _read_object(args.baseline)
            expected_metadata = {"suite_commit": CONFIG["suite"]["commit"],
                                 "seed": CONFIG["determinism"]["seed"], "runs_per_case": runs,
                                 "statistic": CONFIG["determinism"]["statistic"]}
            if any(calibration.get(k) != v for k, v in expected_metadata.items()):
                raise ValueError("baseline configuration does not match this evaluation")
            candidate = _object(calibration["cases"], "baseline.cases")
            for cid, entry in candidate.items():
                _object(entry, f"baseline.cases.{cid}")
            if args.evaluation_profile == 'student':
                if (calibration.get('evaluation_profile') != 'student'
                        or calibration.get('gate_version') != STUDENT_GATE_VERSION):
                    raise ValueError('student baseline needs a newly measured student-v2 whole evaluation')
                baseline_label = calibration.get('evaluation_label')
                if not isinstance(baseline_label, str) or not baseline_label:
                    raise ValueError('student baseline evaluation label is missing')
                reference_path = args.baseline_evaluation_report or calibration_report_path(baseline_label, calibration.get('evaluation_id', 'missing'))
                reference = validate_student_evaluation(
                    _read_object(reference_path),
                    label=baseline_label, mode='baseline',
                    folders={cid: reference_path.parent / 'archives' / cid for cid in ranked})
                if calibration.get('evaluation_id') != reference['evaluation_id']:
                    raise ValueError('baseline manifest belongs to a different complete evaluation')
                if candidate != calibration_payload(reference['cases'], 'student', reference)['cases']:
                    raise ValueError('baseline manifest measurements differ from its complete evaluation')
                if calibration.get('versions') != reference['versions']:
                    raise ValueError('baseline software differs from its complete evaluation')
                if evaluation is not None and evaluation['versions'] != reference['versions']:
                    evaluation_error = 'software versions differ from calibration; recalibrate on judging hardware'
            baseline = candidate
        except (OSError, ValueError, KeyError, TypeError) as error:
            baseline_error = str(error)
    else:
        baseline_error = "baseline manifest missing; speed points unavailable"

    verdict, execution, speed = {}, 0.0, 0.0
    for case in cases:
        cid = case["id"]
        summary = measured.get(cid, {})
        if "error" in summary:
            verdict[cid] = {"passed": 0, "reasons": [summary["error"]], "points": 0.0}
            continue
        passed, reasons = gate(summary, gates)
        if summary['baseline_profile'] != args.evaluation_profile:
            passed = 0
            reasons.append('archive ring profile differs from selected evaluation profile')
        if evaluation_error:
            passed = 0
            reasons.append('student whole evaluation: ' + evaluation_error)
        base = baseline.get(cid, {})
        if (args.evaluation_profile == 'student' and base
                and base.get('identity') != summary.get('identity')):
            passed = 0
            reasons.append('model or ranked input identity differs from calibration')
        if (args.evaluation_profile == 'student' and base
                and (base.get('execution') or {}).get('device_kinds') != (summary.get('execution') or {}).get('device_kinds')):
            passed = 0
            reasons.append('TPU device kind differs from calibration')
        ratio = None
        base_rate = base.get("throughput_images_per_s")
        if (passed and base.get("verified") == "PASS" and base.get("images") == case["batch"]
                and base.get("baseline_profile", "legacy") == summary["baseline_profile"]
                and summary['baseline_profile'] == args.evaluation_profile
                and _finite_number(base_rate) and base_rate > 0):
            ratio = summary["throughput_images_per_s"] / base["throughput_images_per_s"]
            if not math.isfinite(ratio) or ratio <= 0:
                ratio = None
                baseline_error = f"nonfinite or underflowed throughput ratio for {cid}; speed points unavailable"
        points_exec = (30.0 / len(ranked)) * passed
        points_speed = 0.0
        if passed and ratio and ratio > 0:
            points_speed = (60.0 / len(ranked)) * min(max(math.log2(ratio) / 2.0, 0.0), 1.0)
        execution += points_exec
        speed += points_speed
        verdict[cid] = {"passed": passed, "reasons": reasons,
                        "throughput_ratio": ratio,
                        "points": round(points_exec + points_speed, 3), **summary}

    ratios = [v["throughput_ratio"] for v in verdict.values()
              if v.get("passed") and v.get("throughput_ratio")]
    success = all(v.get('passed') for v in verdict.values()) and not baseline_error and not evaluation_error
    print(json.dumps({
        'passed': bool(success),
        "suite_commit": CONFIG["suite"]["commit"],
        "seed": CONFIG["determinism"]["seed"],
        "runs_per_case": runs,
        "statistic": CONFIG["determinism"]["statistic"],
        "cases": verdict,
        "evaluation_profile": args.evaluation_profile,
        "ranked_case_ids": ranked,
        "baseline_error": baseline_error,
        'evaluation_error': evaluation_error,
        "execution_points": round(execution, 3),
        "speed_points": round(speed, 3),
        "evidence_points": "assessed by the judge from REPORT.md (10 max)",
        "subtotal_without_evidence": round(execution + speed, 3),
        "geometric_mean_throughput_ratio":
            round(math.exp(sum(math.log(r) for r in ratios) / len(ratios)), 4) if ratios else None,
        "note": "organizer arithmetic over harness-written measurement files; "
                "not an official score without a judge rerun",
    }, indent=2, allow_nan=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
