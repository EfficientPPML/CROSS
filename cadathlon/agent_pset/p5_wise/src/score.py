"""Score complete, matched CPU/TPU WISE NN benchmark reports."""
import argparse
import json
import math
from pathlib import Path
import statistics


def checked(report, backend, *, native=False):
    if report.get("status") != "completed" or report.get("passed") is not True:
        raise ValueError("experiment not completed successfully")
    if report.get("mode") != "HE-WISE-NN" or report.get("backend") != backend:
        raise ValueError("wrong encrypted execution backend")
    if report.get("warmups", 0) < 1 or report.get("repeats", 0) < 3:
        raise ValueError("one warmup and at least three repeats required")
    samples = report.get("samples", [])
    warmups = report.get("warmup_samples", [])
    if len(samples) != report["repeats"] or len(warmups) != report["warmups"]:
        raise ValueError("incomplete timing samples")
    if not report.get("sources_before") or report["sources_before"] != report.get("sources_after"):
        raise ValueError("missing or changed sources")
    pretrained = report.get("case", {}).get("network") == "wise_mnist"
    if pretrained:
        from pretrained_mnist import ASSETS
        from wise_source import WISE_COMMIT
        batch = report["case"].get("batch")
        if (type(batch) is not int or not 1 <= batch <= 64
                or report.get("pretrained_checkpoint_evaluated") is not True
                or report.get("samples_scored") != batch):
            raise ValueError("missing complete pretrained checkpoint/sample evidence")
        wise = report.get("schedule", {}).get("wise", {})
        if (wise.get("commit") != WISE_COMMIT or not wise.get("source_sha256")
                or wise.get("asset_sha256") != ASSETS):
            raise ValueError("missing or changed pinned pretrained assets")
    for sample in samples + warmups:
        quality = sample.get("quality", {})
        for key in ("server_s", "client_to_client_s"):
            t = sample.get(key)
            if type(t) not in (int, float) or not math.isfinite(t) or t <= 0:
                raise ValueError("invalid timing")
        if (sample.get("passed") is not True or quality.get("top1_agreement") != 1
                or not 0 <= quality.get("max_abs_error", math.inf) <= (1e-3 if pretrained else 1e-5)):
            raise ValueError("failed numerical gate")
        if pretrained and not 0 <= quality.get("nrmse", math.inf) <= 1e-4:
            raise ValueError("failed pretrained normalized-error gate")
    if native:
        if report.get("engine") != "WISE/OpenFHE" or report.get("profile", {}).get("security_level") != "HEStd_128_classic":
            raise ValueError("missing native 128-bit security enforcement")
        binaries = report.get("native_binary_hashes_before")
        if not binaries or binaries != report.get("native_binary_hashes_after"):
            raise ValueError("missing or changed native runtime binaries")
    elif report.get("crypto_profile", {}).get("security_bits") != 128:
        raise ValueError("missing secure profile")
    return statistics.median(s["server_s"] for s in samples)


def compare(cpu, tpu):
    cpu_s, tpu_s = checked(cpu, "cpu"), checked(tpu, "tpu")
    for name in ("case", "crypto_profile"):
        if cpu[name] != tpu[name]:
            raise ValueError(f"CPU/TPU {name} differs")
    if cpu["schedule"]["sha256"] != tpu["schedule"]["sha256"]:
        raise ValueError("CPU/TPU WISE schedule differs")
    return {"cpu_server_s": cpu_s, "tpu_server_s": tpu_s, "speedup": cpu_s / tpu_s}


def compare_native(cpu, tpu):
    """Equal logical work/security/accuracy; allow different CPU/TPU packing."""
    cpu_s, tpu_s = checked(cpu, "cpu", native=True), checked(tpu, "tpu")
    if cpu["case"] != tpu["case"]:
        raise ValueError("native CPU/TPU workload differs")
    from wise_source import WISE_COMMIT
    left, right = cpu.get("schedule", {}).get("wise"), tpu.get("schedule", {}).get("wise")
    if (not left or left != right or left.get("commit") != WISE_COMMIT or not left.get("source_sha256")):
        raise ValueError("native CPU/TPU pinned WISE identity differs or is missing")
    return {"native_cpu_server_s": cpu_s, "tpu_server_s": tpu_s,
            "speedup_over_native_cpu": cpu_s/tpu_s,
            "comparison": "same network and numerical gate; independently secure profiles and schedules"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", type=Path, nargs="+")
    p.add_argument("--tpu", type=Path, nargs="+", required=True)
    p.add_argument("--native-cpu", type=Path, nargs="+",
                   help="optional tuned native CPU reports, in the same workload order")
    p.add_argument("--native-only", action="store_true",
                   help="compare native CPU directly with TPU, without CROSS CPU reports or contest points")
    p.add_argument("--pretrained", action="store_true",
                   help="compare complete MNIST batches separately, without synthetic-suite points")
    args = p.parse_args()
    if args.native_only:
        if args.cpu or not args.native_cpu:
            p.error("--native-only requires --native-cpu and excludes --cpu")
        reference_paths = args.native_cpu
    else:
        if not args.cpu:
            p.error("--cpu is required unless --native-only is selected")
        reference_paths = args.cpu
    if len(reference_paths) != len(args.tpu) or (args.native_cpu and len(args.native_cpu) != len(reference_paths)):
        p.error("provide equal numbers of CPU, TPU and optional native reports")
    if not args.pretrained and len(reference_paths) != 6:
        p.error("the synthetic suite requires six CPU/TPU pairs")
    references = [json.loads(path.read_text()) for path in reference_paths]
    tpu_reports = [json.loads(path.read_text()) for path in args.tpu]
    cases = [report["case"] for report in references]
    if args.pretrained:
        if (any(c.get("network") != "wise_mnist" for c in cases)
                or len({c["batch"] for c in cases}) != len(cases)):
            raise ValueError("pretrained comparison requires distinct complete MNIST batches")
    else:
        expected = [json.loads(p.read_text()) for p in sorted((Path(__file__).parent.parent / "test").glob("test*.in"))]
        if sorted(json.dumps(c, sort_keys=True) for c in cases) != sorted(json.dumps(c, sort_keys=True) for c in expected):
            raise ValueError("all six fixed workloads required, each exactly once")
    result = {"workloads": cases,
              "scope": "pretrained MNIST batches" if args.pretrained else "six fixed synthetic networks"}
    if args.native_only:
        result.update(comparison_mode="native CPU versus TPU", contest_scoring_eligible=False)
    else:
        pairs = [compare(c, t) for c, t in zip(references, tpu_reports)]
        speedup = math.exp(statistics.mean(math.log(pair["speedup"]) for pair in pairs))
        result.update(cases=pairs, geomean_speedup=speedup)
        if not args.pretrained:
            result.update(correctness_points=40, performance_points=40 * max(0, 1 - 1 / speedup),
                          evidence_points="up to 20, organizer-reviewed; never awarded automatically")
    if args.native_cpu:
        native_reports = references if args.native_only else [json.loads(path.read_text()) for path in args.native_cpu]
        native_pairs = [compare_native(c, t) for c, t in zip(native_reports, tpu_reports)]
        result["native_cpu_comparison"] = native_pairs
        result["geomean_speedup_over_native_cpu"] = math.exp(statistics.mean(
            math.log(pair["speedup_over_native_cpu"]) for pair in native_pairs))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
