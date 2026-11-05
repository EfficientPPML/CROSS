"""Measure the actual WISE/OpenFHE CPU backend at enforced 128-bit security."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
from types import SimpleNamespace
import statistics
import sys
import time
import traceback
import numpy as np

from runner import save
from schedule import Schedule, build_schedule, execute, pack_inputs, NumpyArithmetic
from workloads import input_values, reference


def run(args, report):
    start = time.perf_counter()
    report["initial_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    source_hashes = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path(__file__).parent.glob("*.py"))}
    report["sources_before"] = source_hashes()
    spec = importlib.util.spec_from_file_location("fhecore", args.module)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    native_files = {args.module.resolve()}
    for row in Path("/proc/self/maps").read_text().splitlines():
        name = row.split()[-1]
        if name.startswith("/") and "libOPENFHE" in name:
            native_files.add(Path(name).resolve())
    native_hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(native_files)}
    report["native_binary_hashes_before"] = native_hashes()
    case = (json.loads(args.case.read_text()) if not args.pretrained else
            {"network": "wise_mnist", "batch": args.batch, "selection": "first test examples in IDX order"})
    slots = args.degree // 2
    planning = time.perf_counter()
    with redirect_stdout(sys.stderr):
        if args.pretrained:
            from pretrained_mnist import build_pretrained
            import torch
            plan, x, labels, model = build_pretrained(args.batch, slots, args.wise_root, convolution=args.convolution)
            with torch.no_grad(): expected = model(torch.from_numpy(x)).numpy()
            report.update(native_correct=int((expected.argmax(-1) == labels).sum()), samples_scored=len(labels))
        else:
            plan = build_schedule(case, slots, args.wise_root, convolution=args.convolution)
            x = input_values(case); expected = reference(case, x)
    report["planning_s"] = time.perf_counter()-planning
    verification = time.perf_counter()
    packed = pack_inputs(plan, x)
    ideal = np.concatenate(execute(plan, packed, NumpyArithmetic()))[:plan.output_size].reshape(expected.shape)
    report["schedule_max_abs_error"] = float(np.max(np.abs(ideal-expected)))
    np.testing.assert_allclose(ideal, expected, atol=1e-8 if args.pretrained else 1e-10, rtol=0)
    report["plaintext_verification_s"] = time.perf_counter()-verification
    report["numerical_gates"] = {"encrypted_max_abs": 1e-3 if args.pretrained else 1e-5,
                                "encrypted_nrmse": 1e-4 if args.pretrained else None,
                                "top1_agreement": 1.0}
    params = native.FHEParams()
    params.log_N = args.degree.bit_length()-1
    params.log_batch_size = slots.bit_length()-1
    params.mult_depth = plan.depth
    params.scale_mod_size = args.scale_bits
    # Trained logits need substantially more integer headroom than the small
    # synthetic cases. Keep the final modulus independent of precision.
    first_bits = args.first_bits or (60 if args.pretrained else args.scale_bits+1)
    params.first_mod_size = first_bits
    params.secret_key_dist = "UNIFORM_TERNARY"
    params.security_level = "HEStd_128_classic"
    params.scaling_technique = "COMPOSITESCALINGAUTO" if args.scaling == "composite" else "FLEXIBLEAUTO"
    params.key_switch_technique = "HYBRID"
    params.composite_degree = 2 if args.scaling == "composite" else 1
    params.register_word_size = 24 if args.scaling == "composite" else 0
    params.level_budget = []
    params.global_rots = plan.rotations()
    report.update(case=case, schedule=plan.summary(), backend="cpu", engine="WISE/OpenFHE",
                  mode="client-layer-diagnostic" if args.diagnose else "HE-WISE-NN",
                  convolution=args.convolution, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  omp_threads=os.environ.get("OMP_NUM_THREADS"),
                  thread_environment={k: os.environ.get(k) for k in
                      ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                  versions={"python": sys.version, "numpy": np.__version__},
                  module_path=str(args.module), module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  profile={"degree": args.degree, "slots": slots, "depth": plan.depth,
                           "scale_bits": args.scale_bits, "first_bits": first_bits,
                           "scaling": args.scaling, "security_level": "HEStd_128_classic"},
                  warmups=args.warmups, repeats=args.repeats, cold_start_included=args.warmups == 0,
                  phase="OpenFHE parameter validation and key generation")
    save(args.output, report)
    begin = time.perf_counter()
    # GenCryptoContext validates the actual chosen ring against its 128-bit
    # standard. An insufficient explicit degree is a failed attempt, retained.
    ctx = native.FHEContext(params)
    report["setup_s"] = time.perf_counter()-begin
    report["warmup_samples"], report["samples"] = [], []
    for trial in range(args.warmups+args.repeats):
        report["phase"] = f"request {trial+1}"
        save(args.output, report)
        begin = time.perf_counter()
        values = ctx.encrypt_batch(packed, 0)
        ideal_values = list(packed) if args.diagnose else None
        if args.diagnose: report["layer_diagnostics"] = []
        server_start = time.perf_counter()
        for level, (kind, layer) in enumerate(plan.layers):
            if kind == "linear":
                values = ctx.eval_linear_transform(values, layer, level)
            elif kind == "winograd":
                values = ctx.eval_winograd(values, layer, level)
            elif kind == "square":
                hp = SimpleNamespace(a0=np.zeros((len(values), slots)), a1=np.zeros((len(values), slots)))
                values = ctx.eval_herpn(values, hp, level)
            elif kind == "herpn":
                values = ctx.eval_herpn(values, layer, level)
            else:
                raise ValueError("unsupported native NN operator")
            if args.diagnose:
                single = Schedule(slots, [(kind, layer)], len(ideal_values)*slots, len(values)*slots, plan.provenance)
                ideal_values = execute(single, ideal_values, NumpyArithmetic())
                decoded = np.asarray(ctx.decrypt_batch(values))
                delta = decoded - np.asarray(ideal_values)
                report["layer_diagnostics"].append({"layer": level+1, "operation": kind,
                    "max_abs_error": float(np.max(np.abs(delta))),
                    "reference_max_abs": float(np.max(np.abs(ideal_values))),
                    "actual_max_abs": float(np.max(np.abs(decoded)))})
            if args.pretrained:
                report["phase"] = f"request {trial+1}, layer {level+1}/{plan.depth}: {kind} finished"
                save(args.output, report)
        server_s = time.perf_counter()-server_start
        actual = np.asarray(ctx.decrypt_batch(values)).ravel()[:plan.output_size].reshape(expected.shape)
        error = float(np.max(np.abs(actual-expected)))
        nrmse = float(np.sqrt(np.mean((actual-expected)**2)) / max(np.sqrt(np.mean(expected**2)), 1e-12))
        top1 = float(np.mean(actual.argmax(-1)==expected.argmax(-1)))
        passed = np.isfinite(error) and error <= report["numerical_gates"]["encrypted_max_abs"] and top1 == 1
        if args.pretrained:
            passed = passed and nrmse <= 1e-4
            report["encrypted_correct"] = int((actual.argmax(-1) == labels).sum())
        sample = {"server_s": server_s, "client_to_client_s": time.perf_counter()-begin,
                  "quality": {"max_abs_error": error, "nrmse": nrmse, "top1_agreement": top1}, "passed": bool(passed)}
        report["warmup_samples" if trial < args.warmups else "samples"].append(sample)
        save(args.output, report)
        if not passed:
            raise ValueError("native encrypted NN failed its unchanged numerical gate")
    report["sources_after"] = source_hashes()
    report["native_binary_hashes_after"] = native_hashes()
    # OpenMP binding may pin the main thread after setup. Preserve each live
    # worker's placement rather than treating that main-thread mask as the
    # whole allocation; the collector also records its prelaunch affinity.
    thread_affinities = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            thread_affinities[task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
    report["thread_cpu_affinities"] = thread_affinities
    if (report["sources_before"] != report["sources_after"] or
            report["native_binary_hashes_before"] != report["native_binary_hashes_after"]):
        raise ValueError("native runtime sources or binaries changed")
    report.update(status="completed", passed=True, elapsed_s=time.perf_counter()-start,
                  peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  timing_scope="internal runner through numerical checks; excludes interpreter startup and process-exit cleanup",
                  pretrained_checkpoint_evaluated=args.pretrained, full_pretrained_test_set_evaluated=False,
                  median_server_s=statistics.median(s["server_s"] for s in report["samples"]),
                  median_client_to_client_s=statistics.median(s["client_to_client_s"] for s in report["samples"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cases = p.add_mutually_exclusive_group(required=True)
    cases.add_argument("--case", type=Path)
    cases.add_argument("--pretrained", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--module", type=Path, required=True, help="built WISE fhecore extension")
    p.add_argument("--wise-root", type=Path, default=os.environ.get("WISE_ROOT"))
    p.add_argument("--degree", type=int, default=16384)
    p.add_argument("--scale-bits", type=int, default=42)
    p.add_argument("--first-bits", type=int, help="final modulus headroom; pretrained defaults to 60")
    p.add_argument("--scaling", choices=("flexible", "composite"), default="flexible")
    p.add_argument("--convolution", choices=("toeplitz", "winograd", "tiled-toeplitz", "hybrid"), default="toeplitz")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnose", action="store_true", help="client decrypts after every layer; not a server timing result")
    args = p.parse_args()
    if args.output.exists(): p.error("output exists; preserve earlier evidence")
    if args.degree < 4096 or args.degree & (args.degree-1): p.error("degree must be a power of two >= 4096")
    if args.warmups < 0 or args.repeats < 1: p.error("invalid sample count")
    if not args.wise_root: p.error("set WISE_ROOT or --wise-root")
    if args.diagnose and (args.warmups != 0 or args.repeats != 1): p.error("diagnostics require --warmups 0 --repeats 1")
    report = {"status": "running", "passed": False}
    save(args.output, report)
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="failed", passed=False, error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
    save(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
