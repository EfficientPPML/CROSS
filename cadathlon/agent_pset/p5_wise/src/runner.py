"""Verify and benchmark an actual WISE-generated encrypted NN schedule."""
import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np

from schedule import build_schedule, execute, pack_inputs, NumpyArithmetic
from workloads import input_values, reference


def save(path, value):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(path)


def run(args, report):
    started = time.perf_counter()
    case = (json.loads(args.case.read_text()) if not args.pretrained else
            {"network": "wise_mnist", "batch": args.batch, "selection": "first test examples in IDX order"})
    report.update(case=case, strategy=args.strategy, convolution=args.convolution,
                  warmups=args.warmups, repeats=args.repeats, cold_start_included=args.warmups == 0,
                  linear_chunk=args.linear_chunk, cache_mib=args.cache_mib,
                  rotation_cache_mib=args.rotation_cache_mib)
    source_dir = Path(__file__).parent
    source_hashes = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(source_dir.glob("*.py"))}
    report["sources_before"] = source_hashes()
    if args.reference:
        slots = args.slots
    else:
        # Explicit selection prevents silently benchmarking CPU fallback as TPU.
        os.environ["JAX_PLATFORMS"] = args.backend
        import cross_backend
        spec = cross_backend.profile(9 if args.pretrained else 3 if case["network"] == "mlp" else 4)
        slots = spec["degree"] // 2
        report["crypto_profile"] = spec
    planning = time.perf_counter()
    with redirect_stdout(sys.stderr):
        if args.pretrained:
            from pretrained_mnist import build_pretrained
            import torch
            schedule, x, labels, model = build_pretrained(args.batch, slots, args.wise_root, convolution=args.convolution)
            with torch.no_grad():
                expected = model(torch.from_numpy(x)).numpy()
            report["native_correct"] = int((expected.argmax(-1) == labels).sum())
            report["samples_scored"] = len(labels)
        else:
            x = input_values(case)
            expected = reference(case, x)
            schedule = build_schedule(case, slots, args.wise_root, convolution=args.convolution)
    report["schedule"] = schedule.summary()
    report["planning_s"] = time.perf_counter() - planning
    verification = time.perf_counter()
    packed = pack_inputs(schedule, x)
    clear = np.concatenate(execute(schedule, packed, NumpyArithmetic(), strategy=args.strategy))
    clear = clear[:schedule.output_size].reshape(expected.shape)
    error = float(np.max(np.abs(clear - expected)))
    report["schedule_max_abs_error"] = error
    report["plaintext_verification_s"] = time.perf_counter() - verification
    schedule_gate = 1e-8 if args.pretrained else 1e-10
    report["numerical_gates"] = {"schedule_max_abs": schedule_gate,
                                "encrypted_max_abs": 1e-3 if args.pretrained else 1e-5,
                                "encrypted_nrmse": 1e-4 if args.pretrained else None, "top1_agreement": 1.0}
    if not np.isfinite(error) or error > schedule_gate:
        raise ValueError(f"WISE schedule disagrees with independent NN oracle: {error}")
    if args.reference:
        report.update(mode="plaintext-reference", passed=True, status="completed")
    else:
        import jax
        from support import require_secure_execution, metrics
        devices = jax.devices()
        if len(devices) < 1 or any(d.platform != args.backend for d in devices):
            raise RuntimeError("requested execution backend unavailable")
        report.update(mode="HE-WISE-NN", backend=args.backend, devices=[str(devices[0])],
                      available_devices=[str(d) for d in devices], execution_device_count=1,
                      device_kind=devices[0].device_kind, cpu_affinity=sorted(os.sched_getaffinity(0)),
                      thread_environment={k: os.environ.get(k) for k in
                          ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                      versions={"python": sys.version, "jax": jax.__version__, "numpy": np.__version__})
        for package in ("jaxlib", "libtpu", "torch"):
            try:
                report["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["versions"][package] = None
        report["phase"] = "key generation and initialization"
        save(args.output, report)
        setup = time.perf_counter()
        with redirect_stdout(sys.stderr), jax.default_device(devices[0]):
            client, server = cross_backend.create_contexts(spec, schedule.rotations(args.strategy))
            arithmetic = cross_backend.make_arithmetic(server, args.cache_mib * 1024 ** 2,
                                                       linear_chunk=args.linear_chunk,
                                                       rotation_cache_bytes=args.rotation_cache_mib * 1024 ** 2)
        report["setup_s"] = time.perf_counter() - setup
        report["samples"], report["warmup_samples"] = [], []
        for trial in range(args.warmups + args.repeats):
            report["phase"] = f"request {trial + 1}"
            save(args.output, report)
            request_start = time.perf_counter()
            with jax.default_device(devices[0]):
                encrypted = [client.encrypt_slots(v) for v in packed]
                jax.block_until_ready([v.polynomial for v in encrypted])
                require_secure_execution(arithmetic, encrypted)
                server_start = time.perf_counter()
                # Catch accidental client work in the evaluator call. This is
                # a regression guard, not an isolation boundary for submissions.
                def forbidden(*a, **kw):
                    raise RuntimeError("client codec called during server evaluation")
                with ExitStack() as stack:
                    for name in ("encrypt_slots", "encrypt_slots_batch", "decrypt_slots", "decrypt_slots_batch"):
                        if hasattr(type(client), name):
                            stack.enter_context(patch.object(type(client), name, forbidden))
                    def progress(index, kind, values):
                        jax.block_until_ready([v.polynomial for v in values])
                        report["phase"] = f"request {trial + 1}, layer {index + 1}/{schedule.depth}: {kind} finished"
                        save(args.output, report)
                    output = execute(schedule, encrypted, arithmetic, strategy=args.strategy,
                                     progress=progress if args.pretrained else None)
                jax.block_until_ready([v.polynomial for v in output])
                server_s = time.perf_counter() - server_start
                require_secure_execution(arithmetic, output)
                if any(next(iter(v.polynomial.devices())).platform != args.backend for v in output):
                    raise RuntimeError("ciphertext output on wrong backend")
                decoded = np.concatenate([client.decrypt_slots(v) for v in output])
            actual = decoded[:schedule.output_size].reshape(expected.shape)
            quality = metrics(actual, expected)
            passed = quality["max_abs_error"] <= report["numerical_gates"]["encrypted_max_abs"] and quality["top1_agreement"] == 1.0
            if args.pretrained:
                passed = passed and quality["nrmse"] <= 1e-4
                report["encrypted_correct"] = int((actual.argmax(-1) == labels).sum())
            sample = {"server_s": server_s, "client_to_client_s": time.perf_counter() - request_start,
                      "quality": quality, "passed": passed}
            report["warmup_samples" if trial < args.warmups else "samples"].append(sample)
            save(args.output, report)
            if not passed:
                raise ValueError(f"encrypted numerical gate failed: {quality}")
            del encrypted, output
        report.update(passed=True, status="completed",
                      median_server_s=statistics.median(s["server_s"] for s in report["samples"]),
                      median_client_to_client_s=statistics.median(s["client_to_client_s"] for s in report["samples"]),
                      peak_public_cache_bytes=arithmetic.peak_cached_bytes,
                      rotations_compiled=arithmetic.rotator.compile_count,
                      peak_rotation_cache_bytes=getattr(arithmetic.rotator, "peak_cached_bytes", 0),
                      rotation_binding_hits=getattr(arithmetic.rotator, "binding_hits", 0),
                      rotation_state_hits=getattr(arithmetic.rotator, "state_hits", 0),
                      linear_fusion_compiled=(arithmetic.linear_fusion.compiler.compile_count
                                              if arithmetic.linear_fusion else 0),
                      pretrained_checkpoint_evaluated=args.pretrained,
                      full_pretrained_test_set_evaluated=False)
        report["device_memory"] = devices[0].memory_stats()
        report["device_memory_note"] = "Allocator statistics at completion; unavailable when null, not total physical-device memory"
    report["sources_after"] = source_hashes()
    if report["sources_before"] != report["sources_after"]:
        raise RuntimeError("runner sources changed during measurement")
    report["elapsed_s"] = time.perf_counter() - started
    report["peak_host_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["timing_scope"] = "internal runner through numerical checks; excludes interpreter startup and process-exit cleanup"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cases = p.add_mutually_exclusive_group(required=True)
    cases.add_argument("--case", type=Path)
    cases.add_argument("--pretrained", action="store_true", help="entire pinned WISE MNIST/HerPN checkpoint")
    p.add_argument("--batch", type=int, default=1, help="pretrained first N test examples, 1..64")
    p.add_argument("--wise-root", type=Path, default=os.environ.get("WISE_ROOT"))
    p.add_argument("--reference", action="store_true")
    p.add_argument("--backend", choices=("cpu", "tpu"), default="cpu")
    p.add_argument("--strategy", choices=("bsgs", "direct"), default="bsgs")
    p.add_argument("--convolution", choices=("toeplitz", "winograd", "tiled-toeplitz", "hybrid"), default="toeplitz")
    p.add_argument("--slots", type=int, default=4096, help="plaintext schedule check only; HE derives secure ring")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--rotation-cache-mib", type=int, default=0,
                   help="bounded public rotation key/control cache; 0 retains the original helper")
    p.add_argument("--linear-chunk", type=int, choices=range(17), default=0,
                   help="fuse up to N public diagonal products; 0 uses separate primitives")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.warmups < 0 or args.repeats < 1 or min(args.cache_mib, args.rotation_cache_mib) < 0:
        p.error("warmups/cache must be nonnegative and repeats positive")
    if args.output and args.output.exists():
        p.error("output already exists; preserve the earlier experiment and choose a fresh path")
    report = {"status": "running", "passed": False}
    save(args.output, report)
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="failed", passed=False, error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
    save(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
