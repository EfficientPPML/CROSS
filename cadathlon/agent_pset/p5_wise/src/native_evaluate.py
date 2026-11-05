"""Preserve every native WISE CPU tuning attempt, including security failures."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate import digest, publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--wise-root", type=Path, default=os.environ.get("WISE_ROOT"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=int, nargs="+", default=list(range(1, 7)))
    p.add_argument("--degrees", type=int, nargs="+", default=[8192, 16384])
    p.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    p.add_argument("--scaling", nargs="+", choices=("flexible", "composite"), default=["flexible", "composite"])
    p.add_argument("--scale-bits", type=int, nargs="+", default=[42])
    p.add_argument("--first-bits", type=int, help="explicit final-modulus headroom; retain the runner default when omitted")
    p.add_argument("--convolution", nargs="+", choices=("toeplitz", "winograd", "tiled-toeplitz", "hybrid"), default=["toeplitz"])
    p.add_argument("--pretrained-batch", type=int)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--diagnose", action="store_true")
    args = p.parse_args()
    started = time.perf_counter()
    if not args.wise_root: p.error("set WISE_ROOT or --wise-root")
    if args.pretrained_batch is not None and not 1 <= args.pretrained_batch <= 64: p.error("pretrained batch must be 1..64")
    if args.warmups < 0 or args.repeats < 1: p.error("invalid sample count")
    if args.diagnose and (args.warmups != 0 or args.repeats != 1): p.error("diagnostics require --warmups 0 --repeats 1")
    if not args.cases or len(set(args.cases)) != len(args.cases) or any(i not in range(1, 7) for i in args.cases):
        p.error("cases must be distinct integers 1 through 6")
    if any(t < 1 or t > len(os.sched_getaffinity(0)) for t in args.threads): p.error("threads exceed CPU affinity")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    shutil.copytree(Path(__file__).resolve().parent.parent, source,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pdf", "*.html", "run"))
    before = digest(source); publish(output / "source_sha256.json", before)
    import hashlib
    module_hash = hashlib.sha256(args.module.read_bytes()).hexdigest()
    report = {"status": "running", "engine": "WISE/OpenFHE", "attempts": {}, "best": {},
              "purpose": "client-layer-diagnostic" if args.diagnose else "encrypted-benchmark",
              "module": str(args.module.resolve()), "module_sha256": module_hash,
              "cpu_affinity": sorted(os.sched_getaffinity(0))}
    requested = [f"mnist{args.pretrained_batch}"] if args.pretrained_batch else args.cases
    report["requested_cases"] = requested
    for i, degree, threads, scaling, bits, convolution in itertools.product(
            requested, args.degrees, args.threads, args.scaling, args.scale_bits, args.convolution):
        name = f"test{i}-n{degree}-threads{threads}-{scaling}-bits{bits}-{convolution}"
        if args.first_bits is not None:
            name += f"-first{args.first_bits}"
        case_args = (["--pretrained", "--batch", str(args.pretrained_batch)] if args.pretrained_batch
                     else ["--case", str(source / f"test/test{i}.in")])
        command = [sys.executable, str(source / "src/native_runner.py"), *case_args,
                   "--module", str(args.module.resolve()), "--degree", str(degree), "--scaling", scaling,
                   "--scale-bits", str(bits), "--convolution", convolution,
                   "--warmups", str(args.warmups), "--repeats", str(args.repeats),
                   "--output", str(output / f"{name}.json")]
        if args.first_bits is not None:
            command.extend(["--first-bits", str(args.first_bits)])
        if args.diagnose: command.append("--diagnose")
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), WISE_ROOT=str(args.wise_root.resolve()))
        start = time.perf_counter()
        with (output / f"{name}.stdout").open("w") as out, (output / f"{name}.stderr").open("w") as err:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err)
            report.update(active=name, active_pid=child.pid); publish(output / "report.json", report)
            try:
                rc = child.wait()
            except BaseException:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                report.update(status="interrupted"); publish(output / "report.json", report)
                raise
        result_file = output / f"{name}.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        passed = rc == 0 and result.get("passed") is True
        report["attempts"][name] = {"passed": passed, "returncode": rc, "process_reaped": True,
            "elapsed_s": time.perf_counter()-start, "median_server_s": result.get("median_server_s"),
            "error": result.get("error"), "command": command}
        if passed and not args.diagnose and (str(i) not in report["best"] or result["median_server_s"] < report["best"][str(i)]["median_server_s"]):
            report["best"][str(i)] = {"attempt": name, "median_server_s": result["median_server_s"]}
        if before != digest(source) or hashlib.sha256(args.module.read_bytes()).hexdigest() != module_hash:
            report.update(status="failed", error="source or native binary changed"); publish(output / "report.json", report)
            return 1
        report["elapsed_s"] = time.perf_counter()-started
        publish(output / "report.json", report)
    successful = (all(a["passed"] for a in report["attempts"].values()) if args.diagnose else
                  all(str(i) in report["best"] for i in requested))
    report.update(status="completed", active=None, active_pid=None, source_files_unchanged=True,
                  all_cases_have_passing_result=successful, elapsed_s=time.perf_counter()-started)
    publish(output / "report.json", report)
    return 0 if report["all_cases_have_passing_result"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
