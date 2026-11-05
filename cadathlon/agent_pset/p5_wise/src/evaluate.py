"""Collect fresh-process NN experiments against an immutable source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
            and p.suffix != ".pyc"}


def publish(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def snapshot_runtime(snapshot):
    """Keep the released planner/helpers and use the explicitly selected core."""
    package = Path(__file__).resolve().parents[3]
    core = Path(os.environ.get("CROSS_ROOT", package.parent)).resolve()
    if not (core / "jaxite_word/API_REFERENCE.md").is_file():
        raise FileNotFoundError("Set CROSS_ROOT to the CROSS checkout when using the extracted release")
    folders = [(core / "jaxite_word", "jaxite_word"),
               (package / "common", "cadathlon/common"),
               (Path(__file__).resolve().parent.parent, "cadathlon/agent_pset/p5_wise")]
    for source, destination in folders:
        shutil.copytree(source, snapshot / destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pdf", "*.html", "run", "*.tar.gz"))
    return digest(snapshot)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wise-root", type=Path, default=os.environ.get("WISE_ROOT"))
    p.add_argument("--strategy", choices=("direct", "bsgs"), default="bsgs")
    p.add_argument("--convolution", choices=("toeplitz", "winograd", "tiled-toeplitz", "hybrid"), default="toeplitz")
    p.add_argument("--cases", type=int, nargs="+", default=list(range(1, 7)))
    p.add_argument("--pretrained-batch", type=int, help="collect full WISE MNIST instead of the six synthetic cases")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--rotation-cache-mib", type=int, default=0)
    p.add_argument("--linear-chunk", type=int, choices=range(17), default=0)
    args = p.parse_args()
    if not args.wise_root:
        p.error("set WISE_ROOT or --wise-root")
    if args.warmups < 0 or args.repeats < 1 or min(args.cache_mib, args.rotation_cache_mib) < 0:
        p.error("warmups/cache must be nonnegative and repeats positive")
    if args.pretrained_batch is not None and not 1 <= args.pretrained_batch <= 64:
        p.error("pretrained batch must be 1..64")
    if not args.cases or len(set(args.cases)) != len(args.cases) or any(i not in range(1, 7) for i in args.cases):
        p.error("cases must be unique numbers 1 through 6")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    before = snapshot_runtime(snapshot)
    publish(output / "source_sha256.json", before)
    requested = [f"mnist{args.pretrained_batch}"] if args.pretrained_batch else args.cases
    report = {"status": "running", "passed": False, "backend": args.backend, "strategy": args.strategy,
              "convolution": args.convolution,
              "linear_chunk": args.linear_chunk, "cache_mib": args.cache_mib,
              "rotation_cache_mib": args.rotation_cache_mib,
              "warmups": args.warmups, "repeats": args.repeats,
              "requested_cases": requested, "complete_six_case_schedule": not args.pretrained_batch and set(args.cases) == set(range(1, 7)),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "cases": {}}
    env = dict(os.environ, CROSS_ROOT=str(snapshot), WISE_ROOT=str(args.wise_root.resolve()),
               JAX_PLATFORMS=args.backend, PYTHONPATH="")
    started = time.perf_counter()
    for i in requested:
        report["active_case"] = i
        publish(output / "report.json", report)
        src = snapshot / "cadathlon/agent_pset/p5_wise"
        case_args = (["--pretrained", "--batch", str(args.pretrained_batch)] if args.pretrained_batch
                     else ["--case", str(src / f"test/test{i}.in")])
        command = [sys.executable, str(src / "src/runner.py"), *case_args,
                   "--backend", args.backend, "--strategy", args.strategy, "--convolution", args.convolution,
                   "--warmups", str(args.warmups), "--repeats", str(args.repeats),
                   "--linear-chunk", str(args.linear_chunk), "--cache-mib", str(args.cache_mib),
                   "--rotation-cache-mib", str(args.rotation_cache_mib),
                   "--output", str(output / f"test{i}.json")]
        start = time.perf_counter()
        with (output / f"test{i}.stdout").open("w") as stdout, (output / f"test{i}.stderr").open("w") as stderr:
            child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            report["active_pid"] = child.pid
            publish(output / "report.json", report)
            try:
                returncode = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                report.update(status="interrupted", passed=False)
                publish(output / "report.json", report)
                raise
        result_file = output / f"test{i}.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        report["cases"][str(i)] = {"returncode": returncode, "process_reaped": True,
            "external_elapsed_s": time.perf_counter() - start, "command": command,
            "passed": returncode == 0 and result.get("passed") is True,
            "median_server_s": result.get("median_server_s"), "error": result.get("error")}
        if digest(snapshot) != before:
            report.update(status="failed", error="frozen runtime sources changed")
            publish(output / "report.json", report)
            return 1
    report.update(status="completed", active_case=None, active_pid=None,
                  source_files_unchanged=True, elapsed_s=time.perf_counter() - started,
                  passed=all(c["passed"] for c in report["cases"].values()))
    publish(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
