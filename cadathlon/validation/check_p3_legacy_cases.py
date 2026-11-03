"""Sequential TPU evidence driver for original P3 public cases 1--4.

This is a review utility, not the compact competition evaluator. --plan emits
the exact commands without starting a backend. Each real case gets a fresh
process, raw stdout/stderr, GNU time resource data and before/after hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cadathlon/non_agent_pset/p3_cnn/src"
CASES = tuple(f"test{number}.in" for number in range(1, 5))


def snapshot():
    paths = [path for folder in (ROOT / "jaxite_word", ROOT / "demos")
             for path in folder.rglob("*.py") if "__pycache__" not in path.parts]
    paths += [SRC / name for name in ("baseline.py", "runner.py", "reference.py")]
    paths += [ROOT / "cadathlon/common/support.py", Path(__file__).resolve()]
    paths += [SRC.parent / "test" / case for case in CASES]
    paths += [(SRC.parent / "test" / case).with_suffix(".out") for case in CASES]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def stop(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    process.wait()


def command(case):
    return [sys.executable, str(SRC / "runner.py"), "--case", str(SRC.parent / "test" / case),
            "--baseline", "--warmups", "0", "--repeats", "1"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_case(case, output, timeout_s):
    folder = output / Path(case).stem
    folder.mkdir()
    before = snapshot()
    write_json(folder / "source-before.json", before)
    started = time.monotonic()
    result = {"case": case, "command": command(case), "passed": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    with (folder / "stdout.json").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            ["/usr/bin/time", "-v", "-o", str(folder / "time.txt"), *command(case)],
            cwd=SRC, env=dict(os.environ, CROSS_ROOT=str(ROOT), JAX_PLATFORMS="tpu", PYTHONUNBUFFERED="1"),
            stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            result["error"] = f"case exceeded {timeout_s} seconds"
            stop(process)
        except BaseException:
            stop(process)
            raise
    result["returncode"] = process.returncode
    result["parent_elapsed_s"] = time.monotonic() - started
    after = snapshot()
    write_json(folder / "source-after.json", after)
    result["source_unchanged"] = before == after
    try:
        report = json.loads((folder / "stdout.json").read_text())
        result["runner"] = report
        result["passed"] = (process.returncode == 0 and report.get("passed") is True
                            and report.get("case") == case and report.get("implementation") == "baseline.py"
                            and result["source_unchanged"] and "error" not in result)
    except (OSError, ValueError, AttributeError) as error:
        result.setdefault("error", f"cannot read runner JSON: {error}")
    time_path = folder / "time.txt"
    if time_path.exists():
        for line in time_path.read_text().splitlines():
            if "Maximum resident set size (kbytes):" in line:
                result["peak_child_rss_kib"] = int(line.rsplit(":", 1)[1])
    write_json(folder / "summary.json", result)
    print(json.dumps({key: result.get(key) for key in
                      ("case", "passed", "returncode", "parent_elapsed_s", "peak_child_rss_kib")} ), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.timeout_s < 1:
        parser.error("--timeout-s must be positive")
    if args.plan:
        print(json.dumps({"cases": CASES, "commands": [command(case) for case in CASES],
                          "backend": "tpu", "fresh_process_per_case": True}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for a real run")
    if not Path("/usr/bin/time").is_file():
        parser.error("GNU /usr/bin/time is required for per-case RSS evidence")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"passed": False, "scope": "original full-model P3 cases 1--4; not compact-v1",
              "review_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("jax", "jaxlib", "libtpu", "numpy", "torch")},
              "python": sys.version, "cases": {}}
    write_json(args.output / "source-before.json", snapshot())
    try:
        for case in CASES:
            report["cases"][case] = run_case(case, args.output, args.timeout_s)
            write_json(args.output / "report.json", report)
        report["passed"] = all(case["passed"] for case in report["cases"].values())
    finally:
        report["parent_elapsed_s"] = time.monotonic() - started
        report["parent_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.output / "source-after.json", snapshot())
        write_json(args.output / "report.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
