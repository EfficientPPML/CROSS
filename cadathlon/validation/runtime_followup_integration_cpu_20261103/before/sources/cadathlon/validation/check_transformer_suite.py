"""Collect encrypted TPU measurements for the six public BERT or Qwen cases.

This invokes the normal runner and its security/numerical gates in fresh,
sequential processes. It never measures pretrained models or awards points.
Use a new output directory for every sweep; failed and partial evidence stays
visible. No deadline is imposed unless --case-timeout-s is explicitly supplied.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import runpy
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = {"bert": "p2_bert", "qwen": "p4_qwen"}
CASES = tuple(f"test{i}.in" for i in range(1, 7))
ENVIRONMENT = ("CROSS_ROOT", "PYTHONPATH", "JAX_PLATFORMS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "CROSS_PUBLIC_KEY_DIRECTORY", "CROSS_LEVEL_PARAMS_MAX",
               "CROSS_BARRETT_NTT_CONTEXT_CACHE_CAPACITY", "XLA_FLAGS")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(problem, implementation):
    paths = {Path(__file__).resolve(), implementation}
    for directory in (ROOT / "jaxite_word", ROOT / "demos", ROOT / "cadathlon/common", problem / "src"):
        paths.update(directory.rglob("*.py"))
    paths.update(implementation.parent.glob("*.py"))
    paths.update(problem / "test" / name for case in CASES
                 for name in (case, case.replace(".in", ".out")))
    return {str(path): sha256(path) for path in sorted(paths)}


def software_versions():
    result = {"python": platform.python_version(), "executable": sys.executable}
    for name in ("jax", "jaxlib", "libtpu", "numpy", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return value


def validate_result(result, *, model, case, implementation, warmups, repeats):
    """Check artifact consistency in addition to the child's actual gates."""
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError("runner did not report a passing encrypted request")
    mode = "HE-synthetic-smoke" if model == "bert" else "HE-toy-smoke"
    if result.get("case") != case or result.get("mode") != mode:
        raise ValueError("wrong case or encrypted runner mode")
    if result.get(f"full_{model}_evaluated") is not False:
        raise ValueError("public fixtures must not claim pretrained evaluation")
    if model == "bert" and result.get("mrpc_accuracy_evaluated") is not False:
        raise ValueError("public fixtures must not claim MRPC accuracy")
    if result.get("implementation") != str(implementation):
        raise ValueError("wrong implementation path")
    if (result.get("requested_platforms") != ["tpu"]
            or type(result.get("requested_device_count")) is not int
            or result["requested_device_count"] != 1):
        raise ValueError("expected the public runner's single selected TPU")
    if result.get("cold_start_included") is not (warmups == 0):
        raise ValueError("inconsistent cold-start declaration")
    elapsed = []
    for field, count in (("samples", repeats), ("warmup_samples", warmups)):
        samples = result.get(field)
        if not isinstance(samples, list) or len(samples) != count:
            raise ValueError(f"wrong {field} count")
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("invalid request sample")
            server = number(sample.get("server_s"), "server_s", positive=True)
            total = number(sample.get("end_to_end_s"), "end_to_end_s", positive=True)
            if total < server:
                raise ValueError("request elapsed time is shorter than server time")
            elapsed.append(total)
    for field in ("server_s", "end_to_end_s"):
        measured = number(result.get(field), field, positive=True)
        median = statistics.median(sample[field] for sample in result["samples"])
        if not math.isclose(measured, median, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{field} differs from sample median")
    setup = number(result.get("setup_s"), "setup_s")
    total = number(result.get("total_s"), "total_s", positive=True)
    if total + 1e-5 < setup + sum(elapsed):
        raise ValueError("runner total omits setup or requests")
    for field in ("max_abs_error", "nrmse", "top1_agreement"):
        number(result.get(field), field)
    if result["top1_agreement"] > 1:
        raise ValueError("invalid agreement")
    if model == "qwen" and (result["max_abs_error"] > .20 or result["nrmse"] > .10):
        raise ValueError("Qwen numeric gate failed")


def stop_process_group(process, *, grace_s=2.):
    """Stop only this owned group, allowing requested graceful cleanup time."""
    number(grace_s, 'cleanup_grace_s')
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_s
    while process_group_alive(process.pid) and time.monotonic() < deadline:
        remaining = min(.05, max(0., deadline - time.monotonic()))
        if process.poll() is None:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        else:
            # The leader may exit before a descendant finishes its cleanup.
            time.sleep(remaining)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def process_group_alive(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def child_environment_overrides():
    # support.cross_root() honors this variable. Bind it to the checkout whose
    # sources we hash rather than inheriting an unrelated developer checkout.
    return {"CROSS_ROOT": str(ROOT), "JAX_PLATFORMS": "tpu", "PYTHONUNBUFFERED": "1"}


@contextmanager
def termination_signals():
    """Turn external termination into the same evidence/cleanup path as Ctrl-C."""
    signals = (signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    def terminate(signum, frame):
        # Repeated termination requests must not interrupt owned-group cleanup.
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        raise SystemExit(128 + signum)
    try:
        for sig in signals:
            signal.signal(sig, terminate)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def runner_command(runner, arguments):
    """Use the same cancellation-aware child shim for every public runner."""
    return [sys.executable, str(Path(__file__).resolve()), '--child-runner', str(runner), *arguments]


def child_runner(runner, arguments):
    runner = Path(runner).resolve(strict=True)
    allowed = {(ROOT/'cadathlon/agent_pset'/problem/'src/runner.py').resolve()
               for problem in PROBLEMS.values()}
    if runner not in allowed:
        raise ValueError("child shim requires one of this checkout's public runners")
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), *arguments]
    with termination_signals():
        runpy.run_path(str(runner), run_name='__main__')


def run_case(command, directory, *, timeout, cleanup_grace_s=2.):
    number(cleanup_grace_s, 'cleanup_grace_s')
    started = time.monotonic()
    record = {"passed": False, "command": command, "timeout_s": timeout,
              "cleanup_grace_s": cleanup_grace_s}
    process = None
    try:
        with (directory / "stdout.json").open("w") as output, (directory / "stderr.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=log,
                                       start_new_session=True,
                                       env=dict(os.environ, **child_environment_overrides()))
            record["pid"] = process.pid
            try:
                record["returncode"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["error"] = "explicit per-case deadline exceeded"
                stop_process_group(process, grace_s=cleanup_grace_s)
                record["returncode"] = process.returncode
            except BaseException:
                stop_process_group(process, grace_s=cleanup_grace_s)
                record["returncode"] = process.returncode
                record["error"] = "case interrupted; owned process group stopped"
                raise
        if record.get("error") is None:
            try:
                record["result"] = json.loads((directory / "stdout.json").read_text())
            except (ValueError, OSError) as error:
                record["error"] = f"invalid runner output: {error}"
        if record.get("returncode") != 0:
            record.setdefault("error", f"runner exited {record.get('returncode')}; see stderr.log")
    finally:
        if process is not None and (process.poll() is None or process_group_alive(process.pid)):
            leader_completed = process.poll() is not None
            stop_process_group(process, grace_s=cleanup_grace_s)
            record["returncode"] = process.returncode
            record.setdefault("error", "runner left descendant processes; owned group stopped" if leader_completed
                              else "case interrupted; owned process group stopped")
        record["process_elapsed_s"] = time.monotonic() - started
        write_json(directory / "process.json", record)
    return record


def summarize(records, selected, *, warmups, repeats):
    complete = tuple(selected) == CASES and set(records) == set(CASES)
    passed = complete and all(record.get("passed") is True for record in records.values())
    return {"all_six_passed": passed, "complete": complete,
            "steady_measurement_schedule": warmups >= 1 and repeats >= 3,
            "scoring_calibrated": False, "full_model_evaluated": False}


def evaluate(args):
    problem = ROOT / "cadathlon/agent_pset" / PROBLEMS[args.model]
    implementation = args.implementation.resolve(strict=True)
    # Atomic creation prevents stale artifacts and concurrent writers.
    args.output.mkdir(parents=True, exist_ok=False)
    initial = source_manifest(problem, implementation)
    selected = tuple(f"test{i}.in" for i in args.cases)
    report = {"scope": "encrypted public synthetic fixtures only",
              "model": args.model, "selected_cases": selected, "cases": {},
              "warmups": args.warmups, "repeats": args.repeats,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "software": software_versions(), "host": platform.node(),
              "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
              "child_environment_overrides": child_environment_overrides(),
              "source_hashes_start": initial,
              "source_scope": "core/demos/common/problem Python, adjacent implementation Python, six input/output fixtures, wrapper; not an arbitrary external import closure"}
    started = time.monotonic()
    try:
        for case in selected:
            directory = args.output / Path(case).stem
            directory.mkdir()
            command = runner_command(problem / "src/runner.py", ["--case", str(problem / "test" / case),
                       "--implementation", str(implementation), "--require-tpu",
                       "--warmups", str(args.warmups), "--repeats", str(args.repeats)])
            report["active_case"] = case
            write_json(args.output / "report.json", report)
            print(f"Starting {case}; raw output: {directory}", flush=True)
            record = run_case(command, directory, timeout=args.case_timeout_s,
                              cleanup_grace_s=getattr(args, "cleanup_grace_s", 30.))
            try:
                current = source_manifest(problem, implementation)
            except OSError as error:
                current = None
                record["source_read_error"] = str(error)
            record["source_files_unchanged"] = current == initial
            if not record["source_files_unchanged"]:
                record["error"] = "source or fixture changed during sweep"
            if "error" not in record:
                try:
                    validate_result(record.get("result"), model=args.model, case=case,
                                    implementation=implementation, warmups=args.warmups, repeats=args.repeats)
                    record["passed"] = True
                except ValueError as error:
                    record["error"] = str(error)
            report["cases"][case] = record
            write_json(directory / "process.json", record)
            if not record["source_files_unchanged"]:
                break
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.pop("active_case", None)
        report["elapsed_s"] = time.monotonic() - started
        try:
            report["source_hashes_end"] = source_manifest(problem, implementation)
        except OSError as error:
            report["source_hashes_end"] = None
            report["source_read_error"] = str(error)
        report["source_files_unchanged"] = report["source_hashes_end"] == initial
        report.update(summarize(report["cases"], selected, warmups=args.warmups, repeats=args.repeats))
        report["passed"] = report["all_six_passed"] and report["source_files_unchanged"] and "error" not in report
        write_json(args.output / "report.json", report)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PROBLEMS, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing paths are rejected")
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-timeout-s", type=float)
    parser.add_argument("--cleanup-grace-s", type=float, default=30.,
                        help="grace after cancellation; not a healthy-work deadline")
    args = parser.parse_args()
    if (args.warmups < 0 or args.repeats < 1 or sorted(set(args.cases)) != args.cases
            or any(case not in range(1, 7) for case in args.cases)):
        parser.error("counts must be nonnegative/positive; cases must be unique ascending integers 1 through 6")
    if args.case_timeout_s is not None and (not math.isfinite(args.case_timeout_s) or args.case_timeout_s <= 0):
        parser.error("--case-timeout-s must be positive and finite")
    if not math.isfinite(args.cleanup_grace_s) or args.cleanup_grace_s <= 0:
        parser.error("--cleanup-grace-s must be positive and finite")
    args.output = args.output.resolve()
    with termination_signals():
        return evaluate(args)


if __name__ == "__main__":
    if sys.argv[1:2] == ['--child-runner']:
        if len(sys.argv) < 3:
            raise SystemExit('child runner path required')
        child_runner(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(main())
