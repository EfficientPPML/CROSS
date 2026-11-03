"""Measurement artifact gates; these CPU tests do not certify encrypted runs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_transformer_suite.py"
spec = importlib.util.spec_from_file_location("transformer_suite_validation", SCRIPT)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def result(model="bert", case="test1.in", warmups=1, repeats=3, implementation=Path("/reference.py")):
    data = dict(passed=True, case=case, mode="HE-synthetic-smoke" if model == "bert" else "HE-toy-smoke",
                implementation=str(implementation), requested_platforms=["tpu"], requested_device_count=1,
                cold_start_included=warmups == 0, setup_s=10., total_s=10. + 3. * (warmups + repeats),
                server_s=2., end_to_end_s=3., max_abs_error=.001, nrmse=.001, top1_agreement=1.,
                samples=[dict(server_s=2., end_to_end_s=3.) for _ in range(repeats)],
                warmup_samples=[dict(server_s=2., end_to_end_s=3.) for _ in range(warmups)])
    data[f"full_{model}_evaluated"] = False
    if model == "bert":
        data["mrpc_accuracy_evaluated"] = False
    return data


@pytest.mark.parametrize("model", ["bert", "qwen"])
@pytest.mark.parametrize("warmups,repeats", [(0, 1), (1, 3)])
def test_accepts_complete_normal_runner_artifact(model, warmups, repeats):
    suite.validate_result(result(model, warmups=warmups, repeats=repeats), model=model,
                          case="test1.in", implementation=Path("/reference.py"), warmups=warmups, repeats=repeats)


@pytest.mark.parametrize("field,value", [
    ("passed", 1), ("passed", False), ("case", "test2.in"), ("mode", "plaintext-reference"),
    ("full_bert_evaluated", True), ("mrpc_accuracy_evaluated", True),
    ("implementation", "/different.py"), ("requested_platforms", ["cpu"]),
    ("requested_device_count", True), ("requested_device_count", 8),
    ("cold_start_included", True), ("samples", []), ("warmup_samples", []),
    ("server_s", float("nan")), ("end_to_end_s", float("inf")),
    ("server_s", 1.9), ("end_to_end_s", 2.9), ("total_s", 10.),
    ("max_abs_error", -1.), ("nrmse", True), ("top1_agreement", 1.1),
])
def test_rejects_invalid_or_inconsistent_measurement(field, value):
    data = result()
    data[field] = value
    with pytest.raises(ValueError):
        suite.validate_result(data, model="bert", case="test1.in", implementation=Path("/reference.py"),
                              warmups=1, repeats=3)


@pytest.mark.parametrize("sample", [None, dict(server_s=-1, end_to_end_s=3),
                                     dict(server_s=4, end_to_end_s=3)])
def test_checks_individual_warmups_too(sample):
    data = result()
    data["warmup_samples"] = [sample]
    with pytest.raises(ValueError):
        suite.validate_result(data, model="bert", case="test1.in", implementation=Path("/reference.py"),
                              warmups=1, repeats=3)


def test_partial_and_single_sample_sweeps_never_claim_calibration():
    records = {case: dict(passed=True) for case in suite.CASES}
    full = suite.summarize(records, suite.CASES, warmups=1, repeats=3)
    assert full["complete"] and full["all_six_passed"] and full["steady_measurement_schedule"]
    assert not full["scoring_calibrated"] and not full["full_model_evaluated"]
    partial = suite.summarize({"test1.in": dict(passed=True)}, ("test1.in",), warmups=0, repeats=1)
    assert not partial["complete"] and not partial["all_six_passed"]
    assert not partial["steady_measurement_schedule"]
    records["test6.in"]["passed"] = False
    assert not suite.summarize(records, suite.CASES, warmups=1, repeats=3)["all_six_passed"]


def test_process_exit_and_raw_artifacts_are_retained(tmp_path):
    record = suite.run_case([sys.executable, "-c", "import json; print(json.dumps({'passed': False})); raise SystemExit(3)"],
                            tmp_path, timeout=None)
    assert record["returncode"] == 3 and not record["passed"]
    assert record["result"] == {"passed": False}
    assert json.loads((tmp_path / "process.json").read_text())["returncode"] == 3


def test_explicit_timeout_reaps_process(tmp_path):
    record = suite.run_case([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, timeout=.1)
    assert record["returncode"] != 0
    assert record["error"] == "explicit per-case deadline exceeded"
    assert record["process_elapsed_s"] < 5


def test_child_uses_hashed_checkout_even_when_parent_points_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSS_ROOT", "/unrelated/checkout")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    code = "import json,os; print(json.dumps({name:os.environ[name] for name in ('CROSS_ROOT','JAX_PLATFORMS')}))"
    record = suite.run_case([sys.executable, "-c", code], tmp_path, timeout=None)
    assert record["returncode"] == 0
    assert record["result"] == {"CROSS_ROOT": str(suite.ROOT), "JAX_PLATFORMS": "tpu"}


def prepare_evaluation(monkeypatch, tmp_path, *, sources=None):
    implementation = tmp_path / "reference.py"
    implementation.write_text("# mock implementation; never executed\n")
    args = SimpleNamespace(model="bert", implementation=implementation, output=tmp_path / "evidence",
                           cases=list(range(1, 7)), warmups=1, repeats=3, case_timeout_s=None)
    monkeypatch.setattr(suite, "source_manifest", sources or (lambda *args: {"source.py": "unchanged"}))
    def child(command, directory, *, timeout, cleanup_grace_s=30.):
        case = Path(command[command.index("--case") + 1]).name
        return dict(passed=False, returncode=0, process_reaped=True, public_key_store_removed=True,
                    result=result(case=case, implementation=implementation))
    monkeypatch.setattr(suite, "run_case", child)
    return args


def test_full_sweep_publishes_all_cases_but_no_score(monkeypatch, tmp_path):
    args = prepare_evaluation(monkeypatch, tmp_path)
    assert suite.evaluate(args) == 0
    report = json.loads((args.output / "report.json").read_text())
    assert report["passed"] and report["all_six_passed"] and report["source_files_unchanged"]
    assert set(report["cases"]) == set(suite.CASES)
    assert not report["scoring_calibrated"]
    with pytest.raises(FileExistsError):
        suite.evaluate(args)


@pytest.mark.parametrize('process_reaped,keys_removed', [(False, True), (True, False), (False, False)])
def test_incomplete_cleanup_prevents_later_case_launch(monkeypatch, tmp_path, process_reaped, keys_removed):
    args = prepare_evaluation(monkeypatch, tmp_path)
    run = suite.run_case
    launched = []
    def child(*arguments, **kwargs):
        launched.append(arguments[0])
        record = run(*arguments, **kwargs)
        record.update(process_reaped=process_reaped, public_key_store_removed=keys_removed)
        return record
    monkeypatch.setattr(suite, 'run_case', child)
    assert suite.evaluate(args) == 1
    report = json.loads((args.output / 'report.json').read_text())
    assert len(launched) == 1 and list(report['cases']) == ['test1.in']
    assert report['stopped_for_incomplete_cleanup'] and not report['passed']
    assert report['cases']['test1.in']['passed'] is False


@pytest.mark.parametrize("missing", [False, True])
def test_changed_or_deleted_source_stops_and_preserves_failed_evidence(monkeypatch, tmp_path, missing):
    calls = 0
    def sources(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"source.py": "before"}
        if missing:
            raise FileNotFoundError("source.py disappeared")
        return {"source.py": "after"}
    args = prepare_evaluation(monkeypatch, tmp_path, sources=sources)
    assert suite.evaluate(args) == 1
    report = json.loads((args.output / "report.json").read_text())
    assert not report["passed"] and not report["source_files_unchanged"]
    assert list(report["cases"]) == ["test1.in"]
    assert not report["cases"]["test1.in"]["passed"]


def _process_is_running(pid):
    """A killed orphan may briefly remain a zombie until its reaper runs."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _await_file(path, timeout=5):
    import time
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), f"child did not create {path}"


def _await_stopped(pid, timeout=5):
    import time
    deadline = time.monotonic() + timeout
    while _process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not _process_is_running(pid), f"owned process {pid} survived cleanup"


def test_normal_leader_exit_cleans_lingering_descendants_and_fails(tmp_path):
    import os
    import signal
    ready = tmp_path / "descendant.pid"
    descendant = ("import os,signal,time; from pathlib import Path; "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
    leader = ("import subprocess,sys,time,json; from pathlib import Path; "
              f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
              f"ready=Path({str(ready)!r}); deadline=time.monotonic()+5\n"
              "while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
              "assert ready.exists()\n"
              "print(json.dumps({'passed':True,'descendant':child.pid}),flush=True)")
    try:
        record = suite.run_case([sys.executable, "-c", leader], tmp_path, timeout=None)
        _await_file(ready)
        descendant_pid = int(ready.read_text())
        assert record["returncode"] == 0
        assert record["passed"] is False and record.get("error")
        _await_stopped(descendant_pid)
        stored = json.loads((tmp_path / "process.json").read_text())
        assert stored["passed"] is False and stored.get("error")
    finally:
        if ready.exists() and _process_is_running(int(ready.read_text())):
            # The grandchild belongs to run_case's new group. Kill only our
            # recorded child if this regression fails before cleanup.
            os.kill(int(ready.read_text()), signal.SIGKILL)


def test_sigterm_outer_wrapper_stops_owned_group_and_preserves_evidence(tmp_path):
    import os
    import signal
    import subprocess
    ready = tmp_path / "child.pid"
    child = ("import os,time; from pathlib import Path; "
             f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import importlib.util,sys\nfrom pathlib import Path\n"
        f"spec=importlib.util.spec_from_file_location('suite',{str(SCRIPT)!r})\n"
        "suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)\n"
        "with suite.termination_signals():\n"
        f"    suite.run_case([sys.executable,'-c',{child!r}],Path({str(tmp_path)!r}),timeout=None)\n")
    process = subprocess.Popen([sys.executable, str(wrapper)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True,
                               env=dict(os.environ, JAX_PLATFORMS="cpu"))
    try:
        _await_file(ready)
        child_pid = int(ready.read_text())
        process.terminate()
        process.communicate(timeout=8)
        assert process.returncode != 0
        _await_stopped(child_pid)
        record = json.loads((tmp_path / "process.json").read_text())
        assert record["passed"] is False and record["returncode"] != 0
        assert "interrupt" in record["error"]
    finally:
        if ready.exists() and _process_is_running(int(ready.read_text())):
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_termination_signal_handlers_are_restored(raise_inside):
    import signal
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        with suite.termination_signals():
            for sig, handler in previous.items():
                assert callable(signal.getsignal(sig))
                assert signal.getsignal(sig) is not handler
            if raise_inside:
                raise RuntimeError("test context cleanup")
    except RuntimeError:
        assert raise_inside
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
