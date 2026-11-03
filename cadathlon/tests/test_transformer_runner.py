"""Transformer timing, readiness and failure cleanup without accelerator setup."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "common"))
    spec = importlib.util.spec_from_file_location("transformer_timing_test", ROOT / "common" / "transformer_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "require_ciphertexts", lambda value: None)
    monkeypatch.setattr(module, "require_secure_execution", lambda owner, value: None)
    monkeypatch.setattr(module, "synchronize", lambda value: None)
    return module


def make_session(clock, expected):
    class Session:
        requests = 0
        encryptions = 0
        closed = False

        def encrypt(self, inputs):
            clock.now += 1
            self.encryptions += 1
            return object()

        def evaluate(self, encrypted, **kwargs):
            clock.now += 100 if self.requests == 0 else 2
            self.requests += 1
            return object()

        def decrypt(self, encrypted):
            clock.now += 3
            return expected.copy()

        def close(self):
            clock.now += 7
            self.closed = True

    session = Session()

    def create(*args, **kwargs):
        clock.now += 10
        return session

    return session, SimpleNamespace(create_session=create)


def run(runner, implementation, **overrides):
    options = dict(config={}, weights={}, embeddings=np.zeros((1, 2, 3)),
                   expected=np.array([[1., 2.]]), devices=(SimpleNamespace(platform="tpu"),),
                   evaluate_kwargs={}, check_output=lambda *args: None, warmups=0, repeats=1)
    options.update(overrides)
    return runner.benchmark_session(implementation, **options)


@pytest.mark.parametrize("warmups", [0, 1])
def test_timing_separates_cold_request_and_warmups(runner, monkeypatch, warmups):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock.now)
    session, implementation = make_session(clock, np.array([[1., 2.]]))
    result = run(runner, implementation, warmups=warmups, repeats=2)
    assert session.closed
    assert session.encryptions == warmups + 2
    assert result["setup_s"] == 10
    assert result["server_s"] == (2 if warmups else 51)
    assert result["end_to_end_s"] == (6 if warmups else 55)
    assert result["cold_start_included"] is (warmups == 0)
    assert result["warmup_samples"] == ([{"server_s": 100, "end_to_end_s": 104}] if warmups else [])
    assert clock.now == (133 if warmups else 127)
    assert result["passed"] is True


@pytest.mark.parametrize("invalid", [dict(warmups=-1), dict(repeats=0), dict(warmups=True), dict(repeats=1.5)])
def test_invalid_counts_fail_before_setup(runner, invalid):
    run_error = SimpleNamespace(create_session=lambda *args, **kwargs: pytest.fail("setup should not run"))
    with pytest.raises(ValueError):
        run(runner, run_error, **invalid)


def test_require_tpu_prevents_cpu_calibration(runner):
    implementation = SimpleNamespace(create_session=lambda *args, **kwargs: pytest.fail("setup should not run"))
    with pytest.raises(RuntimeError, match="TPU required"):
        run(runner, implementation, devices=(SimpleNamespace(platform="cpu"),), require_tpu=True)


def test_smoke_retains_experimental_status_without_claiming_device_utilization(runner):
    _, implementation = make_session(SimpleNamespace(now=0), np.array([[1., 2.]]))
    implementation.__file__ = "/explicit/experimental_reference.py"
    implementation.IMPLEMENTATION_STATUS = "experimental-unvalidated"
    result = run(runner, implementation, devices=(SimpleNamespace(platform="tpu", device_kind="test TPU"),) * 2)
    assert result["implementation"] == implementation.__file__
    assert result["implementation_status"] == "experimental-unvalidated"
    assert result["requested_device_count"] == 2
    assert result["requested_platforms"] == ["tpu"]
    assert result["requested_device_kinds"] == ["test TPU", "test TPU"]
    assert "device_count" not in result


@pytest.mark.parametrize("phase", ["encrypt", "evaluate", "decrypt", "security", "numerical"])
def test_all_failure_paths_close_session(runner, monkeypatch, phase):
    session, implementation = make_session(SimpleNamespace(now=0), np.array([[1., 2.]]))

    def fail(*args, **kwargs):
        raise ValueError(phase)

    options = {}
    if phase in ("encrypt", "evaluate", "decrypt"):
        setattr(session, phase, fail)
    elif phase == "security":
        monkeypatch.setattr(runner, "require_secure_execution", fail)
    else:
        options["check_output"] = fail
    with pytest.raises(ValueError, match=phase):
        run(runner, implementation, **options)
    assert session.closed


def test_warmups_must_also_pass_numeric_gates(runner):
    session, implementation = make_session(SimpleNamespace(now=0), np.array([[1., np.nan]]))
    with pytest.raises(ValueError, match="Invalid output"):
        run(runner, implementation, warmups=1, repeats=1)
    assert session.requests == 1
    assert session.closed


def test_completed_request_ciphertexts_are_released_before_next_encryption(runner):
    class Payload:
        pass

    class Session:
        previous_input = previous_result = None
        requests = 0

        def encrypt(self, inputs):
            assert self.previous_input is None or self.previous_input() is None
            assert self.previous_result is None or self.previous_result() is None
            value = Payload()
            self.previous_input = weakref.ref(value)
            return value

        def evaluate(self, encrypted, **kwargs):
            self.requests += 1
            value = Payload()
            self.previous_result = weakref.ref(value)
            return value

        def decrypt(self, result):
            # Synchronized server output no longer needs the input payload.
            assert self.previous_input() is None
            return np.array([[1., 2.]])

        def close(self):
            assert self.previous_result() is None

    session = Session()
    run(runner, SimpleNamespace(create_session=lambda *args, **kwargs: session),
        warmups=1, repeats=2)
    assert session.requests == 3


@pytest.mark.parametrize('cleanup_error', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_cleanup_does_not_hide_primary_failure(runner, cleanup_error):
    session, implementation = make_session(SimpleNamespace(now=0), np.array([[1., np.nan]]))

    def broken_close():
        raise cleanup_error("cleanup")

    session.close = broken_close
    with pytest.raises(ValueError, match="Invalid output") as error:
        run(runner, implementation)
    assert error.value.__notes__ == ["Session cleanup also failed: cleanup"]


@pytest.mark.parametrize("problem,filename", [("p2_bert", "bert_he.py"), ("p4_qwen", "qwen_he.py")])
def test_starter_and_oracle_preflight_are_explicit(problem, filename):
    src = ROOT / "agent_pset" / problem / "src"
    for suffix, status in [("", "unimplemented"), (".golden", "plaintext-oracle")]:
        result = subprocess.run([sys.executable, str(src / "runner.py"), "--preflight", "--implementation", str(src / (filename + suffix))],
                                text=True, capture_output=True)
        assert result.returncode == 1, result.stderr
        report = json.loads(result.stdout)
        assert report["implementation_status"] == status
        assert report["ready_for_execution"] is False
        assert report["encryption_validated"] is False


@pytest.mark.parametrize("problem,filename", [("p2_bert", "bert_he.py"), ("p4_qwen", "qwen_he.py")])
def test_goldens_fail_with_oracle_diagnostic_before_jax(problem, filename):
    src = ROOT / "agent_pset" / problem / "src"
    result = subprocess.run([sys.executable, str(src / "runner.py"), "--case", str(src.parent / "test" / "test1.in"),
                             "--implementation", str(src / (filename + ".golden"))],
                            text=True, capture_output=True)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["passed"] is False
    assert "plaintext numerical oracle" in report["error"]
    assert "--reference" in report["error"]


def test_stale_starter_marker_does_not_block_implemented_session(runner, tmp_path):
    module_path = tmp_path / "implementation.py"
    module_path.write_text("IMPLEMENTATION_STATUS = 'unimplemented'\ndef create_session(*args, **kwargs):\n    return 'implemented'\n")
    module = runner.load_encrypted_implementation(module_path)
    assert module.create_session() == "implemented"
