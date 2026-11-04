"""Cold setup must remain visible when smoke runners exclude warmup samples."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("problem", ["p1_benchmark", "p3_cnn"])
@pytest.mark.parametrize("warmups", [0, 1])
def test_cold_request_and_total_wall_time(monkeypatch, problem, warmups):
    clock = SimpleNamespace(now=0.0)
    expected = np.array([[0.5, 0.2]])

    class Service:
        requests = 0
        released = False

        def encrypt(self, value):
            clock.now += 1.0
            return object()

        encrypt_input = encrypt

        def execute(self, value):
            clock.now += 100.0 if self.requests == 0 else 2.0
            self.requests += 1
            return object()

        def decrypt(self, value):
            clock.now += 3.0
            return expected

        decrypt_output = decrypt

        def release(self):
            clock.now += 7.0
            self.released = True

    service = Service()

    def build(*args, **kwargs):
        clock.now += 10.0
        return service

    def make_case(case):
        clock.now += 5.0
        return object(), (1, 2), expected, expected

    implementation = SimpleNamespace(build_service=build, build_mapping=build)
    support = SimpleNamespace(
        cross_root=lambda: None, load_module=lambda path: implementation,
        read_json=lambda path: ({"logits": expected.tolist()} if path.suffix == ".out"
                                else {"batch": 1, "devices": 1}),
        require_ciphertexts=lambda value: None,
        require_secure_execution=lambda *args: None,
        synchronize=lambda value: None,
        metrics=lambda actual, wanted: {"top1_agreement": 1.0},
    )
    monkeypatch.setitem(sys.modules, "support", support)
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(make_case=make_case))
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(
        config=SimpleNamespace(update=lambda *args: None), devices=lambda: ("device",)))
    # Each runner inserts the shared support path while importing.
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = (Path(__file__).resolve().parents[1] / "agent_pset" /
            problem / "src" / "runner.py")
    spec = importlib.util.spec_from_file_location(f"timing_{problem}", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    result = runner.run(SimpleNamespace(
        case=Path("test1.in"), reference=False, baseline=True, warmups=warmups, repeats=2))

    assert service.released
    assert result["setup_s"] == 10.0
    assert result["cold_start_included"] is (warmups == 0)
    assert result["warmup_s"] == ([104.0] if warmups else [])
    assert result["warmup_server_s"] == ([100.0] if warmups else [])
    assert result["end_to_end_s"] == ([6.0, 6.0] if warmups else [104.0, 6.0])
    assert result["server_s"] == ([2.0, 2.0] if warmups else [100.0, 2.0])
    assert result["median_s"] == (6.0 if warmups else 55.0)
    # Include fixture preparation and release, even though neither is part
    # of the historical constructor or request timing fields.
    assert result["total_s"] == (138.0 if warmups else 132.0)
