"""Cold setup must remain visible when smoke runners exclude warmup samples."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("problem,prepared", [
    ("p1_benchmark", False), ("p3_cnn", False), ("p3_cnn", True),
])
@pytest.mark.parametrize("warmups", [0, 1])
def test_cold_request_and_total_wall_time(monkeypatch, problem, prepared, warmups):
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
            assert not self.released, 'release is not required to be idempotent'
            clock.now += 7.0
            self.released = True

    service = Service()

    def build(*args, **kwargs):
        assert not prepared, 'a prepared Mapping must not be constructed twice'
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
    path = (Path(__file__).resolve().parents[1] / "non_agent_pset" /
            problem / "src" / "runner.py")
    spec = importlib.util.spec_from_file_location(f"timing_{problem}", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    kwargs = {"prepared": (service, 10.0)} if prepared else {}
    result = runner.run(SimpleNamespace(
        case=Path("test1.in"), reference=False, baseline=True, warmups=warmups, repeats=2), **kwargs)

    assert service.released is not prepared
    assert result["setup_s"] == 10.0
    assert result["cold_start_included"] is (warmups == 0)
    assert result["warmup_s"] == ([104.0] if warmups else [])
    assert result["warmup_server_s"] == ([100.0] if warmups else [])
    assert result["end_to_end_s"] == ([6.0, 6.0] if warmups else [104.0, 6.0])
    assert result["server_s"] == ([2.0, 2.0] if warmups else [100.0, 2.0])
    assert result["median_s"] == (6.0 if warmups else 55.0)
    # A prepared mapping is borrowed: its suite owner measures construction
    # and cleanup outside this invocation and releases it exactly once.
    assert result["total_s"] == (138.0 if warmups else 132.0) - (17.0 if prepared else 0.0)
    if prepared:
        assert result["mapping_release_included"] is False
        service.release()
