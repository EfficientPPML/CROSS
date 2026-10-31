"""Warm the fixed physical lane group without encrypting the ranked batch."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / "non_agent_pset" / "p1_benchmark" / "src"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("lanes", [1, 8])
def test_baseline_warms_one_complete_lane_group(lanes):
    baseline = load(ROOT / "baseline.py", "baseline_warmup")
    service = baseline.BaselineService.__new__(baseline.BaselineService)
    service.batch, service.lanes = 1000, lanes
    logical_shape = (1, 28, 28)
    packed_slots = object()
    service.packed = SimpleNamespace(
        logical_input_shape=logical_shape,
        ring_config=SimpleNamespace(scaling_factor=2**40),
        pack=Mock(return_value=packed_slots))
    ciphertext = object()
    codec = SimpleNamespace(encrypt_slots=Mock(return_value=ciphertext))
    service._context = Mock(return_value=codec)
    service.encrypt = Mock(side_effect=AssertionError("full batch encryption is unnecessary"))

    warm = service.warmup_input()

    assert len(warm) == lanes
    assert all(value is ciphertext for value in warm)
    service.encrypt.assert_not_called()
    codec.encrypt_slots.assert_called_once_with(packed_slots, scale=float(2**40))
    zeros = service.packed.pack.call_args.args[0]
    assert zeros.shape == logical_shape
    assert np.count_nonzero(zeros) == 0


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    paths = SimpleNamespace(pubkeydir=tmp_path / "keys", manifest=tmp_path / "manifest.json")
    service = SimpleNamespace(encrypt=Mock(return_value=["full-input"]),
                              execute=Mock(return_value=["output"]))
    lib = SimpleNamespace(Paths=lambda size: paths, load_keys=lambda paths: {},
                          build_service=lambda *args: service)
    checks = Mock(return_value=[{"security_bits": 128}])
    monkeypatch.setitem(sys.modules, "stage_lib", lib)
    monkeypatch.setitem(sys.modules, "bundle_io", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "support", SimpleNamespace(require_secure_execution=checks))
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(
        devices=lambda: tuple(range(8)), block_until_ready=lambda value: None))
    monkeypatch.setattr(sys, "path", list(sys.path))
    module = load(ROOT / "suite" / "service_daemon.py", "warmup_daemon")
    return module.Server(2, 8), service, checks


def test_daemon_uses_hook_and_checks_both_boundaries(daemon):
    server, service, checks = daemon
    warm = ["warm-input"] * 8
    service.warmup_input = Mock(return_value=warm)
    server.materialize(batch=1000)
    service.warmup_input.assert_called_once_with()
    service.encrypt.assert_not_called()
    service.execute.assert_called_once_with(warm)
    assert checks.call_args_list[0].args == (service, warm)
    assert checks.call_args_list[1].args == (service, (warm, ["output"]))
    assert server.ready
    assert server.batch == 1000


def test_daemon_falls_back_to_full_batch_without_hook(daemon):
    server, service, checks = daemon
    server.materialize(batch=1000)
    zeros = service.encrypt.call_args.args[0]
    assert zeros.shape == (1000, 784)
    assert np.count_nonzero(zeros) == 0
    service.execute.assert_called_once_with(["full-input"])
    assert checks.call_count == 2
    assert server.ready


@pytest.mark.parametrize("reject_output", [False, True])
def test_hook_does_not_bypass_security_checks(daemon, reject_output):
    server, service, checks = daemon
    service.warmup_input = Mock(return_value=["warm-input"] * 8)
    error = ValueError("insecure execution ring")
    checks.side_effect = [[{"security_bits": 128}], error] if reject_output else error
    with pytest.raises(ValueError, match="insecure execution ring"):
        server.materialize(batch=1000)
    assert not server.ready
    assert not server.paths.manifest.exists()
    assert service.execute.call_count == int(reject_output)
