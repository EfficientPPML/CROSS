"""A multi-device fixture must not pass by silently running on fewer devices."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("requested", [2, 8, 0, -1, True, 1.5])
def test_p1_rejects_missing_or_invalid_devices_before_build(monkeypatch, requested):
    expected = np.array([[0.5, 0.2]])
    load = Mock(side_effect=AssertionError("must reject before loading submission"))
    support = SimpleNamespace(
        cross_root=lambda: None, load_module=load,
        read_json=lambda path: ({"logits": expected.tolist()} if path.suffix == ".out"
                                else {"batch": 1, "devices": requested}),
        require_ciphertexts=lambda value: None,
        require_secure_execution=lambda *args: None,
        synchronize=lambda value: None,
        metrics=lambda actual, wanted: {"top1_agreement": 1.0},
    )
    monkeypatch.setitem(sys.modules, "support", support)
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(
        make_case=lambda case: (object(), (1, 2), expected, expected)))
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(
        config=SimpleNamespace(update=lambda *args: None), devices=lambda: ("cpu",)))
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = (Path(__file__).parents[1] / "non_agent_pset/p1_benchmark/src/runner.py")
    spec = importlib.util.spec_from_file_location("device_test_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with pytest.raises((ValueError, RuntimeError), match="devices"):
        runner.run(SimpleNamespace(case=Path("test1.in"), reference=False, baseline=True,
                                   warmups=0, repeats=1))
    load.assert_not_called()
