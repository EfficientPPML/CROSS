"""Prepared mappings remain owned by the enclosing competition evaluation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def load_runner(monkeypatch, service, *, prepared):
    expected = np.array([[.5, .2]])
    def load(path):
        assert not prepared, 'borrowed mapping must not reimport the implementation'
        return SimpleNamespace(build_mapping=lambda *a, **k: service)
    support = SimpleNamespace(
        cross_root=lambda: None, load_module=load,
        read_json=lambda path: ({'logits': expected.tolist()} if path.suffix == '.out' else {'batch': 1}),
        require_ciphertexts=lambda value: None, require_secure_execution=lambda *a: [],
        synchronize=lambda value: None,
        metrics=lambda actual, wanted: {'top1_agreement': 1., 'max_abs_error': 0., 'nrmse': 0.})
    monkeypatch.setitem(sys.modules, 'support', support)
    monkeypatch.setitem(sys.modules, 'reference', SimpleNamespace(
        make_case=lambda case: (object(), (1, 2), expected, expected)))
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(
        config=SimpleNamespace(update=lambda *a: None), devices=lambda: ('device',)))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    path = Path(__file__).resolve().parents[1] / 'non_agent_pset/p3_cnn/src/runner.py'
    spec = importlib.util.spec_from_file_location('cnn_lifecycle_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, SimpleNamespace(case=Path('test1.in'), reference=False, baseline=False, warmups=0, repeats=1)


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('cleanup_error', [None, RuntimeError, KeyboardInterrupt, SystemExit])
def test_request_failure_releases_only_owned_mapping_and_preserves_primary(monkeypatch, prepared, cleanup_error):
    class Service:
        release_calls = 0
        def encrypt_input(self, value):
            return object()
        def execute(self, value):
            raise ValueError('request failed')
        def release(self):
            self.release_calls += 1
            if cleanup_error:
                raise cleanup_error('cleanup failed')
    service = Service()
    runner, args = load_runner(monkeypatch, service, prepared=prepared)
    with pytest.raises(ValueError, match='request failed') as error:
        runner.run(args, **({'prepared': (service, 10.)} if prepared else {}))
    assert service.release_calls == (0 if prepared else 1)
    assert getattr(error.value, '__notes__', []) == (
        ['Mapping cleanup also failed: cleanup failed'] if cleanup_error and not prepared else [])


def test_owned_cleanup_failure_prevents_success(monkeypatch):
    class Service:
        def encrypt_input(self, value):
            return object()
        def execute(self, value):
            return object()
        def decrypt_output(self, value):
            return np.array([[.5, .2]])
        def release(self):
            raise RuntimeError('cleanup failed')
    runner, args = load_runner(monkeypatch, Service(), prepared=False)
    with pytest.raises(RuntimeError, match='cleanup failed'):
        runner.run(args)
