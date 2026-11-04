"""Regress the standalone baseline's independently initialized key pairs."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def baseline(monkeypatch):
    path = (Path(__file__).resolve().parents[1] / "agent_pset" /
            "p1_benchmark" / "src" / "baseline.py")
    spec = importlib.util.spec_from_file_location("p1_baseline_keys", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = []

    def runtime(ring, *, keys=None):
        if keys is None:
            keys = {"public_key": object(), "secret_key": object()}
            generated.append(keys)
        return dict(keys)

    mapping_factory = Mock(side_effect=lambda *a, **kw: SimpleNamespace(
        keys=kw["keys"], release=Mock()))
    monkeypatch.setitem(sys.modules, "mapping", SimpleNamespace(
        ring_runtime_parameters=runtime))
    monkeypatch.setitem(sys.modules, "ckks_ctx", SimpleNamespace(
        CKKSContext=lambda params: SimpleNamespace(**params),
        evict_context_caches=Mock()))
    monkeypatch.setitem(sys.modules, "jaxite_word", SimpleNamespace(
        Mapping=mapping_factory))

    # Bypass model vectorization: this regression concerns ownership of the
    # codec/evaluator keys, not frontend or cryptographic arithmetic.
    service = module.BaselineService.__new__(module.BaselineService)
    service.packed = SimpleNamespace(ring_config=object())
    service.lanes = 1
    service.devices = (object(),)
    service.keys = None
    service._ctx = service._mapping = None
    return service, generated


@pytest.mark.parametrize("server_first", [False, True])
def test_lazy_client_and_server_share_one_pair(baseline, server_first):
    service, generated = baseline
    if server_first:
        service._compiled()
    context = service._context()
    evaluator = service._compiled()
    assert len(generated) == 1
    for name in ("public_key", "secret_key"):
        assert getattr(context, name) is evaluator.keys[name]


def test_external_pair_is_preserved(baseline):
    service, generated = baseline
    keys = {"public_key": object(), "secret_key": object()}
    service.keys = keys
    assert service._compiled().keys is keys
    assert service._context().secret_key is keys["secret_key"]
    assert not generated


def test_release_rebuild_keeps_the_same_identity(baseline):
    service, generated = baseline
    first = service._context()
    evaluator = service._compiled()
    service.release()
    evaluator.release.assert_called_once()
    sys.modules["ckks_ctx"].evict_context_caches.assert_called_once_with(first)
    assert service._context().secret_key is first.secret_key
    assert service._compiled().keys["secret_key"] is first.secret_key
    assert len(generated) == 1


@pytest.mark.parametrize("release_fails", [False, True])
def test_release_evicts_client_caches_even_if_mapping_release_fails(monkeypatch, release_fails):
    from jaxite_word import ckks_ctx

    path = (Path(__file__).resolve().parents[1] / "agent_pset" /
            "p1_benchmark" / "src" / "baseline.py")
    spec = importlib.util.spec_from_file_location("p1_baseline_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "ckks_ctx", ckks_ctx)
    client_ctx, other_ctx = object(), object()
    other_encrypt, other_decrypt = object(), object()
    encrypt_cache = {(id(client_ctx), 4): object(), (id(other_ctx), 4): other_encrypt}
    decrypt_cache = {(id(client_ctx), 1): object(), (id(other_ctx), 1): other_decrypt}
    monkeypatch.setattr(ckks_ctx, "_encrypt_cache", encrypt_cache)
    monkeypatch.setattr(ckks_ctx, "_decrypt_cache", decrypt_cache)
    service = module.BaselineService.__new__(module.BaselineService)
    service._ctx = client_ctx
    release = Mock(side_effect=RuntimeError("release failed") if release_fails else None)
    service._mapping = SimpleNamespace(release=release)
    keys = {"public_key": object(), "secret_key": object()}
    service.keys = keys

    if release_fails:
        with pytest.raises(RuntimeError, match="release failed"):
            service.release()
    else:
        service.release()
    assert service._ctx is None
    assert service._mapping is None
    assert encrypt_cache == {(id(other_ctx), 4): other_encrypt}
    assert decrypt_cache == {(id(other_ctx), 1): other_decrypt}
    assert service.keys is keys
    service.release()  # Cleanup is idempotent even after a failed Mapping release.
    release.assert_called_once()
