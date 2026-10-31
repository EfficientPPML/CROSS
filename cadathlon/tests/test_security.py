"""Actual-ring gates use complete live contexts, including auxiliary P."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxite_word import CKKSContext, Polynomial

_SPEC = importlib.util.spec_from_file_location(
    "contest_security_support", Path(__file__).parents[1] / "common" / "support.py")
support = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(support)


def context(degree=4096, q=(40961, 65537), p=(114689,), **extra):
    return CKKSContext(dict(degree=degree, q_towers=list(q), p_towers=list(p),
                            scaling_factor=65537, **extra))


def ciphertext(ctx, moduli=None, degree=None, components=2):
    moduli = list(ctx.q_towers if moduli is None else moduli)
    degree = ctx.degree if degree is None else degree
    # The gate is structural; no arithmetic tables or encryption are needed.
    ntt = SimpleNamespace(ff_ctx=SimpleNamespace(moduli=moduli), moduli=moduli)
    return Polynomial.from_array(
        np.zeros((1, components, 1, degree, len(moduli)), dtype=np.uint32),
        dict(batch=1, num_elements=components, degree=degree,
             num_moduli=len(moduli), precision=32, degree_layout=(1, degree)),
        dict(moduli=moduli, ntt_ctx=ntt))


def test_accepts_live_context_and_reduced_q_output():
    ctx = context()
    manifests = support.require_secure_execution(
        SimpleNamespace(_mapping=SimpleNamespace(ctx=ctx)),
        {"input": ciphertext(ctx), "output": [ciphertext(ctx, moduli=ctx.q_towers[:1]), 1]})
    assert manifests[0]["degree"] == 4096
    assert manifests[0]["security_bits"] == 128
    assert manifests[0]["p_towers"] == [114689]


def test_rejects_reported_insecure_ring_even_with_secure_declared_profile():
    # Reproduces N=1024 and roughly 1750 bits of QP from the issue.
    ctx = context(degree=1024, q=tuple((1 << 30) - 2*i - 1 for i in range(57)))
    owner = SimpleNamespace(ctx=ctx, ring_config=SimpleNamespace(degree=131072))
    with pytest.raises(ValueError, match="Actual ring fails 128-bit security: N=1024"):
        support.require_secure_execution(owner, ciphertext(ctx))


def test_security_counts_full_q_and_auxiliary_p_not_output_level():
    ctx = context(p=tuple((1 << 30) - 2*i - 1 for i in range(5)))
    with pytest.raises(ValueError, match="128-bit security"):
        support.require_secure_execution(ctx, ciphertext(ctx, moduli=ctx.q_towers[:1]))


@pytest.mark.parametrize("change", [dict(degree=2048), dict(moduli=[40961, 114689])])
def test_ciphertext_must_match_actual_context(change):
    ctx = context()
    with pytest.raises(ValueError, match="Ciphertext ring does not match"):
        support.require_secure_execution(ctx, ciphertext(ctx, **change))


def test_ring_declaration_without_context_is_rejected():
    ctx = context()
    with pytest.raises(ValueError, match="expose live CKKSContext"):
        support.require_secure_execution(SimpleNamespace(ring_config=ctx.parameters), ciphertext(ctx))


def test_checks_lazy_server_context_even_if_client_ring_is_secure():
    client, server = context(), context(degree=1024)
    service = SimpleNamespace(_ctx=client, _mapping=SimpleNamespace(ctx=server))
    with pytest.raises(ValueError, match="128-bit security"):
        support.require_secure_execution(service, ciphertext(client))


def test_custom_wrapper_can_export_actual_contexts():
    ctx = context()
    owner = SimpleNamespace(security_contexts=lambda: [ctx])
    assert support.require_secure_execution(owner, ciphertext(ctx))


@pytest.mark.parametrize("bundle", [[], {}, [1, 2]])
def test_empty_or_metadata_only_bundle_is_rejected(bundle):
    with pytest.raises(ValueError, match="at least one ciphertext"):
        support.require_secure_execution(context(), bundle)


def test_plaintext_polynomial_is_not_a_ciphertext():
    ctx = context()
    with pytest.raises(TypeError, match="at least two components"):
        support.require_secure_execution(ctx, ciphertext(ctx, components=1))


def test_noise_assumption_is_checked():
    ctx = context(sigma=0.1)
    with pytest.raises(ValueError, match="noise sigma"):
        support.require_secure_execution(ctx, ciphertext(ctx))


@pytest.mark.parametrize("insecure_server", [False, True])
def test_suite_stage3_checks_live_lazy_server_and_records_it(monkeypatch, tmp_path,
                                                            insecure_server):
    import json
    import sys
    import jax

    client = context()

    class Service:
        def __init__(self):
            self._ctx = client
            self._mapping = None

        def encrypt(self, images):
            return ciphertext(self._ctx)

        def execute(self, value):
            self._mapping = SimpleNamespace(ctx=context(degree=1024) if insecure_server
                                            else context())
            return ciphertext(self._mapping.ctx)

    paths = SimpleNamespace(pubkeydir=tmp_path, manifest=tmp_path / "ring.json")
    lib = SimpleNamespace(Paths=lambda size: paths, load_keys=lambda paths: {},
                          build_service=lambda *args: Service())
    # Stage 3 must derive the manifest from execution, not lib.pack_program().
    monkeypatch.setitem(sys.modules, "stage_lib", lib)
    monkeypatch.setitem(sys.modules, "bundle_io", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "support", support)
    monkeypatch.setattr(jax, "devices", lambda: ("cpu",))
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.syspath_prepend(str(tmp_path))
    path = (Path(__file__).parents[1] / "non_agent_pset/p1_benchmark/src/suite"
            / "service_daemon.py")
    spec = importlib.util.spec_from_file_location("security_test_daemon", path)
    daemon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daemon)
    server = daemon.Server(size=0, devices=1)
    if insecure_server:
        with pytest.raises(ValueError, match="128-bit security"):
            server.materialize(batch=1)
        assert not server.ready
        assert not paths.manifest.exists()
    else:
        server.materialize(batch=1)
        assert server.ready
        manifest = json.loads(paths.manifest.read_text())
        assert manifest["degree"] == client.degree
        assert len(manifest["execution_rings"]) == 2
        assert all(ring["p_towers"] == list(client.p_towers)
                   for ring in manifest["execution_rings"])
