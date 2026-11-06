"""Release all resources after partial setup or individual cleanup failures."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import encrypted_reference as reference


@pytest.mark.parametrize('failure', [None, 'client', 'server', 'jax', 'keys'])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_close_attempts_all_resources_and_is_idempotent(monkeypatch, tmp_path, failure, error_type):
    import jax
    from jaxite_word import ckks_ctx

    events = []
    public_file = tmp_path / 'public-key.npy'
    public_file.write_bytes(b'public test key')

    def record(label):
        events.append(label)
        if label == 'keys':
            public_file.unlink()
        if failure == label:
            raise error_type(f'{label} failed')

    monkeypatch.setattr(ckks_ctx, 'evict_context_caches', record)
    monkeypatch.setattr(jax, 'clear_caches', lambda: record('jax'))
    monkeypatch.setattr(reference.gc, 'collect', lambda: record('gc'))
    session = object.__new__(reference.ToyReferenceSession)
    session.client, session.ctx = 'client', 'server'
    session.key_store = SimpleNamespace(close=lambda: record('keys'))
    session.evaluator = session.ops = session.refresher = object()
    if failure is None:
        session.close()
    else:
        with pytest.raises(error_type, match=f'{failure} failed'):
            session.close()
    assert events == ['client', 'server', 'jax', 'gc', 'keys']
    assert not public_file.exists()
    assert session.closed
    assert all(getattr(session, name) is None for name in
               ('client', 'ctx', 'evaluator', 'ops', 'refresher', 'key_store'))
    session.close()
    assert events == ['client', 'server', 'jax', 'gc', 'keys']


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_setup_error_survives_multiple_cleanup_errors(monkeypatch, error_type):
    import jax
    from jaxite_word import ckks_ctx

    events = []

    def fail(label):
        events.append(label)
        raise error_type(f'{label} failed')

    def setup(session, *args, **kwargs):
        session.client, session.ctx = 'client', 'server'
        session.key_store = SimpleNamespace(close=lambda: fail('keys'))
        raise ValueError('setup failed')

    monkeypatch.setattr(reference.ToyReferenceSession, '__init__', setup)
    monkeypatch.setattr(ckks_ctx, 'evict_context_caches', fail)
    monkeypatch.setattr(jax, 'clear_caches', lambda: events.append('jax'))
    monkeypatch.setattr(reference.gc, 'collect', lambda: events.append('gc'))
    with pytest.raises(ValueError, match='setup failed') as caught:
        reference.create_toy_session('bert', {}, {}, batch=1, sequence_length=1, devices=[])
    assert events == ['client', 'server', 'jax', 'gc', 'keys']
    assert any('client failed' in note for note in caught.value.__notes__)
