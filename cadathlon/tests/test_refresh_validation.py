"""Fail-closed ownership checks for the explicit refresh validation driver."""
from pathlib import Path
import json
import os
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
import check_secure_refresh as checker


@pytest.mark.parametrize('after', [{'core.py': 'changed'}, {}, {'core.py': 'same', 'new.py': 'added'}])
def test_source_drift_fails_even_when_numerical_gates_pass(after):
    result = checker.measurement_status(output_finite=True, error=1e-5, refresh_error=1e-5,
                                        before={'core.py': 'same'}, after=after)
    assert result['numerical_passed'] is True
    assert result['source_hashes_unchanged'] is False
    assert result['passed'] is False


@pytest.mark.parametrize('error', [1e-3, float('nan'), float('inf'), -.1])
def test_provenance_cannot_override_failed_numerical_gate(error):
    result = checker.measurement_status(output_finite=True, error=error, refresh_error=1e-5,
                                        before={'core.py': 'same'}, after={'core.py': 'same'})
    assert result['source_hashes_unchanged'] is True
    assert result['numerical_passed'] is False and result['passed'] is False


def test_snapshot_includes_lazy_runtime_dependencies():
    hashes = checker.source_hashes()
    assert {'jaxite_word/polynomial.py', 'jaxite_word/ntt_mm.py', 'cadathlon/common/support.py',
            'cadathlon/validation/check_secure_refresh.py'} <= hashes.keys()


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failed_attempt_invalidates_previous_pass_before_running(monkeypatch, tmp_path, error_type):
    report_path = tmp_path / 'report.json'
    report_path.write_text('{"passed": true}')
    def fail(**kwargs):
        assert json.loads(report_path.read_text())['passed'] is False
        raise error_type('interrupted run')
    monkeypatch.setattr(checker, 'run', fail)
    monkeypatch.setattr(sys, 'argv', ['check_secure_refresh.py', '--report', str(report_path)])
    with pytest.raises(error_type, match='interrupted run'):
        checker.main()
    assert json.loads(report_path.read_text())['passed'] is False


@pytest.mark.parametrize('sig', [signal.SIGTERM, signal.SIGHUP])
def test_termination_uses_cleanup_path_and_restores_handlers(sig):
    before = {value: signal.getsignal(value) for value in (signal.SIGTERM, signal.SIGHUP)}
    with pytest.raises(InterruptedError, match='cancelled by signal'):
        with checker.cancellation_handlers():
            os.kill(os.getpid(), sig)
    assert {value: signal.getsignal(value) for value in before} == before


@pytest.mark.parametrize('fail_on_context', [1, 2])
def test_disk_public_keys_close_after_partial_context_setup(monkeypatch, tmp_path, fail_on_context):
    from jaxite_word import ckks_ctx, key_gen
    import jaxite_word

    events = []
    public_file = tmp_path / 'public-key-placeholder.npy'
    public_file.write_bytes(b'public test fixture')

    def close_store():
        events.append('key store closed')
        public_file.unlink()

    store = SimpleNamespace(keys={}, required_bytes=public_file.stat().st_size, close=close_store)
    monkeypatch.setattr(checker, 'ClientRotationKeyStore', SimpleNamespace(generate=lambda *a, **k: store))
    # Crypto arithmetic is deliberately outside this setup-failure unit. The
    # real secure primitive and optional bootstrap checks live separately.
    monkeypatch.setattr(key_gen, 'gen_pke_pair', lambda *a, **k: {'secret_key': object()})
    monkeypatch.setattr(key_gen, 'gen_evaluation_key', lambda *a, **k:
                        {'a': np.zeros((1, 2, 2), np.uint32), 'b': np.zeros((1, 2, 2), np.uint32)})
    attempts = 0

    def context(_):
        nonlocal attempts
        attempts += 1
        if attempts == fail_on_context:
            raise RuntimeError('context setup failed')
        return SimpleNamespace()

    monkeypatch.setattr(jaxite_word, 'CKKSContext', context)
    monkeypatch.setattr(ckks_ctx, 'evict_context_caches', lambda _: events.append('context released'))
    with pytest.raises(RuntimeError, match='context setup failed'):
        checker.run(disk_keys=True)
    assert events == ['context released'] * (fail_on_context - 1) + ['key store closed']
    assert not public_file.exists()


@pytest.mark.parametrize('primary_error', [None, ValueError('evaluation failed')])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_cleanup_attempts_every_context_and_key_store(monkeypatch, primary_error, error_type):
    from jaxite_word import ckks_ctx

    events = []

    def release(context):
        events.append(context)
        if context == 'second':
            raise error_type('context release failed')

    monkeypatch.setattr(ckks_ctx, 'evict_context_caches', release)
    store = SimpleNamespace(close=lambda: events.append('key store'))
    contexts = ['first', 'second']
    if primary_error is None:
        with pytest.raises(error_type, match='context release failed'):
            checker._cleanup(contexts, store, None)
    else:
        checker._cleanup(contexts, store, primary_error)
        assert any('context release failed' in note for note in primary_error.__notes__)
    assert contexts == []
    assert events == ['second', 'first', 'key store']
