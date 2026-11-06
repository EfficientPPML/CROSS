"""Engine mask transitions with post-mutation failures; no native backend."""
import signal
import pytest
import full_model_engine as engine
from test_full_model_engine import fake


class SignalState:
    SIGINT = signal.SIGINT
    SIGTERM = signal.SIGTERM
    SIGHUP = signal.SIGHUP
    SIG_BLOCK = signal.SIG_BLOCK
    SIG_SETMASK = signal.SIG_SETMASK

    def __init__(self, phase, error, query=False):
        self.mask = {signal.SIGUSR1}
        self.original_mask = set(self.mask)
        self.handlers = {s: object() for s in (self.SIGINT, self.SIGTERM, self.SIGHUP)}
        self.original_handlers = dict(self.handlers)
        self.phase = phase
        self.error = error
        self.query = query
        self.blocks = self.queries = 0
        self.calls = []

    def getsignal(self, signum):
        return self.handlers[signum]

    def signal(self, signum, handler):
        original = self.handlers[signum]
        self.handlers[signum] = handler
        return original

    def pthread_sigmask(self, how, values):
        old = set(self.mask)
        self.calls.append((how, tuple(values)))
        if how == self.SIG_BLOCK:
            self.mask.update(values)
            if values:
                self.blocks += 1
                if not self.query and self.blocks == self.phase:
                    raise self.error
            else:
                self.queries += 1
                if self.query and self.queries == self.phase:
                    raise self.error
        else:
            self.mask = set(values)
        return old


@pytest.mark.parametrize('phase', [1, 2], ids=['installation', 'restoration'])
@pytest.mark.parametrize('kind', [OSError, KeyboardInterrupt])
@pytest.mark.parametrize('body_failure', [False, True])
def test_post_mutation_block_failure_restores_previous_mask(fake, monkeypatch, phase, kind, body_failure):
    ctx, calls, _ = fake
    error = kind('block changed the mask before raising')
    signals = SignalState(phase, error)
    monkeypatch.setattr(engine, 'signal', signals)
    primary = ValueError('stage primary')
    caught = None
    try:
        with engine.server_engine(ctx, manifest_bytes=b'fixed'):
            if body_failure:
                raise primary
    except BaseException as raised:
        caught = raised
    assert caught is (primary if body_failure else error)
    assert calls[-1] == ('close_provider',)
    assert len([c for c in calls if c[0] == 'clear']) == 17
    assert signals.mask == signals.original_mask
    assert signals.handlers == signals.original_handlers
    if body_failure:
        assert any('block changed' in note for note in primary.__notes__)


@pytest.mark.parametrize('phase', [1, 2], ids=['installation', 'restoration'])
@pytest.mark.parametrize('body_failure', [False, True])
def test_failed_mask_query_does_not_mutate_unknown_mask(fake, monkeypatch, phase, body_failure):
    ctx, calls, _ = fake
    error = OSError('query failed without changing mask')
    signals = SignalState(phase, error, query=True)
    monkeypatch.setattr(engine, 'signal', signals)
    primary = RuntimeError('body failure')
    caught = None
    try:
        with engine.server_engine(ctx, manifest_bytes=b'fixed'):
            if body_failure:
                raise primary
    except BaseException as raised:
        caught = raised
    assert caught is (primary if body_failure else error)
    assert calls[-1] == ('close_provider',)
    assert len([c for c in calls if c[0] == 'clear']) == 17
    assert signals.mask == signals.original_mask
    assert signals.handlers == signals.original_handlers
    # No mutating block is allowed without a successful query for that phase.
    assert signals.blocks == (0 if phase == 1 else 1)
