import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('CLEANUP_SOURCE', HERE / 'candidate/cleanup.py'))


def load():
    spec = importlib.util.spec_from_file_location('tested_cleanup', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Signals:
    SIGINT = signal.SIGINT
    SIGTERM = signal.SIGTERM
    SIGHUP = signal.SIGHUP
    SIG_BLOCK = signal.SIG_BLOCK
    SIG_SETMASK = signal.SIG_SETMASK
    SIG_IGN = signal.SIG_IGN

    def __init__(self):
        self.original = {s: object() for s in (self.SIGINT, self.SIGTERM, self.SIGHUP)}
        self.handlers = dict(self.original)
        self.mask = {signal.SIGUSR1}
        self.old_mask = set(self.mask)
        self.events = []
        self.install_faults = {}
        self.restore_faults = {}
        self.mask_faults = {}
        self.mask_calls = 0

    def getsignal(self, signum):
        return self.handlers[signum]

    def signal(self, signum, handler):
        restoring = handler is self.original[signum]
        self.events.append(('restore' if restoring else 'install', signum))
        previous = self.handlers[signum]
        faults = self.restore_faults if restoring else self.install_faults
        fault = faults.pop(signum, None)
        if fault and not fault[0]:
            raise fault[1]
        self.handlers[signum] = handler
        if fault:
            raise fault[1]
        return previous

    def pthread_sigmask(self, how, values):
        self.mask_calls += 1
        previous = set(self.mask)
        if how == self.SIG_BLOCK:
            self.mask.update(values)
        else:
            self.mask = set(values)
        fault = self.mask_faults.pop(self.mask_calls, None)
        if fault:
            raise fault
        return previous


@pytest.fixture
def fake(monkeypatch):
    module = load()
    signals = Signals()
    monkeypatch.setattr(module, 'signal', signals)
    return module, signals


def restored(signals):
    assert signals.handlers == signals.original
    assert signals.mask == signals.old_mask


def test_cleanup_install_failure_still_runs_body_and_restores_mutated_handler(fake):
    module, signals = fake
    error = OSError('setter failed after mutation')
    signals.install_faults[signal.SIGTERM] = (True, error)
    entered = []
    with pytest.raises(OSError) as caught:
        with module.cleanup_signals():
            entered.append('published')
    assert caught.value is error
    assert entered == ['published']
    restored(signals)


def test_install_failure_still_publishes_owned_terminal_record(fake, tmp_path):
    module, signals = fake
    signals.install_faults[signal.SIGTERM] = (True, OSError('install failure'))
    primary = RuntimeError('stage failed')
    report = tmp_path / 'report.json'
    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        finally:
            with module.cleanup_signals():
                report.write_text(json.dumps({'passed': False, 'cleanup_attempted': True}))
    assert caught.value is primary
    assert json.loads(report.read_text()) == {'passed': False, 'cleanup_attempted': True}
    assert any('install failure' in note for note in primary.__notes__)
    restored(signals)


@pytest.mark.parametrize('scope', ['cleanup_signals', 'termination_signals'])
def test_failed_initial_mask_query_does_not_skip_owned_cleanup(fake, scope):
    module, signals = fake
    signals.mask_faults[1] = OSError('query failed')
    entered = []
    with pytest.raises(OSError):
        with getattr(module, scope)():
            entered.append(True)
    assert bool(entered) == (scope == 'cleanup_signals')
    assert signals.events == []
    restored(signals)


@pytest.mark.parametrize('scope', ['cleanup_signals', 'termination_signals'])
def test_restore_failure_before_mutation_attempts_other_handlers(fake, scope):
    module, signals = fake
    error = OSError('restore cannot finish')
    signals.restore_faults[signal.SIGHUP] = (False, error)
    with pytest.raises(OSError) as caught:
        with getattr(module, scope)():
            pass
    assert caught.value is error
    assert signals.handlers[signal.SIGINT] is signals.original[signal.SIGINT]
    assert signals.handlers[signal.SIGTERM] is signals.original[signal.SIGTERM]
    assert signals.handlers[signal.SIGHUP] is not signals.original[signal.SIGHUP]
    assert signals.mask == signals.old_mask


@pytest.mark.parametrize('scope', ['cleanup_signals', 'termination_signals'])
def test_body_primary_survives_restore_cancellation(fake, scope):
    module, signals = fake
    body = ValueError('body primary')
    signals.restore_faults[signal.SIGINT] = (True, KeyboardInterrupt('cleanup cancel'))
    caught = None
    try:
        with getattr(module, scope)():
            raise body
    except BaseException as error:
        caught = error
    assert caught is body
    assert any('cleanup cancel' in note for note in body.__notes__)
    restored(signals)


@pytest.mark.parametrize('scope', ['cleanup_signals', 'termination_signals'])
def test_restore_failure_attempts_all_handlers_and_mask(fake, scope):
    module, signals = fake
    error = OSError('restore failed after mutation')
    # Fail whichever signal is first in either implementation's restore order.
    signals.restore_faults = {s: (True, error) for s in module.SIGNALS}
    with pytest.raises(OSError):
        with getattr(module, scope)():
            pass
    assert {s for phase, s in signals.events if phase == 'restore'} == set(module.SIGNALS)
    restored(signals)


@pytest.mark.parametrize('cancel', [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('scope', ['cleanup_signals', 'termination_signals'])
def test_cleanup_only_cancel_outranks_ordinary_restore_error(fake, cancel, scope):
    module, signals = fake
    ordinary = OSError('ordinary restore')
    cancellation = cancel('later cancellation')
    signals.restore_faults[signal.SIGHUP] = (True, ordinary)
    signals.restore_faults[signal.SIGTERM] = (True, cancellation)
    with pytest.raises(cancel) as caught:
        with getattr(module, scope)():
            pass
    assert caught.value is cancellation
    assert any('ordinary restore' in note for note in cancellation.__notes__)
    restored(signals)


def test_incoming_primary_survives_body_and_restoration_failures(fake):
    module, signals = fake
    primary = LookupError('original failure')
    signals.restore_faults[signal.SIGHUP] = (True, KeyboardInterrupt('restore cancellation'))
    with pytest.raises(LookupError) as caught:
        try:
            raise primary
        finally:
            with module.cleanup_signals():
                raise OSError('cleanup body error')
    assert caught.value is primary
    assert len(primary.__notes__) == 2
    restored(signals)


def test_incoming_primary_is_not_newly_raised_when_cleanup_body_succeeds(fake):
    module, signals = fake
    primary = LookupError('handled failure')
    signals.restore_faults[signal.SIGHUP] = (True, OSError('restore error'))
    handled = []
    try:
        raise primary
    except LookupError:
        with module.cleanup_signals():
            handled.append(True)
        handled.append(True)
    assert handled == [True, True]
    assert 'restore error' in primary.__notes__[0]
    restored(signals)


@pytest.mark.parametrize('post_mutation', [False, True])
def test_termination_setup_failure_never_yields_work(fake, post_mutation):
    module, signals = fake
    failure = OSError('setup failure')
    signals.install_faults[signal.SIGTERM] = (post_mutation, failure)
    entered = []
    with pytest.raises(OSError) as caught:
        with module.termination_signals():
            entered.append(True)
    assert caught.value is failure and not entered
    restored(signals)


@pytest.mark.parametrize('scope', ['cleanup_signals', 'termination_signals'])
@pytest.mark.parametrize('mask_call', [2, 3])
def test_mask_failure_after_mutation_restores_original(fake, scope, mask_call):
    module, signals = fake
    failure = OSError('mask setter failure')
    signals.mask_faults[mask_call] = failure
    entered = []
    with pytest.raises(OSError):
        with getattr(module, scope)():
            entered.append(True)
    assert bool(entered) == (scope == 'cleanup_signals')
    restored(signals)


def test_deferred_signals_do_not_interrupt_cleanup_body(fake):
    module, signals = fake
    steps = []
    with pytest.raises(KeyboardInterrupt):
        with module.cleanup_signals():
            for _ in range(3):
                for sig in module.SIGNALS:
                    signals.handlers[sig](sig, None)
                    steps.append(sig)
            steps.append('publication')
    assert len(steps) == 10 and steps[-1] == 'publication'
    restored(signals)


def test_termination_handler_is_single_fire(fake):
    module, signals = fake
    with pytest.raises(SystemExit) as caught:
        with module.termination_signals():
            try:
                signals.handlers[signal.SIGTERM](signal.SIGTERM, None)
            finally:
                for sig in module.SIGNALS:
                    signals.handlers[sig](sig, None)
    assert caught.value.code == 128 + signal.SIGTERM
    assert 'repeat signals deferred' in caught.value.__notes__[0]
    restored(signals)


@pytest.mark.parametrize('scope', ['cleanup_signals', 'termination_signals'])
def test_nonmain_thread_does_not_mutate_signal_state(scope):
    module = load()
    result = []
    def worker():
        try:
            with getattr(module, scope)():
                result.append('body')
        except BaseException as error:
            result.append(type(error).__name__)
    thread = threading.Thread(target=worker)
    thread.start();thread.join()
    assert result == (['body', 'RuntimeError'] if scope == 'cleanup_signals' else ['RuntimeError'])


@pytest.mark.parametrize('mode', ['repeated_process', 'pending_main', 'body_primary', 'nested'])
def test_real_signals_in_isolated_single_thread_process(mode):
    script = HERE / 'signal_probe.py'
    result = subprocess.run([sys.executable, str(script), str(SOURCE), mode],
                            capture_output=True, text=True, timeout=15,
                            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS' in result.stdout
