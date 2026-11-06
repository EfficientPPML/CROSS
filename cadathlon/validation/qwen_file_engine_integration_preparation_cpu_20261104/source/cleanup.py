"""Local signal scopes for dedicated stage collectors.

Cleanup bodies always run, including after shield-installation errors. Signal
masks protect main-thread transitions; they do not constitute a process-wide
atomicity guarantee for arbitrary native worker threads.
"""
from contextlib import contextmanager
import signal
import sys
import threading
from cleanup_origin import mark,select,protect_body

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_MISSING = object()


def _attempt(errors, label, function):
    protect_body(sys.exception())
    try:
        return function()
    except BaseException as error:
        errors.append((label, mark(error)))
        return _MISSING


def _finish(primary, errors):
    chosen=select(primary,errors)
    if chosen is not None and chosen is not primary:raise chosen


def _capture_mask(errors):
    if threading.current_thread() is not threading.main_thread():
        errors.append(('signal thread', RuntimeError('signal scopes require the main thread')))
        return _MISSING
    # Query first: even a subsequently partially mutating block/setter failure
    #must not lose the mask that needs restoring.
    return _attempt(errors, 'capture signal mask',
                    lambda: signal.pthread_sigmask(signal.SIG_BLOCK, ()))


def _install(originals, errors, handler):
    for signum in SIGNALS:
        original = _attempt(errors, 'capture signal handler', lambda s=signum: signal.getsignal(s))
        if original is _MISSING:
            continue
        # A signal setter can mutate process state before raising.
        originals.append((signum, original))
        _attempt(errors, 'install signal handler', lambda s=signum: signal.signal(s, handler))


def _restore(originals, previous_mask, errors):
    if previous_mask is _MISSING:
        return
    _attempt(errors, 'block restoration signals',
             lambda: signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS))
    for signum, original in reversed(originals):
        _attempt(errors, 'restore signal handler', lambda s=signum, h=original: signal.signal(s, h))
    # All handlers have been attempted before pending signals can call a
    #restored caller handler. Record a raised caller cancellation, too.
    _attempt(errors, 'restore signal mask',
             lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))


@contextmanager
def cleanup_signals():
    """Defer cancellation through cleanup/publication, then report all failures."""
    incoming = protect_body(sys.exception())
    caught = None
    errors = []
    originals = []
    deferred = []
    previous_mask = _capture_mask(errors)

    def remember(signum, frame):
        deferred.append(signum)

    try:
        if previous_mask is not _MISSING:
            _attempt(errors, 'block installation signals',
                     lambda: signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS))
            _install(originals, errors, remember)
            if not errors:
                _attempt(errors, 'release installation mask',
                         lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))
        # An installation failure must never skip this owned cleanup body.
        try:
            yield
        except BaseException as error:
            caught = protect_body(error)
    finally:
        _restore(originals, previous_mask, errors)
    if deferred:
        errors.append(('deferred cleanup cancellation',
                       KeyboardInterrupt('signals deferred through cleanup: ' + repr(deferred))))
    if incoming is not None:
        if caught is not None and caught is not incoming:
            errors.insert(0, ('cleanup body', caught))
        _finish(incoming, errors)
        if caught is not None:
            raise incoming
    elif caught is not None:
        _finish(caught, errors)
        raise caught
    else:
        _finish(None, errors)


@contextmanager
def termination_signals():
    """Install cancellation before work; repeat signals cannot reenter it."""
    incoming = protect_body(sys.exception())
    primary = None
    errors = []
    originals = []
    deferred = []
    terminating = False
    previous_mask = _capture_mask(errors)

    def terminate(signum, frame):
        nonlocal terminating
        if terminating:
            deferred.append(signum)
            return
        terminating = True
        if signum == signal.SIGINT:
            raise KeyboardInterrupt('stage component interrupted')
        raise SystemExit(128 + signum)

    try:
        if previous_mask is not _MISSING:
            _attempt(errors, 'block installation signals',
                     lambda: signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS))
            _install(originals, errors, terminate)
            if not errors:
                _attempt(errors, 'release installation mask',
                         lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))
        _finish(None, errors)
        yield
    except BaseException as error:
        primary = protect_body(error)
    finally:
        _restore(originals, previous_mask, errors)
    if deferred and primary is not None:
        primary.add_note('repeat signals deferred: ' + repr(deferred))
    if incoming is not None and primary is not None:
        if primary is not incoming:
            errors.insert(0, ('termination scope', primary))
        _finish(incoming, errors)
        raise incoming
    if primary is not None:
        _finish(primary, errors)
        raise primary
    _finish(incoming, errors)
