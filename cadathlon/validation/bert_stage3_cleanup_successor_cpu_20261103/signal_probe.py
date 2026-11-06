"""Real signal probe in a fresh Python process without NumPy/JAX/worker threads."""
import importlib.util
import os
import signal
import sys
import threading

source, mode = sys.argv[1:]
spec = importlib.util.spec_from_file_location('signal_cleanup_candidate', source)
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert threading.active_count() == 1
originals = {s: signal.getsignal(s) for s in module.SIGNALS}
mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
steps = []
primary = ValueError('original stage failure')

if mode in ('repeated_process', 'body_primary'):
    try:
        with module.cleanup_signals():
            for _ in range(4):
                for signum in module.SIGNALS:
                    os.kill(os.getpid(), signum)
                    steps.append(signum)
            steps.append('published')
            if mode == 'body_primary':
                raise primary
    except BaseException as error:
        assert (error is primary) if mode == 'body_primary' else isinstance(error, KeyboardInterrupt)
    else:
        raise AssertionError('cancellation was lost')
    assert steps[-1] == 'published' and len(steps) == 13
elif mode == 'pending_main':
    def raising_handler(signum, frame):
        steps.append('caller handler')
        raise KeyboardInterrupt('pending caller cancellation')
    signal.signal(signal.SIGINT, raising_handler)
    originals[signal.SIGINT] = raising_handler
    try:
        with module.cleanup_signals():
            signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT,))
            signal.pthread_kill(threading.get_ident(), signal.SIGINT)
            steps.append('published')
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('pending cancellation was lost')
    assert steps == ['published', 'caller handler']
elif mode == 'nested':
    try:
        with module.termination_signals():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            finally:
                with module.cleanup_signals():
                    for _ in range(4):
                        for signum in module.SIGNALS:
                            os.kill(os.getpid(), signum)
                    steps.append('published')
    except SystemExit as error:
        assert error.code == 128 + signal.SIGTERM
    else:
        raise AssertionError('primary termination was lost')
    assert steps == ['published']
else:
    raise AssertionError(mode)

assert {s: signal.getsignal(s) for s in module.SIGNALS} == originals
assert signal.pthread_sigmask(signal.SIG_BLOCK, ()) == mask
print('PASS', mode)
