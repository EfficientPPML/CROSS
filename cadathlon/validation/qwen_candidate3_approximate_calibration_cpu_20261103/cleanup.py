"""Local cleanup signal policy; shared live collectors are left unchanged."""
from contextlib import contextmanager
import signal
SIGNALS=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)

@contextmanager
def cleanup_signals():
    """Do not let repeated cancellation interrupt owned cleanup/publication."""
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);old={}
    try:
        for signum in SIGNALS:old[signum]=signal.signal(signum,signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        for signum,handler in old.items():signal.signal(signum,handler)
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)

@contextmanager
def termination_signals():
    old={s:signal.getsignal(s) for s in SIGNALS}
    def terminate(signum,frame):
        for s in SIGNALS:signal.signal(s,signal.SIG_IGN)
        if signum==signal.SIGINT:raise KeyboardInterrupt('key component interrupted')
        raise SystemExit(128+signum)
    try:
        for s in SIGNALS:signal.signal(s,terminate)
        yield
    finally:
        for s,handler in old.items():signal.signal(s,handler)
