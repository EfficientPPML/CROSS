"""Real pending signals and post-mutation setter fault regressions."""
import os
import signal
import pytest
import full_model_engine as engine
from test_full_model_engine import fake

SIGS=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)


@pytest.mark.parametrize('failure',['real_pending','setter_after_mutation'])
def test_installation_is_atomic_and_every_mutation_restored(fake,monkeypatch,failure):
    ctx,calls,state=fake
    old={s:signal.getsignal(s) for s in SIGS};real_set=signal.signal
    beforemask=signal.pthread_sigmask(signal.SIG_BLOCK,[])
    def earlier_handler(signum,frame):raise KeyboardInterrupt('old handler during install')
    for s in SIGS:real_set(s,earlier_handler)
    injected=[]
    def mutate_then_fail(signum,handler):
        result=real_set(signum,handler)
        if signum==signal.SIGINT and handler is not earlier_handler and not injected:
            injected.append(True)
            if failure=='real_pending':os.kill(os.getpid(),signal.SIGTERM)
            else:raise OSError('setter failed after changing process handler')
        return result
    monkeypatch.setattr(signal,'signal',mutate_then_fail)
    try:
        with pytest.raises(KeyboardInterrupt if failure=='real_pending' else OSError):
            with engine.server_engine(ctx,manifest_bytes=b'fixed'):pass
        assert injected==[True]
        assert all(signal.getsignal(s) is earlier_handler for s in SIGS)
        assert signal.pthread_sigmask(signal.SIG_BLOCK,[])==beforemask
        assert calls[-1]==('close_provider',)
        assert len([c for c in calls if c[0]=='clear'])==17
    finally:
        for s,h in old.items():real_set(s,h)
        signal.pthread_sigmask(signal.SIG_SETMASK,beforemask)


@pytest.mark.parametrize('body_primary',[False,True])
def test_signal_arriving_during_restoration_preserves_primary_and_final_mask(fake,monkeypatch,body_primary):
    ctx,calls,state=fake
    old={s:signal.getsignal(s) for s in SIGS};real_set=signal.signal
    beforemask=signal.pthread_sigmask(signal.SIG_BLOCK,[]);primary=ValueError('original primary')
    def earlier_handler(signum,frame):raise KeyboardInterrupt('old handler after cleanup')
    for s in SIGS:real_set(s,earlier_handler)
    injected=[]
    def restore_then_signal(signum,handler):
        result=real_set(signum,handler)
        if signum==signal.SIGHUP and handler is earlier_handler and not injected:
            injected.append(True);os.kill(os.getpid(),signal.SIGHUP)
        return result
    monkeypatch.setattr(signal,'signal',restore_then_signal)
    try:
        with pytest.raises(ValueError if body_primary else KeyboardInterrupt) as caught:
            with engine.server_engine(ctx,manifest_bytes=b'fixed'):
                if body_primary:raise primary
        if body_primary:assert caught.value is primary and any('old handler after cleanup' in n for n in primary.__notes__)
        assert all(signal.getsignal(s) is earlier_handler for s in SIGS)
        assert signal.pthread_sigmask(signal.SIG_BLOCK,[])==beforemask
        assert calls[-1]==('close_provider',)
    finally:
        for s,h in old.items():real_set(s,h)
        signal.pthread_sigmask(signal.SIG_SETMASK,beforemask)
