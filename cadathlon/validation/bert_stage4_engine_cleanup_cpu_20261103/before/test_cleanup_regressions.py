"""Engine cleanup only; fabricated public owners, no crypto/model imports."""
import os
import signal
import pytest
import full_model_engine as engine
from test_full_model_engine import fake


@pytest.mark.parametrize('interrupt',[KeyboardInterrupt('later interrupt'),SystemExit(143)])
def test_later_cleanup_interrupt_takes_precedence_and_finishes(fake,interrupt):
    ctx,calls,state=fake
    ordinary=OSError('first ordinary cleanup error')
    with pytest.raises(type(interrupt)) as caught:
        with engine.server_engine(ctx,manifest_bytes=b'fixed') as graph:
            first,second=graph.refresher.compiled_primitives[:2]
            def fail_first(**kw):calls.append(('first_failed',));raise ordinary
            def fail_second(**kw):calls.append(('second_failed',));raise interrupt
            first.clear=fail_first;second.clear=fail_second
    assert caught.value is interrupt
    assert any('first ordinary' in note for note in interrupt.__notes__)
    assert calls[-1]==('close_provider',)
    assert len([c for c in calls if c[0]=='clear'])==15


@pytest.mark.parametrize('primary',[RuntimeError('body failure'),KeyboardInterrupt('body cancellation'),SystemExit(130)])
def test_existing_primary_preserved_through_cleanup_interrupts(fake,primary):
    ctx,calls,state=fake
    with pytest.raises(type(primary)) as caught:
        with engine.server_engine(ctx,manifest_bytes=b'fixed') as graph:
            first,second=graph.refresher.compiled_primitives[:2]
            def ordinary(**kw):raise OSError('cleanup ordinary')
            def interrupt(**kw):raise KeyboardInterrupt('cleanup interrupt')
            first.clear=ordinary;second.clear=interrupt
            raise primary
    assert caught.value is primary
    assert len(primary.__notes__)==2 and 'cleanup ordinary' in primary.__notes__[0] and 'cleanup interrupt' in primary.__notes__[1]
    assert calls[-1]==('close_provider',)


@pytest.mark.parametrize('body_failure',[False,True])
def test_repeated_real_signals_are_deferred_until_all_cleanup_finishes(fake,body_failure):
    ctx,calls,state=fake
    signs=(signal.SIGTERM,signal.SIGHUP,signal.SIGINT)
    old={s:signal.getsignal(s) for s in signs}
    delivered=[];completed=[];primary=RuntimeError('original body failure')
    def original_handler(signum,frame):
        delivered.append(signum)
        raise KeyboardInterrupt('unshielded signal')
    for s in signs:signal.signal(s,original_handler)
    try:
        with pytest.raises(RuntimeError if body_failure else KeyboardInterrupt) as caught:
            with engine.server_engine(ctx,manifest_bytes=b'fixed') as graph:
                helper=graph.refresher.compiled_primitives[0];real=helper.clear
                def send_during_cleanup(**kw):
                    for s in signs*2:os.kill(os.getpid(),s)
                    completed.append(True);real(**kw)
                helper.clear=send_during_cleanup
                if body_failure:raise primary
        if body_failure:assert caught.value is primary
        assert completed==[True], 'cleanup operation itself was interrupted'
        assert delivered==[], 'old signal handler ran during cleanup'
        assert len([c for c in calls if c[0]=='clear'])==17
        assert calls[-1]==('close_provider',)
        assert all(signal.getsignal(s) is original_handler for s in signs)
    finally:
        for s,h in old.items():signal.signal(s,h)


def test_signal_handlers_restored_after_success(fake):
    ctx,calls,state=fake
    signals=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)
    old={s:signal.getsignal(s) for s in signals}
    with engine.server_engine(ctx,manifest_bytes=b'fixed'):pass
    assert {s:signal.getsignal(s) for s in signals}==old


def test_deferred_signal_takes_precedence_over_ordinary_cleanup_failure(fake):
    ctx,calls,state=fake;old=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise KeyboardInterrupt('unshielded')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            with engine.server_engine(ctx,manifest_bytes=b'fixed') as graph:
                first,second=graph.refresher.compiled_primitives[:2]
                def fail(**kw):raise OSError('earlier ordinary')
                def cancel(**kw):os.kill(os.getpid(),signal.SIGTERM)
                first.clear=fail;second.clear=cancel
        assert any('earlier ordinary' in note for note in caught.value.__notes__)
        assert calls[-1]==('close_provider',)
        assert signal.getsignal(signal.SIGTERM) is interrupted
    finally:signal.signal(signal.SIGTERM,old)
