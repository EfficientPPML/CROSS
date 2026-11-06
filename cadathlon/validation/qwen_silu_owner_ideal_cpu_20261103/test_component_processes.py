"""Mock orphan/failure cleanup and a real clean child; no crypto or orphan job."""
import os,signal,subprocess,sys
from types import SimpleNamespace
import pytest
import component_processes as p

@pytest.fixture
def fake(monkeypatch):
    state=SimpleNamespace(group=True,leader=True,ignore_term=False,stubborn=False,code=0,
        first_error=None,waits=[],signals=[],clock=0.,group_error=None,wait_error=None)
    class Child:
        pid=876543210
        def __init__(self,command,**kwargs):
            assert kwargs['start_new_session'] is True and kwargs['umask']==0o077
            self.returncode=None;state.options=kwargs
        def wait(self,timeout=None):
            state.waits.append(timeout)
            if timeout is None:
                if state.first_error is not None:raise state.first_error
                state.leader=False;self.returncode=state.code
                if not state.ignore_term and not state.stubborn:state.group=False
                return self.returncode
            if state.wait_error is not None:
                error=state.wait_error;state.wait_error=None;raise error
            if state.leader and not state.signals:raise subprocess.TimeoutExpired('mock',timeout)
            if state.leader and state.ignore_term and signal.SIGKILL not in state.signals:
                state.clock+=timeout;raise subprocess.TimeoutExpired('mock',timeout)
            state.leader=False;self.returncode=-9 if signal.SIGKILL in state.signals else -15
            return self.returncode
    def group(pid,sig):
        assert pid==Child.pid
        if sig==0:
            if state.group_error is not None:raise state.group_error
            if not state.group:raise ProcessLookupError()
        else:
            state.signals.append(sig)
            if not state.stubborn and (sig==signal.SIGKILL or not state.ignore_term):state.group=False
    def sleep(seconds):state.clock+=seconds
    monkeypatch.setattr(p.subprocess,'Popen',Child);monkeypatch.setattr(p.os,'killpg',group)
    monkeypatch.setattr(p.time,'monotonic',lambda:state.clock);monkeypatch.setattr(p.time,'sleep',sleep)
    return state


def child():return p.OwnedChild(['mock'],cwd='.',stdout=None,term_grace=.02,kill_grace=.02)


def test_healthy_wait_has_no_deadline_and_records_separate_facts(fake):
    owner=child();assert owner.wait()==0
    assert fake.waits==[None] and fake.signals==[]
    assert owner.record['direct_child_reaped'] and owner.record['group_empty_verified'] and owner.record['cleanup_completed']
    assert owner.record['direct_returncode']==0 and 'descendant' in owner.record['scope']
    owner.close();assert fake.waits==[None]
    with pytest.raises(RuntimeError,match='repeat'):owner.wait()


def test_nonzero_clean_leader_is_still_reaped(fake):
    fake.code=7;owner=child();assert owner.wait()==7
    assert owner.record['direct_returncode']==7 and owner.record['group_empty_verified']


def test_terminal_orphan_fails_even_when_cleanup_succeeds(fake):
    fake.ignore_term=True;owner=child()
    with pytest.raises(RuntimeError,match='surviving owned'):owner.wait()
    assert fake.waits==[None] # no claim of reaping descendants
    assert fake.signals==[signal.SIGTERM,signal.SIGKILL]
    assert owner.record['direct_child_reaped'] and owner.record['group_empty_verified']
    assert owner.record['cleanup_completed'] and fake.clock<=.041


@pytest.mark.parametrize('kind',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_cancel_or_failure_preserves_primary_and_reaps_direct_child(fake,kind):
    primary=kind('original');fake.first_error=primary;owner=child()
    with pytest.raises(kind) as found:owner.wait()
    assert found.value is primary and fake.waits[0] is None
    assert fake.signals==[signal.SIGTERM] and owner.record['direct_child_reaped'] and owner.record['group_empty_verified']


def test_stubborn_terminal_group_cannot_claim_cleanup(fake):
    fake.stubborn=True;owner=child()
    with pytest.raises(RuntimeError,match='surviving owned') as found:owner.wait()
    assert owner.record['direct_child_reaped'] and not owner.record['group_empty_verified']
    assert not owner.record['cleanup_completed'] and owner.record['cleanup_errors']
    assert found.value.__notes__ and fake.clock<=.041


def test_cleanup_only_later_cancellation_has_priority(fake):
    owner=child();fake.wait_error=OSError('first wait error')
    original=owner._await_group_empty;calls=[]
    def fault(grace):
        calls.append(grace)
        if len(calls)==1:raise KeyboardInterrupt('later cancel')
        return original(grace)
    owner._await_group_empty=fault
    with pytest.raises(KeyboardInterrupt,match='later cancel'):owner.close()
    assert owner.record['direct_child_reaped'] and owner.record['group_empty_verified']
    assert not owner.record['cleanup_completed'] and len(owner.record['cleanup_errors'])>=2


def test_cleanup_unknown_group_check_clears_stale_absence_flag(fake):
    owner=child();owner.record['group_empty_verified']=True;fake.group_error=PermissionError('unknown group')
    primary=ValueError('body');owner.close(primary=primary)
    assert not owner.record['group_empty_verified'] and not owner.record['cleanup_completed']
    assert len(primary.__notes__)>=1


def test_close_with_body_primary_never_replaces_it(fake):
    owner=child();fake.wait_error=KeyboardInterrupt('cleanup cancel')
    primary=ValueError('body');owner.close(primary=primary)
    assert owner.record['direct_child_reaped'] and primary.__notes__
    assert not owner.record['cleanup_completed']


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),31.,True,1])
def test_invalid_cleanup_grace_rejected_before_launch(monkeypatch,value):
    monkeypatch.setattr(p.subprocess,'Popen',lambda *a,**k:pytest.fail('must reject first'))
    with pytest.raises(ValueError):p.OwnedChild([],cwd='.',stdout=None,term_grace=value)


def test_real_clean_child_wait_and_group_absence(tmp_path):
    with (tmp_path/'child.log').open('wb') as log:
        owner=p.OwnedChild([sys.executable,'-c','print("public clean probe")'],cwd=tmp_path,stdout=log)
        assert owner.wait()==0
    assert owner.record['direct_child_reaped'] and owner.record['direct_returncode']==0
    assert owner.record['group_empty_verified'] and owner.record['signals_sent']==[]
    with pytest.raises(ProcessLookupError):os.killpg(owner.pid,0)
    assert (tmp_path/'child.log').read_text().strip()=='public clean probe'
