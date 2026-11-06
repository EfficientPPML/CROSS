import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
import pytest

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE/"execution"))
import entry
import freeze_io
import freeze_process as p

@pytest.mark.parametrize('module,function',[(entry,'bounded'),(freeze_io,'read')])
def test_real_read_body_primary_survives_close_cancellation(monkeypatch,module,function):
    body=ValueError('body');cancel=KeyboardInterrupt('cleanup')
    def fail_body(fd):raise body
    def fail_close(fd):raise cancel
    real=module.os
    fake=NS(**{n:getattr(real,n) for n in dir(real)})
    fake.open=lambda *a,**k:7;fake.fstat=fail_body;fake.close=fail_close
    monkeypatch.setattr(module,'os',fake)
    observed=None
    try:getattr(module,function)('/synthetic/file',32) if module is entry else module.read('/synthetic/file')
    except BaseException as error:observed=error
    assert observed is body


def test_terminal_leader_remaining_owned_group_is_retired(monkeypatch):
    calls=[];live=[True]
    class Child:
        pid=123;returncode=0
        def poll(self):return 0
        def wait(self,*a,**k):calls.append(('wait',));return 0
    real=p.os;fake=NS(**{n:getattr(real,n)for n in dir(real)})
    def kill(pid,sig):calls.append(('killpg',pid,sig));live[0]=False
    fake.killpg=kill
    monkeypatch.setattr(p,'os',fake);monkeypatch.setattr(p,'group_absent',lambda pid:not live[0])
    p.retire(Child(),None)
    assert any(x[0]=='killpg'for x in calls)and ('wait',)in calls and not live[0]


def test_body_output_close_failure_cannot_leave_success_report(monkeypatch):
    receipts=[];events=[]
    class Output:
        path='public';owner=(1,2)
        def close(self):events.append('close-body');raise OSError('body output close')
        @classmethod
        def reopen(cls,path,owner):
            assert path=='public'and owner==(1,2);events.append('reopen');return Publication()
    class Publication:
        def json(self,name,report):events.append('publish');receipts.append(dict(report))
        def close(self):events.append('close-publication')
    monkeypatch.setattr(p,'Output',Output)
    @__import__('contextlib').contextmanager
    def scopes(state=None,shield=False):
        yield
        if state is not None:state['restored']=True
    monkeypatch.setattr(p,'signals',scopes);monkeypatch.setattr(p,'source_end',lambda *a:{'a':'b'})
    report={'passed':False}
    with pytest.raises(OSError,match='body output close'):
        p.supervise(Output(),report,lambda:report.update(passed=True,complete=True),[],Path('/unused'),{'a':'b'})
    assert receipts[-1]['passed']is False and receipts[-1]['cleanup_completed']is False
    assert events==['close-body','reopen','publish','close-publication']
