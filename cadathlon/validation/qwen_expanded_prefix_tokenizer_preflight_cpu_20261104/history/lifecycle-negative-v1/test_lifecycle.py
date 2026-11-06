import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
import pytest

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
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
    calls=[]
    class Child:
        pid=123;returncode=0
        def poll(self):return 0
        def wait(self,*a,**k):calls.append(('wait',));return 0
    real=p.os;fake=NS(**{n:getattr(real,n) for n in dir(real)})
    fake.killpg=lambda pid,sig:calls.append(('killpg',pid,sig))
    monkeypatch.setattr(p,'os',fake)
    monkeypatch.setattr(p,'group_absent',lambda pid:False)
    try:p.retire(Child(),None)
    except RuntimeError:pass
    assert any(x[0]=='killpg' for x in calls),calls


def test_output_close_failure_cannot_leave_success_report(monkeypatch):
    receipts=[]
    class Output:
        def __init__(self,path):self.path=path
        def json(self,name,report):receipts.append(dict(report))
        def close(self,primary=None):
            if self.path=='public':raise OSError('final output close')
    monkeypatch.setattr(p,'Output',Output)
    monkeypatch.setattr(p,'grant_check',lambda *a:{})
    monkeypatch.setattr(p,'install_audit',lambda *a:[])
    monkeypatch.setattr(p,'signals',nullcontext)
    monkeypatch.setattr(p,'source_end',lambda *a:{'a':'b'})
    monkeypatch.setattr(p.freeze_logic,'execute',lambda *a:{'passed':True,'complete':True})
    monkeypatch.setattr(p,'importlib',NS(metadata=NS(version=lambda *a:'synthetic')))
    monkeypatch.setitem(sys.modules,'tokenizers',NS(Tokenizer=NS(from_str=lambda *a:None)))
    args=NS(output='public',private_output='private',manifest_sha256='x',grant_sha256='y')
    with pytest.raises(OSError,match='final output close'):
        p.worker(args,{'a':'b'},{},{},Path('/unused'))
    assert receipts and receipts[-1]['passed'] is False
