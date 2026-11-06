from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
import sys,stat,pytest
sys.path.insert(0,str(Path(__file__).parent))
import freeze_process as p
import freeze_io as io

def test_later_cleanup_cancellation_precedes_earlier_read_close_only_error(monkeypatch):
    first=OSError('metadata file close only');cancel=KeyboardInterrupt('directory close only')
    state=NS(st_mode=stat.S_IFREG|0o600,st_dev=1,st_ino=2,st_size=1,st_mtime_ns=3,st_ctime_ns=4)
    chunks=iter((b'x',b''));fake=NS(**{n:getattr(io.os,n)for n in dir(io.os)})
    fake.open=lambda *a,**k:7;fake.fstat=lambda fd:state;fake.stat=lambda *a,**k:state;fake.read=lambda *a:next(chunks)
    def close(fd):raise first
    fake.close=close;monkeypatch.setattr(io,'os',fake)
    class Output:
        path='synthetic';owner=(3,4)
        def close(self):pass
        @classmethod
        def reopen(cls,*a):return cls()
        def json(self,*a):pass
    @contextmanager
    def signals(state=None,shield=False):
        yield
        if state is not None:state['restored']=True
    monkeypatch.setattr(p,'Output',Output);monkeypatch.setattr(p,'signals',signals);monkeypatch.setattr(p,'source_end',lambda *a:{})
    def second_close():raise cancel
    observed=None
    try:p.supervise(Output(),{'passed':False},lambda:io.read('/synthetic/read'),[second_close],Path('/unused'),{})
    except BaseException as error:observed=error
    assert observed is cancel
