from pathlib import Path
from types import SimpleNamespace
import ast,signal
import pytest


@pytest.mark.parametrize('failure',['poll','TERM'])
def test_cleanup_fault_still_attempts_kill_wait_group(failure):
    calls=[];state={'empty':False,'poll':0,'clock':0}
    def now():state['clock']+=100;return state['clock']
    def absent(pid):calls.append('group');return state['empty']
    def kill(pid,sig):
        calls.append(sig)
        if sig==signal.SIGTERM and failure=='TERM':raise OSError('TERM failure')
        if sig==signal.SIGKILL:state['empty']=True
    class Child:
        pid=123;returncode=None
        def poll(self):
            calls.append('poll');state['poll']+=1
            if failure=='poll' and state['poll']==1:raise OSError('poll failure')
        def wait(self,*args,**kwargs):calls.append('wait');self.returncode=-9;return -9
    namespace=dict(signal=signal,os=SimpleNamespace(killpg=kill),time=SimpleNamespace(monotonic=now,sleep=lambda _:None),
        group_absent=absent)
    root=Path(__file__).parent
    exec(compile((root/'cleanup_origin.py').read_text(),str(root/'cleanup_origin.py'),'exec'),namespace)
    tree=ast.parse((root/'observer_process.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='retire')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'observer_process.py'),'exec'),namespace)
    with pytest.raises(OSError):namespace['retire'](Child())
    assert signal.SIGKILL in calls and 'wait' in calls and calls[-1]=='group'
