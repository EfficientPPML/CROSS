"""Real first SIGTERM injected at exact collector finally boundary; fake child."""
from pathlib import Path
import json,signal,sys
import pytest
import collect as c
from cleanup import termination_signals

@pytest.mark.parametrize('prior_failure',[False,True])
def test_first_signal_at_cleanup_entry_cannot_skip_owned_child(tmp_path,monkeypatch,prior_failure):
    body=ValueError('synthetic child body failure');closed=[];leaders=[];output=tmp_path/'attempt'
    monkeypatch.setattr(c,'sources',lambda:{'stub.py':b'pass\n'})
    class Child:
        def __init__(self,*a,**kw):
            self.record=dict(direct_child_reaped=False,group_empty_verified=False,cleanup_completed=False);leaders.append(self)
        def wait(self):
            if prior_failure:raise body
            (output/'tests.xml').write_text('<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0"><testcase name="synthetic"/></testsuite></testsuites>')
            self.record.update(direct_child_reaped=True,group_empty_verified=True,cleanup_completed=True)
            return 0
        def close(self,primary=None):
            closed.append(primary);self.record.update(direct_child_reaped=True,group_empty_verified=True,cleanup_completed=True)
    monkeypatch.setattr(c,'OwnedChild',Child)
    lines=Path(c.__file__).read_text().splitlines()
    target=next(i for i,line in enumerate(lines,1) if 'if child is not None:attempt(' in line)
    delivered=[]
    def trace(frame,event,arg):
        if (event=='line' and frame.f_code.co_filename==c.__file__ and frame.f_lineno==target and not delivered):
            delivered.append(True);signal.raise_signal(signal.SIGTERM)
        return trace
    old=sys.gettrace();sys.settrace(trace)
    try:
        with pytest.raises(BaseException) as got:
            with termination_signals():c.run(output,False)
    finally:sys.settrace(old)
    assert delivered and leaders
    assert closed, 'first SIGTERM skipped the owned-child cleanup call'
    if prior_failure:assert got.value is body

@pytest.mark.parametrize('source',['export_decoder.py','collect.py'])
@pytest.mark.parametrize('restore_failure',[False,True])
def test_actual_cli_directory_output_boundary(tmp_path,monkeypatch,source,restore_failure):
    import ast,argparse
    from contextlib import contextmanager,nullcontext
    from types import SimpleNamespace as NS
    import export_decoder as e
    import finish as f
    original=f.finish;out=tmp_path/'output';out.mkdir();report=out/'report.json'
    @contextmanager
    def scope():
        yield
        if restore_failure:raise RuntimeError('chosen outer restore fault')
    monkeypatch.setattr(f,'finish',lambda path,call,write,read:original(path,call,write,read,termination=scope,shield=nullcontext))
    class Parser:
        def add_argument(self,*a,**kw):pass
        def parse_args(self):return NS(output=out,body=tmp_path/'bodies',execute=False)
    def run(*args):
        value=dict(passed=True,operation_complete=True,termination_scope_completed=False,elapsed_s=1.)
        e.write_json(report,value);return value
    tree=ast.parse((Path(c.__file__).parent/source).read_bytes());main=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
    namespace=dict(__name__='__main__',argparse=NS(ArgumentParser=Parser),Path=Path,
        json=json,run=run,e=e,write_json=e.write_json,public_report=e.public_report)
    if restore_failure:
        with pytest.raises(RuntimeError,match='chosen outer restore fault'):
            exec(compile(ast.Module(body=[main],type_ignores=[]),source,'exec'),namespace)
    elif source=='collect.py':
        with pytest.raises(SystemExit) as got:exec(compile(ast.Module(body=[main],type_ignores=[]),source,'exec'),namespace)
        assert got.value.code==0
    else:exec(compile(ast.Module(body=[main],type_ignores=[]),source,'exec'),namespace)
    value=json.loads(report.read_bytes())
    assert value['passed'] is (not restore_failure)
    assert value['termination_scope_completed'] is (not restore_failure)

@pytest.mark.parametrize('value',[None,[],7])
def test_malformed_terminal_return_keeps_intended_failure(tmp_path,value):
    from contextlib import nullcontext
    import export_decoder as e
    import finish as f
    report=tmp_path/'report.json'
    with pytest.raises(ValueError,match='terminal report object required'):
        f.finish(report,lambda:value,e.write_json,e.public_report,termination=nullcontext,shield=nullcontext)
    r=json.loads(report.read_bytes())
    assert r['passed'] is False and r['termination_scope_completed'] is False and r['error']['type']=='ValueError'
