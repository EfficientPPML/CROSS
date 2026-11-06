"""Exact source AST boundary tests; synthetic owned files, no payload/model."""
import ast,json,os,uuid,sys
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
import pytest
HERE=Path(__file__).resolve().parent

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_attempt_write_body_keeps_primary_after_close_error(tmp_path,monkeypatch,kind):
    tree=ast.parse((HERE/'ranked_execution.py').read_bytes())
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize_attempt')
    class Case:pass
    primary=kind('body interruption');close=OSError('cleanup close')
    original_close=os.close
    calls=[]
    def fail_write(*a):raise primary
    def fail_close(fd):original_close(fd);calls.append(fd);raise close
    namespace=dict(Path=Path,RankedCase=Case,outside=lambda root,bases:Path(root),protected=lambda case:(),
        case_binding=lambda case:{'synthetic':True},uuid=uuid,os=os,sys=sys,QUARANTINE=[],
        key_gate=NS(canonical=lambda x:json.dumps(x).encode(),digest=lambda x:'unused'))
    exec(compile(ast.Module(body=[body],type_ignores=[]),'ranked_execution.py','exec'),namespace)
    with monkeypatch.context() as m:
        m.setattr(os,'write',fail_write);m.setattr(os,'close',fail_close)
        with pytest.raises(BaseException) as got:namespace['initialize_attempt'](tmp_path/'fresh',Case())
    assert calls and got.value is primary
    assert any('close' in n for n in primary.__notes__)

@pytest.mark.parametrize('source',['ranked_worker.py','ranked_collect.py'])
@pytest.mark.parametrize('restore_failure',[False,True])
def test_exact_entrypoint_file_directory_shape(tmp_path,monkeypatch,source,restore_failure):
    from contextlib import nullcontext
    from types import ModuleType
    tree=ast.parse((HERE/source).read_bytes());main=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
    report=tmp_path/'report.json';argument=tmp_path if source=='ranked_collect.py' else report
    def run(**kwargs):
        assert kwargs['output']==argument
        data=dict(passed=True,execute=True,operation_complete=True,terminal_scope_completed=True)
        report.write_text(json.dumps(data));return data
    @contextmanager
    def scope():
        yield
        if restore_failure:raise RuntimeError('termination restore failed')
    class Parser:
        def add_argument(self,*a,**kw):pass
        def parse_args(self):return NS(output=argument)
    def write(path,value):
        path.write_text(json.dumps(value))
    gate=NS(read=lambda p,n:p.read_bytes(),digest=lambda raw:'public_digest')
    module=ModuleType('ranked_terminal');module.__dict__.update(Path=Path,json=json,key_gate=gate,termination_signals=scope,cleanup_signals=nullcontext)
    finish=next(n for n in ast.parse((HERE/'ranked_terminal.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='finish')
    exec(compile(ast.Module(body=[finish],type_ignores=[]),'ranked_terminal.py','exec'),module.__dict__)
    monkeypatch.setitem(sys.modules,'ranked_terminal',module)
    namespace=dict(__name__='__main__',Path=Path,argparse=NS(ArgumentParser=Parser),
        termination_signals=scope,run=run,json=json,write=write,key_gate=gate)
    with pytest.raises(SystemExit) as got:exec(compile(ast.Module(body=[main],type_ignores=[]),source,'exec'),namespace)
    assert got.value.code==(1 if restore_failure else 0)
    assert json.loads(report.read_bytes())['passed'] is (not restore_failure)
