"""Exact source AST boundary tests; synthetic owned files, no payload/model."""
import ast,json,os,uuid
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
        case_binding=lambda case:{'synthetic':True},uuid=uuid,os=os,
        key_gate=NS(canonical=lambda x:json.dumps(x).encode(),digest=lambda x:'unused'))
    exec(compile(ast.Module(body=[body],type_ignores=[]),'ranked_execution.py','exec'),namespace)
    with monkeypatch.context() as m:
        m.setattr(os,'write',fail_write);m.setattr(os,'close',fail_close)
        with pytest.raises(BaseException) as got:namespace['initialize_attempt'](tmp_path/'fresh',Case())
    assert calls and got.value is primary
    assert any('close' in n for n in primary.__notes__)

@pytest.mark.parametrize('source',['ranked_worker.py','ranked_collect.py'])
def test_termination_restore_failure_republishes_failed_terminal(tmp_path,source):
    tree=ast.parse((HERE/source).read_bytes());main=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
    report=tmp_path/'report.json'
    def run(**kwargs):
        data=dict(passed=True,execute=True,operation_complete=True,terminal_scope_completed=True)
        report.write_text(json.dumps(data));return data
    @contextmanager
    def scope():
        yield
        raise RuntimeError('termination restore failed')
    class Parser:
        def add_argument(self,*a,**kw):pass
        def parse_args(self):return NS(output=report)
    namespace=dict(__name__='__main__',Path=Path,argparse=NS(ArgumentParser=Parser),
        termination_signals=scope,run=run,json=json)
    with pytest.raises(SystemExit) as got:exec(compile(ast.Module(body=[main],type_ignores=[]),source,'exec'),namespace)
    assert got.value.code==1
    assert json.loads(report.read_bytes())['passed'] is False
