from types import SimpleNamespace
import pytest
import q5_checkpoint_entry as entry


@pytest.mark.parametrize('role',['setup','server','client'],ids=['setup','server','client'])
def test_actual_dispatch_immediately_denies_without_helper_import(role,monkeypatch):
    import builtins
    original=builtins.__import__;attempts=[]
    def guard(name,*args,**kwargs):
        if name.startswith(('q5_','jax','numpy','torch')):
            attempts.append(name);raise AssertionError('actual dispatch imported helper')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(PermissionError,match='no actual role authority'):
        entry.dispatch(SimpleNamespace(role=role))
    assert attempts==[]


@pytest.mark.parametrize('case',['late_oserror','systemexit0','systemexit1','keyboardinterrupt','malformed_bool'],ids=['late_oserror','systemexit0','systemexit1','keyboardinterrupt','malformed_bool'])
def test_terminal_boundary_does_not_promote_escaped_failures(case,monkeypatch):
    errors={'late_oserror':OSError('private synthetic text'),'systemexit0':SystemExit(0),'systemexit1':SystemExit(1),'keyboardinterrupt':KeyboardInterrupt()}
    def invoke():
        if case=='malformed_bool':return True
        raise errors[case]
    def broken_diagnostic(*args,**kwargs):raise SystemExit(1)
    monkeypatch.setattr('builtins.print',broken_diagnostic)
    assert entry.terminal_status(invoke)==70
