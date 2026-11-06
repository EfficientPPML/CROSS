import ast
from pathlib import Path
import pytest
P=Path(__file__).resolve().parent
policy={};exec(compile((P/"cleanup_origin.py").read_bytes(),"exact-cleanup-origin","exec"),policy)
t=ast.parse((P/"finalizer_source.py").read_bytes());node=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=="Finalizer")
ns=dict(policy);exec(compile(ast.Module(body=[node],type_ignores=[]),"exact-Finalizer-AST","exec"),ns);Finalizer=ns["Finalizer"]
def fail(error):raise error
@pytest.mark.parametrize("cancel",[KeyboardInterrupt,SystemExit])
def test_marked_incoming_cleanup_yields_to_later_cancellation(cancel):
    ordinary=policy["mark"](OSError("publisher close"));later=cancel("later cleanup")
    final=Finalizer({},ordinary);final.attempt("later",lambda:fail(later))
    with pytest.raises(cancel)as caught:final.raise_if_failed()
    assert caught.value is later
def test_genuine_body_wins():
    body=ValueError("genuine body");final=Finalizer({},body);final.attempt("later",lambda:fail(KeyboardInterrupt()))
    with pytest.raises(ValueError)as caught:final.raise_if_failed()
    assert caught.value is body
def test_sticky_same_body_rethrow_keeps_origin():
    body=ValueError("genuine body");final=Finalizer({},body);final.attempt("sticky",lambda:fail(body));final.attempt("later",lambda:fail(SystemExit(7)))
    with pytest.raises(ValueError)as caught:final.raise_if_failed()
    assert caught.value is body
def test_no_incoming_cleanup_cancellation_control():
    later=KeyboardInterrupt();final=Finalizer({});final.attempt("first",lambda:fail(OSError()));final.attempt("later",lambda:fail(later))
    with pytest.raises(KeyboardInterrupt)as caught:final.raise_if_failed()
    assert caught.value is later
