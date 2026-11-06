"""Real frozen marker + exact Finalizer AST; no component/backend imports."""
from pathlib import Path
import ast
import pytest

ROOT=Path(__file__).parent/'execution'
namespace={}
exec(compile((ROOT/'cleanup_origin.py').read_text(),str(ROOT/'cleanup_origin.py'),'exec'),namespace)
tree=ast.parse((ROOT/'component_protocol.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Finalizer')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'component_protocol.py'),'exec'),namespace)
Finalizer=namespace['Finalizer']


def throwing(error):
    def action():raise error
    return action


@pytest.mark.parametrize('later',[KeyboardInterrupt('later cleanup'),SystemExit('later cleanup')])
def test_repeated_true_body_primary_remains_body(later):
    primary=ValueError('genuine body failed before cleanup')
    final=Finalizer(primary)
    final.attempt('sticky barrier reraises same error',throwing(primary))
    final.attempt('later cleanup cancellation',throwing(later))
    try:final.raise_failure()
    except BaseException as actual:assert actual is primary
    else:pytest.fail('missing primary')


def test_distinct_cleanup_error_body_control():
    primary=ValueError('genuine body')
    final=Finalizer(primary)
    final.attempt('different error',throwing(OSError('cleanup')))
    final.attempt('later cancellation',throwing(KeyboardInterrupt('later')))
    with pytest.raises(ValueError) as result:final.raise_failure()
    assert result.value is primary
