"""Saved child callback must reject replacement before another operation."""
import pytest
from full_synthetic import build
import encrypted_qwen_full_model as runtime
class ReachedReplacement(RuntimeError):pass
@pytest.fixture(autouse=True)
def clear():
    runtime._QUARANTINE.clear();yield;runtime._QUARANTINE.clear()
@pytest.mark.parametrize('when',['before','first'])
def test_saved_child_callback_is_construction_bound(monkeypatch,when):
    held={};counts=[]
    def replacement(*args):raise ReachedReplacement('unbound saved callback invoked')
    def progress(*args):
        counts.append(dict(held['a'].calls));held['o']._child_progress=replacement
    o,a,b,r,p,x=build(monkeypatch,progress=progress if when=='first' else None);held.update(o=o,a=a)
    if when=='before':o._child_progress=replacement
    with pytest.raises(ValueError):o.evaluate(x)
    assert (not a.calls and not p.calls) if when=='before' else dict(a.calls)==counts[0]
