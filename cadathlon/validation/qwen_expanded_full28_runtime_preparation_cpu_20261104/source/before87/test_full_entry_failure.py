"""No-work state collection failure must leave owner reusable/inactive."""
import pytest
from full_synthetic import build
import encrypted_qwen_full_model as runtime

def test_initial_snapshot_failure_leaves_inactive(monkeypatch):
    o,a,b,r,p,x=build(monkeypatch);original=o._state
    def fail(*args):raise MemoryError('initial state')
    o._state=fail
    with pytest.raises(MemoryError,match='initial state'):o.evaluate(x)
    assert not o._active and o._borrowed==() and not a.calls and not p.calls and not runtime._QUARANTINE
    o._state=original;o._full_guard()
