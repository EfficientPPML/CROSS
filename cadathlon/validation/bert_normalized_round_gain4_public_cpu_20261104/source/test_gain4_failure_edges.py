import pytest
from test_gain4_ideal import primitives,setup,g

def test_source_snapshot_failure_does_not_leave_active(monkeypatch):
    owner,ops,source=setup();primary=MemoryError('source snapshot')
    monkeypatch.setattr(owner,'_fingerprint',lambda v:(_ for _ in ()).throw(primary))
    with pytest.raises(MemoryError)as seen:owner.evaluate_tile(source)
    assert seen.value is primary and owner._active is False and not ops.calls

def test_later_entered_failure_clears_previous_plan():
    fail=[False]
    def progress(*args):
        if fail[0]:raise KeyboardInterrupt('later invocation')
    owner,ops,source=setup(progress=progress);owner.evaluate_tile(source)
    assert owner.last_plan is not None;fail[0]=True
    with pytest.raises(KeyboardInterrupt):owner.evaluate_tile(source)
    assert owner.last_plan is None and not owner._active

def test_entered_failure_forbids_new_owner_reuse():
    owner,ops,source=setup(progress=lambda *args:(_ for _ in ()).throw(KeyboardInterrupt('entered failure')))
    with pytest.raises(KeyboardInterrupt):owner.evaluate_tile(source)
    with pytest.raises(RuntimeError):g.BertNormalizationRoundGain4(ops,owner.layout,owner.contract)
