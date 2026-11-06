import pytest
import qwen_backend_composition as b
from test_backend_composition import setup

def test_ref_list_replacement_preserves_original_failure(public):
    c=setup(public);e=b.BorrowedServerEngine(**c.args);primary=RuntimeError("publisher primary")
    def publish(*args):e.refs=None;raise primary
    with pytest.raises(RuntimeError)as caught:e.run(publish)
    assert caught.value is primary and b._QUARANTINE[-1][0]is e

def test_owned_arithmetic_replacement_rejected_and_original_retired_or_quarantined(public):
    c=setup(public);e=b.BorrowedServerEngine(**c.args)
    def publish(*args):row=c.publish(*args);e.arithmetic=None;return row
    with pytest.raises(ValueError,match="owned") :e.run(publish)
    assert e.failed is not None and not c.params.retired

def test_clean_control(public):
    c=setup(public);e=b.BorrowedServerEngine(**c.args);e.run(c.publish);assert e.retired

def test_run_local_borrowed_fields_and_retained_list_survive_callback_primary(public):
    c=setup(public);e=b.BorrowedServerEngine(**c.args);primary=RuntimeError('callback body')
    original=e._binding
    def publish(*args):
        e._binding=();e._retained=();e.refs=None;e.parameters=None;e.upload=None;e.ctx=None;e.key_owner=None;e.shield=None;e.cleanup=None
        raise primary
    with pytest.raises(RuntimeError)as caught:e.run(publish)
    assert caught.value is primary and b._QUARANTINE[-1][1]==original
    assert ('parameters','readback')in c.events and ('upload','readback')in c.events
    assert any(x[0]=='barrier'for x in c.events)and not c.params.retired

def test_partial_snapshot_failure_preserves_raw_arrays_and_primary(public):
    from dataclasses import replace
    c=setup(public);primary=ValueError('partial return snapshot');materialize=c.args['dependencies'].materialize
    def broken(record):
        value=materialize(record)
        def fail():raise primary
        value.get_moduli=fail
        return value
    c.args['dependencies']=replace(c.args['dependencies'],materialize=broken)
    e=b.BorrowedServerEngine(**c.args)
    with pytest.raises(ValueError)as caught:e.run(c.publish)
    assert caught.value is primary
    assert ('barrier','input0')in c.events and ('barrier','input0.q')in c.events
    retained=b._QUARANTINE[-1][3]
    assert len(retained)==1 and retained[0][1]is c.state.inputs[0].polynomial

def test_original_file_readback_bound_before_publication(public):
    c=setup(public);e=b.BorrowedServerEngine(**c.args)
    def publish(*args):
        row=c.publish(*args);c.params.readback=lambda:(_ for _ in ()).throw(AssertionError('replacement invoked'))
        return row
    with pytest.raises(ValueError,match='file accessor'):e.run(publish)
    assert ('parameters','readback')in c.events and e.failed is not None
