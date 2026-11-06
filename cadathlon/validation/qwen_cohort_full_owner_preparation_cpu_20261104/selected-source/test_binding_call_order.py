"""The replaced wrapper must be rejected before it can execute."""
import pytest
import test_softmax_cohorts as helper
from test_softmax_cohorts import isolated_ideal_process

@pytest.mark.parametrize('when',['before','callback'])
def test_substituted_check_never_invoked(when):
    calls=[];holder={}
    def replace():holder['owner'].refresh_binding.check=lambda:calls.append('invoked')
    def callback(label,value,metadata):
        if when=='callback'and label.endswith('normalized_t.refresh'):replace()
    _,_,owner=helper.make(2,callback=callback);holder['owner']=owner
    if when=='before':replace()
    with pytest.raises(ValueError):owner.evaluate_tile(helper.Value(helper.scores(2,0),21),query_tile=0)
    assert calls==[]

def test_actual_returned_refresh_value_cannot_replace_wrapper():
    holder={};refs=[]
    def callback(label,value,metadata):
        if label.endswith('normalized_t.refresh'):
            assert value.level==25
            refs.append(value)
            holder['owner'].refresh_binding.refresh=lambda source,*,public_bound:value
    _,_,owner=helper.make(2,callback=callback);holder['owner']=owner
    with pytest.raises(ValueError,match='wrapper accessor'):owner.evaluate_tile(helper.Value(helper.scores(2,0),21),query_tile=0)
    assert refs and helper.batched._QUARANTINE
