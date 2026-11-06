"""Actual ideal owner callback seams; no model/backend or encrypted execution."""
import numpy as np
import pytest
import test_softmax_cohorts as helper
from test_softmax_cohorts import isolated_ideal_process

@pytest.mark.parametrize('method',['refresh','check'])
@pytest.mark.parametrize('when',['before','callback'])
def test_refresh_wrapper_accessors_bound(method,when):
    holder={}
    def substitute():
        binding=holder['owner'].refresh_binding
        if method=='refresh':binding.refresh=lambda value,**kwargs:helper.Value(value.polynomial,25)
        else:binding.check=lambda:None
    def callback(label,value,metadata):
        if when=='callback'and label.endswith('source_branch_mask'):substitute()
    arithmetic,_,owner=helper.make(8,callback=callback);holder['owner']=owner
    if when=='before':substitute()
    with pytest.raises(ValueError):owner.evaluate_tile(helper.Value(helper.scores(8,0),21),query_tile=0)
    if when=='before':assert not arithmetic.calls

@pytest.mark.parametrize('when',['before','callback'])
def test_original_cohort_vector_factory_bound(monkeypatch,when):
    original=helper.batched._vectors
    def wrong(*args):
        values=list(original(*args));values[3]=np.frombuffer((values[3]*.5).tobytes(),dtype='<f8')
        return tuple(values)
    def substitute():monkeypatch.setattr(helper.batched,'_vectors',wrong)
    def callback(label,value,metadata):
        if when=='callback'and label.startswith('C0.')and label.endswith('.base_exp'):substitute()
    arithmetic,_,owner=helper.make(8,callback=callback)
    if when=='before':substitute()
    with pytest.raises(ValueError):owner.evaluate_tile(helper.Value(helper.scores(8,0),21),query_tile=0)
    if when=='before':assert not arithmetic.calls

@pytest.mark.parametrize('length',[2,8])
def test_clean_real_owner_control(length):
    _,_,owner=helper.make(length);data=helper.scores(length,0)
    result=owner.evaluate_tile(helper.Value(data,21),query_tile=0)
    helper.assert_result(result,data,length,0,3)
