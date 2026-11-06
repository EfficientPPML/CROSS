"""Expected failures of the exact reviewed draft; no backend or HE."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from test_bsgs import Arithmetic,Value,BSGSScoreRouting,FactoredScoreLayout,SLOTS

@pytest.fixture(autouse=True)
def doubles(monkeypatch):
    monkeypatch.setitem(sys.modules,'encrypted_packed_qwen',SimpleNamespace(CompiledArithmetic=Arithmetic))
    monkeypatch.setitem(sys.modules,'jaxite_word',SimpleNamespace(Polynomial=Value))
    monkeypatch.setitem(sys.modules,'support',SimpleNamespace(require_secure_execution=lambda *args:None))
    monkeypatch.setitem(sys.modules,'jax',SimpleNamespace(block_until_ready=lambda arrays:None))

def test_missing_supplied_keys_rejected_before_work():
    a=Arithmetic();k=a.ctx._param_cache;k.raw_rotation_keys={};k.coef_maps={};k.per_level_rotation_keys=False
    with pytest.raises(ValueError):BSGSScoreRouting(a,FactoredScoreLayout(0,0))
    assert not a.calls

def test_changed_field_class_rejected_before_work():
    a=Arithmetic();a.ctx._param_cache.ff_context_cls=type('OriginalBarrett',(),{})
    owner=BSGSScoreRouting(a,FactoredScoreLayout(0,0))
    a.ctx._param_cache.ff_context_cls=type('ChangedFamily',(),{})
    with pytest.raises(ValueError):owner._guard()
    assert not a.calls

def test_later_callback_cannot_leave_owned_prior_annotations():
    source=Value(np.arange(SLOTS,dtype=np.float64));saved=[]
    def callback(label,value,counts):
        if label=='head compaction':saved.append(value)
        else:saved[0]._public_bound='stale callback assertion'
    owner=BSGSScoreRouting(Arithmetic(),FactoredScoreLayout(0,0),progress=callback)
    result=owner.forward(source)
    assert source._public_bound=='caller annotation'
    assert not hasattr(result,'_public_bound')
    assert not hasattr(saved[0],'_public_bound')
