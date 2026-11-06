"""Early boundary regressions; arithmetic spy proves rejection before work."""
import numpy as np
import pytest
import copy
from test_first_attention_ideal import arrays,primitives,setup
from test_initial_ideal import ideal_primitives
from test_softmax_tile import refresh_and_tracked
from ideal_primitives import DELTA
from ideal_primitives import Value
from encrypted_bert_first_attention import BertFirstAttention
from test_first_attention_ideal import RAW

@pytest.mark.parametrize('kind',('float','numpy_int'))
def test_original_positions_type_alias_fails_before_any_primitive(arrays,monkeypatch,kind):
    owner,ops,refresh,sources,model=setup(arrays,15)
    convert=float if kind=='float' else np.int64
    owner.positions=tuple(tuple(convert(x) for x in row) for row in owner.positions)
    def forbidden(*a,**k):raise AssertionError('arithmetic reached')
    monkeypatch.setattr(owner.linear,'linear_tile',forbidden)
    with pytest.raises(ValueError):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls

@pytest.mark.parametrize('scale',(float(np.nextafter(DELTA,0.)),float(np.nextafter(DELTA,np.inf)),np.float64(DELTA)))
def test_fresh_input_schema_requires_exact_typed_canonical_scale(arrays,monkeypatch,scale):
    owner,ops,refresh,sources,model=setup(arrays,15)
    for value in sources:value._ckks_scale=scale
    def forbidden(*a,**k):raise AssertionError('arithmetic reached')
    monkeypatch.setattr(owner.linear,'linear_tile',forbidden)
    with pytest.raises(ValueError):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls

def test_callback_forgets_annotations_on_current_and_earlier_owned_outputs(arrays):
    earlier=Value(np.zeros(65536),9,DELTA)
    current=Value(np.ones(65536),9,DELTA)
    def progress(*args):
        current._public_bound='unjustified current bound'
        earlier._public_relation='unjustified retained relation'
    owner,ops,refresh,sources,model=setup(arrays,15,progress=progress)
    sources[0]._public_client_annotation='preserve borrowed input annotation'
    owner._sources=tuple(sources)
    owner._source_states=tuple(owner._fingerprint(v) for v in sources)
    owner._retained.append((earlier,owner._fingerprint(earlier)))
    owner._callback('final output',current,{})
    assert not any(k.startswith('_public_') for k in vars(current))
    assert not any(k.startswith('_public_') for k in vars(earlier))
    assert sources[0]._public_client_annotation=='preserve borrowed input annotation'
    assert not ops.calls and not refresh.calls

def test_distinct_bound_aware_linear_arithmetic_rejected_before_work(arrays):
    owner,ops,refresh,sources,model=setup(arrays,15)
    separate=copy.copy(ops)
    separate.declare_bound=lambda *args:None
    assert separate is not ops and separate.ctx is ops.ctx
    owner.linear.arithmetic=separate
    with pytest.raises((TypeError,ValueError),match='bound'):
        BertFirstAttention(ops,owner.linear,refresh,model,owner.contract,RAW,arrays['gamma'],arrays['beta'])
    assert not ops.calls and not separate.calls and not refresh.calls
