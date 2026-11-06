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

@pytest.mark.parametrize('kind',('empty_plan','missing_rotation','mode_int','mode_float','no_helpers',
    'helper_context','helper_method','evaluator_refresher','evaluator_bound','binding_replaced'))
@pytest.mark.parametrize('when',('entry','callback'))
def test_refresh_binding_rejects_changes_before_continuation(arrays,monkeypatch,kind,when):
    observed=[]
    def mutate(owner,ops,refresh):
        if kind=='empty_plan':refresh.plan['rotations']=()
        elif kind=='missing_rotation':ops.ctx._param_cache.raw_rotation_keys.pop(refresh.plan['rotations'][0])
        elif kind=='mode_int':ops.ctx._param_cache.per_level_rotation_keys=0
        elif kind=='mode_float':ops.ctx._param_cache.per_level_rotation_keys=0.
        elif kind=='no_helpers':refresh.compiled_primitives=()
        elif kind=='helper_context':refresh.compiled_primitives[0].ctx=copy.copy(ops.ctx)
        elif kind=='helper_method':refresh.compiled_primitives[0].rotate=None
        elif kind=='evaluator_refresher':refresh.evaluator.refresher=object()
        elif kind=='evaluator_bound':refresh.evaluator.declare_bound=lambda *a:None
        else:owner.refresh_binding=copy.copy(owner.refresh_binding)
    def progress(*args):mutate(owner,ops,refresh)
    owner,ops,refresh,sources,model=setup(arrays,15,progress=progress if when=='callback' else None)
    def project(value,*args,**kwargs):
        observed.append('projection')
        return Value(np.zeros(65536),48,DELTA)
    monkeypatch.setattr(owner.linear,'linear_tile',project)
    if when=='entry':mutate(owner,ops,refresh)
    with pytest.raises((ValueError,TypeError,AttributeError)):owner.evaluate(sources)
    assert observed==([] if when=='entry' else ['projection'])
    assert not refresh.calls and not owner._active and not owner._sources and not owner._retained

def test_public_refresh_inventory_uses_unchanged_bootstrap_planner():
    import json
    from pathlib import Path
    from encrypted_refresh import refresh_plan
    expected=json.loads((Path(__file__).parent/'p60_profile_and_refresh.json').read_text())['refresh_plan']
    actual=refresh_plan(131072,65536,(4,4),max_level=49)
    assert json.loads(json.dumps(actual))==expected
    assert type(actual['rotations'])is tuple and type(actual['level_budget'])is tuple
