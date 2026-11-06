"""Public-boundary and continuation guards; arithmetic spies are not HE proof."""
import copy,math
import numpy as np
import pytest
from test_layer_attention_ideal import arrays,primitives,setup,RAW
from test_initial_ideal import ideal_primitives
from test_softmax_tile import refresh_and_tracked
from ideal_primitives import Value,DELTA

@pytest.mark.parametrize('layer',(0,1,6,11))
@pytest.mark.parametrize('kind',('layer_alias','positions_alias','contract','bound_aware','linear_bound_aware',
    'attention_callback','norm_callback','softmax_callback','refresh_binding','secret','missing_key'))
def test_construction_bindings_reject_before_any_work(arrays,layer,kind):
    owner,ops,refresh,sources=setup(arrays,layer,15)
    if kind=='layer_alias':owner.layer=float(layer)
    elif kind=='positions_alias':owner.positions=tuple(tuple(float(x) for x in row) for row in owner.positions)
    elif kind=='contract':owner.contract['key_count']=16
    elif kind=='bound_aware':ops.declare_bound=lambda *a:None
    elif kind=='linear_bound_aware':owner.linear.arithmetic=copy.copy(ops);owner.linear.arithmetic.declare_bound=lambda *a:None
    elif kind=='attention_callback':owner.attention.progress=lambda *a:None
    elif kind=='norm_callback':owner.norm.progress=lambda *a:None
    elif kind=='softmax_callback':owner.softmaxes[0].progress=lambda *a:None
    elif kind=='refresh_binding':owner.refresh_binding=copy.copy(owner.refresh_binding)
    elif kind=='secret':ops.ctx.secret_key=object()
    else:ops.ctx._param_cache.raw_rotation_keys.pop(owner.attention.required_rotation_indices()[0])
    with pytest.raises((ValueError,TypeError,AttributeError)):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls and not owner._active

@pytest.mark.parametrize('layer,delta',((0,-1),(0,1),(1,-1),(1,1),(11,1)))
def test_exact_input_inventory(arrays,layer,delta):
    owner,ops,refresh,sources=setup(arrays,layer,15)
    wrong=sources[:-1] if delta<0 else sources+[sources[0]]
    with pytest.raises(ValueError):owner.evaluate(wrong)
    assert not ops.calls and not refresh.calls

@pytest.mark.parametrize('scale',(math.nextafter(DELTA,math.inf),math.nextafter(DELTA,-math.inf),np.float64(DELTA)))
def test_fresh_layer0_scale_contract_is_exact(arrays,scale):
    owner,ops,refresh,sources=setup(arrays,0,15)
    for source in sources:source._ckks_scale=scale
    with pytest.raises(ValueError):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls

@pytest.mark.parametrize('error_type',(KeyboardInterrupt,SystemExit))
@pytest.mark.parametrize('layer',(0,1))
def test_cancel_after_first_owned_output_preserves_primary_and_sources(arrays,monkeypatch,error_type,layer):
    error=error_type('expected cancellation')
    def progress(*args):raise error
    owner,ops,refresh,sources=setup(arrays,layer,15,progress)
    if layer==0:
        monkeypatch.setattr(owner.linear,'linear_tile',lambda value,*a,**k:Value(np.zeros(65536),48,DELTA))
    with pytest.raises(error_type) as caught:owner.evaluate(sources)
    assert caught.value is error and owner.arithmetic is ops and owner.refresher is refresh
    assert not owner._active and not owner._sources and not owner._retained
    assert len(refresh.calls)==int(layer!=0)

@pytest.mark.parametrize('target',('original_input','earlier_refresh','zero_frame'))
def test_later_entry_retains_sources_refreshes_and_real_zero_tiles(arrays,monkeypatch,target):
    seen={};observed=[]
    def progress(label,value,counts):
        if label=='layer input B128 refresh':seen['refresh']=value
        if label=='public padded ciphertext tile' and 'zero' not in seen:seen['zero']=value
        if label=='key projection':
            changed={'original_input':sources[0],'earlier_refresh':seen['refresh'],'zero_frame':seen['zero']}[target]
            changed.polynomial=copy.copy(changed.polynomial);observed.append(len(ops.calls))
    owner,ops,refresh,sources=setup(arrays,1,15,progress)
    monkeypatch.setattr(owner.linear,'linear_tile',lambda value,*a,**k:Value(np.zeros(65536),24,DELTA))
    with pytest.raises(ValueError):owner.evaluate(sources)
    assert observed and len(ops.calls)==observed[0]
    assert len(refresh.calls)==1 and ops.calls.count('add')==7
    assert not owner._active and not owner._sources and not owner._retained

def test_callback_strip_preserves_borrowed_annotations(arrays):
    old=Value(np.zeros(65536),9);current=Value(np.zeros(65536),9)
    def progress(*args):old._public_bound=0.;current._public_relation=0.
    owner,ops,refresh,sources=setup(arrays,1,15,progress)
    sources[0]._public_input='preserved';owner._sources=tuple(sources);owner._source_states=tuple(owner._fingerprint(v) for v in sources)
    owner._retained.append((old,owner._fingerprint(old)));owner._callback('returned',current,{})
    assert not any(k.startswith('_public_') for v in (old,current) for k in vars(v))
    assert sources[0]._public_input=='preserved' and not ops.calls and not refresh.calls

@pytest.mark.parametrize('scale',(-1.,0.,float('nan'),float('inf')))
def test_later_entry_invalid_actual_scale_fails_before_refresh(arrays,scale):
    owner,ops,refresh,sources=setup(arrays,1,15);sources[0]._ckks_scale=scale
    with pytest.raises((ValueError,TypeError)):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls
