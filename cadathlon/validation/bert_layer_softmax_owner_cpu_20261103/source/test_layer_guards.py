"""Ownership/callback regressions for complete layer softmax; ideal values only."""
import copy, math
import numpy as np
import pytest
from test_layer_softmax import primitives, ideal_primitives, refresh_and_tracked, setup, fixture
from ideal_primitives import Value

def mutate(owner,ops,refresh,kind):
    if kind in ('arithmetic','layout','reduction','contract','refresher','refresh_binding','initial','round'):
        setattr(owner,kind,copy.copy(getattr(owner,kind)))
    elif kind=='positions':owner.original_positions=tuple(float(x) for x in owner.original_positions)
    elif kind=='progress':owner.progress=lambda *a:None
    elif kind=='norm_depth':owner.attention_norm_depth=float(owner.attention_norm_depth)
    elif kind=='continuation':owner.continuation_levels+=1
    elif kind=='contract_layer':owner.contract.__dict__['layer']=float(owner.contract.layer)
    elif kind=='contract_divisor':owner.contract.__dict__['normalized_divisor']=float(owner.contract.normalized_divisor)
    elif kind=='contract_squarings':owner.contract.__dict__['squarings']=float(owner.contract.squarings)
    elif kind=='initial_callback':owner.initial.reduction.progress=lambda *a:None
    elif kind=='round_callback':owner.round.reduction.progress=lambda *a:None
    elif kind=='reduction_callback':owner.reduction.progress=lambda *a:None
    elif kind=='bound_aware':ops.declare_bound=lambda *a:None
    elif kind=='implicit_refresh':ops.refresher=object()
    elif kind=='secret':ops.ctx.secret_key=object()
    elif kind=='refresh_plan':refresh.plan['rotations']=()
    elif kind=='refresh_helpers':refresh.compiled_primitives=()
    elif kind=='missing_key':ops.ctx._param_cache.raw_rotation_keys.pop(refresh.plan['rotations'][0])
    elif kind=='mode_alias':ops.ctx._param_cache.per_level_rotation_keys=0
    else:raise AssertionError(kind)

@pytest.mark.parametrize('kind',('arithmetic','layout','reduction','contract','refresher','refresh_binding','initial','round',
    'positions','progress','norm_depth','continuation','contract_layer','contract_divisor','contract_squarings',
    'initial_callback','round_callback','reduction_callback','bound_aware','implicit_refresh','secret',
    'refresh_plan','refresh_helpers','missing_key','mode_alias'))
@pytest.mark.parametrize('when',('entry','first_callback','final_callback'))
def test_bindings_stop_before_next_primitive(kind,when):
    stopped=[]
    def progress(label,value,counts):
        if (when=='first_callback' and not stopped) or (when=='final_callback' and label=='layer softmax final masked probabilities'):
            stopped.append(len(ops.calls));mutate(owner,ops,refresh,kind)
    owner,ops,refresh,layout=setup(1,15,7,progress)
    source=Value(fixture(owner,layout,15),21)
    if when=='entry':mutate(owner,ops,refresh,kind)
    with pytest.raises((ValueError,TypeError,AttributeError)):owner.evaluate_tile(source,capture_checkpoints=True)
    assert len(ops.calls)==(0 if when=='entry' else stopped[0])
    assert not owner._active and owner._source is None and not owner._retained and not owner._captured

@pytest.mark.parametrize('target,at',(('source','layer softmax initial'),('initial','layer softmax round0'),
    ('round0','layer softmax round1'),('before round1','layer softmax round1'),
    ('final continuation refresh','layer softmax final masked probabilities')))
@pytest.mark.parametrize('field',('polynomial','_ckks_scale'))
def test_source_and_retained_checkpoints_are_guarded(target,at,field):
    seen={};stopped=[]
    def progress(label,value,counts):
        if label.startswith('layer softmax '):seen[label.removeprefix('layer softmax ')]=value
        if label==at:
            changed=source if target=='source' else seen[target]
            if field=='polynomial':changed.polynomial=copy.copy(changed.polynomial)
            else:changed._ckks_scale=math.nextafter(changed._ckks_scale,math.inf)
            stopped.append(len(ops.calls))
    owner,ops,refresh,layout=setup(1,15,7,progress);source=Value(fixture(owner,layout,15),21)
    with pytest.raises(ValueError):owner.evaluate_tile(source,capture_checkpoints=True)
    assert len(ops.calls)==stopped[0] and not owner._active and not owner._captured

def test_all_returned_annotations_are_removed_after_each_callback():
    seen=[]
    def progress(label,value,counts):
        if label.startswith('layer softmax '):seen.append(value)
        for earlier in seen:earlier._public_abs_bound=0.
    owner,ops,refresh,layout=setup(0,15,7,progress);source=Value(fixture(owner,layout,15),45)
    source._public_borrowed='keep'
    output,stages=owner.evaluate_tile(source,capture_checkpoints=True)
    assert source._public_borrowed=='keep'
    assert not any(n.startswith('_public_') for value in (output,*stages.values()) for n in vars(value))

@pytest.mark.parametrize('layer',(0,1))
@pytest.mark.parametrize('stage',('initial','refresh','round','final'))
@pytest.mark.parametrize('error_type',(KeyboardInterrupt,SystemExit))
def test_cancellation_keeps_primary_and_borrowed_owners(layer,stage,error_type):
    error=error_type('expected cancellation')
    def progress(label,value,counts):
        if (stage=='initial' or (stage=='round' and label=='layer softmax round0')
            or (stage=='final' and label=='layer softmax final masked probabilities')):raise error
    owner,ops,refresh,layout=setup(layer,15,7,progress);source=Value(fixture(owner,layout,15),45 if layer==0 else 21)
    if stage=='refresh':refresh.inject=lambda *a:(_ for _ in ()).throw(error)
    with pytest.raises(error_type) as caught:owner.evaluate_tile(source,capture_checkpoints=True)
    assert caught.value is error and owner.arithmetic is ops and owner.refresher is refresh
    assert not owner._active and owner._source is None and not owner._retained and not owner._captured

@pytest.mark.parametrize('capture',(0,1,np.bool_(True),'yes'))
def test_capture_requires_bool(capture):
    owner,ops,refresh,layout=setup(0,1,1)
    with pytest.raises(ValueError):owner.evaluate_tile(Value(np.zeros(65536),45),capture_checkpoints=capture)
    assert not ops.calls and not refresh.calls

def test_reentry_fails_without_another_operation():
    entered=[]
    def progress(*args):
        if not entered:
            entered.append(len(ops.calls))
            with pytest.raises(ValueError,match='reentered'):owner.evaluate_tile(source)
            assert len(ops.calls)==entered[0]
    owner,ops,refresh,layout=setup(0,1,1,progress);source=Value(np.zeros(65536),45)
    owner.evaluate_tile(source);assert entered
