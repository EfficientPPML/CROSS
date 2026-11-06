"""Boundary/lifetime regressions for split initial softmax; backend-free doubles."""
import copy,math
import numpy as np
import pytest
from test_split_initial import primitives,ideal_primitives,refresh_and_tracked,setup,fixture,RAW,Q
from ideal_primitives import Value,DELTA
from encrypted_bert_split_initial import BertSplitInitialSoftmax,prefix_scale_plan,tail_scale_plan
from encrypted_bert_initial_softmax import initial_softmax_plan

@pytest.mark.parametrize('scale',(DELTA,math.nextafter(DELTA,math.inf),math.nextafter(DELTA,-math.inf)))
def test_tail_preserves_exact_existing_circuit_and_encoding_targets(scale):
    original=initial_softmax_plan(Q[:58],input_level=28,canonical_scale=DELTA,input_scale=scale)
    prefix=prefix_scale_plan(Q[:58],input_level=28,canonical_scale=DELTA,input_scale=scale)
    tail=tail_scale_plan(Q[:52],input_level=25,canonical_scale=DELTA,input_scale=original['scales']['normalized_t'])
    assert prefix['scales']=={n:original['scales'][n] for n in prefix['scales']}
    assert tail['scales']=={n:original['scales'][n] for n in tail['scales']}
    assert tail['levels']=={n:original['levels'][n] for n in tail['levels']}
    assert tail['static_inverse_scale_plan']==original['static_inverse_scale_plan']
    assert tail['nominal_inverse_target_plan']==original['nominal_inverse_target_plan']

def mutate(owner,ops,refresh,kind):
    if kind=='arithmetic':owner.arithmetic=copy.copy(ops)
    elif kind=='layout':owner.layout=copy.copy(owner.layout)
    elif kind=='reduction':owner.reduction=copy.copy(owner.reduction)
    elif kind=='contract':owner.contract=copy.copy(owner.contract)
    elif kind=='refresher':owner.refresher=copy.copy(refresh)
    elif kind=='binding':owner.refresh_binding=copy.copy(owner.refresh_binding)
    elif kind=='positions':owner.original_positions=tuple(float(x) for x in owner.original_positions)
    elif kind=='callback':owner.progress=lambda *a:None
    elif kind=='reduction_callback':owner.reduction.progress=lambda *a:None
    elif kind=='contract_layer':owner.contract.__dict__['layer']=0.
    elif kind=='contract_domain':owner.contract.__dict__['centered_domain']=list(owner.contract.centered_domain)
    elif kind=='contract_divisor':owner.contract.__dict__['normalized_divisor']=16.
    elif kind=='contract_squarings':owner.contract.__dict__['squarings']=5.
    elif kind=='contract_updates':owner.contract.__dict__['initial_updates']=6.
    elif kind=='contract_seed':owner.contract.__dict__['initial_seed']=np.float64(.8)
    elif kind=='contract_coefficients':owner.contract.__dict__['exponential_coefficients']=list(owner.contract.exponential_coefficients)
    elif kind=='bound_aware':ops.declare_bound=lambda *a:None
    elif kind=='implicit_refresh':ops.refresher=object()
    elif kind=='secret':ops.ctx.secret_key=object()
    elif kind=='refresh_plan':refresh.plan['rotations']=()
    elif kind=='refresh_helpers':refresh.compiled_primitives=()
    elif kind=='missing_key':ops.ctx._param_cache.raw_rotation_keys.pop(refresh.plan['rotations'][0])
    elif kind=='mode_alias':ops.ctx._param_cache.per_level_rotation_keys=0
    else:raise AssertionError(kind)

@pytest.mark.parametrize('kind',('arithmetic','layout','reduction','contract','refresher','binding','positions',
    'callback','reduction_callback','contract_layer','contract_domain','contract_divisor','contract_squarings',
    'contract_updates','contract_seed','contract_coefficients','bound_aware','implicit_refresh','secret',
    'refresh_plan','refresh_helpers','missing_key','mode_alias'))
@pytest.mark.parametrize('when',('entry','first_callback','final_callback'))
def test_construction_and_contract_changes_stop_before_next_primitive(kind,when):
    captured=[]
    def progress(label,value,counts):
        if (when=='first_callback' and not captured) or (when=='final_callback' and label=='split initial probabilities'):
            captured.append(len(ops.calls));mutate(owner,ops,refresh,kind)
    owner,ops,refresh,layout=setup(0,15,7,progress=progress)
    source=Value(fixture(owner,layout,15),21)
    if when=='entry':mutate(owner,ops,refresh,kind)
    with pytest.raises((ValueError,TypeError,AttributeError)):owner.evaluate_tile(source)
    assert len(ops.calls)==(0 if when=='entry' else captured[0])
    assert not owner._active and not owner._sources and not owner._retained and not owner._captured

@pytest.mark.parametrize('target,at',(('score','split initial refreshed_normalized_t'),
    ('refreshed_normalized_t','split initial base_exp'),('base_exp','split initial mean_exp'),
    ('mean_exp','split initial inverse_update1'),('score_mean','split initial probabilities')))
@pytest.mark.parametrize('field',('_ckks_scale','polynomial'))
def test_prefix_source_and_all_retained_branches_are_immutable_across_callbacks(target,at,field):
    seen={};stopped=[]
    def progress(label,value,counts):
        if label.startswith('split initial '):seen[label.removeprefix('split initial ')]=value
        if label==at:
            changed=source if target=='score' else seen[target]
            if field=='_ckks_scale':changed._ckks_scale=math.nextafter(changed._ckks_scale,math.inf)
            else:changed.polynomial=copy.copy(changed.polynomial)
            stopped.append(len(ops.calls))
    owner,ops,refresh,layout=setup(0,15,7,progress=progress);source=Value(fixture(owner,layout,15),21)
    with pytest.raises(ValueError):owner.evaluate_tile(source,capture_checkpoints=True)
    assert len(ops.calls)==stopped[0] and not owner._active and not owner._retained and not owner._captured

def test_current_and_earlier_returned_annotations_are_cleared_after_callbacks():
    seen=[]
    def progress(label,value,counts):
        if label.startswith('split initial '):seen.append(value)
        for earlier in seen:earlier._public_abs_bound=0.
    owner,ops,refresh,layout=setup(4,15,7,progress=progress);source=Value(fixture(owner,layout,15),21)
    source._public_source='keep'
    output,stages=owner.evaluate_tile(source,capture_checkpoints=True)
    assert source._public_source=='keep'
    assert not any(n.startswith('_public_') for value in stages.values() for n in vars(value))
    assert not any(n.startswith('_public_') for n in vars(output))

@pytest.mark.parametrize('stage',('prefix','refresh','tail'))
@pytest.mark.parametrize('error_type',(KeyboardInterrupt,SystemExit))
def test_cancellation_preserves_primary_and_releases_only_local_references(stage,error_type):
    error=error_type('expected cancellation')
    def progress(label,value,counts):
        if stage=='prefix' or (stage=='tail' and label=='split initial base_exp'):raise error
    owner,ops,refresh,layout=setup(1,15,7,progress=progress);source=Value(fixture(owner,layout,15),21)
    if stage=='refresh':refresh.inject=lambda *a:(_ for _ in ()).throw(error)
    with pytest.raises(error_type) as caught:owner.evaluate_tile(source,capture_checkpoints=True)
    assert caught.value is error and owner.arithmetic is ops and owner.refresher is refresh
    assert not owner._active and not owner._sources and not owner._retained and not owner._captured

@pytest.mark.parametrize('capture',(0,1,np.bool_(True),'yes'))
def test_capture_flag_is_explicit_boolean(capture):
    owner,ops,refresh,layout=setup(0,1,1)
    with pytest.raises(ValueError):owner.evaluate_tile(Value(np.zeros(65536),21),capture_checkpoints=capture)
    assert not ops.calls and not refresh.calls

def test_reentry_is_rejected_without_another_primitive():
    entered=[]
    def progress(*args):
        if not entered:
            entered.append(len(ops.calls))
            with pytest.raises(ValueError,match='reentered'):owner.evaluate_tile(source)
            assert len(ops.calls)==entered[0]
    owner,ops,refresh,layout=setup(0,1,1,progress=progress);source=Value(np.zeros(65536),21)
    owner.evaluate_tile(source);assert entered and len(refresh.calls)==1
