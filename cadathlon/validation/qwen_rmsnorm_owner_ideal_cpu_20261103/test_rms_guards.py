import gc,weakref,math
import numpy as np
import pytest
from test_rmsnorm import setup,fixture,gammas,QwenRMSContract,RAW,META,CONFIG
from encrypted_qwen_rmsnorm import QwenRMSNorm

def mutate(o,which):
    r=o.refresher;a=o.arithmetic
    if which=='refresh':r.refresh=lambda *args,**kwargs:None
    elif which=='bootstrap':r._bootstrap=lambda *args,**kwargs:None
    elif which=='engine':r.engine.meta_bootstrap=lambda *args,**kwargs:None
    elif which=='evaluator':r.evaluator.weighted_sum=lambda *args,**kwargs:None
    elif which=='helper':r.compiled_primitives[2].rescale=lambda *args,**kwargs:None
    elif which=='arithmetic':a.weighted_sum=lambda *args,**kwargs:None
    elif which=='arithmetic_helper':a.rotation.rotate=lambda *args,**kwargs:None
    elif which=='profile':a.ctx.dnum=8.
    elif which=='binding':o.refresh_binding=object()
    elif which=='gamma':o.gamma=o.gamma.copy()
    elif which=='boundaware':a.declare_bound=lambda *args:None
    elif which=='plan':r.plan['period']=128
    elif which=='engine_sparse':r.engine._sparse=True
    elif which=='bootstrap_key':
        key=r.plan['rotations'][0];a.ctx._param_cache.raw_rotation_keys[key]={'a':object(),'b':object()}
    else:raise AssertionError(which)

@pytest.mark.parametrize('which',['refresh','bootstrap','engine','evaluator','helper','arithmetic','arithmetic_helper','profile','binding','gamma','boundaware','plan','engine_sparse','bootstrap_key'])
@pytest.mark.parametrize('phase',['before','first','final'])
def test_bound_substitutions_rejected_without_next_primitive(which,phase,gammas):
    held={}
    def callback(label,value,meta):
        if (phase=='first' and label=='square') or (phase=='final' and label=='output'):
            held['calls']=dict(held['a'].calls);mutate(held['o'],which)
    o,a,r=setup('10.input.inverse_sqrt',gammas,callback=callback);held.update(o=o,a=a)
    if phase=='before':mutate(o,which);held['calls']={}
    with pytest.raises((ValueError,TypeError)):o.evaluate_tile(fixture(o),tile=0)
    assert dict(a.calls)==held['calls'] and o.last_run is None and not o._active

@pytest.mark.parametrize('field,bad',[('epsilon',True),('width',1024.),('groups',True),('iterations',22.),('first_weight',0.),('first_bias',True),('domain',[.1,1.]),('events',()),('output_level',True),('method','polynomial'),('gamma_sha256','0'*64)])
def test_complete_typed_contract_mutation_before_work(field,bad,gammas):
    o,a,r=setup('10.input.inverse_sqrt',gammas);x=fixture(o)
    object.__setattr__(o.contract,field,bad)
    with pytest.raises(ValueError):o.evaluate_tile(x,tile=0)
    assert not a.calls

@pytest.mark.parametrize('target',['source','basis','captured_inverse'])
def test_retained_wrappers_and_original_payload_refs_survive_callback(target,gammas):
    held={}
    def callback(label,current,meta):
        if label=='map':held['basis']=current;held['basis_payload']=weakref.ref(current.polynomial)
        if label=='variance':held['variance']=current
        if label=='inverse':held['inverse']=current
        selected=(target=='source' and label=='square') or (target=='basis' and label=='T4.product') or (target=='variance' and label=='map') or (target=='captured_inverse' and label=='normalized')
        if selected:
            key={'source':'source','basis':'basis','variance':'variance','captured_inverse':'inverse'}[target]
            prior=held[key];old=weakref.ref(prior.polynomial);prior.polynomial=prior.polynomial.copy();gc.collect()
            # Strong payload retention is a lifetime property, independently
            # of later fingerprint rejection of the replaced wrapper field.
            assert old() is not None
    o,a,r=setup('0.input.inverse_sqrt',gammas,callback=callback);x=fixture(o);held['source']=x
    with pytest.raises(ValueError):o.evaluate_tile(x,tile=0,capture_checkpoints=True)
    assert o.last_run is None and not o._active

def test_late_callback_captured_checkpoint_typed_metadata(gammas):
    held={}
    def callback(label,value,meta):
        if label=='map':held['variance']=held['o']._seen_variance
        if label=='output':held['variance']._ckks_nsd=True
    o,a,r=setup('0.input.inverse_sqrt',gammas)
    # Capture the complete broadcast variance object when the real mapper
    # consumes it, without substituting any primitive or arithmetic result.
    original=a.weighted_sum
    def weighted(values,weights,bias=0.):
        if len(values)==1 and values[0].level==23:o._seen_variance=values[0]
        return original(values,weights,bias=bias)
    # Install before construction so the accessor is legitimately bound.
    a.weighted_sum=weighted
    o=QwenRMSNorm(a,o.contract,CONFIG,gamma=gammas[o.contract.site],positions=(0,2,5),progress=callback);held['o']=o
    with pytest.raises(ValueError):o.evaluate_tile(fixture(o),tile=0,capture_checkpoints=True)
    assert not o._active

def test_public_annotations_removed_and_original_not_mutated(gammas):
    def callback(label,value,meta):value._public_abs_bound=0
    o,a,r=setup('0.input.inverse_sqrt',gammas,callback=callback);x=fixture(o);x._public_abs_bound=123
    result=o.evaluate_tile(x,tile=0,capture_checkpoints=True)
    assert x._public_abs_bound==123
    assert all(not any(n.startswith('_public_') for n in vars(v)) for v in (*result.checkpoints.values(),result.output))

def test_zero_payload_retained_during_first_callback_only(gammas):
    held={}
    def callback(label,value,meta):
        assert label=='square'
        old=weakref.ref(held['source'].polynomial)
        held['source'].polynomial=np.zeros(65536);gc.collect()
        assert old() is not None
    o,a,r=setup('0.input.inverse_sqrt',gammas,callback=callback)
    from ideal_primitives import Value
    held['source']=Value(np.zeros(65536))
    # This is solely an early guard/lifetime test. Zero variance+epsilon is
    # outside this site's declared domain; no inverse or valid output is run.
    with pytest.raises(ValueError):o.evaluate_tile(held['source'],tile=0)
    assert a.calls['multiply']==1 and a.calls['weighted']==0

def test_cancel_and_reentry_preserve_borrowed_objects(gammas):
    held={};primary=KeyboardInterrupt('fixed callback cancellation')
    def callback(label,value,meta):
        with pytest.raises(ValueError):held['o'].evaluate_tile(held['x'],tile=0)
        raise primary
    o,a,r=setup('10.input.inverse_sqrt',gammas,callback=callback);x=fixture(o);held.update(o=o,x=x);cache=a.ctx._param_cache
    with pytest.raises(KeyboardInterrupt) as caught:o.evaluate_tile(x,tile=0)
    assert caught.value is primary and a.ctx._param_cache is cache and not o._active and o.last_run is None

@pytest.mark.parametrize('mutation',['bool_nsd','float_level','low_level','bad_gamma','eventless_refresher'])
def test_invalid_entry_inputs_rejected_before_primitives(mutation,gammas):
    o,a,r=setup('0.input.inverse_sqrt',gammas);x=fixture(o)
    if mutation=='bool_nsd':x._ckks_nsd=True
    elif mutation=='float_level':x.level=25.
    elif mutation=='low_level':x.level=24
    elif mutation=='bad_gamma':
        with pytest.raises(ValueError):QwenRMSNorm(a,o.contract,CONFIG,gamma=np.zeros(1024),positions=(0,))
        assert not a.calls;return
    else:
        from ideal_refresh import IdentityRefresh
        with pytest.raises(ValueError):QwenRMSNorm(a,o.contract,CONFIG,gamma=gammas[o.contract.site],positions=(0,),refresher=IdentityRefresh(a))
        assert not a.calls;return
    with pytest.raises((ValueError,TypeError)):o.evaluate_tile(x,tile=0)
    assert not a.calls
