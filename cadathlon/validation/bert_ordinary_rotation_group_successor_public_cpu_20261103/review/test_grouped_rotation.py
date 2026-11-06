"""Tiny mocked metadata/lifecycle tests; no JAX, contexts, keys or HE."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys,importlib
import numpy as np
import pytest
HERE=Path(__file__).parent
sys.path[:0]=[str(HERE),str(HERE/'base')]
from lease_test_helpers import env as lease_env,build,Array,Value,Field

class Binding:
    def __init__(self,fn):self.executable=fn;self.controls=()
    def __call__(self,*args):return self.executable(*args)
class Compiler:
    def __init__(self,**kwargs):self.compile_count=0
    def bind(self,fn,operands,**kwargs):self.compile_count+=1;return Binding(fn)
    def clear_executables(self):pass

@pytest.fixture
def env(lease_env,monkeypatch):
    e=lease_env;e.decompositions=0;e.ordinary=0
    monkeypatch.setitem(sys.modules,'finite_field',SimpleNamespace(BarrettContext=Field))
    monkeypatch.setattr(sys.modules['encrypted_compilation'],'ExplicitArrayCompiler',Compiler)
    monkeypatch.setattr(sys.modules['jax'],'default_device',lambda device:nullcontext(),raising=False)
    module=importlib.import_module('grouped_rotation');monkeypatch.setattr(module,'_FAILURE_QUARANTINE',e.quarantine)
    e.group_module=module
    return e

def setup(env,**limits):
    _,ctx,lease,backing=build(env)
    old=ctx.he_rot
    class Kernel:
        def __init__(self):
            self.r=2;self.c=4;self.batch=1;self.sizeQl=4;self.sizeP=2;self.sizeQlP=6;self.numPartQl=2
            self.overall_moduli=[17,19,23,29,31,37];self.extend_moduli=[31,37];self._controls_ready=True;self.ff_context_cls=Field
            self.ct_full=object();self.ct_in=object();self.ct_approx=object();self.bconv=object()
            self.select_tower_index=[[0,1],[2,3]];self.non_select_tower_index=[];self.restore_indices=[]
        def _decompose_array(self,payload):
            env.decompositions+=1
            return Array(np.zeros((2,1,1,2,4,6),dtype='uint32'))
        def _rotate_from_digits_array(self,payload,digits,a,b,coef):
            env.work+=1
            flat=payload.data.reshape(1,2,8,-1)
            return Array((np.take(flat,coef.data,axis=2)+np.uint32(a.data.flat[0]+b.data.flat[0])).reshape(payload.shape))
    class Accessor:
        def __init__(self):self._instances={}
        def __getitem__(self,key):
            operation=old[key];operation._herot=Kernel();self._instances[key]=operation;return operation
        def take_explicit_rotation_fn(self,key):
            self._instances.pop(key,None);return old.take_explicit_rotation_fn(key)
    ctx.he_rot=Accessor()
    value=Value();value.moduli=[17,19,23,29];value.degree=8;value.degree_layout=(2,4);value.batch=1;value.num_elements=2;value.precision=32;value.ntt_ctx=object()
    owner=env.group_module.GroupedCompiledRotation(ctx,source_lease=lease,**limits)
    return owner,ctx,lease,backing,value

def test_same_input_zero_negative_duplicate_order_and_metadata(env):
    owner,ctx,_,_,value=setup(env);before=value.polynomial.data.copy()
    outputs=owner.rotate_many(value,(0,1,-1,1,4))
    assert outputs[0] is outputs[4] is value and outputs[1] is not outputs[3]
    assert env.decompositions==1 and env.work==3 and owner.last_group['mode']=='digits'
    for index,result in zip((0,1,3,1,0),outputs):
        assert result._ckks_scale==17.25 and result._ckks_nsd==1
        if index:
            baseline=env.module.CompiledRotation(ctx).rotate(value,index)
            np.testing.assert_array_equal(result.polynomial.data,baseline.polynomial.data)
    np.testing.assert_array_equal(value.polynomial.data,before)
    assert owner._group is None and not owner._active

def test_two_groups_recompute_digits_and_clear_releases(env):
    owner,_,_,_,value=setup(env)
    for _ in range(2):owner.rotate_many(value,(1,2));owner.clear()
    assert env.decompositions==2 and owner._group is None
    owner.close()
    with pytest.raises(RuntimeError):owner.rotate_many(value,(1,))

def test_zero_only_never_constructs_controls(env):
    owner,_,_,_,value=setup(env);assert owner.rotate_many(value,(0,4,-4))==(value,)*3
    assert env.work==env.decompositions==0

@pytest.mark.parametrize('limit', ['results','bytes'])
def test_result_budget_rejected_before_work(env,limit):
    args={'max_results':1} if limit=='results' else {'max_result_bytes':1}
    owner,_,_,_,value=setup(env,**args)
    with pytest.raises(ValueError):owner.rotate_many(value,(1,2))
    assert env.work==env.decompositions==0

def test_digit_budget_falls_back_before_decomposition(env):
    owner,_,_,_,value=setup(env,max_digit_bytes=1)
    out=owner.rotate_many(value,(1,2))
    assert len(out)==2 and env.decompositions==0 and owner.last_group['mode']=='ordinary'

@pytest.mark.parametrize('what',['secret_ctx','secret_cache','secret_parameters','key_missing','key_dtype','input_moduli','input_nsd','source_scale','lease_closed'])
def test_invalid_shared_contract_fails_before_work(env,what):
    owner,ctx,lease,_,value=setup(env)
    if what=='secret_ctx':ctx.secret_key=object()
    elif what=='secret_cache':ctx._param_cache.secret_key=object()
    elif what=='secret_parameters':ctx.parameters['secret_key']=object()
    elif what=='key_missing':ctx._param_cache.raw_rotation_keys.pop(1)
    elif what=='key_dtype':ctx._param_cache.raw_rotation_keys[1]['a']=np.zeros((2,6,8))
    elif what=='input_moduli':value.moduli[0]=41
    elif what=='input_nsd':value._ckks_nsd=True
    elif what=='source_scale':value._ckks_scale=float('nan')
    elif what=='lease_closed':lease.retire()
    with pytest.raises((ValueError,TypeError)):owner.rotate_many(value,(1,2))
    assert env.work==env.decompositions==0

@pytest.mark.parametrize('what',['payload','scale','moduli','context','key','controls','retained','binding'])
def test_callback_mutation_rejected_before_next_primitive_and_quarantined(env,what):
    owner,ctx,_,_,value=setup(env);seen=[]
    def callback(n,index,result):
        seen.append(env.work)
        if what=='payload':value.polynomial=Array(value.polynomial.data)
        elif what=='scale':value._ckks_scale+=1
        elif what=='moduli':value.moduli[0]=41
        elif what=='context':ctx._param_cache.q_towers[0]=41
        elif what=='key':ctx._param_cache.raw_rotation_keys[2]['a']=ctx._param_cache.raw_rotation_keys[1]['a']
        elif what=='controls':owner._group.kernel.sizeQl=2
        elif what=='retained':result.polynomial=Array(result.polynomial.data)
        elif what=='binding':owner._group.bindings[0].executable=lambda *args:None
    with pytest.raises(ValueError):owner.rotate_many(value,(1,2),progress=callback)
    assert env.work==seen[0]==1 and owner._failed is not None and env.quarantine
    with pytest.raises(RuntimeError):owner.clear()

@pytest.mark.parametrize('action',['clear','close','rotate','nested'])
def test_callback_reentry_and_clear_rejected_without_invalidating_group(env,action):
    owner,_,_,_,value=setup(env)
    def callback(*args):
        with pytest.raises(RuntimeError):
            if action=='clear':owner.clear()
            elif action=='close':owner.close()
            elif action=='rotate':owner.rotate(value,1)
            else:owner.rotate_many(value,(1,))
    assert len(owner.rotate_many(value,(1,2),progress=callback))==2
    assert owner._failed is None

def test_callback_cancel_preserved_and_original_backing_retained(env):
    owner,_,_,_,value=setup(env);error=KeyboardInterrupt('cancel')
    def callback(*args):raise error
    with pytest.raises(KeyboardInterrupt) as caught:owner.rotate_many(value,(1,2),progress=callback)
    assert caught.value is error and env.work==1
    assert env.quarantine[0][2].source is value and owner._group is not None

def test_readiness_failure_attempts_all_backing_and_quarantines(env):
    owner,_,_,_,value=setup(env)
    env.fail=lambda item:getattr(item,'shape',None)==(2,1,1,2,4,6)
    with pytest.raises(TimeoutError):owner.rotate_many(value,(1,2))
    assert env.work==0 and env.decompositions==1 and owner._failed is not None
    assert any(item is value.polynomial for item in env.barriers)

@pytest.mark.parametrize('what',['getter','q_selector','level_selector','accessor'])
def test_bound_key_control_methods_cannot_change_during_callback(env,what):
    owner,ctx,_,_,value=setup(env)
    def callback(*args):
        if what=='getter':ctx._param_cache.get_rot_key=lambda *args:None
        elif what=='q_selector':ctx._param_cache.q_moduli_at_level=lambda level:[17,19,23,29]
        elif what=='level_selector':ctx._param_cache.level_for_num_q=lambda n:1
        else:ctx.he_rot=SimpleNamespace()
    with pytest.raises((ValueError,AttributeError)):owner.rotate_many(value,(1,2),progress=callback)
    assert env.work==1 and owner._failed is not None

def test_mixed_result_readiness_failure_preserves_input_control_and_results(env):
    owner,_,_,_,value=setup(env)
    env.fail=lambda item:env.work>0 and getattr(item,'shape',None)==value.shape
    with pytest.raises(TimeoutError):owner.rotate_many(value,(1,2))
    assert env.work==1 and owner._failed is not None
    group=env.quarantine[0][2]
    assert group.source is value and group.kernel is not None and group.bindings and group.digits is not None
    assert any(getattr(v,'shape',None)==(2,2,4,6) for v in env.barriers)

def test_compile_failure_retains_original_kernel_and_key_backing(env,monkeypatch):
    owner,_,_,_,value=setup(env);error=RuntimeError('compiler failed')
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(owner._group_compiler,'bind',fail)
    with pytest.raises(RuntimeError) as caught:owner.rotate_many(value,(1,2))
    assert caught.value is error and env.work==0
    group=env.quarantine[0][2];assert group.source is value and group.kernel is not None and group.states

def test_mutable_public_array_uses_original_path_before_decomposition(env):
    owner,ctx,lease,backing,value=setup(env)
    original=ctx._param_cache.raw_rotation_keys[1]['a']
    # Construct a new authenticated mock lease for the mutable producer bytes.
    ctx._param_cache.raw_rotation_keys[1]['a']=original.copy()
    from prepared_rotation import AuthenticatedRotationLease
    lease=AuthenticatedRotationLease(ctx,context_id='b'*64,backing_guard=lambda *args:None)
    owner=env.group_module.GroupedCompiledRotation(ctx,source_lease=lease)
    owner.rotate_many(value,(1,2));assert env.decompositions==0 and owner.last_group['mode']=='ordinary'

def test_opt_in_arithmetic_factory_preserves_default_rotate_and_zero_alias(env,monkeypatch):
    owner,ctx,lease,_,value=setup(env)
    class Arithmetic:
        def __init__(self,context,**kwargs):self.ctx=context;self.rotation=env.module.CompiledRotation(context);self.prepares=[]
        def _level(self,value):return 1
        def _prepare(self,level):self.prepares.append(level);self.rotation.clear()
        def _finish(self,value):raise AssertionError('group must not finish/mutate borrowed zero aliases')
        def rotate(self,value,index):return value if not index%self.ctx.num_slots else self.rotation.rotate(value,index%self.ctx.num_slots)
        def _clear_working_caches(self):self.rotation.clear()
    monkeypatch.setitem(sys.modules,'encrypted_packed_qwen',SimpleNamespace(CompiledArithmetic=Arithmetic))
    arithmetic=env.group_module.create_grouped_arithmetic(ctx,source_lease=lease)
    outputs=arithmetic.rotate_many(value,(0,1,1))
    assert outputs[0] is value and outputs[1] is not outputs[2] and arithmetic.prepares==[1]
    assert arithmetic.rotate(value,0) is value
    assert arithmetic.rotate.__func__ is Arithmetic.rotate

@pytest.mark.parametrize('change',['dtype','shape','device','ntt'])
def test_bad_primitive_output_shape_dtype_device_or_ntt_rejected(env,monkeypatch,change):
    owner,ctx,_,_,value=setup(env,max_digit_bytes=1)
    original=env.module.CompiledRotation.rotate
    def bad(self,v,index):
        out=original(self,v,index)
        if change=='dtype':out.polynomial=Array(out.polynomial.data.astype('float64'))
        elif change=='shape':out.polynomial=Array(out.polynomial.data[:,:,:,:1])
        elif change=='device':out.polynomial.device='cpu1'
        else:out.ntt_ctx=object()
        return out
    monkeypatch.setattr(env.module.CompiledRotation,'rotate',bad)
    with pytest.raises(ValueError):owner.rotate_many(value,(1,2))
    assert env.work==1 and owner._failed is not None

def test_original_runtime_and_lease_bytes_unchanged():
    import hashlib,json
    pins=json.loads((HERE/'origins.json').read_bytes())['fixed_input_hashes']
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest for name,digest in pins.items())
