"""Mocked execution/lifecycle tests; no JAX/context/key generation or HE."""
from pathlib import Path
import importlib
import math
import sys
from types import SimpleNamespace
import numpy as np
import pytest

HERE=Path(__file__).parent
sys.path[:0]=[str(HERE),str(HERE/'base')]


class Array:
    def __init__(self,value,device='cpu0'):
        self.data=np.array(value,copy=True);self.device=device
    @property
    def shape(self):return self.data.shape
    @property
    def dtype(self):return self.data.dtype
    def devices(self):return {self.device}
    def __getitem__(self,key):return Array(self.data[key],self.device)
    def astype(self,dtype):return Array(self.data.astype(dtype),self.device)


class Value:
    def __init__(self,q=4,device='cpu0'):
        self.polynomial=Array(np.arange(1*2*2*4*q,dtype='uint32').reshape(1,2,2,4,q),device)
        self._ckks_scale=17.25;self._ckks_nsd=1;self.marker='preserve'
    @property
    def num_moduli(self):return self.polynomial.shape[-1]
    @property
    def shape(self):return self.polynomial.shape
    def _clone_with_payload(self,payload):
        clone=object.__new__(Value);clone.__dict__=self.__dict__.copy();clone.polynomial=payload;return clone


class Compiler:
    def __init__(self):self.compile_count=0;self.cleared=0
    def bind(self,fn,operands,**kwargs):self.compile_count+=1;return fn
    def clear_executables(self):self.cleared+=1


class Field:
    computation_format_is_standard=True
    def __init__(self,moduli):self.moduli=moduli
    def to_computation_format(self,value):return value


@pytest.fixture
def env(monkeypatch):
    state=SimpleNamespace(barriers=[],fail=None,work=0,quarantine=[])
    def asarray(x,dtype=None):
        return Array(x.data if isinstance(x,Array) else x).astype(dtype or (x.dtype if hasattr(x,'dtype') else 'int64'))
    def ready(value):
        state.barriers.append(value)
        if state.fail is not None and state.fail(value):raise TimeoutError('mock late device readiness')
        return value
    jnp=SimpleNamespace(asarray=asarray,uint64=np.uint64,uint32=np.uint32,int32=np.int32)
    jax=SimpleNamespace(Array=Array,numpy=jnp,block_until_ready=ready)
    monkeypatch.setitem(sys.modules,'jax',jax);monkeypatch.setitem(sys.modules,'jax.numpy',jnp)
    monkeypatch.setitem(sys.modules,'encrypted_polynomial',SimpleNamespace(EncryptedPolynomialEvaluator=lambda ctx:None))
    monkeypatch.setitem(sys.modules,'encrypted_compilation',SimpleNamespace(ExplicitArrayCompiler=Compiler))
    def secure(owner,value):
        ctx=getattr(owner,'ctx',owner)
        if ctx.secret_key is not None:raise ValueError('secret')
    monkeypatch.setitem(sys.modules,'support',SimpleNamespace(require_secure_execution=secure))
    import prepared_rotation as module
    monkeypatch.setattr(module,'_FAILURE_QUARANTINE',state.quarantine)
    state.module=module
    return state


def build(env,*,mutable=False,max_entries=2,max_bytes=2<<30,source_guard=None):
    def host(x):return x.copy() if mutable else np.frombuffer(x.tobytes(),dtype=x.dtype).reshape(x.shape)
    raw={i:{p:host(np.full((2,6,8),i+(p=='b'),dtype='uint32')) for p in ('a','b')} for i in (1,2,3)}
    maps={i:Array(np.roll(np.arange(8,dtype='int32'),-i)) for i in raw}
    cache=SimpleNamespace(degree=8,batch=1,degree_layout=(2,4),dnum=2,q_towers=[17,19,23,29],p_towers=[31,37],
        per_level_rotation_keys=False,cache_rotation_keys=True,composite_degree=2,secret_key=None,
        raw_rotation_keys=raw,coef_maps=maps,ff_context_cls=Field,_formatted_rotation_keys={},prepares=0)
    cache.q_moduli_at_level=lambda level:cache.q_towers[:2*(level+1)]
    cache.level_for_num_q=lambda q:q//2-1
    def get(index,level):
        cache.prepares+=1;n=2*(level+1)
        pair=[]
        for part in ('a','b'):
            a=np.transpose(cache.raw_rotation_keys[index][part],(0,2,1))
            pair.append(Array(np.concatenate((a[...,:n],a[...,-2:]),axis=-1).reshape(2,2,4,n+2)))
        cache._formatted_rotation_keys[index]=tuple(pair)
        return *pair,cache.coef_maps[index]
    cache.get_rot_key=get
    ctx=SimpleNamespace(_param_cache=cache,degree=8,num_slots=4,degree_layout=(2,4),dnum=2,composite_degree=2,
                        parameters={'secret_key':None},secret_key=None)
    class Accessor:
        def __init__(self):self._instances={}
        def __getitem__(self,key):
            level,index=key
            obj=SimpleNamespace(_rotation_state=lambda:env.module.rotation_state_without_controls(cache,level,index))
            self._instances[key]=obj;return obj
        def take_explicit_rotation_fn(self,key):
            self._instances.pop(key,None)
            def fn(payload,a,b,coef):
                env.work+=1
                # Distinct explicit operands affect the independent fake result.
                flat=payload.data.reshape(1,2,8,-1)
                result=np.take(flat,coef.data,axis=2)+np.uint32(a.data.flat[0]+b.data.flat[0])
                return Array(result.reshape(payload.shape),payload.device)
            return fn
    ctx.he_rot=Accessor()
    pinned={i:tuple(id(x) for x in (raw[i]['a'],raw[i]['b'],maps[i])) for i in raw}
    guard_state=SimpleNamespace(retired=False,changed=False,calls=0)
    def guard(index,a,b,m):
        guard_state.calls+=1
        if guard_state.retired or guard_state.changed or tuple(map(id,(a,b,m)))!=pinned[index]:
            raise ValueError('authenticated backing changed or retired')
        if source_guard is not None:source_guard(index,a,b,m)
    lease=env.module.AuthenticatedRotationLease(ctx,context_id='a'*64,backing_guard=guard)
    owner=env.module.PreparedCompiledRotation(ctx,source_lease=lease,max_entries=max_entries,max_bytes=max_bytes)
    return owner,ctx,lease,guard_state


def test_exact_mock_payload_and_metadata_against_unchanged_baseline(env):
    owner,ctx,lease,_=build(env);baseline=env.module.CompiledRotation(ctx);value=Value()
    before=value.polynomial.data.copy()
    for index in (1,1,2,2,1,3,1):
        left=owner.rotate(value,index);right=baseline.rotate(value,index)
        np.testing.assert_array_equal(left.polynomial.data,right.polynomial.data)
        assert left._ckks_scale==right._ckks_scale==17.25
        assert left._ckks_nsd==right._ckks_nsd==1
        assert left.marker==right.marker=='preserve'
    np.testing.assert_array_equal(value.polynomial.data,before)
    assert owner.stats['hits']==4
    assert owner.retained_entries==2


def test_63_identical_baby_keys_prepare_once(env):
    owner,ctx,_,_=build(env);v=Value()
    for _ in range(63):v=owner.rotate(v,1)
    assert ctx._param_cache.prepares==1 and owner.stats['hits']==62
    assert owner.compile_count==1
    assert ctx._param_cache._formatted_rotation_keys=={}
    assert ctx.he_rot._instances=={}


def test_two_entry_lru_and_accounting(env):
    owner,ctx,_,_=build(env);v=Value()
    for i in (1,2,1,3,1):owner.rotate(v,i)
    expected=2*(2*2*2*4*6*4+8*4)
    assert owner.retained_bytes==expected and owner.retained_entries==2
    assert [key[-1] for key in owner._entries]==[3,1]
    assert owner.stats['evictions']==1 and owner.stats['hits']==2


def test_oversized_never_published(env):
    owner,ctx,_,_=build(env,max_bytes=1)
    owner.rotate(Value(),1);owner.rotate(Value(),1)
    assert owner.retained_entries==0 and ctx._param_cache.prepares==2
    assert owner.stats['oversized']==2


def test_byte_cap_evicts_before_second_entry(env):
    owner,ctx,_,_=build(env,max_bytes=900)
    owner.rotate(Value(),1);owner.rotate(Value(),2)
    assert owner.retained_entries==1 and owner.retained_bytes<=900


def test_mutable_host_fallback_reflects_new_contents(env):
    owner,ctx,_,_=build(env,mutable=True)
    first=owner.rotate(Value(),1)
    ctx._param_cache.raw_rotation_keys[1]['a']+=7
    second=owner.rotate(Value(),1)
    assert owner.retained_entries==0 and ctx._param_cache.prepares==2
    np.testing.assert_array_equal(second.polynomial.data,first.polynomial.data+7)


@pytest.mark.parametrize('mode',['key','map','guard','context_id','raw_inventory','cache','field','moduli','secret','per_level'])
def test_provenance_or_context_drift_fails_before_next_primitive(env,mode):
    owner,ctx,lease,guard=build(env);owner.rotate(Value(),1);work=env.work
    if mode=='key':ctx._param_cache.raw_rotation_keys[1]['a']=ctx._param_cache.raw_rotation_keys[2]['a']
    elif mode=='map':ctx._param_cache.coef_maps[1]=ctx._param_cache.coef_maps[2]
    elif mode=='guard':guard.changed=True
    elif mode=='context_id':lease.context_id='b'*64
    elif mode=='raw_inventory':ctx._param_cache.raw_rotation_keys=dict(ctx._param_cache.raw_rotation_keys)
    elif mode=='cache':ctx._param_cache=SimpleNamespace(**vars(ctx._param_cache))
    elif mode=='field':ctx._param_cache.ff_context_cls=type('OtherField',(Field,),{})
    elif mode=='moduli':ctx._param_cache.q_towers[0]=41
    elif mode=='secret':ctx._param_cache.secret_key=object()
    elif mode=='per_level':ctx._param_cache.per_level_rotation_keys=0
    with pytest.raises(ValueError):owner.rotate(Value(),1)
    assert env.work==work


def test_level_change_clears_prepared_state_and_rebinds(env):
    owner,ctx,_,_=build(env);owner.rotate(Value(),1)
    owner.rotate(Value(q=2),1)
    assert owner.retained_entries==1 and owner.compile_count==2
    assert next(iter(owner._entries))[0][0]==0
    assert ctx._param_cache.prepares==2


def test_device_change_rejects_cross_device_state_before_publication(env):
    owner,ctx,_,_=build(env);owner.rotate(Value(),1)
    with pytest.raises(ValueError,match='device'):owner.rotate(Value(device='cpu1'),1)
    assert owner.retained_entries==0 and owner._failed is not None


def test_clear_releases_entries_controls_then_retire(env):
    owner,ctx,lease,guard=build(env);owner.rotate(Value(),1);owner.rotate(Value(),2)
    owner.clear(executables=True)
    assert owner.retained_entries==owner.retained_bytes==0 and owner._binding is None
    assert owner._compiler.cleared==1
    lease.retire();guard.retired=True
    work=env.work
    with pytest.raises(ValueError):owner.rotate(Value(),1)
    assert env.work==work


def test_close_idempotent_rejects_old_owner(env):
    owner,ctx,lease,_=build(env);owner.rotate(Value(),1);owner.close();owner.close();lease.retire()
    with pytest.raises(RuntimeError,match='closed'):owner.rotate(Value(),1)


def test_failed_result_readiness_quarantines_unpublished_buffers(env):
    owner,ctx,lease,_=build(env)
    env.fail=lambda value:isinstance(value,Array)
    with pytest.raises(TimeoutError):owner.rotate(Value(),1)
    assert owner.retained_entries==0 and len(env.quarantine)==1
    assert env.quarantine[0][0] is owner and owner._lease is lease
    with pytest.raises(RuntimeError,match='quarantined'):owner.clear()
    with pytest.raises(RuntimeError,match='quarantined'):owner.rotate(Value(),1)


def test_failed_state_readiness_never_publishes(env):
    owner,ctx,_,_=build(env);env.fail=lambda value:isinstance(value,tuple)
    with pytest.raises(TimeoutError):owner.rotate(Value(),1)
    assert owner.retained_entries==0 and owner._failed is not None


def test_late_clear_failure_retains_all_entries_and_backing(env):
    owner,ctx,lease,_=build(env);owner.rotate(Value(),1);owner.rotate(Value(),2)
    before=tuple(owner._entries.values());env.fail=lambda value:isinstance(value,tuple)
    with pytest.raises(TimeoutError):owner.clear()
    assert tuple(owner._entries.values())==before and owner.retained_entries==2
    assert env.quarantine[0][0]._lease is lease


def test_eviction_readiness_failure_retains_old_and_new_pending_state(env):
    owner,ctx,_,_=build(env);owner.rotate(Value(),1);owner.rotate(Value(),2)
    old=next(iter(owner._entries.values())).state
    env.fail=lambda value:value is old
    with pytest.raises(TimeoutError):owner.rotate(Value(),3)
    assert owner.retained_entries==2 and owner._failed is not None


def test_backing_changed_during_operation_rejected_quarantined(env):
    owner,ctx,lease,guard=build(env)
    original=owner._compiler.bind
    def bind(*args,**kwargs):
        fn=original(*args,**kwargs)
        def mutate(*operands):
            result=fn(*operands);guard.changed=True;return result
        return mutate
    owner._compiler.bind=bind
    with pytest.raises(ValueError,match='backing'):owner.rotate(Value(),1)
    assert owner.retained_entries==0 and owner._failed is not None


@pytest.mark.parametrize('value',[True,0,-1,1.5])
def test_typed_budget_rejections(env,value):
    with pytest.raises(ValueError):build(env,max_entries=value)
    with pytest.raises(ValueError):build(env,max_bytes=value)


def test_public_memory_bound():
    one=2*8*131072*114*4+131072*4
    assert one==956825600 and 2*one<=2<<30
    assert 3*one>2<<30


def test_readonly_flag_on_mutable_owner_does_not_establish_immutability(env):
    arr=np.arange(10,dtype='uint32');arr.flags.writeable=False
    assert env.module._array_state(arr)[2] is False
    immutable=np.frombuffer(arr.tobytes(),dtype='uint32')
    assert env.module._array_state(immutable)[2] is True


def test_readonly_memmap_requires_external_lease_for_file_changes(env,tmp_path):
    path=tmp_path/'key.bin';np.arange(12,dtype='uint32').tofile(path)
    arr=np.memmap(path,mode='r',dtype='uint32')
    assert env.module._array_state(arr)[2] is True
    before=path.stat()
    with path.open('r+b') as stream:stream.write(b'xxxx')
    after=path.stat()
    assert (before.st_mtime_ns,before.st_ctime_ns)!=(after.st_mtime_ns,after.st_ctime_ns)
    # Cache deliberately delegates these authenticated file-state comparisons
    # to the loader guard; mmap's readonly flag alone is not its provenance.
    arr._mmap.close()
