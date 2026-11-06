"""Real loader/authentication code with tiny fake producer/context/device work."""
from pathlib import Path
import os,sys
from types import SimpleNamespace
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'base')]
from test_key_contexts import case,assert_restored
from test_client_keys import mocked,PROFILE,RAW_PLAN
from test_prepared_rotation import env,Array,Value,Field,_actual_arithmetic_class
import key_contexts as loaders
import client_keys as keys


@pytest.fixture
def integrated(case,env,monkeypatch):
    c=case
    original=c.runtime.Context.program_initialization
    def initialize(ctx,rotations,**kwargs):
        original(ctx,rotations,**kwargs)
        p=c.m.params;cache=ctx._param_cache
        for name in ('degree','num_slots','dnum','composite_degree'):setattr(ctx,name,p[name])
        for name in ('degree','dnum','batch','composite_degree'):setattr(cache,name,p[name])
        cache.degree_layout=tuple(p['degree_layout']);cache.q_towers=p['q_towers'];cache.p_towers=p['p_towers']
        cache.ff_context_cls=Field;cache.cache_rotation_keys=True;cache._formatted_rotation_keys={}
        cache.coef_maps={i:Array(np.roll(np.arange(p['degree'],dtype='int32'),-i)) for i in rotations}
        cache.q_moduli_at_level=lambda level:cache.q_towers[:2*(level+1)]
        cache.level_for_num_q=lambda count:count//2-1;cache.prepares=0
        def get(index,level):
            cache.prepares+=1;n=2*(level+1);pair=[]
            for part in ('a','b'):
                raw=cache.raw_rotation_keys[index][part]
                a=np.transpose(raw,(0,2,1))
                pair.append(Array(np.concatenate((a[...,:n],a[...,-len(cache.p_towers):]),axis=-1).reshape(cache.dnum,*cache.degree_layout,n+len(cache.p_towers))))
            cache._formatted_rotation_keys[index]=tuple(pair)
            return *pair,cache.coef_maps[index]
        cache.get_rot_key=get
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
                    flat=payload.data.reshape(1,2,p['degree'],-1)
                    return Array(np.take(flat,coef.data,axis=2).reshape(payload.shape))
                return fn
        ctx.he_rot=Accessor()
    monkeypatch.setattr(c.runtime.Context,'program_initialization',initialize)
    _actual_arithmetic_class(env,monkeypatch)
    return c,env


def loader(c,**kwargs):
    return loaders.load_server_prepared(c.m.public,profile_bytes=PROFILE,plan_bytes=RAW_PLAN,
                                       dedicated_no_write=kwargs.pop('dedicated_no_write',True),**kwargs)


def value():
    v=Value();v.polynomial=Array(np.arange(16,dtype='uint32').reshape(1,2,2,2,2));return v


def mapped(ctx):return [part for pair in ctx._param_cache.raw_rotation_keys.values() for part in pair.values()]


def test_real_authentication_then_prepared_cache_success_and_retirement(integrated):
    c,env=integrated
    with loader(c) as handle:
        context=handle.ctx;cache=context._param_cache;maps=mapped(context)
        arithmetic=handle.prepared_arithmetic(low_memory=True,preserve_dispatch_cache=True)
        manager=handle.prepared_arithmetic.__self__
        for _ in range(4):arithmetic.rotation.rotate(value(),1)
        assert cache.prepares==1 and arithmetic.rotation.stats['hits']==3
        assert all(not a._mmap.closed for a in maps)
        assert manager.lease.context_id==handle.public_manifest['context_id']
    assert manager.cleanup_completed and manager.lease._closed and manager.lease.context is None
    assert arithmetic.rotation.retained_entries==0 and arithmetic.rotation._closed
    assert all(a._mmap.closed for a in maps) and not vars(context)
    with pytest.raises(RuntimeError):handle.prepared_arithmetic()
    assert_restored(c)


def test_owner_barriers_happen_before_context_disposal_and_mapping_close(integrated,monkeypatch):
    c,env=integrated;events=c.events
    with loader(c) as handle:
        maps=mapped(handle.ctx)
        base=sys.modules['encrypted_packed_qwen'].CompiledArithmetic.__mro__[1]
        original=base._clear_working_caches
        def clear(owner):
            assert all(not x._mmap.closed for x in maps)
            assert handle.ctx._param_cache is not None
            events.append('prepared_clear');return original(owner)
        monkeypatch.setattr(base,'_clear_working_caches',clear)
        arithmetic=handle.prepared_arithmetic();arithmetic.rotation.rotate(value(),1)
    assert events.index('prepared_clear')<events.index('evict')


@pytest.mark.parametrize('claim',[False,None,0,1,'yes'])
def test_no_write_ownership_must_be_explicit_before_payload_reads(integrated,monkeypatch,claim):
    c,_=integrated
    monkeypatch.setattr(loaders,'_map_array',lambda *a,**k:pytest.fail('no payload before policy'))
    with pytest.raises(ValueError):
        with loader(c,dedicated_no_write=claim):pass
    assert 'crypto' not in c.events and not loaders._ACTIVE.locked()


def test_unauthenticated_changed_bytes_rejected_before_context_or_lease(integrated):
    c,_=integrated;path=c.m.public/'rot_1_a.npy'
    with path.open('r+b') as out:out.seek(-1,2);out.write(b'X')
    with pytest.raises(ValueError,match='identity'):
        with loader(c):pass
    assert 'crypto' not in c.events


@pytest.mark.parametrize('kind',['file','directory','metadata','map','manifest','retired'])
def test_authenticated_backing_changes_reject_cached_hit_before_next_work(integrated,kind):
    c,env=integrated;failure=None
    with pytest.raises((ValueError,RuntimeError)):
        with loader(c) as handle:
            ctx=handle.ctx;arithmetic=handle.prepared_arithmetic();arithmetic.rotation.rotate(value(),1)
            manager=handle.prepared_arithmetic.__self__;before=env.work;path=c.m.public/'rot_1_a.npy'
            if kind=='file':path.rename(path.with_name('old.npy'));path.write_bytes(b'replacement')
            elif kind=='directory':c.m.public.rename(c.m.public.with_name('old_public'));c.m.public.mkdir()
            elif kind=='metadata':
                st=path.stat();os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns+1_000_000_000))
            elif kind=='map':ctx._param_cache.coef_maps[1]=Array(np.arange(4,dtype='int32'))
            elif kind=='manifest':handle.public_manifest['keygen_source_sha256']='f'*64
            else:manager._active=False
            with pytest.raises((ValueError,RuntimeError)) as caught:arithmetic.rotation.rotate(value(),1)
            assert env.work==before
            failure=caught.value;raise failure
    assert failure is not None and not loaders._ACTIVE.locked()
    assert_restored(c)


@pytest.mark.parametrize('caught_inside',[False,True])
def test_readiness_failure_keeps_all_public_backing_and_primary_identity(integrated,caught_inside):
    c,env=integrated;original_error=None
    with pytest.raises(TimeoutError) as outer:
        with loader(c) as handle:
            maps=mapped(handle.ctx);arithmetic=handle.prepared_arithmetic()
            env.fail=lambda x:isinstance(x,tuple)
            try:arithmetic.rotation.rotate(value(),1)
            except TimeoutError as error:
                original_error=error
                if not caught_inside:raise
    assert outer.value is original_error
    assert all(not x._mmap.closed for x in maps)
    assert loaders._FAILURE_QUARANTINE[0]['prepared_rotation_owner'] is handle.prepared_arithmetic.__self__
    assert not handle.prepared_arithmetic.__self__.cleanup_completed
    assert_restored(c)


def test_late_clear_failure_attempts_other_owner_and_retains_maps(integrated,monkeypatch):
    c,env=integrated;error=KeyboardInterrupt('late clear cancellation')
    with pytest.raises(KeyboardInterrupt) as caught:
        with loader(c) as handle:
            maps=mapped(handle.ctx);a=handle.prepared_arithmetic();b=handle.prepared_arithmetic()
            a.rotation.rotate(value(),1);b.rotation.rotate(value(),2)
            state=next(iter(a.rotation._entries.values())).state
            def fail(value):
                if value is state:raise error
                return False
            env.fail=fail
    assert caught.value is error and b.base_clears==1
    assert all(not x._mmap.closed for x in maps) and not handle.prepared_arithmetic.__self__.cleanup_completed
    assert_restored(c)


def test_body_primary_preserved_if_prepared_cleanup_also_fails(integrated,monkeypatch):
    c,env=integrated;primary=SystemExit('body primary')
    with pytest.raises(SystemExit) as caught:
        with loader(c) as handle:
            arithmetic=handle.prepared_arithmetic();arithmetic.rotation.rotate(value(),1)
            env.fail=lambda x:isinstance(x,tuple)
            raise primary
    assert caught.value is primary
    assert any('prepared rotation owner cleanup' in note for note in primary.__notes__)
    assert_restored(c)


def test_caught_factory_failure_still_forces_incomplete_exit(integrated,monkeypatch):
    c,env=integrated;failure=RuntimeError('partial owner construction')
    def fail(*args,**kwargs):raise failure
    monkeypatch.setattr(env.module,'create_prepared_arithmetic',fail)
    with pytest.raises(RuntimeError) as caught:
        with loader(c) as handle:
            maps=mapped(handle.ctx)
            with pytest.raises(RuntimeError):handle.prepared_arithmetic()
    assert caught.value is failure and all(not x._mmap.closed for x in maps)
    assert not handle.prepared_arithmetic.__self__.cleanup_completed
    assert_restored(c)


def test_failed_session_blocks_any_new_loader(integrated):
    c,env=integrated
    with pytest.raises(TimeoutError):
        with loader(c) as handle:
            arithmetic=handle.prepared_arithmetic();env.fail=lambda x:isinstance(x,tuple)
            arithmetic.rotation.rotate(value(),1)
    before=len(c.contexts)
    with pytest.raises(RuntimeError,match='quarantined'):
        with loader(c):pass
    assert len(c.contexts)==before


def test_explicit_prior_rotation_close_is_safe_for_loader_attempt_all_cleanup(integrated):
    c,env=integrated
    with loader(c) as handle:
        maps=mapped(handle.ctx);arithmetic=handle.prepared_arithmetic()
        arithmetic.rotation.rotate(value(),1);arithmetic.rotation.close()
    assert all(x._mmap.closed for x in maps) and handle.prepared_arithmetic.__self__.cleanup_completed
    assert_restored(c)


@pytest.mark.parametrize('broken',['missing_rotation','raising_rotation','missing_clear'])
def test_cleanup_attribute_resolution_cannot_skip_later_owners(integrated,monkeypatch,broken):
    c,env=integrated
    with pytest.raises((ValueError,RuntimeError,AttributeError)):
        with loader(c) as handle:
            maps=mapped(handle.ctx);a=handle.prepared_arithmetic();b=handle.prepared_arithmetic()
            if broken=='missing_rotation':del a.rotation
            elif broken=='missing_clear':
                a._clear_working_caches=None
            else:
                class Broken(type(a)):
                    @property
                    def rotation(self):raise RuntimeError('rotation property failed')
                a.__class__=Broken
    assert b.base_clears==1 and all(not x._mmap.closed for x in maps)
    assert_restored(c)


def test_cleanup_lookup_cancellation_supersedes_ordinary_failure_but_all_owners_run(integrated,monkeypatch):
    c,env=integrated;cancel=KeyboardInterrupt('rotation lookup cancellation')
    with pytest.raises(KeyboardInterrupt) as caught:
        with loader(c) as handle:
            a=handle.prepared_arithmetic();b=handle.prepared_arithmetic()
            manager=handle.prepared_arithmetic.__self__;manager._failure=RuntimeError('earlier failure')
            class Broken(type(a)):
                @property
                def rotation(self):raise cancel
            a.__class__=Broken
    assert caught.value is cancel and b.base_clears==1
    assert_restored(c)


@pytest.mark.parametrize('replacement',['rotation','arithmetic_clear','rotation_clear','rotation_close'])
def test_replaced_cleanup_cannot_hide_original_failed_rotation(integrated,monkeypatch,replacement):
    c,env=integrated;original_error=None;replacement_calls=[]
    with pytest.raises(TimeoutError) as caught:
        with loader(c) as handle:
            maps=mapped(handle.ctx);a=handle.prepared_arithmetic();original_rotation=a.rotation
            env.fail=lambda x:isinstance(x,tuple)
            try:a.rotation.rotate(value(),1)
            except TimeoutError as error:original_error=error
            env.fail=None
            replacement_fn=lambda *args,**kwargs:replacement_calls.append('called')
            if replacement=='rotation':
                a.rotation=SimpleNamespace(_failed=None,clear=replacement_fn,close=replacement_fn)
            elif replacement=='arithmetic_clear':a._clear_working_caches=replacement_fn
            else:setattr(original_rotation,replacement.removeprefix('rotation_'),replacement_fn)
    assert caught.value is original_error and not replacement_calls
    assert all(not x._mmap.closed for x in maps)
    manager=handle.prepared_arithmetic.__self__
    assert not manager.cleanup_completed and manager._registered[0].rotation is original_rotation
    assert any('replaced' in note for note in original_error.__notes__)
    assert_restored(c)
