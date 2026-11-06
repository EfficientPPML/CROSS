"""Real authenticated-loader code with tiny mocked producer/context arrays."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'joined/checkout/cadathlon/common'),str(HERE/'runtime')]
from test_prepared_loader import integrated,mapped
from test_key_contexts import case,assert_restored
from test_client_keys import mocked,PROFILE,RAW_PLAN
from test_prepared_rotation import env,Array,Value,Field
import key_contexts as loaders

class Compiler:
    def __init__(self,**kwargs):pass
    def clear_executables(self):pass

@pytest.fixture
def group_case(integrated,monkeypatch):
    c,e=integrated
    monkeypatch.setitem(sys.modules,'finite_field',SimpleNamespace(BarrettContext=Field))
    monkeypatch.setattr(sys.modules['encrypted_compilation'],'ExplicitArrayCompiler',Compiler)
    import grouped_rotation as groups
    monkeypatch.setattr(groups,'_FAILURE_QUARANTINE',[])
    return c,e,groups

def loader(c,**kwargs):
    return loaders.load_server_grouped(c.m.public,profile_bytes=PROFILE,plan_bytes=RAW_PLAN,
        dedicated_no_write=kwargs.pop('dedicated_no_write',True),**kwargs)

def value(ctx):
    v=Value();v.polynomial=Array(np.arange(16,dtype='uint32').reshape(1,2,2,2,2))
    v.moduli=list(ctx._param_cache.q_towers);v.degree=4;v.degree_layout=(2,2);v.batch=1;v.num_elements=2;v.precision=32;v.ntt_ctx=object()
    return v

def test_authenticated_group_factory_original_cleanup_and_zero_alias(group_case):
    c,e,groups=group_case
    with loader(c) as handle:
        ctx=handle.ctx;maps=mapped(ctx);owner=handle.grouped_arithmetic();manager=handle.grouped_arithmetic.__self__
        assert type(owner.rotation)is groups.GroupedCompiledRotation
        original=value(ctx);assert owner.rotation.rotate_many(original,(0,0))==(original,original)
        assert all(not m._mmap.closed for m in maps)
        assert manager.lease.context_id==handle.public_manifest['context_id']
        assert not hasattr(handle,'prepared_arithmetic') and not hasattr(handle,'private_manifest')
    assert manager.cleanup_completed and manager.lease._closed and owner.rotation._closed
    assert all(m._mmap.closed for m in maps) and not vars(ctx)
    with pytest.raises(RuntimeError):handle.grouped_arithmetic()
    assert_restored(c)

def test_original_owner_retired_before_context_and_maps(group_case,monkeypatch):
    c,e,groups=group_case;events=[]
    with loader(c) as handle:
        maps=mapped(handle.ctx)
        old=groups.GroupedCompiledRotation.clear
        def clear(self,**kwargs):
            assert vars(handle.ctx) and all(not m._mmap.closed for m in maps)
            events.append('original_group_clear');return old(self,**kwargs)
        monkeypatch.setattr(groups.GroupedCompiledRotation,'clear',clear)
        owner=handle.grouped_arithmetic()
        owner.rotation.rotate_many(value(handle.ctx),(0,))
    assert events and all(m._mmap.closed for m in maps)
    assert_restored(c)

def test_caught_factory_failure_still_quarantines_public_maps(group_case,monkeypatch):
    c,e,groups=group_case;error=RuntimeError('factory failed')
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(groups,'create_grouped_arithmetic',fail)
    with pytest.raises(RuntimeError) as caught:
        with loader(c) as handle:
            maps=mapped(handle.ctx)
            with pytest.raises(RuntimeError) as inner:handle.grouped_arithmetic()
            assert inner.value is error
    assert caught.value is error and all(not m._mmap.closed for m in maps)
    assert loaders._FAILURE_QUARANTINE and not loaders._ACTIVE.locked()
    assert_restored(c)

def test_caught_readiness_failure_preserves_original_group_backing(group_case):
    c,e,groups=group_case;error=TimeoutError('unknown group readiness')
    with pytest.raises(TimeoutError) as caught:
        with loader(c) as handle:
            maps=mapped(handle.ctx);owner=handle.grouped_arithmetic()
            owner.rotation._failed=error
    assert caught.value is error and all(not m._mmap.closed for m in maps)
    assert loaders._FAILURE_QUARANTINE and not loaders._ACTIVE.locked()
    assert_restored(c)

def test_original_rotation_replacement_cannot_authorize_unmapping(group_case):
    c,e,groups=group_case
    with pytest.raises(ValueError):
        with loader(c) as handle:
            maps=mapped(handle.ctx);owner=handle.grouped_arithmetic();original=owner.rotation
            owner.rotation=SimpleNamespace(clear=lambda **kwargs:None)
    # Identity failure retires the original clear hook, but deliberately does
    # not call uncertain owner/close continuations or claim completed closure.
    assert not original._closed and original._binding is None and all(not m._mmap.closed for m in maps)
    assert loaders._FAILURE_QUARANTINE
    assert_restored(c)

@pytest.mark.parametrize('policy',[False,1,None])
def test_dedicated_no_write_policy_is_exact_and_precontext(group_case,policy):
    c,e,groups=group_case
    with pytest.raises(ValueError):
        with loader(c,dedicated_no_write=policy):pass
    assert not c.contexts

def test_mutually_exclusive_owner_flags_precede_context(group_case):
    c,e,groups=group_case
    with pytest.raises(ValueError):
        with loaders._load(c.m.public,None,profile_bytes=PROFILE,plan_bytes=RAW_PLAN,
            prepared_rotation=True,grouped_rotation=True,dedicated_no_write=True):pass
    assert not c.contexts

def test_ordinary_and_prepared_loader_modes_remain_separate(group_case):
    c,e,groups=group_case
    with loaders.load_server(c.m.public,profile_bytes=PROFILE,plan_bytes=RAW_PLAN) as handle:
        assert not hasattr(handle,'grouped_arithmetic') and not hasattr(handle,'prepared_arithmetic')
    with loaders.load_server_prepared(c.m.public,profile_bytes=PROFILE,plan_bytes=RAW_PLAN,dedicated_no_write=True) as handle:
        assert not hasattr(handle,'grouped_arithmetic') and hasattr(handle,'prepared_arithmetic')
