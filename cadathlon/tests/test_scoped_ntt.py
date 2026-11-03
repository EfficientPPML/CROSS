"""Owner-scoped NTT subset contracts; second-device guard is explicitly opt-in."""
from pathlib import Path
import gc, importlib, sys, weakref
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import jax
from jaxite_word import he_params
from encrypted_ntt import FIELDS,ScopedNTTSubsets
jax.config.update('jax_enable_x64',True)
ntt=he_params.ntt_mm
class Owner:pass

def make_owner():
    owner=Owner();owner.q_towers=[12289,40961,65537,114689];owner.p_towers=[147457,163841]
    owner.r=16;owner.c=16;owner.degree=256;owner.ff_context_cls=he_params.BarrettContext;owner.perf_test=False
    for attr,moduli in [('ntt_q_max',owner.q_towers),('ntt_p',owner.p_towers)]:
        field=owner.ff_context_cls(moduli)
        setattr(owner,attr,ntt.NTTCiphertextBarrettContext(moduli,{'r':16,'c':16,'finite_field_context':field}))
    return owner

def params(owner,moduli):
    return {'r':owner.r,'c':owner.c,'finite_field_context':owner.ff_context_cls(moduli)}

def test_ordered_subsets_exact_and_lifecycle():
    owner=make_owner();provider=ScopedNTTSubsets(owner,max_entries=2);device=jax.devices()[0]
    choices=[owner.q_towers[:-1],owner.q_towers[1:]+owner.p_towers,
             [owner.p_towers[1],owner.q_towers[2],owner.q_towers[0],owner.p_towers[0]]]
    retained=[]
    for moduli in choices:
        p=params(owner,moduli);actual=provider.get(moduli,p,device=device)
        assert actual is not None
        expected=ntt.NTTCiphertextBarrettContext(moduli,p)
        for name in FIELDS:np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
        assert provider.get(moduli,params(owner,moduli),device=device) is actual
        retained.append(actual)
        jax.clear_caches();ntt._barrett_ntt_context_cache.clear();gc.collect()
    assert provider.live_entries==2 and len(provider._entries)==2
    refs=[weakref.ref(x) for x in retained]
    del actual,expected,retained
    jax.clear_caches();gc.collect()
    assert all(ref() is None for ref in refs)
    assert provider.live_entries==0
    moduli=choices[-1];actual=provider.get(moduli,params(owner,moduli),device=device)
    baseline=np.asarray(actual.ntt_tf_step2).copy()
    provider.close();np.testing.assert_array_equal(actual.ntt_tf_step2,baseline)
    assert provider.get(moduli,params(owner,moduli),device=device) is None

def test_owner_not_retained_and_replacement_falls_back():
    owner=make_owner();provider=ScopedNTTSubsets(owner);device=jax.devices()[0]
    moduli=owner.q_towers[:2]
    actual=provider.get(moduli,params(owner,moduli),device=device)
    owner_ref=weakref.ref(owner);master_ref=weakref.ref(owner.ntt_q_max)
    del owner
    jax.clear_caches();ntt._barrett_ntt_context_cache.clear();gc.collect()
    assert owner_ref() is None and master_ref() is None
    assert provider.get(moduli,{'r':16,'c':16,'finite_field_context':he_params.BarrettContext(moduli)},device=device) is None
    assert actual.ntt_tf_step2.shape[-1]==2

@pytest.mark.parametrize('case',['duplicate','unknown','floatmodulus','layout','precision','extra_parameter','wrong_ff_moduli','changed_field_array','changed_field_metadata','changed_source_array','changed_source_moduli','changed_owner_degree','changed_owner_field','perf_test','nonstandard_field','other_device'])
def test_guards_fall_back(case):
    owner=make_owner();provider=ScopedNTTSubsets(owner);device=jax.devices()[0];moduli=owner.q_towers[:2];p=params(owner,moduli);precision=32
    if case=='duplicate':moduli=[moduli[0]]*2
    elif case=='unknown':moduli=[12289,786433];p=params(owner,moduli)
    elif case=='floatmodulus':moduli=[float(x) for x in moduli]
    elif case=='layout':p['r'],p['c']=8,32
    elif case=='precision':precision=64
    elif case=='extra_parameter':p['perf_test']=False
    elif case=='wrong_ff_moduli':p['finite_field_context']=owner.ff_context_cls(owner.q_towers[1:3])
    elif case=='changed_field_array':p['finite_field_context'].m=jax.numpy.zeros_like(p['finite_field_context'].m)
    elif case=='changed_field_metadata':p['finite_field_context'].barrett_m=[0,0]
    elif case=='changed_source_array':owner.ntt_q_max.ntt_tf_step2=jax.numpy.zeros_like(owner.ntt_q_max.ntt_tf_step2)
    elif case=='changed_source_moduli':owner.ntt_q_max.moduli=list(reversed(owner.q_towers))
    elif case=='changed_owner_degree':owner.degree=512
    elif case=='changed_owner_field':owner.ff_context_cls=type('OtherBarrett',(owner.ff_context_cls,),{})
    elif case=='perf_test':owner.perf_test=True
    elif case=='nonstandard_field':p['finite_field_context'].computation_format_is_standard=False
    elif case=='other_device':
        if len(jax.devices()) < 2:
            pytest.skip('requires two logical CPU devices; run with --xla_force_host_platform_device_count=2')
        device=jax.devices()[1]
        with jax.default_device(device):p=params(owner,moduli)
    assert provider.get(moduli,p,precision=precision,device=device) is None

def test_mutated_cached_field_is_not_reused():
    owner=make_owner();provider=ScopedNTTSubsets(owner);device=jax.devices()[0];moduli=owner.q_towers[:2]
    actual=provider.get(moduli,params(owner,moduli),device=device)
    actual.ff_ctx.m=jax.numpy.zeros_like(actual.ff_ctx.m)
    rebuilt=provider.get(moduli,params(owner,moduli),device=device)
    assert rebuilt is not actual
    np.testing.assert_array_equal(rebuilt.ff_ctx.m,he_params.BarrettContext(moduli).m)


def test_contiguous_view_keeps_backing_until_last_holder_drops():
    owner=make_owner();provider=ScopedNTTSubsets(owner);device=jax.devices()[0]
    moduli=owner.q_towers[:2]
    actual=provider.get(moduli,params(owner,moduli),device=device)
    master=owner.ntt_q_max.ntt_bat_tf_step1
    view=actual.ntt_bat_tf_step1
    assert isinstance(master,np.ndarray) and isinstance(view,np.ndarray)
    assert np.shares_memory(master,view)
    assert view.nbytes < master.nbytes
    backing=master
    while isinstance(backing.base,np.ndarray):backing=backing.base
    backing_ref=weakref.ref(backing);wrapper_ref=weakref.ref(owner.ntt_q_max)
    del owner,master,backing,view
    jax.clear_caches();gc.collect()
    assert wrapper_ref() is None and backing_ref() is not None
    del actual
    provider.clear();jax.clear_caches();gc.collect()
    assert backing_ref() is None

@pytest.mark.parametrize('failure',[None,RuntimeError,KeyboardInterrupt,SystemExit])
def test_reference_session_releases_provider_even_if_cleanup_fails(monkeypatch,failure):
    import encrypted_reference as reference
    from types import SimpleNamespace
    from jaxite_word import ckks_ctx
    events=[]
    def release():
        events.append('ntt')
        if failure:raise failure('ntt cleanup')
    monkeypatch.setattr(ckks_ctx,'evict_context_caches',lambda value:events.append(value))
    monkeypatch.setattr(jax,'clear_caches',lambda:events.append('jax'))
    monkeypatch.setattr(reference.gc,'collect',lambda:events.append('gc'))
    session=object.__new__(reference.ToyReferenceSession)
    session.client='client';session.ctx='server';session.ntt_subsets=SimpleNamespace(close=release)
    session.key_store=SimpleNamespace(close=lambda:events.append('keys'))
    if failure:
        with pytest.raises(failure,match='ntt cleanup'):session.close()
    else:session.close()
    assert events==['ntt','client','server','jax','gc','keys']
    assert session.ntt_subsets is None and session.closed
    session.close();assert len(events)==6

def test_reference_option_is_explicit_before_keygen(monkeypatch):
    import encrypted_reference as reference
    from jaxite_word import key_gen
    def forbidden(*args,**kwargs):raise AssertionError('keygen happened before opt-in validation')
    monkeypatch.setattr(key_gen,'gen_pke_pair',forbidden)
    with pytest.raises(TypeError,match='ntt_subsets must be an explicit boolean'):
        reference.create_toy_session('bert',{}, {},batch=1,sequence_length=1,devices=[],ntt_subsets='yes')
