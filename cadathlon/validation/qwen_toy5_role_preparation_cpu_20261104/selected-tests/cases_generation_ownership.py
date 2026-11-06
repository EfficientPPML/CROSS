"""Original generation/setup ownership with explicit small recording objects."""
from contextlib import contextmanager
from types import SimpleNamespace
import case_fixtures as f

def test_recorded_203_generation_preserves_original_secret_and_literal_rotations():
    from q5_key_producer import _generate_bound
    x=f.generation_fixture();result=_generate_bound(x.intent,x.deps)
    assert result.terminal_ready is False and result.execution_authorized is False
    assert x.calls[0]==('pke',)and x.calls[1]==('evaluation',id(x.secret))
    assert x.calls[2:]==[('rotation',i,id(x.secret))for i in x.p.rotations]
    assert [len(s.rows)for s in x.stores]==[203,1]and [s.close_count for s in x.stores]==[1,1]
    events=f.labels();assert events.index('source')<events.index('generation_security')<events.index('capacity')<events.index('store_acquire')<events.index('pke')
    assert max(i for i,e in enumerate(events)if e=='write_array')<events.index('manifest')
    assert events[-1]=='source'and max(i for i,e in enumerate(events)if e=='store_close')<len(events)-1

def _prerequisite(stage):
    from q5_key_producer import _generate_bound,_QUARANTINE
    x=f.generation_fixture(stage);f.expect(ValueError,lambda:_generate_bound(x.intent,x.deps),same=x.failure)
    assert x.stores==[]and x.calls==[]and _QUARANTINE
    assert 'store_acquire'not in f.labels()and 'pke'not in f.labels()
    if stage=='security':assert 'source'in f.labels()
    if stage=='capacity':assert 'generation_security'in f.labels()

def test_generation_source_denial_precedes_store_and_rng():_prerequisite('source')
def test_generation_security_denial_precedes_store_and_rng():_prerequisite('security')
def test_generation_capacity_denial_precedes_store_and_rng():_prerequisite('capacity')

def test_partial_store_acquisition_keeps_original_owner_for_cleanup():
    from q5_key_producer import _generate_bound,_QUARANTINE
    x=f.generation_fixture(partial_store=True);f.expect(ValueError,lambda:_generate_bound(x.intent,x.deps),same=x.failure)
    assert len(x.stores)==2 and all(s.close_count==1 for s in x.stores)and x.calls==[]
    retained=_QUARANTINE[-1];assert any(all(any(s is v for v in item)for s in x.stores)for item in retained if type(item)is tuple)

def test_generation_postclose_failure_does_not_return_ready_result():
    from q5_key_producer import _generate_bound,_QUARANTINE
    error=OSError('SYNTHETIC final source');x=f.generation_fixture(late_source_error=error)
    f.expect(OSError,lambda:_generate_bound(x.intent,x.deps),same=error)
    assert len(x.calls)==102 and all(s.close_count==1 for s in x.stores)and _QUARANTINE
    assert f.labels()[-1]=='source'

def _setup(fail=None):
    from q5_actual_process import _EncryptedInput
    from q5_checkpoint_contract import numerical_plan
    plan=numerical_plan();q=tuple(plan['profile']['q_towers']);ctx=SimpleNamespace();handle=SimpleNamespace(ctx=ctx);slots=object()
    payload=f.Array('setup_payload',(1,2,256,512,100));moduli=f.Array('setup_moduli',(100,));ffarray=f.Array('setup_ff');nttarray=f.Array('setup_ntt')
    ff=SimpleNamespace(moduli=list(q),tables={'original':ffarray});ntt=SimpleNamespace(ff_ctx=ff,tables=[nttarray],r=256,c=512)
    value=f.Value(payload,q,ntt);value.moduli_array=moduli;value._ckks_scale=plan['profile']['scale']
    def check_handle(h):assert h is handle and h.ctx is ctx;f.event('check_setup_handle')
    def encrypt(c,original_slots,*,scale):
        assert c is ctx and original_slots is slots and scale==plan['profile']['scale'];f.event('encrypt_original',ctx=id(c),slots=id(original_slots),scale=scale)
        c._polynomial_ntt_ctx=ntt
        if fail is not None:raise fail
        return value
    b=SimpleNamespace(profile=SimpleNamespace(parameters={'scaling_factor':plan['profile']['scale']}),runtime=SimpleNamespace(check_loaded_handle=check_handle),encrypt_slots=encrypt,check=lambda:f.event('setup_backend_check'),secure=lambda c,v:f.event('setup_security'))
    owner=_EncryptedInput(b,handle,slots)
    return SimpleNamespace(owner=owner,ctx=ctx,handle=handle,b=b,value=value,ff=ff,ntt=ntt,arrays=(payload,moduli,ffarray,nttarray),partial_arrays=(ffarray,nttarray))

@contextmanager
def _key_scope(x,arrays):
    f.event('setup_key_enter')
    try:yield
    except BaseException as error:
        assert all(a.waits>0 for a in arrays);f.event('setup_key_failed_exit',error=type(error).__name__);raise
    else:
        assert all(a.waits>0 for a in arrays);f.event('setup_key_clean_exit')

def test_setup_encrypt_success_barriers_original_moduli_and_ntt_tables():
    import q5_actual_process as m
    x=_setup()
    with _key_scope(x,x.arrays):
        assert x.owner.load()is x.value;x.owner.check();x.owner.close()
        assert id(x.owner)not in m._INPUTS and all(getattr(x.owner,n)is None for n in('backend','handle','ctx','slots','value'))
        x.owner.close()
    assert all(a.waits>0 for a in x.arrays)and m._QUARANTINE==[]

def test_setup_encrypt_attached_partial_failure_barriers_before_key_exit():
    import q5_actual_process as m
    error=ValueError('SYNTHETIC attached partial');x=_setup(error)
    def run():
        with _key_scope(x,x.partial_arrays):
            try:x.owner.load()
            except BaseException as e:
                assert e is error;x.owner.close(e)
    f.expect(ValueError,run,same=error)
    assert x.owner.value is None and m._INPUTS[id(x.owner)]['fixed'][2]is x.ctx and m._QUARANTINE
    assert all(a.waits>0 for a in x.partial_arrays)and x.arrays[0].waits==0 and x.arrays[1].waits==0

def _late(kind):
    import q5_actual_process as m
    x=_setup();foreign=f.Array('FOREIGN-'+kind);original=x.ff if kind=='ff_owner'else x.ff.tables['original']
    def run():
        with _key_scope(x,x.arrays):
            x.owner.load();x.owner.check()
            if kind=='ff_owner':x.ntt.ff_ctx=SimpleNamespace(moduli=list(x.ff.moduli),tables={'original':foreign})
            else:x.ff.tables['original']=foreign
            x.owner.close()
    error=f.expect(ValueError,run,'finite-field owner changed'if kind=='ff_owner'else'table or control changed')
    assert foreign.waits==0 and all(a.waits>0 for a in x.arrays)
    record=m._INPUTS[id(x.owner)];assert all(a is not foreign for a,_ in record['barriers'])and record['failure']is error
    if kind=='ff_owner':x.ntt.ff_ctx=original
    else:x.ff.tables['original']=original
    assert record['failure']is error and m._QUARANTINE

def test_setup_late_ff_owner_replacement_rejects_before_foreign_barrier():_late('ff_owner')
def test_setup_late_table_array_replacement_rejects_before_foreign_barrier():_late('table_array')
