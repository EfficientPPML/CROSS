"""Exact owner bodies; all numerical/backend values are labelled recordings."""
from types import SimpleNamespace
import weakref,gc
import case_fixtures as f

def test_six_constructor_lookup_is_exact_and_body_scope_precedes_imports():
    import q5_actual_dependencies as m
    table=(('BoundedArithmetic','encrypted_bounded_arithmetic'),('BoundedPackedTransformer','encrypted_bounded_transformer'),('BoundedSecureRefresh','encrypted_refresh'),('PackedPairLayout','encrypted_packed_transformer'),('PolynomialApproximation','encrypted_polynomial'),('enable_scoped_ntt_subsets','encrypted_ntt'))
    values={name:object()for name,_ in table};seen=[]
    def importer(name):
        seen.append(name);f.event('constructor_lookup',name=name);selected=[n for n,mod in table if mod==name];assert len(selected)==1
        return SimpleNamespace(**{selected[0]:values[selected[0]]})
    def definitions(v):assert all(v[n]is values[n]for n in values);f.event('check_definitions')
    receiver=SimpleNamespace(check=lambda:f.event('backend_recording_check'),runtime=SimpleNamespace(require_server_body_scope=lambda:f.event('body_scope'),check_common_definitions=definitions))
    with f.patch(m.importlib,'import_module',importer):result=m.ActualBackend.arithmetic(receiver)
    assert seen==[mod for _,mod in table]and all(result[n]is values[n]for n in values)
    assert f.labels().index('body_scope')<f.labels().index('constructor_lookup')<f.labels().index('check_definitions')

def _owner(x):
    from q5_ciphertext_owner import CiphertextOwner
    return CiphertextOwner(x.np,x.b,x.ctx,x.payload,x.row,check=lambda:f.event('ct_guard'))

def test_server_cache_passes_three_argument_polynomial_abi():
    x=f.backend_fixture();o=_owner(x);value=o.load();o.check()
    assert x.getter_calls==[x.row['level']]and value.ntt_ctx is x.original_ntt and value.polynomial.shape==x.payload.shape
    assert x.ntt_objects==[];assert f.labels().index('cache_get')<f.labels().index('from_array')
    o.close();assert value.moduli_array.waits==1

def test_cache_replacement_after_getter_rejects_before_polynomial():
    x=f.backend_fixture();original=x.ctx._param_cache;original_get=original.get_sliced_ntt_q
    def replacement_get(level):
        result=original_get(level);x.ctx._param_cache=SimpleNamespace(get_sliced_ntt_q=lambda n:(_ for _ in()).throw(AssertionError('foreign getter invoked')));return result
    original.get_sliced_ntt_q=replacement_get;o=_owner(x)
    error=f.expect(ValueError,o.load,'original cache changed during NTT acquisition')
    assert x.values==[]and'from_array'not in f.labels();f.expect(ValueError,lambda:o.close(error),same=error)

def test_client_explicit_ntt_acquires_before_original_initializers():
    import q5_ciphertext_owner as m
    x=f.backend_fixture(cache=False);holder={};FF=x.b.finite_field.BarrettContext;NTT=x.b.ntt_mm.NTTCiphertextBarrettContext;ffi=FF.__init__;ni=NTT.__init__
    def ff(self,*,moduli):
        assert any(v is self for v in m._OWNERS[id(holder['owner'])]['objects']);ffi(self,moduli=moduli)
    def ntt(self,*,moduli,parameters):
        record=m._OWNERS[id(holder['owner'])];assert any(v is self for v in record['objects'])and parameters['finite_field_context']is record['objects'][0];ni(self,moduli=moduli,parameters=parameters)
    with f.patch(FF,'__init__',ff),f.patch(NTT,'__init__',ntt):
        o=m.PublicNTTOwner(x.b,x.q,check=lambda:f.event('ntt_guard'));holder['owner']=o;value=o.load();assert value.ff_ctx is x.ntt_objects[0];o.check();o.close()
    assert id(o)not in m._OWNERS and o.value is None

def _partial(which):
    import q5_ciphertext_owner as m
    error=ValueError('SYNTHETIC '+which+' initializer');x=f.backend_fixture(cache=False,**{which+'_error':error});o=m.PublicNTTOwner(x.b,x.q,check=lambda:None)
    f.expect(ValueError,o.load,same=error);record=m._OWNERS[id(o)];objects=tuple(record['objects']);arrays=tuple(record['arrays']);assert arrays and len(objects)==(1 if which=='ff'else 2)
    f.expect(ValueError,lambda:o.close(error),same=error)
    assert all(a.waits>=1 for a in arrays)and record['failure']is error and m._QUARANTINE

def test_barrett_initializer_partial_arrays_are_retained_and_barred():_partial('ff')
def test_ntt_initializer_partial_arrays_are_retained_and_barred():_partial('ntt')

def _returned_failure(stage):
    import q5_ciphertext_owner as m
    error=ValueError('SYNTHETIC '+stage);x=f.backend_fixture(**{stage+'_error':error});o=_owner(x)
    f.expect(ValueError,o.load,same=error);value=x.values[0];record=m._OWNERS[id(o)]
    assert any(a is value.moduli_array for a in record['arrays'])
    f.expect(ValueError,lambda:o.close(error),same=error)
    assert value.moduli_array.waits==1 and value.polynomial.waits>=1
    assert any(a is value.moduli_array for a in record['arrays'])and record['failure']is error
    f.event('recording_key_failed_exit',after_original_barriers=True)

def test_returned_moduli_array_is_barred_after_validation_failure():_returned_failure('validate')
def test_returned_moduli_array_is_barred_after_security_failure():_returned_failure('security')

def test_barrier_failure_attempts_remaining_original_owners_and_keeps_primary():
    import q5_ciphertext_owner as m
    x=f.backend_fixture(cache=False);o=_owner(x);value=o.load();record=m._OWNERS[id(o)];child=record['ntt_owner'];child_record=m._OWNERS[id(child)];tables=tuple(child_record['arrays'])
    value.polynomial.fail=OSError('SYNTHETIC barrier');primary=ValueError('SYNTHETIC body')
    f.expect(ValueError,lambda:o.close(primary),same=primary)
    assert value.moduli_array.waits==1 and all(a.waits>=2 for a in tables)
    assert record['failure']is primary and m._QUARANTINE

def _input_fixture(load_failure=None):
    import q5_ciphertext_owner as m,q5_checkpoint_transport as t
    rows,_=f.checkpoint_fixture();record=dict(root=f.root_meta('input'),artifact=dict(metadata=rows[0]));children=[];original_value=object()
    class Reader:
        def __init__(self,np,raw):self.closed=False;self.closes=0;children.append(self)
        def read(self,artifact,*,retain_payload=False):f.event('reader_read');return object()
        def check(self):assert not self.closed
        def close(self,primary=None):self.closed=True;self.closes+=1;f.event('reader_close')
    class Consumer:
        def __init__(self,*a,**kw):self.value=None;self.closes=0;children.append(self)
        def load(self):
            f.event('consumer_load')
            if load_failure is not None:raise load_failure
            self.value=original_value;return self.value
        def check(self):return self.value
        def close(self,primary=None):self.closes+=1;f.event('consumer_close');self.value=None
    return m,t,record,Reader,Consumer,children,original_value

def test_input_close_tombstone_is_idempotent_and_preserves_primary():
    m,t,record,R,C,children,value=_input_fixture()
    with f.patch(t,'ArtifactReader',R),f.patch(m,'CiphertextOwner',C):
        o=m.InputOwner(object(),object(),object(),record,check=lambda:None);assert o.load()is value;o.close();o.close();primary=ValueError('SYNTHETIC supplied primary');f.expect(ValueError,lambda:o.close(primary),same=primary)
    assert id(o)not in m._OWNERS and o.reader is None and o.consumer is None and o.value is None and all(c.closes==1 for c in children)

def test_input_partial_consumer_failure_closes_original_children_inside_key_scope():
    error=ValueError('SYNTHETIC consumer load');m,t,record,R,C,children,value=_input_fixture(error)
    with f.patch(t,'ArtifactReader',R),f.patch(m,'CiphertextOwner',C):
        o=m.InputOwner(object(),object(),object(),record,check=lambda:None);f.event('key_enter');f.expect(ValueError,o.load,same=error);f.expect(ValueError,lambda:o.close(error),same=error);f.event('key_failed_exit')
    assert f.labels().index('consumer_close')<f.labels().index('reader_close')<f.labels().index('key_failed_exit')and all(c.closes==1 for c in children)
    assert m._OWNERS[id(o)]['failure']is error

def test_clean_ct_ntt_release_never_disposes_borrowed_context():
    import q5_ciphertext_owner as m
    x=f.backend_fixture(cache=False);dispose=[];x.ctx.dispose=lambda:dispose.append(True);o=_owner(x);value=o.load();ntt=m._OWNERS[id(o)]['ntt_owner'];weak=weakref.ref(ntt)
    o.close();o.close();assert dispose==[]and id(o)not in m._OWNERS and id(ntt)not in m._OWNERS
    assert o.ctx is None and o.payload is None and o.value is None and ntt.value is None
    del ntt;gc.collect();assert weak()is None
