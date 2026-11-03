"""Exact residue and boundary checks for the guarded public constant codec."""
import warnings
import numpy as np
import pytest
from jaxite_word import ckks_ctx


def outcome(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        try:
            result, error = function(*args, **kwargs), None
        except Exception as exc:
            result, error = None, (type(exc).__name__, str(exc))
    return result, error, [(item.category.__name__, str(item.message)) for item in seen]


@pytest.mark.parametrize('slot_count', [1, 16])
@pytest.mark.parametrize('scalar', [-.75, .75])
@pytest.mark.parametrize('scale', [2.**20, 2.**80])
@pytest.mark.parametrize('nsd', [1, 2])
@pytest.mark.parametrize('word', [31, 61])
def test_private_uniform_avoids_ntt_and_matches_independent_scalar_codec(monkeypatch, slot_count, scalar, scale, nsd, word):
    args = dict(slots=[complex(scalar)]*slot_count,cycl_order=64,q_towers=[65537,114689],p_towers=[],scale=scale,
                noise_scale_degree=nsd,max_bits_in_word=word)
    oracle = np.asarray(ckks_ctx._ckks_encode_fall_back(**dict(args,slots=[complex(scalar)]*16)),dtype=np.uint64)
    def forbidden(*args, **kwargs):
        raise AssertionError('uniform public scalar rebuilt host FFT/NTT')
    monkeypatch.setattr(ckks_ctx,'_fast_encode',forbidden)
    actual = ckks_ctx._ckks_encode_array(**args)
    np.testing.assert_array_equal(actual,oracle)
    assert actual.flags.c_contiguous and actual.flags.writeable
    historical = ckks_ctx._ckks_encode(**args)
    assert type(historical) is list and type(historical[0]) is list and type(historical[0][0]) is int
    np.testing.assert_array_equal(historical,oracle)


@pytest.mark.parametrize('updates', [
    {'slots':[1j]*16}, {'slots':np.linspace(-.75,.75,16)}, {'slots':[float('nan')]*16}, {'slots':[1.]*3},
    {'max_bits_in_word':0}, {'max_bits_in_word':62}, {'max_bits_in_word':61.5},
    {'max_bits_value':127}, {'max_bits_value':2**64}, {'noise_scale_degree':3},
    {'scale':0.}, {'scale':-1.}, {'scale':float('inf')}, {'scale':1<<20},
    {'slots':[2.**-80]*16},
])
def test_private_guarded_fallback_preserves_results_errors_and_warnings(monkeypatch, updates):
    args = dict(slots=[-.75]*16,cycl_order=64,q_towers=[65537,114689],p_towers=[],scale=2.**20)
    args.update(updates)
    actual, error, warns = outcome(ckks_ctx._ckks_encode_array,**args)
    monkeypatch.setattr(ckks_ctx,'_try_uniform_real_encode',lambda *a,**k:None)
    expected, expected_error, expected_warns = outcome(ckks_ctx._ckks_encode_array,**args)
    assert error == expected_error and warns == expected_warns
    if error is None:np.testing.assert_array_equal(actual,expected)


def test_nondefault_floating_error_policy_delegates(monkeypatch):
    with np.errstate(all='raise'):
        assert ckks_ctx._try_uniform_real_encode([.75]*16,64,[65537,114689],[],2.**20) is None


@pytest.fixture(scope='module')
def context():
    import jax
    from jaxite_word import CKKSContext,key_gen,util,he_params
    jax.config.update('jax_enable_x64',True)
    degree,dnum=4096,4;q=[40961,65537,114689,147457];p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
    assert he_params.qp_is_secure(q,p,degree,he_params.SecurityTarget(128,'classical'))
    keys=key_gen.gen_pke_pair(q,[],degree);ek=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
    params=dict(degree=degree,num_slots=degree//2,scaling_factor=2.**20,output_scale=2.**20,q_towers=q,p_towers=p,
                composite_degree=2,degree_layout=(64,64),p=20,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
    result=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(ek[name],dtype='uint32').transpose(0,2,1) for name in ('a','b')]))
    result.program_initialization([],dnum=dnum,r=64,c=64,batch=1)
    return result


@pytest.mark.parametrize('level',[0,1])
@pytest.mark.parametrize('scale',[None,2.**40])
@pytest.mark.parametrize('scalar',[-.75,0.])
def test_initialized_uniform_boundary_keeps_payload_and_metadata(monkeypatch,context,level,scale,scalar):
    helper=ckks_ctx._try_uniform_real_encode
    monkeypatch.setattr(ckks_ctx,'_try_uniform_real_encode',lambda *a,**k:None)
    expected=context.encode_at_level([scalar]*context.num_slots,level,scale=scale)
    monkeypatch.setattr(ckks_ctx,'_try_uniform_real_encode',helper)
    def forbidden(*args,**kwargs):raise AssertionError('uniform level constant rebuilt host NTT')
    monkeypatch.setattr(ckks_ctx,'_fast_encode',forbidden)
    actual=context.encode_at_level([scalar]*context.num_slots,level,scale=scale)
    np.testing.assert_array_equal(actual.polynomial,expected.polynomial)
    assert actual._ckks_scale==expected._ckks_scale
    assert actual.shape==expected.shape and tuple(actual.moduli)==tuple(expected.moduli)
    assert actual.ntt_ctx is expected.ntt_ctx
    assert actual.__dict__.keys()==expected.__dict__.keys()


@pytest.mark.parametrize('slots',[[.75]*128,[1j]*2048,[],np.ones((1,2048)),['.75']*2048])
def test_initialized_boundary_keeps_exact_slot_and_complex_validation(monkeypatch,context,slots):
    actual,error,warns=outcome(context.encode_at_level,slots,1)
    monkeypatch.setattr(ckks_ctx,'_try_uniform_real_encode',lambda *a,**k:None)
    expected,expected_error,expected_warns=outcome(context.encode_at_level,slots,1)
    assert error is not None and error==expected_error and warns==expected_warns


def test_initialized_uses_normalized_context_word_parameter(monkeypatch,context):
    monkeypatch.setitem(context.parameters,'max_bits_in_word',31.5)
    helper=ckks_ctx._try_uniform_real_encode
    monkeypatch.setattr(ckks_ctx,'_try_uniform_real_encode',lambda *a,**k:None)
    expected=context.encode_at_level([-.75]*context.num_slots,1,scale=2.**100)
    monkeypatch.setattr(ckks_ctx,'_try_uniform_real_encode',helper)
    def forbidden(*args,**kwargs):raise AssertionError('normalized word parameter should use proven integer envelope')
    monkeypatch.setattr(ckks_ctx,'_fast_encode',forbidden)
    actual=context.encode_at_level([-.75]*context.num_slots,1,scale=2.**100)
    np.testing.assert_array_equal(actual.polynomial,expected.polynomial)


def test_other_context_codec_entrypoints_keep_existing_fast_encode(monkeypatch,context):
    calls=[];original=ckks_ctx._fast_encode
    def observed(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(ckks_ctx,'_fast_encode',observed)
    context.encode([.75]*context.num_slots)
    assert calls==[True]
