"""Cache policy mechanics on tiny public arrays; not a security profile proof."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1]/'common')]
import jax
from jaxite_word import Polynomial
from encrypted_polynomial import EncryptedPolynomialEvaluator
import polynomial
jax.config.update('jax_enable_x64',True)

@pytest.fixture
def primitive():
    shape=dict(batch=1,num_elements=2,degree=256,num_moduli=2,precision=32,degree_layout=(16,16))
    source=Polynomial(shape,{'moduli':[12289,40961]})
    source.polynomial=jax.numpy.ones(source.shape,dtype='uint32')
    output=Polynomial(dict(shape,num_moduli=1),{'moduli':[12289]})
    yield source,output
    jax.clear_caches();polynomial._POLYNOMIAL_REGISTRY.clear()


def evaluator(source,preserve):
    # The real secure constructor and full array path are exercised separately.
    # This object isolates the policy's effect on JAX's actual compilation cache.
    value=object.__new__(EncryptedPolynomialEvaluator)
    cache=SimpleNamespace(ntt_q_max=source.ntt_ctx,_formatted_rotation_keys={},level_params={})
    value.ctx=SimpleNamespace(_param_cache=cache)
    value._multiplication_bindings={};value._plaintext_multiply=value._rescale=None
    value._constant_cache={};value.preserve_dispatch_cache=preserve
    return value


@pytest.mark.parametrize('output_kind',['different_level','different_scale'])
def test_registered_polynomial_output_retraces_safely(primitive,output_kind):
    source,output=primitive;traces=[]
    if output_kind=='different_scale':
        output=source._clone_with_payload(source.polynomial)
        source._ckks_scale=2.;output._ckks_scale=4.
    def operation(value):
        traces.append(True)
        return output._clone_with_payload(value.polynomial[...,:output.num_moduli])
    compiled=jax.jit(operation)
    first=compiled(source);jax.block_until_ready(first.polynomial)
    assert polynomial._POLYNOMIAL_REGISTRY and len(traces)==1
    evaluator(source,True)._clear_working_caches()
    assert not polynomial._POLYNOMIAL_REGISTRY
    second=compiled(source);jax.block_until_ready(second.polynomial)
    np.testing.assert_array_equal(second.polynomial,first.polynomial)
    assert len(traces)==2
    if output_kind=='different_scale':assert second._ckks_scale==4.


@pytest.mark.parametrize('preserve',[False,True])
def test_empty_registry_policy_preserves_array_dispatch_only_when_opted_in(primitive,preserve):
    source,_=primitive;traces=[]
    def operation(value):traces.append(True);return value+1
    compiled=jax.jit(operation)
    np.testing.assert_array_equal(compiled(jax.numpy.array([2])),[3])
    assert len(traces)==1 and not polynomial._POLYNOMIAL_REGISTRY
    evaluator(source,preserve)._clear_working_caches()
    np.testing.assert_array_equal(compiled(jax.numpy.array([3])),[4])
    assert len(traces)==(1 if preserve else 2)


def test_option_requires_explicit_low_memory_contract():
    with pytest.raises(TypeError,match='explicit boolean'):
        EncryptedPolynomialEvaluator(None,preserve_dispatch_cache='yes')
    with pytest.raises(ValueError,match='dedicated low_memory'):
        EncryptedPolynomialEvaluator(None,preserve_dispatch_cache=True)
