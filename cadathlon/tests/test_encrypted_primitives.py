"""Secure-ring primitive parity, metadata and compiled-control lifetime."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
import jax
import numpy as np
from jaxite_word import CKKSContext,he_params,key_gen,util
from encrypted_primitives import CompiledRescale,CompiledPlaintextMultiply,CompiledRelinearization

def test_secure_compiled_primitives_match_facades_and_release_controls():
    jax.config.update('jax_enable_x64',True)
    degree,dnum=4096,4
    q=[40961,65537,114689,147457]
    p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
    assert he_params.qp_is_secure(q,p,degree,he_params.SecurityTarget(128,'classical'))
    keys=key_gen.gen_pke_pair(q,[],degree)
    evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
    params=dict(degree=degree,num_slots=degree//2,scaling_factor=float(1<<20),
        output_scale=float(1<<20),q_towers=q,p_towers=p,composite_degree=2,
        degree_layout=(64,64),p=20,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
    client=CKKSContext(dict(params,**keys))
    for batch,nsd in [(1,1),(2,2)]:
        server=CKKSContext(dict(params,evaluation_key=[
            jax.numpy.asarray(evaluation[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
        server.program_initialization([],dnum=dnum,r=64,c=64,batch=batch)
        slots=np.linspace(-.75,.75,degree//2)
        value=client.encrypt_slots_batch([slots*(i+1)/batch for i in range(batch)],scale=float(1<<24))
        value._ckks_nsd=nsd
        rescale=CompiledRescale(server)
        plaintext=CompiledPlaintextMultiply(server)
        expected=server.he_rescale[1,0].rescale(value)
        actual=rescale.rescale(value,1,0)
        np.testing.assert_array_equal(actual.polynomial,expected.polynomial)
        assert actual._ckks_scale==expected._ckks_scale
        assert actual._ckks_nsd==expected._ckks_nsd==nsd
        assert actual.moduli==expected.moduli
        for level in [1,0]:
            source=value if level==1 else server.he_level_reduce[1,0].level_reduce(value)
            encoded=np.arange(source.num_moduli*degree,dtype=np.uint64).reshape(1,1,64,64,source.num_moduli)
            encoded=encoded%np.asarray(source.moduli,np.uint64)
            expected=server.ptct_mul[level]._mul_encoded(source,encoded)
            actual=plaintext.multiply_encoded(source,encoded,level)
            np.testing.assert_array_equal(actual.polynomial,expected.polynomial)
            assert not hasattr(actual,'_ckks_scale')
            assert actual._ckks_nsd==expected._ckks_nsd==nsd
            plaintext.clear()
            jax.clear_caches()
            actual=plaintext.multiply_encoded(source,encoded,level)
            np.testing.assert_array_equal(actual.polynomial,expected.polynomial)
        for level in (1,0):
            source=value if level==1 else server.he_level_reduce[1,0].level_reduce(value)
            encoded=server.encode_at_level(np.full(degree//2,.5),level,scale=float(1<<10))
            expected=server.ptct_mul[level].mul(source,encoded)
            actual=plaintext.multiply(source,encoded,level)
            np.testing.assert_array_equal(actual.polynomial,expected.polynomial)
            assert actual._ckks_scale==expected._ckks_scale
            assert actual._ckks_nsd==expected._ckks_nsd==nsd+1
        assert plaintext.compiler.compile_count==4
        rescale.clear()
        jax.clear_caches()
        actual=rescale.rescale(value,1,0)
        np.testing.assert_array_equal(actual.polynomial,server.he_rescale[1,0].rescale(value).polynomial)
        assert rescale.compiler.compile_count==1
        with pytest.raises(ValueError):
            rescale.rescale(value,1,1)
        with pytest.raises(ValueError):
            rescale.rescale(value,True,0)
        with pytest.raises(ValueError):
            plaintext.multiply_encoded(value,encoded.polynomial,0)
        with pytest.raises((TypeError,ValueError)):
            plaintext.multiply(value,value,1)
        value._ckks_nsd=1
        relinearization=CompiledRelinearization(server)
        operation=server.he_mul[0]
        tensor=operation.hemul_no_relin(value,value)
        expected_relin=operation.relinearize(tensor)
        actual_relin=relinearization.relinearize(tensor)
        np.testing.assert_array_equal(actual_relin.polynomial,expected_relin.polynomial)
        assert actual_relin._ckks_scale==expected_relin._ckks_scale
        assert actual_relin._ckks_nsd==expected_relin._ckks_nsd
        relinearization.clear()
        jax.clear_caches()
        actual_relin=relinearization.relinearize(tensor)
        np.testing.assert_array_equal(actual_relin.polynomial,expected_relin.polynomial)
        assert relinearization.compiler.compile_count==1
        with pytest.raises(ValueError):
            relinearization.relinearize(value)
        for parameter, invalid, pattern in [('sigma',0.,'noise sigma'),('secret_distribution','gaussian','uniform_ternary')]:
            original=server.parameters.get(parameter,None)
            server.parameters[parameter]=invalid
            try:
                with pytest.raises(ValueError,match=pattern):
                    relinearization.relinearize(tensor)
                with pytest.raises(ValueError,match=pattern):
                    rescale.rescale(value,1,0)
                with pytest.raises(ValueError,match=pattern):
                    plaintext.multiply_encoded(value,encoded.polynomial,1)
            finally:
                if original is None:
                    server.parameters.pop(parameter,None)
                else:
                    server.parameters[parameter]=original
    import gc,weakref
    from jaxite_word import ckks_ctx,polynomial
    import ntt_mm
    # Keep actual executable code alive while dropping every control owner.
    compilers=(rescale.compiler,plaintext.compiler,relinearization.compiler)
    refs=[weakref.ref(value) for value in (*rescale.binding[1].controls,*plaintext.binding[1].controls,*relinearization.binding[1].controls)]
    rescale.clear()
    plaintext.clear()
    relinearization.clear()
    ckks_ctx.evict_context_caches(client)
    ckks_ctx.evict_context_caches(server)
    del rescale,plaintext,relinearization,client,server,value,source,actual,expected,encoded
    del operation,tensor,actual_relin,expected_relin
    jax.clear_caches()
    polynomial._POLYNOMIAL_REGISTRY.clear()
    ntt_mm._barrett_ntt_context_cache.clear()
    gc.collect()
    assert tuple(compiler.compile_count for compiler in compilers) == (1, 4, 1)
    alive=[ref() for ref in refs if ref() is not None]
    assert not alive,[(tuple(value.shape),str(value.dtype)) for value in alive]
