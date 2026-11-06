"""Exact encrypted parity for the isolated explicit-key rotation compiler."""
from pathlib import Path
import gc
import sys
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from encrypted_rotation import CompiledRotation
from encrypted_compilation import CompiledTensorProduct


@pytest.mark.parametrize('batch,scale_bits,nsd', [(1, 20, 1), (2, 24, 2)])
def test_compiled_rotation_reuses_arithmetic_across_keys_and_preserves_metadata(batch, scale_bits, nsd, monkeypatch):
    import jax
    from jaxite_word import CKKSContext, he_params, key_gen, util
    jax.config.update('jax_enable_x64', True)
    degree, dnum = 4096, 4
    q = [40961, 65537, 114689, 147457]
    p = util.generate_p_towers(q, dnum, degree, aux_bits=30)
    assert he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, 'classical'))
    keys = key_gen.gen_pke_pair(q, [], degree)
    evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=dnum)
    indices = (1, 7)
    rotations = {index: key_gen.gen_rotation_key(keys['secret_key'], q, p, index, dnum=dnum)[index]
                 for index in indices}
    params = dict(degree=degree, num_slots=degree // 2, scaling_factor=float(1 << 20),
                  output_scale=float(1 << 20), q_towers=q, p_towers=p, composite_degree=2,
                  degree_layout=(64, 64), p=20, CKKS_M_FACTOR=1, max_bits_in_word=61,
                  noise_scale_degree=1)
    client = CKKSContext(dict(params, **keys))
    server = CKKSContext(dict(params, evaluation_key=[
        jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1) for name in ('a', 'b')]))
    server.program_initialization(list(indices), dnum=dnum, r=64, c=64, batch=batch,
                                  pregenerated_rotation_keys=rotations)
    slots = np.linspace(-.75, .75, degree // 2)
    ciphertext = client.encrypt_slots_batch([slots * (i + 1) / batch for i in range(batch)],
                                            scale=float(1 << scale_bits))
    # Bootstrap rotations also carry nsd=2, so the compiler must preserve it
    # rather than impose the polynomial evaluator's nsd=1 input convention.
    ciphertext._ckks_nsd = nsd
    compiler = CompiledRotation(server)
    for index in indices:
        expected = server.he_rot[server.max_level, index].rotate(ciphertext)
        if index == indices[0]:
            actual = compiler.rotate(ciphertext, index)
        else:
            # A bound level/shape only needs this key and its map. Forbid even
            # accessor lookup: it would reconstruct discarded NTT controls.
            def forbidden_lookup(*args, **kwargs):
                raise AssertionError('binding hit rebuilt a keyed rotation facade')
            with monkeypatch.context() as patch:
                patch.setattr(type(server.he_rot), '__getitem__', forbidden_lookup)
                actual = compiler.rotate(ciphertext, index)
        assert (server.max_level, index) not in server.he_rot._instances
        assert index not in server._param_cache._formatted_rotation_keys
        np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
        assert actual._ckks_scale == ciphertext._ckks_scale
        assert actual._ckks_nsd == nsd
        assert tuple(actual.moduli) == tuple(ciphertext.moduli)
    assert compiler.compile_count == 1
    lowered = server.he_level_reduce[server.max_level, 0].level_reduce(ciphertext)
    expected = server.he_rot[0, indices[0]].rotate(lowered)
    actual = compiler.rotate(lowered, indices[0])
    np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
    assert compiler.compile_count == 2
    compiler.clear()
    assert compiler._binding is None
    jax.clear_caches()
    actual = compiler.rotate(lowered, indices[0])
    np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
    assert compiler.compile_count == 2
    compiler.clear(executables=True)
    actual = compiler.rotate(lowered, indices[0])
    np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
    assert compiler.compile_count == 3
    with pytest.raises(ValueError, match='not supplied.*unavailable'):
        compiler.rotate(lowered, 19)

    # Bootstrap uses the no-rescale tensor/relinearize path rather than the
    # ordinary multiply facade that consumes a composite level.
    ciphertext._ckks_nsd = 1
    product = CompiledTensorProduct(server)
    operation = server.he_mul[server.max_level - 1]
    expected = operation.relinearize(operation.hemul_no_relin(ciphertext, ciphertext))
    actual = product.multiply(ciphertext, ciphertext)
    np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
    assert actual.num_moduli == ciphertext.num_moduli
    assert actual._ckks_scale == ciphertext._ckks_scale ** 2
    assert actual._ckks_nsd == 2
    product.clear()
    jax.clear_caches()
    actual = product.multiply(ciphertext, ciphertext)
    np.testing.assert_array_equal(actual.polynomial, expected.polynomial)
    assert product.compile_count == 1

    # Retain the real compiled HE executables while releasing the entire
    # evaluator and all rebuildable control caches. Nested jit constants must
    # not pin the retired rotation/NTT/relinearization device arrays.
    rotation_code, tensor_code = compiler._compiler, product._compiler
    references = [weakref.ref(value) for value in
                  (*compiler._binding[1].controls, *product._binding[1].controls)]
    compiler.clear()
    product.clear()
    from jaxite_word import ckks_ctx, polynomial
    import ntt_mm  # HEParameterCache uses the historical flat module.
    ckks_ctx.evict_context_caches(client)
    ckks_ctx.evict_context_caches(server)
    del compiler, product, operation, client, server, ciphertext, lowered, expected, actual
    jax.clear_caches()
    polynomial._POLYNOMIAL_REGISTRY.clear()
    ntt_mm._barrett_ntt_context_cache.clear()
    gc.collect()
    assert rotation_code.compile_count == 3 and tensor_code.compile_count == 1
    alive = [reference() for reference in references if reference() is not None]
    assert not alive, [(tuple(value.shape), str(value.dtype)) for value in alive]
