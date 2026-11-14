"""Boundary regressions for the correctness-only Jaxite patch.

Run with the patched checkout on PYTHONPATH; skip if Jaxite is not installed.
"""
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
import pytest

blind_rotate = pytest.importorskip('jaxite.jaxite_ckks.blind_rotate')
from jaxite.jaxite_ckks import key_gen, ntt_cpu, types

jax.config.update('jax_enable_x64', True)


def test_hybrid_reduces_31_bit_products_before_accumulation():
    n, theta, q = 16, 8, 2013265921
    moduli = jnp.array([q], jnp.uint32)
    pt = types.Plaintext(jnp.full((n, 1), q-1, jnp.uint32), moduli)
    key = types.Ciphertext(jnp.full((2, n, 1), q-1, jnp.uint32), moduli)
    kernel = blind_rotate.BlindRotation()
    kernel.all_moduli = moduli
    kernel.mul_kernel.precompute_constants([q])
    kernel.rescale_kernel = mock.Mock()
    kernel.rescale_kernel.rescale.side_effect = lambda ct: ct
    kernel.brot_mux = mock.Mock(side_effect=lambda **kwargs: kwargs['ct_in'])
    out = kernel.brot_hybrid((pt,)*4, [[key]*theta for _ in range(4)], None, theta)
    # Every (q-1)^2 is 1 modulo q, independently of evaluation order.
    np.testing.assert_array_equal(np.asarray(out.data), np.full((2, n, 1), 4*theta))


def test_signed_secret_extension_preserves_centered_coefficients():
    n = 16
    q, p = 1073742881, 1073742721
    coefficients = np.array([-1, 0, 1, -1] + [0]*(n-4), np.int64)
    key = types.SecretKey(ntt_cpu.ntt_negacyclic_poly((coefficients[:,None]%q).astype(np.uint64),[q]),np.array([q],np.uint32))
    extended = key_gen.extend_secret_key(key,[q,p])
    decoded = ntt_cpu.intt_negacyclic_poly(extended.data,[q,p])
    np.testing.assert_array_equal(decoded,np.column_stack([coefficients%q,coefficients%p]))
