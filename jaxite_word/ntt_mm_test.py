import jax
import finite_field as ff_context
import ntt_mm as ntt
from absl.testing import absltest
from absl.testing import parameterized
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import util
import os

# Negacyclic NTT
NTT = [
    (
        "0",
        [2147483489, 2147483137, 2147482817],
        None,
        3,
        4,
        4,
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [7, 0, 0, 0, 11, 0, 0, 0, 13, 0, 0, 0, 17, 0, 0, 0], [16, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8]],
        [[1927271639, 1976851019, 961431098, 750997937, 1858820200, 1645119999, 137255436, 1444884120, 379729034, 612811897, 1855784060, 1875694634, 102353194, 1048752242, 1711958033, 1037636875], [1409600302, 1409600302, 1409600302, 1409600302, 1397968663, 1397968663, 1397968663, 1397968663, 1703050717, 1703050717, 1703050717, 1703050717, 1931829757, 1931829757, 1931829757, 1931829757], [675784815, 337807662, 1571641390, 400235048, 1145917821, 477891965, 1006973034, 524384318, 1690087058, 137036486, 1954663574, 889829120, 1108726548, 2133686119, 1387723090, 1737474744]]
    ),
]

class NTTTest(parameterized.TestCase):
  def __init__(self, *args, **kwargs):
    super(NTTTest, self).__init__(*args, **kwargs)
    self.random_key = jax.random.key(0)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Barrett(self, q, psi, batch, r, c, coef_in, eval_in):
    b = 2 # batch size
    coef_in = jnp.concatenate(
        [jnp.array(coef_in, dtype=jnp.uint64).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    ).astype(jnp.uint32)
    eval_in = jnp.concatenate(
        [jnp.array(eval_in, dtype=jnp.uint32).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    ).astype(jnp.uint32)
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.BarrettContext(moduli=q),
    }
    ntt_ctx = ntt.NTTCiphertextBarrettContext(moduli=q, parameters=parameters)
    # bit_reverse_indices = jnp.array(util.bit_reverse_indices(r*c), jnp.uint32)
    ntt_result_cf = ntt_ctx.ntt(coef_in.reshape(b, r, c, -1))
    # coef_in_br = jnp.take(ntt_result_cf.reshape(b, r*c, -1), bit_reverse_indices, axis=-2)
    np.testing.assert_array_equal(eval_in, ntt_result_cf.reshape(b, r*c, -1))
    intt_result = ntt_ctx.intt(ntt_result_cf)
    np.testing.assert_array_equal(coef_in, intt_result.reshape(b, r*c, -1).tolist())

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Montgomery(self, q, psi, batch, r, c, coef_in, eval_in):
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.MontgomeryContext(moduli=q),
    }
    b = 2 # batch size
    coef_in = jnp.concatenate(
        [jnp.array(coef_in, dtype=jnp.uint64).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    )
    eval_in = jnp.concatenate(
        [jnp.array(eval_in, dtype=jnp.uint32).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    )

    ntt_ctx = ntt.NTTCiphertextMontgomeryContext(moduli=q, parameters=parameters)
    # bit_reverse_indices = jnp.array(util.bit_reverse_indices(r*c), jnp.uint32)
    test_in_cf = ntt_ctx.to_computation_format(coef_in).astype(jnp.uint32).reshape(b, r, c, -1)
    ntt_result_cf = ntt_ctx.ntt(test_in_cf)
    eval_recovered = ntt_ctx.to_original_format(ntt_result_cf.reshape(b, r*c, -1).astype(jnp.uint64))
    # coef_in_br = jnp.take(eval_recovered, bit_reverse_indices, axis=-2)
    np.testing.assert_array_equal(eval_in, eval_recovered)
    intt_result = ntt_ctx.intt(ntt_result_cf)
    x_recovered = ntt_ctx.to_original_format(intt_result.reshape(b, r*c, -1))
    np.testing.assert_array_equal(coef_in, x_recovered.reshape(b, r*c, -1).tolist())
    jit_ntt = jax.jit(ntt_ctx.ntt)
    jit_ntt(test_in_cf)
    profile_name = f"NTT_Montgomery_Performance"
    file_path = os.path.join(os.path.dirname(__file__), "log", profile_name)
    with jax.profiler.trace(file_path):
      jit_ntt(test_in_cf)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Shoup(self, q, psi, batch, r, c, coef_in, eval_in):
    b = 2
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.ShoupContext(moduli=q),
    }
    coef_in = jnp.concatenate(
        [jnp.array(coef_in, dtype=jnp.uint64).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    )
    eval_in = jnp.concatenate(
        [jnp.array(eval_in, dtype=jnp.uint32).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    )

    ntt_ctx = ntt.NTTCiphertextShoupContext(moduli=q, parameters=parameters)
    ntt_result_cf = ntt_ctx.ntt(jnp.array(coef_in, dtype=jnp.uint32).reshape(b, r, c, -1))
    eval_recovered = ntt_ctx.to_original_format(ntt_result_cf)
    np.testing.assert_array_equal(eval_in, eval_recovered.reshape(b, r*c, -1))
    intt_result = ntt_ctx.intt(ntt_result_cf)
    x_recovered = ntt_ctx.to_original_format(intt_result)
    np.testing.assert_array_equal(coef_in, x_recovered.reshape(b, r*c, -1).tolist())

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_BATLazy(self, q, psi, batch, r, c, coef_in, eval_in):
    b = 2 # batch size
    coef_in = jnp.concatenate(
        [jnp.array(coef_in, dtype=jnp.uint64).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    ).astype(jnp.uint32)
    eval_in = jnp.concatenate(
        [jnp.array(eval_in, dtype=jnp.uint32).transpose(1, 0).reshape(1, r * c, -1) for _ in range(b)],
        axis=0,
    ).astype(jnp.uint32)
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.BarrettContext(moduli=q),
    }
    ntt_ctx = ntt.NTTCiphertextBATLazyContext(moduli=q, parameters=parameters)
    ntt_result_cf = ntt_ctx.ntt(coef_in.reshape(b, r, c, -1))
    np.testing.assert_array_equal(eval_in, ntt_result_cf.reshape(b, r*c, -1))
    intt_result = ntt_ctx.intt(ntt_result_cf)
    np.testing.assert_array_equal(coef_in, intt_result.reshape(b, r*c, -1).tolist())


if __name__ == "__main__":
  absltest.main()
