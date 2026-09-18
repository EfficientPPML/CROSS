import jax
import finite_field as ff_context
import ntt_sm as ntt
from absl.testing import absltest
from absl.testing import parameterized
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import util

# Negacyclic NTT golden reference (psi factors fused into twiddles)
NTT = [
    (
        "0",
        2147483489,
        None,
        1,
        4,
        4,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [1927271639, 1976851019, 961431098, 750997937, 1858820200, 1645119999, 137255436, 1444884120, 379729034, 612811897, 1855784060, 1875694634, 102353194, 1048752242, 1711958033, 1037636875]
    )
]

class NTTTest(parameterized.TestCase):
  def __init__(self, *args, **kwargs):
    super(NTTTest, self).__init__(*args, **kwargs)
    self.random_key = jax.random.key(0)

  @parameterized.named_parameters(*NTT)
  def test_NTT_Barrett(self, q, psi, batch, r, c, coef_in, eval_in):
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.BarrettContext(moduli=q),
    }
    ntt_ctx = ntt.NTTContextBase(moduli=q, parameters=parameters)
    ntt_result_cf = ntt_ctx.ntt(jnp.array(coef_in, dtype=jnp.uint32).reshape(-1, r, c))
    np.testing.assert_array_equal(eval_in, ntt_result_cf.flatten().tolist())
    intt_result = ntt_ctx.intt(ntt_result_cf)
    np.testing.assert_array_equal(coef_in, intt_result[0].flatten().tolist())

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Montgomery(self, q, psi, batch, r, c, coef_in, eval_in):
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.MontgomeryContext(moduli=q),
    }
    ntt_ctx = ntt.NTTContextBase(moduli=q, parameters=parameters)
    test_in_cf = ntt_ctx.to_computation_format(jnp.array(coef_in, dtype=jnp.uint64)).reshape(-1, r, c)
    ntt_result_cf = ntt_ctx.ntt(test_in_cf)
    eval_recovered = ntt_ctx.to_original_format(ntt_result_cf.astype(jnp.uint64).flatten())
    np.testing.assert_array_equal(eval_in, eval_recovered.tolist())
    intt_result = ntt_ctx.intt(ntt_result_cf)
    x_recovered = ntt_ctx.to_original_format(intt_result.astype(jnp.uint64))
    np.testing.assert_array_equal(coef_in, x_recovered[0].flatten().tolist())


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_BATLazy(self, q, psi, batch, r, c, coef_in, eval_in):
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.BATLazyContext(moduli=q),
    }
    ntt_ctx = ntt.NTTBATLazyContext(moduli=q, parameters=parameters)
    ntt_result_cf = ntt_ctx.ntt(jnp.array(coef_in, dtype=jnp.uint32).reshape(-1, r, c))
    np.testing.assert_array_equal(eval_in, ntt_result_cf.flatten() % q)
    intt_result = ntt_ctx.intt(ntt_result_cf)
    x_recovered = ntt_ctx.to_original_format(intt_result)
    np.testing.assert_array_equal(coef_in, x_recovered[0].flatten() % q)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Shoup(self, q, psi, batch, r, c, coef_in, eval_in):
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.ShoupContext(moduli=q),
    }
    ntt_ctx = ntt.NTTShoupContext(moduli=q, parameters=parameters)
    ntt_result_cf = ntt_ctx.ntt(jnp.array(coef_in, dtype=jnp.uint32).reshape(r, c))
    eval_recovered = ntt_ctx.to_original_format(ntt_result_cf.flatten())
    np.testing.assert_array_equal(eval_in, eval_recovered)
    intt_result = ntt_ctx.intt(ntt_result_cf)
    x_recovered = ntt_ctx.to_original_format(intt_result)
    np.testing.assert_array_equal(coef_in, x_recovered.flatten().tolist())

if __name__ == "__main__":
  absltest.main()
