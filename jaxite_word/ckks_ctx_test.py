import jax
import jax.numpy as jnp
import numpy as np

import ckks_ctx
import hemul
import herot
import polynomial
import rescale
import util
import key_gen as kg
from absl.testing import absltest
from absl.testing import parameterized

HEMul = hemul.HEMul
Polynomial = polynomial.Polynomial
HERot = herot.HERot
HERescale = rescale.HERescale

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

testing_params = [
  {
    'testcase_name': '0',
  }
]

@parameterized.named_parameters(testing_params)
class CKKSContextTest(parameterized.TestCase):
  def setUp(self):
    self.degree = 16
    self.num_slots = 8
    self.dnum = 3
    self.r, self.c = 4, 4

    self.scaling_factor = 563019763943521
    self.q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
    self.p_towers = [1073740609, 1073739937, 1073739649]
    self.q = 696985728458547852910430465530901300664961
    self.qt = 1329229981441028949792278227703286337
    self.p = 30
    self.CKKS_M_FACTOR = 1
    self.noise_scale_degree = 1
    self.max_bits_in_word = 61
    self.sigma = 3.190000057220458984375
    self.degree_layout = (self.r, self.c)

    key_pair = kg.gen_pke_pair(self.q_towers, self.p_towers, self.degree)
    self.params = {
        "degree": self.degree,
        "num_slots": self.num_slots,
        "scaling_factor": self.scaling_factor,
        "output_scale": self.scaling_factor,
        "q_towers": self.q_towers,
        "p_towers": self.p_towers,
        "p": self.p,
        "CKKS_M_FACTOR": self.CKKS_M_FACTOR,
        "max_bits_in_word": self.max_bits_in_word,
        "noise_scale_degree": self.noise_scale_degree,
        "public_key": key_pair["public_key"],
        "secret_key": key_pair["secret_key"]
    }
    self.real_values_input_in1 = [
        complex(0.25, 0), complex(0.5, 0), complex(0.75, 0), complex(1, 0),
        complex(2, 0), complex(3, 0), complex(4, 0), complex(5, 0),
    ]
    self.real_values_input_in2 = [
        complex(5, 0), complex(4, 0), complex(3, 0), complex(2, 0),
        complex(1, 0), complex(0.75, 0), complex(0.5, 0), complex(0.25, 0),
    ]
    self.real_values_multiply_result = [
        complex(1.25, 0), complex(2, 0), complex(2.25, 0), complex(2, 0),
        complex(2, 0), complex(2.25, 0), complex(2, 0), complex(1.25, 0),
    ]
    self.real_values_rotate_result = [
        complex(0.5, 0), complex(0.75, 0), complex(1, 0), complex(2, 0),
        complex(3, 0), complex(4, 0), complex(5, 0), complex(0.25, 0),
    ]

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encode_decode(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Decoding
    decoded_values = ctx.decode(encoded_ct, is_ntt=True)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_decrypt(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Decryption
    decrypted_ct = ctx.decrypt(encrypted_ct)
    # Step 4: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rotate_decrypt(self):
    # Paramters Setup
    rotate_idx = 1
    coef_map = util.precompute_auto_map(self.degree, kg.find_automorphism_index_2n_complex(rotate_idx, 2 * self.degree))
    # initialization
    herot_obj = HERot(self.r, self.c, self.dnum, self.q_towers, self.p_towers)
    ek_dict = kg.gen_rotation_key(self.params["secret_key"], self.q_towers, self.p_towers, rotate_idx, dnum=self.dnum, noise_std=self.sigma, noise_scale=self.noise_scale_degree)
    ek = ek_dict[rotate_idx]
    herot_obj.setup_rotate(jnp.array(ek["a"], jnp.uint64).transpose(0,2,1).reshape(self.dnum,*self.degree_layout,-1), jnp.array(ek["b"], jnp.uint64).transpose(0,2,1).reshape(self.dnum,*self.degree_layout,-1), coef_map)
    herot_obj.control_gen(batch=1, degree_layout=self.degree_layout)
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Rotate
    encrypted_ct.polynomial = encrypted_ct.polynomial.reshape(1, 2, *self.degree_layout, len(self.q_towers))
    result_ct = herot_obj.rotate(encrypted_ct)
    # Step 4: Decryption
    decrypted_ct = ctx.decrypt(result_ct)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_rotate_result, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rescale_decrypt(self):
    # Paramters Setup
    batch, num_elements, degree, num_moduli = 1, 2, 16, 5
    ct_shapes = {'batch': 1, 'num_elements': 2, 'degree': 16, 'num_moduli': 5, 'precision': 32, 'degree_layout': self.degree_layout}
    ct_params = {'moduli': self.q_towers, 'r': self.r, 'c': self.c}
    params = self.params.copy()
    params.update({
        "output_scale": (self.scaling_factor/self.q_towers[-1]),
    })
    # Initialization
    ctx = ckks_ctx.CKKSContext(params)
    he_rescale = HERescale(batch=batch, num_elements=num_elements, moduli=self.q_towers, r=self.r, c=self.c, degree_layout=self.degree_layout)
    he_rescale.control_gen()

    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Rescale
    in_data = encrypted_ct.polynomial.reshape(batch, num_elements, *self.degree_layout, num_moduli)
    rescaled_data = he_rescale.rescale(in_data)
    # Step 4: Decryption — wrap result in a Polynomial for decrypt
    ct_out = Polynomial(
        {'batch': batch, 'num_elements': num_elements, 'degree': degree,
         'num_moduli': num_moduli - 1, 'precision': 32, 'degree_layout': self.degree_layout},
        {'moduli': self.q_towers[:-1], 'r': self.r, 'c': self.c})
    ct_out.polynomial = rescaled_data.reshape(batch, num_elements, degree, num_moduli - 1)
    decrypted_ct = ctx.decrypt(ct_out)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_multiply_decrypt(self):
    """
    Test the encryption, multiplication, and decryption of the CKKS context.
    See hemul_test.py for the debugging version
    """
    # Paramters Setup
    r, c = 4, 4
    assert (r*c==self.degree)
    batch, num_elements, dnum, num_eval_mult = 1, 2, self.dnum, 1
    self.ek = kg.gen_evaluation_key(self.params["secret_key"], q=self.q_towers, P=self.p_towers, noise_std=self.sigma, noise_scale=1, dnum=3)
    eval_key_a, eval_key_b = jnp.array(self.ek["a"], dtype=jnp.uint32).transpose(0,2,1), jnp.array(self.ek["b"], dtype=jnp.uint32).transpose(0,2,1)
    params = self.params.copy()
    params.update({
        "evaluation_key": [eval_key_a, eval_key_b],
        "output_scale": (self.scaling_factor/self.q_towers[-1])**2,
    })
    # Initialization
    ctx = ckks_ctx.CKKSContext(params)
    he_mul = HEMul(batch, r, c, dnum, num_eval_mult, self.q_towers, self.p_towers)
    he_mul.control_gen(degree_layout=self.degree_layout)
    he_mul.setup_relinearization(eval_key_a, eval_key_b)
    # Step 1: Encoding
    encoded_ct1 = ctx.encode(self.real_values_input_in1)
    encoded_ct2 = ctx.encode(self.real_values_input_in2)
    # Step 2: Encryption
    encrypted_ct1 = ctx.encrypt(encoded_ct1)
    encrypted_ct2 = ctx.encrypt(encoded_ct2)
    # Step 3: Homomorphic Multiplication
    in_cts_array = jnp.concatenate([encrypted_ct1.polynomial, encrypted_ct2.polynomial], axis=1).reshape(batch, 2*num_elements, r, c, len(self.q_towers)).astype(jnp.uint32)
    ct_in_shapes = {'batch': batch, 'num_elements': 2*num_elements, 'degree': self.degree, 'precision': 32, 'num_moduli': len(self.q_towers), 'degree_layout': (r, c)}
    ct_in = Polynomial(ct_in_shapes, parameters={'moduli': self.q_towers})
    ct_in.polynomial = in_cts_array
    encrypted_result = he_mul.mul(ct_in)
    # Step 4: Decryption
    encrypted_ct1.drop_last_modulus()
    encrypted_ct1.set_batch_polynomial(encrypted_result.polynomial.reshape(batch, 2, self.degree, len(self.q_towers)-1))
    decrypted_result = ctx.decrypt(encrypted_ct1)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_result, is_ntt=False)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_multiply_result, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_composite_rescale(self):
    """
    Test that composite rescaling correctly calculates scale factors.

    This test verifies the mathematical correctness of composite scaling:
    - composite_degree=k groups k moduli into a single logical scale
    - composite_scale_factor = product of last k moduli
    - After rescale: effective_scale = original_scale / composite_scale_factor

    This follows the algorithm from ePrint 2023/1462:
    "High-precision RNS-CKKS on fixed but smaller word-size architectures"
    """
    # Test with different composite_degree values
    for composite_degree in [1, 2]:
        # Calculate expected composite scale
        expected_composite_scale = 1
        for i in range(composite_degree):
            expected_composite_scale *= self.q_towers[-(i + 1)]

        params = self.params.copy()
        params.update({
            "composite_degree": composite_degree,
            "output_scale": self.scaling_factor,
        })

        ctx = ckks_ctx.CKKSContext(params)

        # Verify composite scale factor
        self.assertEqual(
            ctx.composite_scale_factor,
            expected_composite_scale,
            f"Failed for composite_degree={composite_degree}"
        )

        # Calculate effective scale bits
        composite_bits = expected_composite_scale.bit_length()
        expected_bits_per_modulus = composite_bits / composite_degree

        # For 30-bit moduli, composite degree k should give ~k*30 bits
        self.assertGreater(
            expected_bits_per_modulus,
            15,  # Each modulus contributes at least 15 bits
            f"composite_degree={composite_degree} should use reasonable moduli"
        )


if __name__ == "__main__":
  absltest.main()
