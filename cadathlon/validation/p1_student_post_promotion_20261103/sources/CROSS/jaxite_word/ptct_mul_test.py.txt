"""Tests for ptct_mul.py."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import ckks_ctx
import key_gen as kg

jax.config.update("jax_enable_x64", True)


PTCT_MUL_TEST_CASES = [
    {
        "testcase_name": "vpu",
        "use_bat": False,
    },
    {
        "testcase_name": "bat",
        "use_bat": True,
    },
]


class PtCtMulTest(parameterized.TestCase):
    """End-to-end ptct_mul correctness against cleartext slot products."""

    def setUp(self):
        super().setUp()
        self.degree = 16
        self.num_slots = 8
        self.dnum = 3
        self.r, self.c = 4, 4
        self.degree_layout = (self.r, self.c)

        self.scaling_factor = 563019763943521
        self.q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
        self.p_towers = [1073740609, 1073739937, 1073739649]
        self.p = 30
        self.CKKS_M_FACTOR = 1
        self.noise_scale_degree = 1
        self.max_bits_in_word = 61
        self.sigma = 3.190000057220458984375

        self.real_values_input_in1 = [
            complex(0.25, 0),
            complex(0.5, 0),
            complex(0.75, 0),
            complex(1.0, 0),
            complex(2.0, 0),
            complex(3.0, 0),
            complex(4.0, 0),
            complex(5.0, 0),
        ]
        self.real_values_input_in2 = [
            complex(5.0, 0),
            complex(4.0, 0),
            complex(3.0, 0),
            complex(2.0, 0),
            complex(1.0, 0),
            complex(0.75, 0),
            complex(0.5, 0),
            complex(0.25, 0),
        ]
        self.expected_slot_product = [
            a * b
            for a, b in zip(self.real_values_input_in1, self.real_values_input_in2)
        ]

    def _build_context(self):
        key_pair = kg.gen_pke_pair(self.q_towers, self.p_towers, self.degree)
        params = {
            "degree": self.degree,
            "num_slots": self.num_slots,
            "scaling_factor": self.scaling_factor,
            "output_scale": self.scaling_factor * self.scaling_factor,
            "q_towers": self.q_towers,
            "p_towers": self.p_towers,
            "p": self.p,
            "CKKS_M_FACTOR": self.CKKS_M_FACTOR,
            "max_bits_in_word": self.max_bits_in_word,
            "noise_scale_degree": self.noise_scale_degree,
            "public_key": key_pair["public_key"],
            "secret_key": key_pair["secret_key"],
        }
        ctx = ckks_ctx.CKKSContext(params)
        ctx.program_initialization(
            total_rotation_indices=[],
            dnum=self.dnum,
            r=self.r,
            c=self.c,
        )
        return ctx

    def _run_ptct_mul(self, ctx: ckks_ctx.CKKSContext, use_bat: bool):
        level = ctx.max_level
        num_q = len(self.q_towers)

        encoded_ct = ctx.encode(self.real_values_input_in1)
        encoded_pt = ctx.encode(self.real_values_input_in2)
        encrypted_ct = ctx.encrypt(encoded_ct)
        encrypted_ct.polynomial = encrypted_ct.polynomial.reshape(
            1, 2, self.r, self.c, num_q
        )

        op = ctx.ptct_mul[level]
        pt_ntt = encoded_pt.polynomial[0, 0].reshape(self.r, self.c, num_q).astype(jnp.uint32)
        if use_bat:
            # BAT remains a private kernel experiment; the supported facade
            # operation is the stateless two-Polynomial form below.
            op._precompute_bat(pt_ntt)

        input_payload = encrypted_ct.polynomial.copy()
        result_ct = (
            op._mul_prepared(encrypted_ct, use_bat=True)
            if use_bat
            else op.mul(encrypted_ct, encoded_pt)
        )
        self.assertIsNot(result_ct, encrypted_ct)
        np.testing.assert_array_equal(encrypted_ct.polynomial, input_payload)

        decrypted = ctx.decrypt(result_ct)
        decoded = ctx.decode(decrypted, is_ntt=False)
        return decoded

    @parameterized.named_parameters(*PTCT_MUL_TEST_CASES)
    def test_encrypt_ptct_mul_decrypt_matches_cleartext(self, use_bat: bool):
        ctx = self._build_context()
        decoded = self._run_ptct_mul(ctx, use_bat=use_bat)
        np.testing.assert_array_almost_equal(
            decoded,
            self.expected_slot_product,
            decimal=3,
        )

    @parameterized.named_parameters(*PTCT_MUL_TEST_CASES)
    def test_montgomery_matches_barrett_and_decodes(self, use_bat: bool):
        """Montgomery and Barrett private kernels agree bit-canonically, and the
        Montgomery result decrypts/decodes to the expected slot products."""
        import finite_field as ff_context
        import ptct_mul as ptct_module
        from polynomial import Polynomial

        ctx = self._build_context()
        num_q = len(self.q_towers)

        encoded_ct = ctx.encode(self.real_values_input_in1)
        encoded_pt = ctx.encode(self.real_values_input_in2)
        encrypted_ct = ctx.encrypt(encoded_ct)
        ct_data = encrypted_ct.polynomial.reshape(
            1, 2, self.r, self.c, num_q).astype(jnp.uint64)
        pt_ntt = encoded_pt.polynomial[0, 0].reshape(
            self.r, self.c, num_q).astype(jnp.uint64)

        def run(ffcls, data):
            op = ptct_module._HEPtCtMulKernel(
                1, self.r, self.c, self.q_towers,
                finite_field_context=ffcls)
            if use_bat:
                op.precompute_plaintext_bat(pt_ntt)
            else:
                op.set_plaintext(pt_ntt)
            ct = Polynomial(
                {'batch': 1, 'num_elements': 2, 'degree': self.degree,
                 'num_moduli': num_q, 'precision': 32,
                 'degree_layout': (self.r, self.c)},
                {'moduli': self.q_towers, 'finite_field_context': ffcls})
            ct.polynomial = data
            return op.mul(ct, use_bat=use_bat).polynomial

        out_barrett = run(ff_context.BarrettContext, ct_data.astype(jnp.uint32))

        mont = ff_context.MontgomeryContext(self.q_towers)
        out_mont_fmt = run(
            ff_context.MontgomeryContext,
            mont.to_computation_format(ct_data).astype(jnp.uint32))
        out_mont = mont.to_original_format(jnp.asarray(out_mont_fmt, jnp.uint64))

        q_arr = jnp.array(self.q_towers, jnp.uint64)
        np.testing.assert_array_equal(
            jnp.asarray(out_barrett, jnp.uint64) % q_arr, out_mont)

        # End-to-end decode of the Montgomery result
        result_ct = Polynomial(
            {'batch': 1, 'num_elements': 2, 'degree': self.degree,
             'num_moduli': num_q, 'precision': 32,
             'degree_layout': self.degree_layout},
            {'moduli': self.q_towers})
        result_ct.polynomial = jnp.asarray(out_mont, jnp.uint32).reshape(
            1, 2, *self.degree_layout, num_q)
        decoded = ctx.decode(ctx.decrypt(result_ct), is_ntt=False)
        np.testing.assert_array_almost_equal(
            decoded, self.expected_slot_product, decimal=3)


if __name__ == "__main__":
    absltest.main()
