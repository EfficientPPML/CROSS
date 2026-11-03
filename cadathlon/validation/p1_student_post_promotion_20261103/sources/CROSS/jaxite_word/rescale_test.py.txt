import functools

import jax
import jax.numpy as jnp
import util

import finite_field as ff_context
from rescale import _HERescaleKernel
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

jax.config.update("jax_enable_x64", True)

class RescaleJaxTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    self.key = jax.random.key(0)
    self.q_towers = [1073742113, 1073740609, 1073741953, 1073741441, 1073741857]
    self.p_towers = [1073739937, 1073739649]
    self.batch, self.r, self.c, self.dnum = 1, 4, 4, 3
    self.in_ciphertexts_str = """
Element 0: 0: EVAL: [423785547 867445182 314281690 387216809 317665009 744769517 854727753 556289972 178254355 728938463 422880370 625974632 839454833 265462911 794994418 971808730] modulus: 1073742113
1: EVAL: [674572399 562016042 689811229 645200620 890951569 589133615 973213608 556537284 422035805 114108925 325645582 634088838 157955718 598501244 264734960 731651488] modulus: 1073740609
2: EVAL: [7101169 227629453 973189237 382116887 436205593 1049470765 618898275 211745856 424574924 346573960 420885112 908930650 155218136 442283445 5245353 655703819] modulus: 1073741953
3: EVAL: [717545614 537748826 732975596 465575012 222502279 540842086 535809083 838630621 143568799 822209840 78316550 364841057 1059974140 272106758 437331910 250156030] modulus: 1073741441
4: EVAL: [960621811 759889476 1065817348 937596980 332871829 378411034 1691508 496620154 932193751 822088981 519113977 295821080 239696052 110480767 1044402272 425743785] modulus: 1073741857
Element 1: 0: EVAL: [93050344 982589074 696085422 903808329 1073011810 882273633 695828671 583700055 717731060 272965092 503830870 771893460 349583234 875122295 468434047 317633096] modulus: 1073742113
1: EVAL: [773420993 1016391708 486533351 907193085 276641678 795453196 1042614922 799111929 1069809506 957528200 258981671 76107404 365378699 1036703896 171269251 598535025] modulus: 1073740609
2: EVAL: [869014237 890954083 305991144 948597079 279786774 1060040146 775843178 725010039 662779830 173756107 797961074 171294358 119622542 832227271 773545209 556481132] modulus: 1073741953
3: EVAL: [523961024 59369515 805484081 142062287 772180452 880685500 36297104 477514495 1044291625 315124203 70786593 506398363 271859057 901297327 248205340 1046163582] modulus: 1073741441
4: EVAL: [490685917 348525422 488819912 171056455 578135610 330980879 7463878 7389941 964683342 827884344 119294455 916674206 896391895 75091622 915263159 812028983] modulus: 1073741857
"""
    self.in_ciphertexts = util.parse_ciphertext_string(self.in_ciphertexts_str)[0]

    self.final_result_str = """
Element 0: 0: EVAL: [124157418 224769381 1020412604 569932777 981325121 470810866 1029735180 113719130 537113111 970877925 440985463 891573079 282851624 388993781 886080475 101309031] modulus: 1073742113
1: EVAL: [624907766 506816351 29862080 728329160 393393842 1072665381 244219748 661516300 191712908 533435850 216404242 206978512 75342479 179480532 969594190 598068547] modulus: 1073740609
2: EVAL: [968736221 819405119 779260731 718396480 902931463 711592629 633115307 294975194 367534575 833863491 42215971 630663680 876237112 466398473 208344709 308403580] modulus: 1073741953
3: EVAL: [920014272 96135897 606341721 201368550 650843777 950281688 969317123 755310602 289763239 26347580 539213324 464126245 20526087 420557824 1065022397 1055579960] modulus: 1073741441
Element 1: 0: EVAL: [187797345 346468403 400091616 779213129 237567707 272698807 849445132 901732556 147999934 456593923 481331552 783750384 356629236 281115942 501544985 691247402] modulus: 1073742113
1: EVAL: [286415726 122441288 468680473 406964333 495076659 202468229 760746605 543463053 669599140 113818155 1048004662 784691089 34456805 46178384 176257778 438475456] modulus: 1073740609
2: EVAL: [112913532 1033686805 589830928 753319193 302065675 194084609 587390065 547338915 244128372 495461986 12041662 957047606 1057381834 792645433 788806231 537244886] modulus: 1073741953
3: EVAL: [64214583 953035458 352178779 71249168 817404617 344838098 261594243 796287197 886003104 362926014 166409376 1062227053 672133715 579323825 559836544 62465017] modulus: 1073741441
"""
    self.final_result_ref = util.parse_ciphertext_string(self.final_result_str)[0]

  def test_control_rejects_invalid_composite_degree(self):
    kernel = _HERescaleKernel(
        batch=1,
        num_elements=2,
        moduli=self.q_towers,
        r=self.r,
        c=self.c,
        degree_layout=(self.r, self.c),
    )
    for invalid in (0, True, len(self.q_towers)):
      with self.subTest(composite_degree=invalid), self.assertRaises(ValueError):
        kernel.control_gen(composite_degree=invalid)

  # @absltest.skip("test a single experiment")
  def test_rescale_ciphertext(self):
    in_ciphertexts_arr = jnp.array(self.in_ciphertexts, jnp.uint32)
    in_ciphertexts_reshaped = in_ciphertexts_arr[None, ...]
    input_shape = in_ciphertexts_reshaped.shape
    output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] - 1)
    degree_layout = (self.r, self.c)
    he_rescale = _HERescaleKernel(batch=1, num_elements=2, moduli=self.q_towers, r=self.r, c=self.c, degree_layout=degree_layout)
    he_rescale.control_gen()
    self.assertEqual(he_rescale._lift_reduction_modes, ('direct',))
    in_data = in_ciphertexts_reshaped.reshape(input_shape[0], input_shape[1], *degree_layout, input_shape[3])
    final_result_custom = he_rescale._rescale_array(in_data).reshape(output_shape)

    np.testing.assert_array_equal(final_result_custom[0], self.final_result_ref)

  @parameterized.parameters(1, 2)
  def test_rescale_ciphertext_montgomery(self, batch):
    """Montgomery-reduction rescale matches the (Barrett-verified) golden.

    Input/output cross the boundary in Montgomery computation format; all
    internal modular reductions are Montgomery reductions.
    """
    in_ciphertexts_arr = jnp.array(self.in_ciphertexts, jnp.uint64)
    in_ciphertexts_arr = jnp.tile(in_ciphertexts_arr[None, ...], (batch, 1, 1, 1))
    input_shape = in_ciphertexts_arr.shape
    output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] - 1)
    degree_layout = (self.r, self.c)
    he_rescale = _HERescaleKernel(
        batch=batch, num_elements=2, moduli=self.q_towers, r=self.r, c=self.c,
        degree_layout=degree_layout,
        finite_field_context=ff_context.MontgomeryContext)
    he_rescale.control_gen()
    in_data = in_ciphertexts_arr.reshape(
        input_shape[0], input_shape[1], *degree_layout, input_shape[3])
    in_mont = ff_context.MontgomeryContext(self.q_towers).to_computation_format(in_data)
    out_mont = he_rescale._rescale_array(in_mont)
    out = ff_context.MontgomeryContext(self.q_towers[:-1]).to_original_format(
        jnp.asarray(out_mont, jnp.uint64)).reshape(output_shape)

    for b in range(batch):
      np.testing.assert_array_equal(out[b], self.final_result_ref)

  def test_rescale_composite_degree2_montgomery_matches_barrett(self):
    """Montgomery composite rescale (cd=2) == Barrett bit-canonically.

    Exercises the multi-iteration constant encoding (gamma*R^2 / beta*R per
    iteration column set) that the cd=1 golden twins do not reach.
    """
    in_ciphertexts_arr = jnp.array(self.in_ciphertexts, jnp.uint64)[None, ...]
    degree_layout = (self.r, self.c)
    in_data = in_ciphertexts_arr.reshape(1, 2, *degree_layout, len(self.q_towers))

    barrett = _HERescaleKernel(batch=1, num_elements=2, moduli=self.q_towers,
                        r=self.r, c=self.c, degree_layout=degree_layout)
    barrett.control_gen(composite_degree=2)
    ref = jnp.asarray(
        barrett._rescale_array(in_data.astype(jnp.uint32)), jnp.uint64
    )

    mont = _HERescaleKernel(batch=1, num_elements=2, moduli=self.q_towers,
                     r=self.r, c=self.c, degree_layout=degree_layout,
                     finite_field_context=ff_context.MontgomeryContext)
    mont.control_gen(composite_degree=2)
    in_mont = ff_context.MontgomeryContext(self.q_towers).to_computation_format(in_data)
    out = ff_context.MontgomeryContext(self.q_towers[:-2]).to_original_format(
        jnp.asarray(mont._rescale_array(in_mont), jnp.uint64))

    q_out = jnp.array(self.q_towers[:-2], jnp.uint64)
    np.testing.assert_array_equal(ref % q_out, out)

  def test_rescale_wide_magnitude_centered_lift(self):
    """Regression: wide-magnitude chain where a remaining modulus is below
    half the dropped tower, so the centered lift hits the u64-underflow
    region that the old `q_j - q_last + last_coeffs` form corrupted (off by
    2^64 mod q_j). Random inputs exercise the upper-half (false) branch
    densely; Barrett and Montgomery use entirely different reduction
    arithmetic (multiply-high vs REDC) on the same lift, so their agreement
    on the canonical result is strong evidence the lift is exact.
    """
    r = c = 4
    degree = r * c
    # find_moduli_ntt(_, _, D) yields primes = 1 mod D; the negacyclic NTT
    # needs = 1 mod 2*degree, so request against 2*degree.
    remaining = util.find_moduli_ntt(2, 28, 2 * degree)  # ~2^28 remaining towers
    dropped = util.find_moduli_ntt(1, 31, 2 * degree)    # ~2^31 dropped tower
    moduli = remaining + dropped                      # dropped tower is last
    q_last = moduli[-1]
    self.assertLess(q_last, 1 << 31)                  # inside Montgomery envelope
    self.assertLess(min(remaining), (q_last - 1) // 2)  # underflow region non-empty
    degree_layout = (r, c)

    key = jax.random.key(7)
    cols = []
    for j, q in enumerate(moduli):
      cols.append(jax.random.randint(
          jax.random.fold_in(key, j), (1, 2, degree, 1), 0, q, dtype=jnp.uint32))
    in_data = jnp.concatenate(cols, axis=-1).reshape(1, 2, *degree_layout, len(moduli))

    barrett = _HERescaleKernel(batch=1, num_elements=2, moduli=moduli,
                        r=r, c=c, degree_layout=degree_layout)
    barrett.control_gen()
    self.assertEqual(barrett._lift_reduction_modes, ('barrett',))
    ref = jnp.asarray(
        barrett._rescale_array(in_data.astype(jnp.uint32)), jnp.uint64
    )

    mont = _HERescaleKernel(batch=1, num_elements=2, moduli=moduli,
                     r=r, c=c, degree_layout=degree_layout,
                     finite_field_context=ff_context.MontgomeryContext)
    mont.control_gen()
    # uint64 before to_computation_format: it shifts left by 32, which would
    # overflow a uint32 input to zero.
    in_mont = ff_context.MontgomeryContext(moduli).to_computation_format(
        in_data.astype(jnp.uint64))
    out = ff_context.MontgomeryContext(moduli[:-1]).to_original_format(
        jnp.asarray(mont._rescale_array(in_mont), jnp.uint64))

    q_out = jnp.array(moduli[:-1], jnp.uint64)
    np.testing.assert_array_equal(ref % q_out, out % q_out)

  def test_rescale_small_dropped_tower_fast_lift_matches_generic(self):
    """The no-reduction lift is exact when q_last <= every remaining q."""
    r = c = 4
    degree = r * c
    remaining = util.find_moduli_ntt(3, 30, 2 * degree)
    dropped = util.find_moduli_ntt(1, 20, 2 * degree)
    moduli = remaining + dropped
    self.assertLessEqual(moduli[-1], min(moduli[:-1]))

    key = jax.random.key(11)
    columns = [
        jax.random.randint(
            jax.random.fold_in(key, index),
            (1, 2, degree, 1),
            0,
            modulus,
            dtype=jnp.uint32,
        )
        for index, modulus in enumerate(moduli)
    ]
    in_data = jnp.concatenate(columns, axis=-1).reshape(
        1, 2, r, c, len(moduli)
    )

    fast = _HERescaleKernel(
        batch=1,
        num_elements=2,
        moduli=moduli,
        r=r,
        c=c,
        degree_layout=(r, c),
    )
    fast.control_gen()
    self.assertEqual(fast._lift_reduction_modes, ('direct',))

    generic = _HERescaleKernel(
        batch=1,
        num_elements=2,
        moduli=moduli,
        r=r,
        c=c,
        degree_layout=(r, c),
    )
    generic.control_gen()
    generic._lift_reduction_modes = ('barrett',)

    np.testing.assert_array_equal(
        fast._rescale_array(in_data), generic._rescale_array(in_data)
    )

  # @absltest.skip("test a single experiment")
  def test_rescale_ciphertext_multibatch(self):
    in_ciphertexts_arr = jnp.array(self.in_ciphertexts, jnp.uint32)
    # Tile input to make it batch size 2
    in_ciphertexts_arr = jnp.tile(in_ciphertexts_arr[None, ...], (2, 1, 1, 1))
    input_shape = in_ciphertexts_arr.shape
    output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] - 1)
    degree_layout = (self.r, self.c)
    he_rescale = _HERescaleKernel(batch=2, num_elements=2, moduli=self.q_towers, r=self.r, c=self.c, degree_layout=degree_layout)
    he_rescale.control_gen()
    in_data = in_ciphertexts_arr.reshape(input_shape[0], input_shape[1], *degree_layout, input_shape[3])
    final_result_custom = he_rescale._rescale_array(in_data).reshape(output_shape)

    # Check both batch elements match the reference
    np.testing.assert_array_equal(final_result_custom[0], self.final_result_ref)
    np.testing.assert_array_equal(final_result_custom[1], self.final_result_ref)


if __name__ == "__main__":
  absltest.main()
