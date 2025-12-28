import functools

import jax
import jax.numpy as jnp
import util

from ciphertext import Ciphertext
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

  # @absltest.skip("test a single experiment")
  def test_rescale_ciphertext(self):
    in_ciphertexts_arr = jnp.array(self.in_ciphertexts, jnp.uint32)
    in_ciphertexts_reshaped = jnp.transpose(in_ciphertexts_arr, (0, 2, 1))[None, ...]
    input_shape = in_ciphertexts_reshaped.shape
    output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] - 1)
    shapes = {'batch': 1, 'num_elements': 2, 'degree': 16, 'num_moduli': 5, 'precision': 32}
    params = {'moduli': self.q_towers, 'r': self.r, 'c': self.c}
    degree_layout = (self.r, self.c)
    ct = Ciphertext(shapes, params)
    ct.modulus_switch_control_gen(degree_layout=degree_layout)
    ct.set_batch_ciphertext(in_ciphertexts_reshaped.reshape(input_shape[0], input_shape[1], *degree_layout,  input_shape[3]))
    ct.rescale()
    final_result_custom = ct.get_batch_ciphertext().reshape(output_shape)

    final_result_custom_reshaped = jnp.transpose(final_result_custom[0], (0, 2, 1))
    np.testing.assert_array_equal(final_result_custom_reshaped, self.final_result_ref)

  # @absltest.skip("test a single experiment")
  def test_rescale_ciphertext_multibatch(self):
    in_ciphertexts_arr = jnp.array(self.in_ciphertexts, jnp.uint32)
    # Tile input to make it batch size 2
    in_ciphertexts_arr = jnp.tile(in_ciphertexts_arr[None, ...], (2, 1, 1, 1))
    # Shape is (Batch, Elements, Moduli, Degree) -> needs to be (Batch, Elements, Degree, Moduli)
    in_ciphertexts_reshaped = jnp.transpose(in_ciphertexts_arr, (0, 1, 3, 2))
    input_shape = in_ciphertexts_reshaped.shape
    output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] - 1)
    degree_layout = (self.r, self.c)
    shapes = {'batch': 2, 'num_elements': 2, 'degree': 16, 'num_moduli': 5, 'precision': 32}
    params = {'moduli': self.q_towers, 'r': self.r, 'c': self.c}
    ct = Ciphertext(shapes, params)
    ct.modulus_switch_control_gen(degree_layout=degree_layout)
    ct.set_batch_ciphertext(in_ciphertexts_reshaped.reshape(input_shape[0], input_shape[1], *degree_layout, input_shape[3]))
    ct.rescale()
    final_result_custom = ct.get_batch_ciphertext().reshape(output_shape)
    final_result_custom_reshaped = jnp.transpose(final_result_custom, (0, 1, 3, 2))

    # Check both batch elements match the reference
    np.testing.assert_array_equal(final_result_custom_reshaped[0], self.final_result_ref)
    np.testing.assert_array_equal(final_result_custom_reshaped[1], self.final_result_ref)


if __name__ == "__main__":
  absltest.main()
