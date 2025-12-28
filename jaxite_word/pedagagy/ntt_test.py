"""A module for operations on test CKKS evaluation kernels including.

Test Naming Convention:
  test_<Algorithm>_<Optimizations>_<Reduction>_<Layout>_<ShardingDimension>

<Algorithm>: 
- C_NTT: Cyclic NTT
- C_INTT: Cyclic INTT
- 4_step_C_NTT: 4-step Cyclic NTT, layout invariant NTT
- br_C_NTT: Cyclic NTT, bit-reverse implementation (cooley-tukey)
- br_C_INTT: Cyclic INTT, bit-reverse implementation (cooley-tukey)

- N_NTT: Negacyclic NTT
- N_INTT: Negacyclic INTT
- br_N_NTT: Negacyclic NTT, bit-reverse implementation
- br_N_INTT: Negacyclic INTT, bit-reverse implementation
- 4_step_N_NTT: 4-step NegacyclicNTT
- 3_step_N_NTT: 3-step Negacyclic NTT, layout invariant NTT

<Optimizations>:
- None: single moduli, no batch
- SMB: single moduli, supporting batch
- MMB: multi moduli, supporting batch, where batch is the second dimension in Einsum
- BMM: multi moduli, supporting batch, where batch is the first dimension in Einsum
- MMSB: multi moduli, supporting batch, sqrt(degree) is integer

<Reduction>: 
- Barrett
- Montgomery

<Layout>:
- BatchFirst: (batch, other dimensions)
- BatchSecond: (moduli, batch)

<ShardingDimension>:
- none: no sharding
- Batch: (batch, other dimensions)
- Moduli: (moduli, batch), sharing over moduli dimension
"""
import functools
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import ntt
import util
import modular_reduction as modred

from absl.testing import absltest
from absl.testing import parameterized
jax.config.update("jax_enable_x64", True)


NTT = [
    (
        "0",
        134219681,
        2766781,
        1,
        4,
        4,
        [105825732, 69968408, 84865210, 120415624, 8886324, 124027644, 118001808, 47736216, 50618461, 50755406, 119028767, 96591031, 12706706, 3258267, 10327190, 17690992],
        [26196696, 45475009, 10055359, 23277424, 69041040, 71916973, 73894069, 3311254, 44646798, 49882443, 28097016, 70484730, 10811958, 11946041, 61318182, 19099272],
    ),
]

NEGACYCLIC_NTT_BATCH = [
    (
        "0",
        [134218433, 134219681],
        [19903273, 2766781],
        1,
        4,
        4,
        [[127362573, 2882815, 69807050, 3295681, 18277548, 130643120, 70030463, 24410996, 64664862, 27221327, 77034770, 49022164, 99192593, 53730438, 61845024, 63103886],
         [68871785 , 73656013, 85854702, 80158002, 33169840, 74377653, 28714220, 90026160, 117786289, 1112408, 104045686, 47797140, 128005255, 71348990, 72093735, 79740457]],
        [[61005260, 109064554, 9706895, 81972073, 108682299, 81299200, 32600679, 104631869, 63845602, 28197148, 59704201, 121314403, 29739520, 85631179, 51866821, 69010434],
         [26889623, 103184208, 55537224, 56765942, 112556220, 79557425, 73703700, 67753919, 117094305, 42422779, 132877395, 82397340, 60847927, 109449147, 21637355, 93493732]],
    ),
]


NEGACYCLIC_INTT = [
   (
        "0",
        524353,
        19017,
        1,
        4,
        4,
        [32503, 440431, 492047, 95506, 208403, 298057, 91982, 521020, 183007, 203269, 112510, 457424, 410322, 80292, 446766, 207541],
        [5391, 367593, 323644, 267506, 321577, 138684, 336724, 241442, 37017, 44767, 140016, 271760, 297656, 243198, 496228, 84777],
   ),
]

NEGACYCLIC_NTT_MULTI_MODULI = [
    (
        "0",
        [134218433, 134219681],
        [19903273, 2766781],
        2,
        4,
        4,
        [[127362573, 2882815, 69807050, 3295681, 18277548, 130643120, 70030463, 24410996, 64664862, 27221327, 77034770, 49022164, 99192593, 53730438, 61845024, 63103886],
         [68871785 , 73656013, 85854702, 80158002, 33169840, 74377653, 28714220, 90026160, 117786289, 1112408, 104045686, 47797140, 128005255, 71348990, 72093735, 79740457]],
        [[61005260, 109064554, 9706895, 81972073, 108682299, 81299200, 32600679, 104631869, 63845602, 28197148, 59704201, 121314403, 29739520, 85631179, 51866821, 69010434],
         [26889623, 103184208, 55537224, 56765942, 112556220, 79557425, 73703700, 67753919, 117094305, 42422779, 132877395, 82397340, 60847927, 109449147, 21637355, 93493732]],
    ),
]


NEGACYCLIC_NTT_MODSW = [ 
    (
        "0",
        [134219681, 134218433, 134219009, 134217409, 134218081, 524353],
        None,
        1,
        4,
        4,
        [[[[105825732,68433452,36629220,126901109],[89469849,106633716,15102657,108374459],[68789927,23451922,93538050,20585372],[30604976,37517995,65644325,102451383]],[[99069710,34257428,52786506,116210383],[26057694,68119619,99174458,86351581],[35413480,114408452,15445355,53539491],[106019559,22946495,43611042,8378523]],[[122238044,29865613,84591434,114334326],[101982316,124788686,45345308,6296238],[75612913,22893021,14048154,126231962],[12915383,118076231,77442049,97997649]],[[4748456,39265883,93217221,126692126],[35748544,60653825,131913969,91968607],[20508620,14450024,3084976,17607150],[14860040,35789101,111095511,107313206]],[[69643005,68902874,15884608,78510831],[97192571,109678384,41445670,107357568],[102520258,62447479,106184373,19694577],[116256926,133808729,103339727,9427348]]],[[[90114427,133438872,104980529,87731625],[115983961,83377921,14405210,30297260],[84308722,68397350,21408743,115163666],[98766748,102167012,30301135,57947749]],[[59665748,19545942,65476858,55715441],[10786394,119906528,44337564,23037678],[95213829,101671488,16935749,4246783],[42646409,33794171,126987436,132909871]],[[104994577,121329083,114585312,132556666],[94023484,105867693,8410688,72163874],[14260026,40582007,62942793,15756748],[114875754,118509058,117046370,83800550]],[[69073640,123640580,7039810,125736888],[98865413,127204532,11132849,93509993],[113859946,81067476,112391529,16480451],[128021450,133983253,89158296,110706136]],[[107919832,45738998,64895884,4945855],[26661208,47740937,112794757,57493947],[125904522,23168711,18848235,99246137],[134013317,13020155,108625986,52632689]]]],
        [[[26196696,45475009,10055359,23277424,69041040,71916973,73894069,3311254,44646798,49882443,28097016,70484730,10811958,11946041,61318182,19099272],[42260745,107584265,34229636,76856528,73149957,51350747,8784787,10556690,95537178,91617682,826227,115667037,115209336,55110970,128915458,40584385],[100907255,27910884,93242445,89400977,85704857,17164816,36914303,57810047,61470328,122692712,40520757,99122771,73057820,20503928,52387777,37463964],[103612805,55654824,108032551,36162460,68496163,45089415,87511207,100149410,6650775,65659157,3605754,2754789,37323665,26373647,93805553,40397575],[34332199,62639348,100048295,3250388,2384201,80330821,111229519,43564766,128380065,111694763,37293776,109813941,113594726,33380475,16213048,126137749]],[[30813801,15967701,111614574,56102546,129485981,130007979,21553968,94520556,49355940,59219781,116354271,21338113,67103858,111639943,7271499,16821278],[13348858,105440518,30864893,12503086,122994520,68208873,115058754,99524949,83798510,58052357,42567274,18655458,122398629,65506849,104756105,25190768],[113733602,74792334,15790221,34547135,59719437,25953433,46123870,46500756,82973579,79422829,9516272,43056937,67592214,20194463,130639692,24042404],[99698152,85648442,111028268,732844,73340328,24833191,28933986,99731614,121481033,61572789,79668760,47344371,79973641,119608453,105444927,100354850],[104124603,87240150,75198866,103469168,92221513,49160688,83214976,68002139,22173025,30332461,88055173,103328154,11067041,43383033,111662992,117211006]]]
    )
]

NEGACYCLIC_NTT_MODSW_MULTI_MODULI = [
    (
        "0",
        [1073741441, 1073741857, 1073740609, 1073739937],
        None,
        4,
        4,
        4,
        [[289958362, 310181825, 829287252, 1046437097, 910799843, 168385317, 313440089, 557255107, 441782342, 646823099, 514041123, 1025036444, 313311490, 777339798, 7632390, 633940400], [24283512, 233105143, 986888805, 844753121, 985234074, 500878713, 140818894, 481276989, 500415677, 653383417, 993101323, 641560507, 708665593, 1030978680, 342838405, 413872803], [822341406, 780984738, 497574822, 569531923, 908067194, 287713760, 11520375, 35247909, 936499896, 1008439059, 100595613, 563443202, 632861344, 502313356, 234913340, 757143397], [1005562785, 1031881758, 676892964, 37386717, 593484003, 278256231, 364745873, 761351971, 999396706, 522472498, 72084093, 599346293, 944129718, 21156711, 725474179, 981807678]],
        [[195720450, 883169532, 602336686, 773474275, 32694214, 986756761, 621910772, 801183008, 651729468, 816984635, 831377463, 25391682, 255405839, 1003234060, 680952134, 845720018], [892120800, 767398447, 956178767, 889988125, 782209800, 53362540, 1051883026, 425796310, 671016132, 798767447, 579320663, 790286845, 339987201, 461988353, 847182206, 818468100], [59266462, 409659266, 713297255, 205709678, 466084010, 762258225, 140350468, 574699548, 278474024, 381347193, 491680048, 793890400, 626156930, 429201978, 106698415, 276244942], [1025510682, 632013831, 74370527, 627258501, 852495649, 80635826, 793826467, 920845393, 742411105, 144562352, 832028297, 150381326, 620934627, 226214689, 578502476, 270833253]]
    )
]

NEGACYCLIC_NTT_64BITs = [
      (
        "reduce_last_tower_NTT",
        576460752303424801,
        None,
        1,
        4,
        4,
        [36028795733201082, 125634782498245228, 252435773917279369, 90271336770773325, 150954262792331003, 473204844454871385, 106226977119228942, 353488136353479560, 0, 222972616335821209, 470233775284859155, 103255907689169864, 425506489544648230, 486189415633314772, 324024978579083416, 450825969805179573],
        [533600933508055103, 347619659595540825, 81150633748659193, 554725690872353490, 332980549301504159, 96080479296532221, 285029104992421243, 362886323764241626, 340095105608222658, 516433470306371252, 29355219934401216, 78811792573421729, 244622621704328213, 96012820295182461, 357397196384003853, 354884395969951677]
    )
]

class CKKSEvalNTTTest(parameterized.TestCase):
  """A base class for running bootstrap tests.

  Example Test Case:
    If use GF(17) and N = 8 (so q=17 and N=8).
    In GF(17), the multiplicative group has order 16.
    Suppose the forward transform used a primitive 8th root of unity.
    For example, we can use omega = 2, since 2^8 mod 17 == 1 and its order is 8.
  """

  def __init__(self, *args, **kwargs):
    super(CKKSEvalNTTTest, self).__init__(*args, **kwargs)
    self.random_key = jax.random.key(0)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_C_NTT_None_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    if psi is not None:
      omega = pow(psi, 2, q)
    else:
      omega = util.root_of_unity(n, q)

    ntt_result = ntt.ntt_original_form(coef_in, q, omega)
    x_recovered = ntt.intt_original_form(ntt_result, q, omega)
    self.assertEqual(coef_in, x_recovered)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_br_C_NTT_None_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    if psi is not None:
      omega = (psi ** 2) % q
      psi = psi
    else:
      omega = util.root_of_unity(n, q)
      psi =  util.root_of_unity(2**n, q)

    ntt_result = ntt.ntt_bit_reverse(coef_in, q, omega)
    x_recovered = ntt.intt_bit_reverse(ntt_result, q, omega)
    self.assertEqual(coef_in, x_recovered)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_4_step_C_NTT_None_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    if psi is not None:
      omega = pow(psi, 2, q)
    else:
      omega = util.root_of_unity(n, q)

    ntt_result = ntt.ntt_four_step(coef_in, q, omega, row_count, col_count)
    x_recovered = ntt.intt_four_step(ntt_result, q, omega, row_count, col_count)
    self.assertEqual(coef_in, x_recovered)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_br_N_NTT_None_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    if psi is not None:
      omega = (psi ** 2) % q
      psi = psi
    else:
      omega = util.root_of_unity(n, q)
      psi =  util.root_of_unity(2**n, q)

    ntt_result = ntt.ntt_negacyclic_bit_reverse(coef_in, q, psi)

    if eval_in is not None:
      ntt_result_test = [i for i in ntt_result]
      bits = int(math.log2(len(ntt_result_test)))
      for i in range(len(ntt_result_test)):
        j = util.bit_reverse(i, bits)
        if i < j:
          ntt_result_test[i], ntt_result_test[j] = ntt_result_test[j], ntt_result_test[i]
      self.assertEqual(ntt_result_test, eval_in)

    x_recovered = ntt.intt_negacyclic_bit_reverse(ntt_result, q, psi)
    self.assertEqual(coef_in, x_recovered)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_br_N_NTT_None_Barrett_BatchFirst_none_Jax(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    if psi is not None:
      omega = (psi ** 2) % q
      psi = psi
    else:
      omega = util.root_of_unity(n, q)
      psi =  util.root_of_unity(2**n, q)

    stage_w_pows, br_idx = ntt.ntt_bit_reverse_control_gen_jax(n, q, omega)
    twist = ntt.ntt_bit_reverse_negacyclic_control_generation(n, q, psi)
    s_w, w_barr, m_barr = modred.barrett_control_generation_s_w(q)
    coef_in = jnp.array(coef_in, jnp.uint32)
    total_iter = int(math.log2(n))
    omega = pow(psi, 2, q)
    ntt_result = ntt.ntt_negacyclic_bit_reverse_jax(coef_in, q, psi, omega, twist, stage_w_pows, br_idx, s_w, w_barr, m_barr, total_iter)

    if eval_in is not None:
      ntt_result_test = [i for i in ntt_result]
      bits = int(math.log2(len(ntt_result_test)))
      for i in range(len(ntt_result_test)):
        j = util.bit_reverse(i, bits)
        if i < j:
          ntt_result_test[i], ntt_result_test[j] = ntt_result_test[j], ntt_result_test[i]
      self.assertEqual(ntt_result_test, eval_in)

    x_recovered = ntt.intt_negacyclic_bit_reverse(ntt_result.tolist(), q, psi)
    self.assertEqual(coef_in.tolist(), x_recovered)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_br_N_NTT_SMB_Barrett_BatchFirst_none_Jax(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    if psi is not None:
      omega = (psi ** 2) % q
      psi = psi
    else:
      omega = util.root_of_unity(n, q)
      psi =  util.root_of_unity(int(2*n), q)

    stage_w_pows, br_idx = ntt.ntt_bit_reverse_control_gen_jax(n, q, omega)
    twist = ntt.ntt_bit_reverse_negacyclic_control_generation(n, q, psi)
    s_w, w_barr, m_barr = modred.barrett_control_generation_s_w(q)
    coef_in=jnp.array([coef_in, coef_in])
    omega = pow(psi, 2, q)
    ntt_result = ntt.ntt_negacyclic_bit_reverse_jax(coef_in, q, psi, omega, twist, stage_w_pows, br_idx, s_w, w_barr, m_barr)

    if eval_in is not None:
      ntt_result_test = [i for i in ntt_result.tolist()[0]]
      bits = int(math.log2(len(ntt_result_test)))
      for i in range(len(ntt_result_test)):
        j = util.bit_reverse(i, bits)
        if i < j:
          ntt_result_test[i], ntt_result_test[j] = ntt_result_test[j], ntt_result_test[i]
      self.assertEqual(ntt_result_test, eval_in)

    x_recovered = ntt.intt_negacyclic_bit_reverse(ntt_result.tolist()[0], q, psi)
    self.assertEqual(coef_in.tolist()[0], x_recovered)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_4_step_N_NTT_None_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    if psi is not None:
      omega = (psi ** 2) % q
      psi = psi
      # psi = pow(psi, 2, q)
    else:
      omega = util.root_of_unity(n, q)
      psi =  util.root_of_unity(2**n, q)

    ntt_result = ntt.ntt_negacyclic_four_step(coef_in, q, psi, row_count, col_count)
    # print("Forward negacyclic NTT of x:", ntt_result)

    x_recovered = ntt.intt_negacyclic_four_step(ntt_result, q, psi, row_count, col_count)
    # print("Recovered x from inverse negacyclic NTT:", x_recovered)

    self.assertEqual(coef_in, x_recovered)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_Reduction_Barrett(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    q = 113 # Must be less than 31 bit
    coef_in = [[[69, 95, 147, 139], [7617, 6977, 8472, 7687]]]
    result_ref = [[[69, 95, 34, 26], [46, 84, 110, 3]]]
    s = 2 * math.ceil(math.log2(q))
    m = math.floor(2**s / q)
    mod_reduced_result = modred.barrett_reduction(
        jnp.array(coef_in, dtype=jnp.uint64), q, s, m
    )
    self.assertEqual(result_ref, mod_reduced_result.tolist())

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_3_step_N_NTT_None_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n

    if psi is not None:
      omega = (psi ** 2) % q
      psi = psi
    else:
      omega = util.root_of_unity(n, q)
      psi =  util.root_of_unity(2**n, q)

    omega_col = pow(omega, c, q)
    omega_row = pow(omega, r, q)
    tf_mat_step1 = jnp.array(
        ntt.gen_twiddle_matrix(r, r, q, omega_col), dtype=int
    )
    coef_step2 = jnp.array(ntt.gen_twiddle_matrix(r, c, q, omega), dtype=int)
    tf_mat_step3 = jnp.array(
        ntt.gen_twiddle_matrix(c, c, q, omega_row), dtype=int
    )

    inv_r = pow(r, -1, q)
    inv_omega_col = pow(
        omega_col, -1, q
    )  # inverse primitive R-th root for columns
    inv_omega_row = pow(
        omega_row, -1, q
    )  # inverse primitive C-th root for rows
    inv_tf_mat_step3 = (jnp.array(
        ntt.gen_twiddle_matrix(r, r, q, inv_omega_col), dtype=int
    ) * inv_r) % q
    inv_coef_step2 = jnp.array(
        ntt.gen_twiddle_matrix_inv(r, c, q, omega), dtype=int
    )
    inv_tf_mat_step1 = jnp.array(
        ntt.gen_twiddle_matrix(c, c, q, inv_omega_row), dtype=int
    )

    np.testing.assert_array_equal(tf_mat_step1.T, tf_mat_step1)
    np.testing.assert_array_equal(tf_mat_step3.T, tf_mat_step3)
    np.testing.assert_array_equal(inv_tf_mat_step1.T, inv_tf_mat_step1)
    np.testing.assert_array_equal(inv_tf_mat_step3.T, inv_tf_mat_step3)

    ntt_result = ntt.ntt_negacyclic_three_step(
        coef_in, q, psi, r, c, tf_mat_step1, coef_step2, tf_mat_step3
    )
    # print("Forward negacyclic NTT of x:", ntt_result)

    x_recovered = ntt.intt_negacyclic_three_step(
        ntt_result,
        q,
        psi,
        r,
        c,
        inv_tf_mat_step1,
        inv_coef_step2,
        inv_tf_mat_step3,
    )
    # print("Recovered x from inverse negacyclic NTT:", x_recovered)
    if x_recovered is jnp.array:
      x_recovered = x_recovered.tolist()
    self.assertEqual(coef_in, x_recovered)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_3_step_N_NTT_SMB_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    s_w, w, m = modred.barrett_control_generation_s_w(q)
    # tf_mat_bat_step1, coef_step2, tf_mat_bat_step3, tf_mat_step1, tf_mat_step3 = ntt.ntt_control_generation(q, r, c)
    tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch = ntt.ntt_negacyclic_three_step_control_generation(q, r, c)
    bat_tf_step1_batch, _, bat_tf_step3_batch = ntt.ntt_three_step_bat_control_generation(q, r, c)

    assert bat_tf_step1_batch.shape == (r, r, 4, 4)
    assert bat_tf_step1_batch.dtype == jnp.uint8
    assert tf_step2_batch.shape == (r, c)
    assert bat_tf_step3_batch.shape == (c, c, 4, 4)
    assert bat_tf_step3_batch.dtype == jnp.uint8

    # if c == r:
    #   np.testing.assert_array_equal(bat_tf_step1_batch, bat_tf_step3_batch)
    np.testing.assert_array_equal(tf_mat_step1_batch.T, tf_mat_step1_batch)
    np.testing.assert_array_equal(tf_mat_step3_batch.T, tf_mat_step3_batch)

    ntt_result = ntt.ntt_negacyclic_three_step(
        coef_in, q, psi, r, c, tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch
    )

    coef_in_twisted = jnp.array(
        [(coef_in[i] * pow(psi, i, q)) % q for i in range(r*c)], jnp.uint32
    )
    coef_in_twisted = coef_in_twisted.reshape(batch, r, c)
    result = ntt.ntt_three_step_bat_barrett_batch(coef_in_twisted, bat_tf_step1_batch, tf_step2_batch, bat_tf_step3_batch, q, s_w, w, m)
    result = np.array(result.T).flatten().tolist()

    self.assertEqual(ntt_result, result)
    result_bit_reverse = util.bit_reverse_array(result)
    self.assertEqual(result_bit_reverse, eval_in)

    # performance measurement
    tasks = [
        (ntt.ntt_three_step_bat_barrett_batch, (coef_in_twisted, bat_tf_step1_batch, tf_step2_batch, bat_tf_step3_batch, q, s_w, w, m)),
    ]
    profile_name = "test_three_step_bat_batch"
    # util.profile_jax_functions(tasks, profile_name)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_INTT)
  def test_3_step_N_INTT_SMB_Barrett_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    s_w, w, m = modred.barrett_control_generation_s_w(q)
    inv_tf_mat_step1_batch, inv_tf_step2_batch, inv_tf_mat_step3_batch = ntt.intt_negacyclic_three_step_control_generation(q, r, c)
    bat_inv_tf_step1_batch, scaled_inv_tf_step2_batch, bat_inv_tf_step3_batch, _ = ntt.intt_three_step_bat_control_generation(q, r, c)

    assert bat_inv_tf_step1_batch.shape == (c, c, 4, 4)
    assert scaled_inv_tf_step2_batch.shape == (r, c)
    assert bat_inv_tf_step3_batch.shape == (r, r, 4, 4)
    coef_in = util.bit_reverse_array(coef_in)

    # INTT
    x_intt_ref = ntt.intt_negacyclic_three_step(
        coef_in,
        q,
        psi,
        r,
        c,
        inv_tf_mat_step1_batch,
        inv_tf_step2_batch,
        inv_tf_mat_step3_batch,
    )
    self.assertEqual(x_intt_ref, eval_in)

    # INTT
    x_intt_current_deploy = ntt.intt_negacyclic_bit_reverse(
        coef_in,
        q,
        psi,
    )
    self.assertEqual(x_intt_current_deploy, eval_in)

    coef_in = jnp.array(coef_in, jnp.uint32).reshape((batch, r, c), order='F')
    # The order='F' tells JAX to fill the new (batch, r, c) array in column‐major (Fortran) order, which has the same effect as doing
    # .reshape(r, c).T.reshape(batch, r, c)
    x_intt = ntt.intt_three_step_bat_barrett_batch(
        coef_in,
        bat_inv_tf_step1_batch,
        scaled_inv_tf_step2_batch,
        bat_inv_tf_step3_batch,
        q,
        s_w,
        w,
        m,
    )
    psi_inv = pow(psi, -1, q)
    x_intt = x_intt.flatten().tolist()
    x_intt_final = [(x_intt[i] * pow(psi_inv, i, q)) % q for i in range(len(x_intt))]
    self.assertEqual(x_intt_final, eval_in)

    # performance measurement
    tasks = [
        (ntt.intt_three_step_bat_barrett_batch, (coef_in, bat_inv_tf_step1_batch, scaled_inv_tf_step2_batch, bat_inv_tf_step3_batch, q, s_w, w, m)),
    ]
    profile_name = "coef_intt_three_step_bat_barrett_batch"
    # util.profile_jax_functions(tasks, profile_name)



  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_3_step_N_NTT_SMB_Barrett_BatchFirst_none_RoundTrip(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    """
    Test the coef_in -> NTT -> x_ref -> INTT -> x_recovered;
    Check coef_in == x_recovered
    """
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    s_w, w, m = modred.barrett_control_generation_s_w(q)
    tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch = ntt.ntt_negacyclic_three_step_control_generation(q, r, c)
    inv_tf_mat_step1_batch, inv_tf_step2_batch, inv_tf_mat_step3_batch = ntt.intt_negacyclic_three_step_control_generation(q, r, c)

    bat_tf_step1_batch, _, bat_tf_step3_batch = ntt.ntt_three_step_bat_control_generation(q, r, c)
    bat_inv_tf_step1_batch, scaled_inv_tf_step2_batch, bat_inv_tf_step3_batch, _ = ntt.intt_three_step_bat_control_generation(q, r, c)

    assert bat_tf_step1_batch.shape == (r, r, 4, 4)
    assert tf_step2_batch.shape == (r, c)
    assert bat_tf_step3_batch.shape == (c, c, 4, 4)
    assert bat_inv_tf_step1_batch.shape == (c, c, 4, 4)
    assert inv_tf_step2_batch.shape == (r, c)
    assert bat_inv_tf_step3_batch.shape == (r, r, 4, 4)

    ntt_result = ntt.ntt_negacyclic_three_step(
        coef_in, q, psi, r, c, tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch
    )

    coef_in_twisted = jnp.array(
        [(coef_in[i] * pow(psi, i, q)) % q for i in range(r*c)], jnp.uint32
    )
    coef_in_twisted = coef_in_twisted.reshape(batch, r, c)
    result_jax = ntt.ntt_three_step_bat_barrett_batch(
      coef_in_twisted,
      bat_tf_step1_batch,
      tf_step2_batch,
      bat_tf_step3_batch,
      q,
      s_w,
      w,
      m
    )
    result = np.array(result_jax.T).flatten().tolist()
    self.assertEqual(ntt_result, result)

    ntt_pure_alg = ntt.ntt_negacyclic_bit_reverse(coef_in, q, psi)
    self.assertEqual(ntt_result, ntt_pure_alg)

    # INTT
    x_ref = ntt.intt_negacyclic_three_step(
        ntt_result,
        q,
        psi,
        r,
        c,
        inv_tf_mat_step1_batch,
        inv_tf_step2_batch,
        inv_tf_mat_step3_batch
    )
    self.assertEqual(x_ref, coef_in)

    x_untwisted = ntt.intt_three_step_bat_barrett_batch(
        result_jax,
        bat_inv_tf_step1_batch,
        scaled_inv_tf_step2_batch,
        bat_inv_tf_step3_batch,
        q,
        s_w,
        w,
        m,
    )
    psi_inv = pow(psi, -1, q)
    x_untwisted =  x_untwisted.flatten().tolist()
    x_recovered = [(x_untwisted[i] * pow(psi_inv, i, q)) % q for i in range(len(x_untwisted))]
    self.assertEqual(x_recovered, coef_in)

    # performance measurement
    tasks = [
        (ntt.ntt_three_step_bat_barrett_batch, (coef_in_twisted, bat_tf_step1_batch, tf_step2_batch, bat_tf_step3_batch, q, s_w, w, m)),
        (ntt.intt_three_step_bat_barrett_batch, (result_jax, bat_inv_tf_step1_batch,  scaled_inv_tf_step2_batch,  bat_inv_tf_step3_batch, q, s_w, w, m)),
    ]
    profile_name = "test_intt_three_step_bat_barrett_batch"
    # util.profile_jax_functions(tasks, profile_name)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MULTI_MODULI)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_none(
      self,
      q_list,
      psi_list,
      num_moduli,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return

    tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch = ntt.ntt_negacyclic_three_step_control_generation(q_list, r, c)
    inv_tf_mat_step1_batch, inv_tf_step2_batch, inv_tf_mat_step3_batch = ntt.intt_negacyclic_three_step_control_generation(q_list, r, c)

    bat_tf_step1_multi_moduli, _, bat_tf_step3_multi_moduli = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_inv_tf_mat_step1_multi_moduli, scaled_inv_tf_step2_multi_moduli, bat_inv_tf_mat_step3_multi_moduli, _ = ntt.intt_three_step_bat_control_generation(q_list, r, c)

    bat_tf_step1_multi_moduli, bat_tf_step3_multi_moduli, bat_inv_tf_mat_step1_multi_moduli, scaled_inv_tf_step2_multi_moduli, bat_inv_tf_mat_step3_multi_moduli = jnp.array(bat_tf_step1_multi_moduli, jnp.uint8), jnp.array(bat_tf_step3_multi_moduli, jnp.uint8), jnp.array(bat_inv_tf_mat_step1_multi_moduli, jnp.uint8), jnp.array(scaled_inv_tf_step2_multi_moduli, jnp.uint32), jnp.array(bat_inv_tf_mat_step3_multi_moduli, jnp.uint8)
    tf_step2_multi_moduli = jnp.array(tf_step2_batch, jnp.uint32)
    s_w_list, w_list, m_list = modred.barrett_control_generation_s_w(q_list)
    psi_inv = [pow(psi, -1, q) for (psi, q) in zip (psi_list, q_list)]

    psi_inv_i_list = jnp.array([
      [pow(psi_inv[idx], i, q_list[idx]) for i in range(r*c)]
      for idx in range(num_moduli)
    ], jnp.uint64)

    assert bat_tf_step1_multi_moduli.shape == (num_moduli, r, r, 4, 4)
    assert tf_step2_multi_moduli.shape == (num_moduli, r, c)
    assert bat_tf_step3_multi_moduli.shape == (num_moduli, c, c, 4, 4)
    assert bat_inv_tf_mat_step1_multi_moduli.shape == (num_moduli, c, c, 4, 4)
    assert scaled_inv_tf_step2_multi_moduli.shape == (num_moduli, r, c)
    assert bat_inv_tf_mat_step3_multi_moduli.shape == (num_moduli, r, r, 4, 4)
    ntt_result = [
      ntt.ntt_negacyclic_three_step(
        coef_in_list[idx], q_list[idx], psi_list[idx], r, c, tf_mat_step1_batch[idx], tf_step2_batch[idx], tf_mat_step3_batch[idx]
      )
      for idx in range(num_moduli)
    ]
    coef_in_twisted = [
        [(coef_in_list[idx][i] * pow(psi_list[idx], i, q_list[idx])) % q_list[idx] for i in range(r*c)]
        for idx in range(num_moduli)
    ]
    coef_in_twisted = jnp.array(coef_in_twisted, jnp.uint32)
    coef_in_twisted = coef_in_twisted.reshape(num_moduli, r, c)
    result_jax = ntt.ntt_three_step_bat_barrett_multi_moduli(
      coef_in_twisted,
      bat_tf_step1_multi_moduli,
      tf_step2_multi_moduli,
      bat_tf_step3_multi_moduli,
      util.to_tuple(q_list),
      util.to_tuple(s_w_list),
      util.to_tuple(w_list),
      util.to_tuple(m_list)
    )
    result = result_jax.T.flatten().tolist()
    if result_jax.ndim == 3:
      result = result_jax.transpose(0,2,1).reshape(num_moduli, -1).tolist()
    self.assertEqual(ntt_result, result)

    # INTT
    x_ref = [
      ntt.intt_negacyclic_three_step(
          ntt_result[idx],
          q_list[idx],
          psi_list[idx],
          r,
          c,
          inv_tf_mat_step1_batch[idx],
          inv_tf_step2_batch[idx],
          inv_tf_mat_step3_batch[idx],
      )
      for idx in range(num_moduli)
    ]
    x_untwisted = ntt.intt_three_step_bat_barrett_multi_moduli(
        result_jax,
        bat_inv_tf_mat_step1_multi_moduli,
        scaled_inv_tf_step2_multi_moduli,
        bat_inv_tf_mat_step3_multi_moduli,
        util.to_tuple(q_list),
        util.to_tuple(s_w_list),
        util.to_tuple(w_list),
        util.to_tuple(m_list)
    )
    x_untwisted =  x_untwisted.reshape(num_moduli, -1)
    x_recovered = jnp.multiply(x_untwisted.astype(jnp.uint64), psi_inv_i_list) % jnp.array(q_list, jnp.uint32)[:, None]

    self.assertEqual(x_recovered.tolist(), coef_in_list)

    # performance measurement
    tasks = [
        (ntt.ntt_three_step_bat_barrett_multi_moduli, (coef_in_twisted, bat_tf_step1_multi_moduli, tf_step2_multi_moduli, bat_tf_step3_multi_moduli, q_list, s_w_list, w_list, m_list)),
        (ntt.intt_three_step_bat_barrett_multi_moduli, (result_jax, bat_inv_tf_mat_step1_multi_moduli, scaled_inv_tf_step2_multi_moduli, bat_inv_tf_mat_step3_multi_moduli, q_list, s_w_list, w_list, m_list)),
    ]
    profile_name = "test_intt_three_step_bat_barrett_multi_moduli"
    # util.profile_jax_functions(tasks, profile_name)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_BATCH)
  def test_3_step_N_NTT_SMB_Barrett_BatchFirst_none_BatchRoundTrip(
      self,
      q_list,
      psi_list,
      batch,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return

    tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch = ntt.ntt_negacyclic_three_step_control_generation(q_list, r, c)
    inv_tf_mat_step1_batch, inv_tf_step2_batch, inv_tf_mat_step3_batch = ntt.intt_negacyclic_three_step_control_generation(q_list, r, c)

    bat_tf_step1_batch, _, bat_tf_step3_batch = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_inv_tf_mat_step1_batch, scaled_inv_tf_step2_batch, bat_inv_tf_mat_step3_batch, power_of_inv_psi_arr = ntt.intt_three_step_bat_control_generation(q_list, r, c)

    s_w_tuple, w_tuple, m_tuple = modred.barrett_reduction_overall_control_generation(q_list)

    for idx in range(len(q_list)):
      assert bat_tf_step1_batch[idx].shape == (r, r, 4, 4)
      assert tf_step2_batch[idx].shape == (r, c)
      assert bat_tf_step3_batch[idx].shape == (c, c, 4, 4)
      assert bat_inv_tf_mat_step1_batch[idx].shape == (c, c, 4, 4)
      assert inv_tf_step2_batch[idx].shape == (r, c)
      assert bat_inv_tf_mat_step3_batch[idx].shape == (r, r, 4, 4)

      ntt_result = ntt.ntt_negacyclic_three_step(
          coef_in_list[idx], q_list[idx], psi_list[idx], r, c, tf_mat_step1_batch[idx], tf_step2_batch[idx], tf_mat_step3_batch[idx]
      )
      coef_in_twisted = jnp.array(
          [(coef_in_list[idx][i] * pow(psi_list[idx], i, q_list[idx])) % q_list[idx] for i in range(r*c)], jnp.uint32
      )
      coef_in_twisted = coef_in_twisted.reshape(batch, r, c)
      result_jax = ntt.ntt_three_step_bat_barrett_batch(
        coef_in_twisted,
        bat_tf_step1_batch[idx],
        tf_step2_batch[idx],
        bat_tf_step3_batch[idx],
        q_list[idx],
        s_w_tuple[idx],
        w_tuple[idx],
        m_tuple[idx]
      )
      result = np.array(result_jax.T).flatten().tolist()
      self.assertEqual(ntt_result, result)

      # INTT
      x_ref = ntt.intt_negacyclic_three_step(
          ntt_result,
          q_list[idx],
          psi_list[idx],
          r,
          c,
          inv_tf_mat_step1_batch[idx],
          inv_tf_step2_batch[idx],
          inv_tf_mat_step3_batch[idx],
      )
      self.assertEqual(x_ref, coef_in_list[idx])

      x_untwisted = ntt.intt_three_step_bat_barrett_batch(
          result_jax,
          bat_inv_tf_mat_step1_batch[idx],
          scaled_inv_tf_step2_batch[idx],
          bat_inv_tf_mat_step3_batch[idx],
          q_list[idx],
          s_w_tuple[idx],
          w_tuple[idx],
          m_tuple[idx],
      )
      x_untwisted = x_untwisted.flatten()
      x_recovered = jnp.multiply(x_untwisted.astype(jnp.uint64), jnp.array(power_of_inv_psi_arr[idx], jnp.uint64)) % q_list[idx]
      self.assertEqual(x_recovered.tolist(), coef_in_list[idx])

      # performance measurement
      tasks = [
          (ntt.ntt_three_step_bat_barrett_batch, (coef_in_twisted, bat_tf_step1_batch[idx], tf_step2_batch[idx], bat_tf_step3_batch[idx], q_list[idx], s_w_tuple[idx], w_tuple[idx], m_tuple[idx])),
          (ntt.intt_three_step_bat_barrett_batch, (result_jax, bat_inv_tf_mat_step1_batch[idx], scaled_inv_tf_step2_batch[idx], bat_inv_tf_mat_step3_batch[idx], q_list[idx], s_w_tuple[idx], w_tuple[idx], m_tuple[idx])),
      ]
      profile_name = "test_intt_three_step_bat_barrett_batch"
      # util.profile_jax_functions(tasks, profile_name)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW_MULTI_MODULI)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_Moduli(
      self,
      q_list,
      psi_list,
      num_moduli,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    batch, rows, cols = 1, r, c
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(batch_partition),
      )
    except RuntimeError as exc:
      self.skipTest(str(exc))
    r,c = rows, cols

    test_in = jnp.array(coef_in_list, dtype=jnp.uint32).reshape(num_moduli, batch, r, c)
    bat_tf_step1_multi_moduli, tf_step2_multi_moduli, bat_tf_step3_multi_moduli = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_inv_tf_mat_step1_multi_moduli, scaled_inv_tf_step2_multi_moduli, bat_inv_tf_mat_step3_multi_moduli, _ = ntt.intt_three_step_bat_control_generation(q_list, r, c)

    bat_tf_step1_multi_moduli, bat_tf_step3_multi_moduli, bat_inv_tf_mat_step1_multi_moduli, scaled_inv_tf_step2_multi_moduli, bat_inv_tf_mat_step3_multi_moduli = jnp.array(bat_tf_step1_multi_moduli, jnp.uint8), jnp.array(bat_tf_step3_multi_moduli, jnp.uint8), jnp.array(bat_inv_tf_mat_step1_multi_moduli, jnp.uint8), jnp.array(scaled_inv_tf_step2_multi_moduli, jnp.uint32), jnp.array(bat_inv_tf_mat_step3_multi_moduli, jnp.uint8)
    tf_step2_multi_moduli = jnp.array(tf_step2_multi_moduli, jnp.uint32)
    s_w_list, w_list, m_list = modred.barrett_control_generation_s_w(q_list)

    assert bat_tf_step1_multi_moduli.shape == (num_moduli, r, r, 4, 4)
    assert tf_step2_multi_moduli.shape == (num_moduli, r, c)
    assert bat_tf_step3_multi_moduli.shape == (num_moduli, c, c, 4, 4)
    assert bat_inv_tf_mat_step1_multi_moduli.shape == (num_moduli, c, c, 4, 4)
    assert scaled_inv_tf_step2_multi_moduli.shape == (num_moduli, r, c)
    assert bat_inv_tf_mat_step3_multi_moduli.shape == (num_moduli, r, r, 4, 4)

    origin_moduli_arr = jnp.array(q_list, dtype=jnp.uint32)
    kernel_name = "jit_ntt_three_step_bat_barrett_batch"
    s_w_arr = jnp.array(s_w_list, dtype=jnp.uint16)
    w_arr = jnp.array(w_list, dtype=jnp.uint16)
    m_arr = jnp.array(m_list, dtype=jnp.uint64)

    jit_ntt_three_step_bat_barrett_batch = jax.jit(jax.named_call(ntt.ntt_three_step_bat_barrett_batch, name=kernel_name))
    project_func_vmap = jax.vmap(jit_ntt_three_step_bat_barrett_batch, in_axes=(0,0,0,0,0,0,0,0), out_axes=0)
    sharded_project_func = jax.jit(
      project_func_vmap,
      in_shardings=(
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
      ),
      out_shardings=batch_sharding,
    )
    coef_in_twisted = jax.device_put(test_in, batch_sharding)
    bat_tf_step1_multi_moduli = jax.device_put(bat_tf_step1_multi_moduli, batch_sharding)
    tf_step2_multi_moduli = jax.device_put(tf_step2_multi_moduli, batch_sharding)
    bat_tf_step3_multi_moduli = jax.device_put(bat_tf_step3_multi_moduli, batch_sharding)
    origin_moduli_arr = jax.device_put(origin_moduli_arr, batch_sharding)
    s_w_arr = jax.device_put(s_w_arr, batch_sharding)
    w_arr = jax.device_put(w_arr, batch_sharding)
    m_arr = jax.device_put(m_arr, batch_sharding)

    result_jax = sharded_project_func(
      coef_in_twisted,
      bat_tf_step1_multi_moduli,
      tf_step2_multi_moduli,
      bat_tf_step3_multi_moduli,
      origin_moduli_arr,
      s_w_arr,
      w_arr,
      m_arr,
    )
    np.testing.assert_array_equal(result_jax.reshape(num_moduli, -1), eval_in)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW_MULTI_MODULI)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_Batch_Vmap(
      self,
      q_list,
      psi_list,
      num_moduli,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    batch, rows, cols = 4, r, c
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(None, batch_partition, None, None),
      )
      replicated_sharding = jax.sharding.NamedSharding(mesh, partition_spec())
    except RuntimeError as exc:
      self.skipTest(str(exc))
    
    test_in = jnp.array(coef_in_list, dtype=jnp.uint32).reshape(num_moduli, 1, r, c)
    test_in = jnp.concatenate([test_in for _ in range(batch)], axis=1).reshape(num_moduli, batch, r, c)
    bat_tf_step1_list, tf_step2_list, bat_tf_step3_list = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_tf_step1_arr, tf_step2_arr, bat_tf_step3_arr = jnp.array(bat_tf_step1_list, jnp.uint8), jnp.array(tf_step2_list, jnp.uint32), jnp.array(bat_tf_step3_list, jnp.uint8)

    origin_moduli_arr = jnp.array(q_list, dtype=jnp.uint32)
    kernel_name = "jit_ntt_three_step_bat_barrett_batch"
    s_w_tuple, w_tuple, m_tuple = modred.barrett_control_generation_s_w(q_list)
    s_w_arr = jnp.array(s_w_tuple, dtype=jnp.uint16)
    w_arr = jnp.array(w_tuple, dtype=jnp.uint16)
    m_arr = jnp.array(m_tuple, dtype=jnp.uint64)
    jit_ntt_three_step_bat_barrett_batch = jax.jit(jax.named_call(ntt.ntt_three_step_bat_barrett_batch, name=kernel_name))
    project_func_vmap = jax.vmap(jit_ntt_three_step_bat_barrett_batch, in_axes=(0,0,0,0,0,0,0,0), out_axes=0)
    sharded_project_func = jax.jit(
      project_func_vmap,
      in_shardings=(
        batch_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
      ),
      out_shardings=batch_sharding,
    )
    test_in = jax.device_put(test_in, batch_sharding)
    bat_tf_step1_arr = jax.device_put(bat_tf_step1_arr, replicated_sharding)
    tf_step2_arr = jax.device_put(tf_step2_arr, replicated_sharding)
    bat_tf_step3_arr = jax.device_put(bat_tf_step3_arr, replicated_sharding)
    origin_moduli_arr = jax.device_put(origin_moduli_arr, replicated_sharding)
    s_w_arr = jax.device_put(s_w_arr, replicated_sharding)
    w_arr = jax.device_put(w_arr, replicated_sharding)
    m_arr = jax.device_put(m_arr, replicated_sharding)
    result_jax = sharded_project_func(
      test_in,
      bat_tf_step1_arr,
      tf_step2_arr,
      bat_tf_step3_arr,
      origin_moduli_arr,
      s_w_arr,
      w_arr,
      m_arr,
    )
    np.testing.assert_array_equal(result_jax.reshape(num_moduli, batch, -1)[:,0,:], eval_in)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW_MULTI_MODULI)
  def test_3_step_N_NTT_BMM_Barrett_BatchFirst_Moduli(
      self,
      q_list,
      psi_list,
      num_moduli,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    batch, rows, cols = 4, r, c
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(None, batch_partition, None, None),
      )
      replicated_sharding = jax.sharding.NamedSharding(mesh, partition_spec())
    except RuntimeError as exc:
      self.skipTest(str(exc))
    
    test_in = jnp.array(coef_in_list, dtype=jnp.uint32).reshape(1, num_moduli, r, c)
    test_in = jnp.concatenate([test_in for _ in range(batch)], axis=0).reshape(batch, num_moduli, r, c)
    bat_tf_step1_list, tf_step2_list, bat_tf_step3_list = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_tf_step1_arr, tf_step2_arr, bat_tf_step3_arr = jnp.array(bat_tf_step1_list, jnp.uint8), jnp.array(tf_step2_list, jnp.uint32), jnp.array(bat_tf_step3_list, jnp.uint8)
    
    origin_moduli_arr = jnp.array(q_list, dtype=jnp.uint32)
    kernel_name = "jit_ntt_three_step_bat_barrett_batch"
    s_w_list, w_list, m_list = modred.barrett_control_generation_s_w(q_list)
    s_w_tuple, w_tuple, m_tuple = util.to_tuple(s_w_list), util.to_tuple(w_list), util.to_tuple(m_list)
    partial_ntt_three_step_bat_barrett_batch_multi_moduli = functools.partial(ntt.ntt_three_step_bat_barrett_batch_multi_moduli, s_w=s_w_tuple, w=w_tuple, m=m_tuple)
    sharded_project_func = jax.jit(
      jax.named_call(partial_ntt_three_step_bat_barrett_batch_multi_moduli, name=kernel_name),
      in_shardings=(
        batch_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
      ),
      out_shardings=batch_sharding,
    )
    test_in = jax.device_put(test_in, batch_sharding)
    bat_tf_step1_arr = jax.device_put(bat_tf_step1_arr, replicated_sharding)
    tf_step2_arr = jax.device_put(tf_step2_arr, replicated_sharding)
    bat_tf_step3_arr = jax.device_put(bat_tf_step3_arr, replicated_sharding)
    origin_moduli_arr = jax.device_put(origin_moduli_arr, replicated_sharding)
    result_jax = sharded_project_func(
      test_in,
      bat_tf_step1_arr,
      tf_step2_arr,
      bat_tf_step3_arr,
      origin_moduli_arr,
    )
    np.testing.assert_array_equal(result_jax.reshape(batch, num_moduli, -1)[0,...], eval_in)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW_MULTI_MODULI)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_none_Batch(
      self,
      q_list,
      psi_list,
      num_moduli,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    batch, rows, cols = 8, r, c
    test_in = jnp.array(coef_in_list, dtype=jnp.uint32).reshape(num_moduli, 1, r, c)
    test_in = jnp.concatenate([test_in for _ in range(batch)], axis=1).reshape(num_moduli, batch, r, c)
    bat_tf_step1_list, tf_step2_list, bat_tf_step3_list = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_tf_step1_arr, tf_step2_arr, bat_tf_step3_arr = jnp.array(bat_tf_step1_list, jnp.uint8), jnp.array(tf_step2_list, jnp.uint32), jnp.array(bat_tf_step3_list, jnp.uint8)

    origin_moduli_arr = jnp.array(q_list, dtype=jnp.uint32)
    kernel_name = "jit_ntt_three_step_bat_barrett_batch"
    s_w_list, w_list, m_list = modred.barrett_control_generation_s_w(q_list)
    s_w_tuple, w_tuple, m_tuple = util.to_tuple(s_w_list), util.to_tuple(w_list), util.to_tuple(m_list)
    partial_ntt_three_step_bat_barrett_batch_multi_moduli = functools.partial(ntt.ntt_three_step_bat_barrett_multi_moduli_batch, s_w=s_w_tuple, w=w_tuple, m=m_tuple)
    sharded_project_func = jax.jit(
      jax.named_call(partial_ntt_three_step_bat_barrett_batch_multi_moduli, name=kernel_name),
    )
    result_jax = partial_ntt_three_step_bat_barrett_batch_multi_moduli(
      test_in,
      bat_tf_step1_arr,
      tf_step2_arr,
      bat_tf_step3_arr,
      origin_moduli_arr,
    )
    np.testing.assert_array_equal(result_jax.reshape(num_moduli, batch, -1)[:, 0, :], eval_in)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW_MULTI_MODULI)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_Batch(
      self,
      q_list,
      psi_list,
      num_moduli,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    batch, rows, cols = 4, r, c
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(None, batch_partition),
      )
      replicated_sharding = jax.sharding.NamedSharding(mesh, partition_spec())
    except RuntimeError as exc:
      self.skipTest(str(exc))
    
    test_in = jnp.array(coef_in_list, dtype=jnp.uint32).reshape(num_moduli, 1, r, c)
    test_in = jnp.concatenate([test_in for _ in range(batch)], axis=1).reshape(num_moduli, batch, r, c)
    bat_tf_step1_list, tf_step2_list, bat_tf_step3_list = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_tf_step1_arr, tf_step2_arr, bat_tf_step3_arr = jnp.array(bat_tf_step1_list, jnp.uint8), jnp.array(tf_step2_list, jnp.uint32), jnp.array(bat_tf_step3_list, jnp.uint8)

    origin_moduli_arr = jnp.array(q_list, dtype=jnp.uint32)
    kernel_name = "jit_inner"
    s_w_list, w_list, m_list = modred.barrett_control_generation_s_w(q_list)
    s_w_tuple, w_tuple, m_tuple = util.to_tuple(s_w_list), util.to_tuple(w_list), util.to_tuple(m_list)
    partial_ntt_three_step_bat_barrett_batch_multi_moduli = functools.partial(ntt.ntt_three_step_bat_barrett_multi_moduli_batch, s_w=s_w_tuple, w=w_tuple, m=m_tuple)
    sharded_project_func = jax.jit(
      jax.named_call(partial_ntt_three_step_bat_barrett_batch_multi_moduli, name=kernel_name),
      in_shardings=(
        batch_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
      ),
      out_shardings=batch_sharding,
    )
    test_in = jax.device_put(test_in, batch_sharding)
    bat_tf_step1_arr = jax.device_put(bat_tf_step1_arr, replicated_sharding)
    tf_step2_arr = jax.device_put(tf_step2_arr, replicated_sharding)
    bat_tf_step3_arr = jax.device_put(bat_tf_step3_arr, replicated_sharding)
    origin_moduli_arr = jax.device_put(origin_moduli_arr, replicated_sharding)
    result_jax = sharded_project_func(
      test_in,
      bat_tf_step1_arr,
      tf_step2_arr,
      bat_tf_step3_arr,
      origin_moduli_arr,
    )
    np.testing.assert_array_equal(result_jax.reshape(num_moduli, batch, -1)[:, 0,...], eval_in)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW_MULTI_MODULI)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_Moduli(
      self,
      q_list,
      psi_list,
      num_moduli,
      r,
      c,
      coef_in_list,
      eval_in,
  ):
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    batch, rows, cols = 4, r, c
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(batch_partition),
      )
    except RuntimeError as exc:
      self.skipTest(str(exc))
    
    test_in = jnp.array(coef_in_list, dtype=jnp.uint32).reshape(num_moduli, 1, r, c)
    test_in = jnp.concatenate([test_in for _ in range(batch)], axis=1).reshape(num_moduli, batch, r, c)
    bat_tf_step1_multi_moduli, tf_step2_multi_moduli, bat_tf_step3_multi_moduli = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_inv_tf_mat_step1_multi_moduli, scaled_inv_tf_step2_multi_moduli, bat_inv_tf_mat_step3_multi_moduli, _ = ntt.intt_three_step_bat_control_generation(q_list, r, c)

    bat_tf_step1_multi_moduli, bat_tf_step3_multi_moduli, bat_inv_tf_mat_step1_multi_moduli, scaled_inv_tf_step2_multi_moduli, bat_inv_tf_mat_step3_multi_moduli = jnp.array(bat_tf_step1_multi_moduli, jnp.uint8), jnp.array(bat_tf_step3_multi_moduli, jnp.uint8), jnp.array(bat_inv_tf_mat_step1_multi_moduli, jnp.uint8), jnp.array(scaled_inv_tf_step2_multi_moduli, jnp.uint32), jnp.array(bat_inv_tf_mat_step3_multi_moduli, jnp.uint8)
    tf_step2_multi_moduli = jnp.array(tf_step2_multi_moduli, jnp.uint32)
    s_w_list, w_list, m_list = modred.barrett_control_generation_s_w(q_list)

    assert bat_tf_step1_multi_moduli.shape == (num_moduli, r, r, 4, 4)
    assert tf_step2_multi_moduli.shape == (num_moduli, r, c)
    assert bat_tf_step3_multi_moduli.shape == (num_moduli, c, c, 4, 4)
    assert bat_inv_tf_mat_step1_multi_moduli.shape == (num_moduli, c, c, 4, 4)
    assert scaled_inv_tf_step2_multi_moduli.shape == (num_moduli, r, c)
    assert bat_inv_tf_mat_step3_multi_moduli.shape == (num_moduli, r, r, 4, 4)

    origin_moduli_arr = jnp.array(q_list, dtype=jnp.uint32)
    kernel_name = "jit_inner"
    s_w_arr = jnp.array(s_w_list, dtype=jnp.uint16)
    w_arr = jnp.array(w_list, dtype=jnp.uint16)
    m_arr = jnp.array(m_list, dtype=jnp.uint64)

    sharded_project_func = jax.jit(
      jax.named_call(ntt.ntt_three_step_bat_barrett_multi_moduli_batch_no_static, name=kernel_name),
      in_shardings=(
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
        batch_sharding,
      ),
      out_shardings=batch_sharding,
    )
    coef_in_twisted = jax.device_put(test_in, batch_sharding)
    bat_tf_step1_multi_moduli = jax.device_put(bat_tf_step1_multi_moduli, batch_sharding)
    tf_step2_multi_moduli = jax.device_put(tf_step2_multi_moduli, batch_sharding)
    bat_tf_step3_multi_moduli = jax.device_put(bat_tf_step3_multi_moduli, batch_sharding)
    origin_moduli_arr = jax.device_put(origin_moduli_arr, batch_sharding)
    s_w_arr = jax.device_put(s_w_arr, batch_sharding)
    w_arr = jax.device_put(w_arr, batch_sharding)
    m_arr = jax.device_put(m_arr, batch_sharding)

    result_jax = sharded_project_func(
      coef_in_twisted,
      bat_tf_step1_multi_moduli,
      tf_step2_multi_moduli,
      bat_tf_step3_multi_moduli,
      origin_moduli_arr,
      s_w_arr,
      w_arr,
      m_arr,
    )
    np.testing.assert_array_equal(result_jax.reshape(num_moduli, batch, -1)[:,0,...], eval_in)
    

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_none_ModSwitch(
      self,
      q_list,
      psi_list,
      batch,
      r,
      c,
      eval_in,
      coef_ref,
  ):
    """
    This script tests the correctness of the NTT used in ModSwitch.
    """
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    ########################
    # reference result generation
    ########################
    # Control Generation for NTT and INTT
    bat_tf_step1_list, tf_step2_list, bat_tf_step3_list = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_tf_step1_arr, tf_step2_arr, bat_tf_step3_arr = jnp.array(bat_tf_step1_list, jnp.uint8), jnp.array(tf_step2_list, jnp.uint32), jnp.array(bat_tf_step3_list, jnp.uint8)
    origin_moduli = q_list
    num_moduli = len(q_list)
    bit_reverse_indices = util.bit_reverse_indices(r*c)
    s_w_tuple, w_tuple, m_tuple = modred.barrett_control_generation_s_w(q_list)
    def project_func(last_poly_switch_modulus_coef_twisted):
        last_poly_switch_modulus_eval = ntt.ntt_three_step_bat_barrett_multi_moduli(
          last_poly_switch_modulus_coef_twisted,
          bat_tf_step1_arr[:-1],
          tf_step2_arr[:-1],
          bat_tf_step3_arr[:-1],
          util.to_tuple(origin_moduli[:-1]),
          util.to_tuple(s_w_tuple[:-1]),
          util.to_tuple(w_tuple[:-1]),
          util.to_tuple(m_tuple[:-1])
        ).transpose(0,2,1).reshape(num_moduli-1, -1)
        return jnp.take(last_poly_switch_modulus_eval, bit_reverse_indices, axis=-1)

    project_func_vmap = jax.vmap(project_func, in_axes=0, out_axes=0)
    new_ciphertext_unreduced = project_func_vmap(jnp.array(eval_in, jnp.uint32))
    np.testing.assert_array_equal(new_ciphertext_unreduced, coef_ref)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NEGACYCLIC_NTT_MODSW)
  def test_3_step_N_NTT_MMB_Barrett_BatchSecond_none_ModSwitchFused(
      self,
      q_list,
      psi_list,
      batch,
      r,
      c,
      eval_in,
      coef_ref,
  ):
    """
    Test the eval_in -> NTT -> coef_out -> bitreverse -> coef_br [-> accumulated -> updated ciphertext] (things in [] are not tested here.);
    Check coef_ref == coef_br
    """
    element_id = 0
    for q in q_list:
      if math.log2(q) > 32:
        print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
        return
    ########################
    # reference result generation
    ########################
    # Control Generation for NTT and INTT
    bat_tf_step1_list, tf_step2_list, bat_tf_step3_list = ntt.ntt_three_step_bat_control_generation(q_list, r, c)
    bat_tf_step1_arr, tf_step2_arr, bat_tf_step3_arr = jnp.array(bat_tf_step1_list, jnp.uint8), jnp.array(tf_step2_list, jnp.uint32), jnp.array(bat_tf_step3_list, jnp.uint8)
    origin_moduli = q_list
    num_moduli = len(q_list)
    bit_reverse_indices = jnp.array(util.bit_reverse_indices(r*c), jnp.uint32)
    merged_indices = (bit_reverse_indices % r) * c + (bit_reverse_indices // r)
    s_w_tuple, w_tuple, m_tuple = modred.barrett_control_generation_s_w(q_list)
    def project_func(last_poly_switch_modulus_coef_twisted):
        last_poly_switch_modulus_eval = ntt.ntt_three_step_bat_barrett_multi_moduli(
          last_poly_switch_modulus_coef_twisted,
          bat_tf_step1_arr[:-1],
          tf_step2_arr[:-1],
          bat_tf_step3_arr[:-1],
          util.to_tuple(origin_moduli[:-1]),
          util.to_tuple(s_w_tuple[:-1]),
          util.to_tuple(w_tuple[:-1]),
          util.to_tuple(m_tuple[:-1])
        ).reshape(num_moduli-1, -1)
        return jnp.take(last_poly_switch_modulus_eval, merged_indices, axis=-1)

    project_func_vmap = jax.vmap(project_func, in_axes=0, out_axes=0)
    new_ciphertext_unreduced = project_func_vmap(jnp.array(eval_in, jnp.uint32))
    np.testing.assert_array_equal(new_ciphertext_unreduced, coef_ref)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_3_step_N_NTT_SMB_Montgomery_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    s_w, w, m = modred.barrett_control_generation_s_w(q)
    tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch = ntt.ntt_negacyclic_three_step_control_generation(q, r, c)
    bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch = ntt.ntt_montgomery_three_step_bat_control_generation(q, r, c)
    parameters = modred.montgomery_control_generation(q, r, c)
    assert bat_tf_step1_montgomery_batch.shape == (r, r, 4, 4)
    assert bat_tf_step1_montgomery_batch.dtype == jnp.uint8
    assert tf_step2_batch.shape == (r, c)
    assert tf_step2_montgomery_batch.dtype == jnp.uint32
    assert bat_tf_step3_montgomery_batch.shape == (c, c, 4, 4)
    assert bat_tf_step3_montgomery_batch.dtype == jnp.uint8
    # if c == r:
    #   np.testing.assert_array_equal(bat_tf_step1_montgomery_batch, bat_tf_step3_montgomery_batch)
    np.testing.assert_array_equal(tf_mat_step1_batch.T, tf_mat_step1_batch)
    np.testing.assert_array_equal(tf_mat_step3_batch.T, tf_mat_step3_batch)

    ntt_result = ntt.ntt_negacyclic_three_step(
        coef_in, q, psi, r, c, tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch
    )
    coef_in = modred.original_format_to_montgomery_computation_format(jnp.array(coef_in, jnp.uint32), q)
    twist_factor = jnp.array([pow(psi, i, q) for i in range(r*c)], jnp.uint64)
    twist_factor_montgomery = modred.original_format_to_montgomery_computation_format(twist_factor, q)
    coef_in_twisted = coef_in.astype(jnp.uint64) * twist_factor_montgomery# % q
    coef_in_twisted = modred.montgomery_reduce_u64_to_u32(coef_in_twisted, *parameters)
    coef_in_twisted = coef_in_twisted.reshape(batch, r, c)
    jit_ntt_three_step_bat_montgomery_batch = jax.jit(
      jax.named_call(ntt.ntt_three_step_bat_montgomery_batch, name="ntt_three_step_bat_montgomery_batch"),
      static_argnames=("q_low", "q_high", "q_inv_32", "q"),
    )
    result = jit_ntt_three_step_bat_montgomery_batch(coef_in_twisted.astype(jnp.uint32), bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch, *parameters)
    result = np.array(result.T).flatten()
    np.testing.assert_array_equal(ntt_result, result)
    result_bit_reverse = util.bit_reverse_array(result)
    np.testing.assert_array_equal(result_bit_reverse, eval_in)

    # performance measurement
    tasks = [
        (ntt.ntt_three_step_bat_montgomery_batch, (coef_in_twisted, bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch, *parameters)),
    ]
    profile_name = "test_modulus_switch_ntt_montgomery_batch"
    # util.profile_jax_functions(tasks, profile_name)


  # @absltest.skip("test single implementation") # TODO: Fix rhs will fix it
  @parameterized.named_parameters(*NTT)
  def test_3_step_N_NTT_MMSB_Montgomery_BatchFirst_none(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    s_w, w, m = modred.barrett_control_generation_s_w(q)
    tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch = ntt.ntt_negacyclic_three_step_control_generation(q, r, c)
    bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch = ntt.ntt_montgomery_three_step_bat_control_generation(q, r, c)
    q_low, q_high, q_inv_32, q = modred.montgomery_control_generation(q, r, c)
    assert bat_tf_step1_montgomery_batch.shape == (r, r, 4, 4)
    assert bat_tf_step1_montgomery_batch.dtype == jnp.uint8
    assert tf_step2_batch.shape == (r, c)
    assert tf_step2_montgomery_batch.dtype == jnp.uint32
    assert bat_tf_step3_montgomery_batch.shape == (c, c, 4, 4)
    assert bat_tf_step3_montgomery_batch.dtype == jnp.uint8
    # if c == r:
    #   np.testing.assert_array_equal(bat_tf_step1_montgomery_batch, bat_tf_step3_montgomery_batch)
    np.testing.assert_array_equal(tf_mat_step1_batch.T, tf_mat_step1_batch)
    np.testing.assert_array_equal(tf_mat_step3_batch.T, tf_mat_step3_batch)

    ntt_result = ntt.ntt_negacyclic_three_step(
        coef_in, q, psi, r, c, tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch
    )
    coef_in = modred.original_format_to_montgomery_computation_format(jnp.array(coef_in, jnp.uint32), q)
    twist_factor = jnp.array([pow(psi, i, q) for i in range(r*c)], jnp.uint64)
    twist_factor_montgomery = modred.original_format_to_montgomery_computation_format(twist_factor, q)
    coef_in_twisted = coef_in.astype(jnp.uint64) * twist_factor_montgomery# % q
    coef_in_twisted = modred.montgomery_reduce_u64_to_u32(coef_in_twisted, q_low, q_high, q_inv_32, q)
    coef_in_twisted = coef_in_twisted.reshape(batch, r, c).astype(jnp.uint32)
    jit_ntt_three_step_bat_montgomery_square_batch = jax.jit(
      ntt.ntt_three_step_bat_montgomery_square_batch,
      static_argnames=("q_low", "q_high", "q_inv_32", "q"),
    )
    result = jit_ntt_three_step_bat_montgomery_square_batch(coef_in_twisted, bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, q_low, q_high, q_inv_32, q)
    result = np.array(result.T).flatten()
    np.testing.assert_array_equal(ntt_result, result)
    result_bit_reverse = util.bit_reverse_array(result)
    np.testing.assert_array_equal(result_bit_reverse, eval_in)

    # performance measurement
    tasks = [
        (jit_ntt_three_step_bat_montgomery_square_batch, (coef_in_twisted, bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch, q_low, q_high, q_inv_32, q)),
    ]
    profile_name = "test_modulus_switch_ntt_montgomery_batch_square"
    # util.profile_jax_functions_xprof(tasks, profile_name, kernel_name="ntt_three_step_bat_montgomery_square_batch")


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_3_step_N_NTT_SMB_Montgomery_BatchFirst_none_Duplicate(
      self,
      q,
      psi,
      batch,
      r,
      c,
      coef_in,
      eval_in,
  ):
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    s_w, w, m = modred.barrett_control_generation_s_w(q)
    tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch = ntt.ntt_negacyclic_three_step_control_generation(q, r, c)
    bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch = ntt.ntt_montgomery_three_step_bat_control_generation(q, r, c)
    parameters = modred.montgomery_control_generation(q, r, c)
    assert bat_tf_step1_montgomery_batch.shape == (r, r, 4, 4)
    assert bat_tf_step1_montgomery_batch.dtype == jnp.uint8
    assert tf_step2_batch.shape == (r, c)
    assert tf_step2_montgomery_batch.dtype == jnp.uint32
    assert bat_tf_step3_montgomery_batch.shape == (c, c, 4, 4)
    assert bat_tf_step3_montgomery_batch.dtype == jnp.uint8
    # if c == r:
    #   np.testing.assert_array_equal(bat_tf_step1_montgomery_batch, bat_tf_step3_montgomery_batch)
    np.testing.assert_array_equal(tf_mat_step1_batch.T, tf_mat_step1_batch)
    np.testing.assert_array_equal(tf_mat_step3_batch.T, tf_mat_step3_batch)

    ntt_result = ntt.ntt_negacyclic_three_step(
        coef_in, q, psi, r, c, tf_mat_step1_batch, tf_step2_batch, tf_mat_step3_batch
    )
    coef_in = modred.original_format_to_montgomery_computation_format(jnp.array(coef_in, jnp.uint32), q)
    twist_factor = jnp.array([pow(psi, i, q) for i in range(r*c)], jnp.uint64)
    twist_factor_montgomery = modred.original_format_to_montgomery_computation_format(twist_factor, q)
    coef_in_twisted = coef_in.astype(jnp.uint64) * twist_factor_montgomery# % q
    coef_in_twisted = modred.montgomery_reduce_u64_to_u32(coef_in_twisted, *parameters)
    coef_in_twisted = coef_in_twisted.reshape(batch, r, c)
    jit_ntt_three_step_bat_montgomery_batch = jax.jit(
      jax.named_call(ntt.ntt_three_step_bat_montgomery_batch, name="ntt_three_step_bat_montgomery_batch"),
      static_argnames=("q_low", "q_high", "q_inv_32", "q"),
    )
    result = jit_ntt_three_step_bat_montgomery_batch(coef_in_twisted.astype(jnp.uint32), bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch, *parameters)
    result = np.array(result.T).flatten()
    np.testing.assert_array_equal(ntt_result, result)
    result_bit_reverse = util.bit_reverse_array(result)
    np.testing.assert_array_equal(result_bit_reverse, eval_in)

    # performance measurement
    tasks = [
        (ntt.ntt_three_step_bat_montgomery_batch, (coef_in_twisted, bat_tf_step1_montgomery_batch, tf_step2_montgomery_batch, bat_tf_step3_montgomery_batch, *parameters)),
    ]
    profile_name = "test_modulus_switch_ntt_montgomery_batch_multi_moduli"
    # util.profile_jax_functions(tasks, profile_name)

 
if __name__ == "__main__":
  absltest.main()
