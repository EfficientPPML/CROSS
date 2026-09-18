"""Tests for the ``packing.py`` phase.

packing.pack turns a frontend VectorizedProgram into PP-ops on a ring whose
security parameters it derives from that program. The properties that matter
here are that the ring is *derived* (not assumed), that depth is recomputed
from the operations actually emitted, and that the packed circuit still
computes what the torch model computed.
"""

import math
import unittest.mock as mock
import pickle
import subprocess
import sys
import typing
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import nn
import packing

torch = None
tnn = None
F = None


def setUpModule():
  global torch, tnn, F
  try:
    import torch as torch_module
    import torch.nn as torch_nn
    import torch.nn.functional as functional
  except ImportError as error:  # pragma: no cover - environment dependent
    raise absltest.SkipTest(f'PyTorch unavailable: {error}')
  torch = torch_module
  tnn = torch_nn
  F = functional
  torch.manual_seed(0)


def _chain():

  class Chain(tnn.Module):

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 2, 3, padding=1)
      self.fc = tnn.Linear(8, 3)

    def forward(self, x):
      x = self.conv(x)
      x = x * x
      return self.fc(torch.flatten(x, 1))

  return Chain().double().eval()


def _residual():

  class Residual(tnn.Module):

    def __init__(self):
      super().__init__()
      self.a = tnn.Conv2d(2, 2, 3, padding=1)
      self.b = tnn.Conv2d(2, 2, 3, padding=1)
      self.fc = tnn.Linear(8, 3)

    def forward(self, x):
      y = self.a(x)
      y = y * y
      y = self.b(y)
      return self.fc(torch.flatten(x + y, 1))

  return Residual().double().eval()


def _pooled():

  class Pooled(tnn.Module):

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 4, 3, padding=1)
      self.fc = tnn.Linear(16, 2)

    def forward(self, x):
      x = F.avg_pool2d(self.conv(x), 2, 2)
      x = x * x
      return self.fc(x.reshape(x.size(0), -1))

  return Pooled().double().eval()


# --------------------------------------------------------------------------
# Cleartext PP-op interpreter, over full-width slot vectors.
# --------------------------------------------------------------------------


def run_packed(packed, sample):
  """Evaluate the emitted PP-ops in cleartext at the ring's slot width."""
  width = packed.num_slots
  flat = np.zeros(width, dtype=np.float64)
  source = np.asarray(sample, dtype=np.float64).ravel()
  flat[:source.size] = source
  values = {'input': flat}
  for value_id, kind, inputs, argument in packed.operations:
    operands = [values[name] for name in inputs]
    if kind == 'matvec':
      matrix = argument[0]
      diagonals = (
          matrix.diagonals if hasattr(matrix, 'diagonals')
          else tuple(matrix.source.as_dict().items())
      )
      result = np.zeros(width, dtype=np.float64)
      for index, diagonal in diagonals:
        result += diagonal * np.roll(operands[0], -index)
      values[value_id] = result
    elif kind == 'add_plain':
      values[value_id] = operands[0] + argument.values
    elif kind == 'mul_plain':
      values[value_id] = operands[0] * argument.values
    elif kind == 'square':
      values[value_id] = operands[0] * operands[0]
    elif kind == 'mul':
      values[value_id] = operands[0] * operands[1]
    elif kind == 'add':
      values[value_id] = operands[0] + operands[1]
    elif kind == 'rescale':
      # Scale bookkeeping only; the represented value does not change.
      values[value_id] = operands[0]
    else:
      raise AssertionError(f'unhandled PP-op {kind}')
  return values[packed.output]


class RingDerivationTest(absltest.TestCase):
  """The ring is derived from this program, and never assumed."""

  def test_slots_are_half_the_degree(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    self.assertEqual(packed.num_slots, packed.ring_config.degree // 2)

  def test_the_degree_is_not_the_next_power_of_two_of_the_slot_demand(self):
    """CKKS packs degree/2 slots; equating the two under-provisions by 2x."""

    class Wide(tnn.Module):

      def __init__(self):
        super().__init__()
        self.conv = tnn.Conv2d(1, 8, 3, padding=1)
        self.fc = tnn.Linear(512, 4)

      def forward(self, x):
        x = self.conv(x)
        x = x * x
        return self.fc(torch.flatten(x, 1))

    program = nn.vectorize(Wide().double().eval(), (1, 8, 8))
    demand = max(
        [program.input_spec.physical_size]
        + [layer.output_spec.physical_size for layer in program.layers]
    )
    packed = packing.pack(program)
    naive = 1 << (demand - 1).bit_length()
    self.assertGreaterEqual(packed.num_slots, demand)
    self.assertGreaterEqual(packed.ring_config.degree, 2 * demand)
    self.assertNotEqual(packed.ring_config.degree, naive)

  def test_num_q_is_the_emitted_depth_plus_one(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    self.assertEqual(
        int(packed.ring_config.logical_num_q), packed.depth + 1
    )
    self.assertEqual(
        int(packed.ring_config.multiplicative_depth), packed.depth
    )

  def test_the_ring_is_composite_degree_two(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    self.assertEqual(int(packed.ring_config.composite_degree), 2)

  def test_a_dnum_of_two_at_num_q_three_is_avoided(self):
    """he_params raises an EstimateLogP mismatch there, aborting generation."""
    self.assertEqual(packing._select_dnum(2), 2)
    self.assertEqual(packing._select_dnum(3), 3)
    self.assertNotEqual(packing._select_dnum(3), 2)
    self.assertEqual(packing._select_dnum(8), 4)
    self.assertEqual(packing._select_dnum(17), 9)

  def test_a_depth_two_program_packs_despite_the_num_q_three_hole(self):

    class Depth2(tnn.Module):

      def __init__(self):
        super().__init__()
        self.fc = tnn.Linear(4, 4)

      def forward(self, x):
        return self.fc(x) * self.fc(x) if False else self.fc(x * x)

    packed = packing.pack(nn.vectorize(Depth2().double().eval(), (4,)))
    self.assertEqual(packed.depth, 2)
    self.assertEqual(int(packed.ring_config.logical_num_q), 3)

  def test_an_explicit_security_target_is_used(self):
    program = nn.vectorize(_chain(), (1, 2, 2))
    default = packing.pack(program)
    self.assertEqual(
        int(default.ring_config.target.bits), packing.DEFAULT_SECURITY_BITS
    )
    self.assertEqual(packing.DEFAULT_SECURITY_BITS, 128)
    stronger = packing.pack(
        program, packing.PackingPolicy(security_bits=256)
    )
    self.assertGreaterEqual(
        stronger.ring_config.degree, default.ring_config.degree
    )


class DepthRecomputationTest(absltest.TestCase):
  """Depth comes from the emitted PP-ops, not the frontend's declaration."""

  def test_packed_depth_matches_the_level_cost_of_the_emitted_dag(self):
    packed = packing.pack(nn.vectorize(_residual(), (2, 2, 2)))
    depth = {'input': 0}
    for value_id, kind, inputs, argument in packed.operations:
      cost = int(argument) if kind == 'rescale' else packing._LEVEL_COST[kind]
      depth[value_id] = max(
          (depth[name] for name in inputs), default=0
      ) + cost
    self.assertEqual(packed.depth, max(depth.values()))

  def test_a_residual_costs_its_deeper_arm_not_their_sum(self):
    packed = packing.pack(nn.vectorize(_residual(), (2, 2, 2)))
    # Deep arm 3, skip arm aligned to 3, join free, fc 1. Summing would give 7.
    self.assertEqual(packed.depth, 4)

  def test_a_residual_join_receives_both_arms_at_one_level(self):
    packed = packing.pack(nn.vectorize(_residual(), (2, 2, 2)))
    add = [op for op in packed.operations if op[1] == 'add']
    self.assertLen(add, 1)
    # Both operands of the join reach it at the same depth.
    depth = {'input': 0}
    for value_id, kind, inputs, argument in packed.operations:
      cost = int(argument) if kind == 'rescale' else packing._LEVEL_COST[kind]
      depth[value_id] = max(
          (depth[name] for name in inputs), default=0
      ) + cost
    self.assertEqual(
        depth[add[0][2][0]], depth[add[0][2][1]],
        'a slotwise add cannot join two different levels',
    )

  def test_fusing_a_pool_lowers_the_packed_depth_and_the_chain(self):
    model = _pooled()
    fused = packing.pack(nn.vectorize(model, (1, 4, 4)))
    separate = packing.pack(
        nn.vectorize(model, (1, 4, 4), fuse_pooling=False)
    )
    self.assertEqual(fused.depth, separate.depth - 1)
    self.assertEqual(fused.kinds().count('matvec'), 2)
    self.assertEqual(separate.kinds().count('matvec'), 3)
    self.assertLess(
        int(fused.ring_config.logical_num_q),
        int(separate.ring_config.logical_num_q),
    )

  def test_a_squaring_chain_emits_one_square_per_level(self):
    for degree in (2, 4, 8, 16):
      polynomial = nn.ActivationPolynomial(
          name=f'x{degree}', coefficients=(0.0,) * degree + (1.0,)
      )
      self.assertTrue(polynomial.is_squaring_chain)
      self.assertEqual(polynomial.depth_cost, int(math.log2(degree)))

  def test_a_registered_quartic_monomial_packs_at_its_declared_depth(self):
    nn.register_activation(
        nn.ActivationPolynomial(
            name='quartic_test', coefficients=(0.0,) * 4 + (1.0,)
        ),
        substitutes=('relu',),
    )
    try:

      class Quartic(tnn.Module):

        def __init__(self):
          super().__init__()
          self.fc1 = tnn.Linear(4, 4)
          self.act = tnn.ReLU()
          self.fc2 = tnn.Linear(4, 2)

        def forward(self, x):
          return self.fc2(self.act(self.fc1(x)))

      with warnings.catch_warnings():
        warnings.simplefilter('ignore', nn.ActivationSubstitutionWarning)
        program = nn.vectorize(Quartic().double().eval(), (4,))
      self.assertEqual(program.layer('act').depth_cost, 2)
      packed = packing.pack(program)
      # fc1(1) + two squarings + fc2(1)
      self.assertEqual(packed.depth, 4)
      self.assertEqual(packed.kinds().count('square'), 2)
      self.assertNotIn('mul', packed.kinds())
    finally:
      nn._ACTIVATIONS.pop('quartic_test', None)
      nn._ACTIVATION_DEFAULTS['relu'] = 'square'

  def test_an_unevaluable_polynomial_is_refused_before_a_model_uses_it(self):
    """It must not vectorize successfully and then surprise pack()."""
    for coefficients in ((0.1, 0.5, 0.3, 0.0, -0.02), (0.0, 0.0, 0.0, 1.0)):
      with self.assertRaises(ValueError) as caught:
        nn.register_activation(
            nn.ActivationPolynomial(
                name='unevaluable_test', coefficients=coefficients
            ),
            substitutes=('relu',),
        )
      self.assertIn('x**(2**k)', str(caught.exception))
    self.assertNotIn('unevaluable_test', nn._ACTIVATIONS)
    self.assertEqual(nn._ACTIVATION_DEFAULTS['relu'], 'square')

  def test_a_multi_term_polynomial_declares_the_deeper_cost(self):
    """Its declaration must not claim ceil(log2 d) it cannot back."""
    polynomial = nn.ActivationPolynomial(
        name='quartic', coefficients=(0.1, 0.5, 0.3, 0.0, -0.02)
    )
    self.assertFalse(polynomial.is_unit_monomial)
    self.assertEqual(polynomial.depth_cost, math.ceil(math.log2(4)) + 1)


class ConstantMaterializationTest(absltest.TestCase):
  """Constants are full width, immutable and content-digested."""

  def test_every_constant_spans_exactly_the_ring_slot_count(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    for _, kind, _, argument in packed.operations:
      if kind == 'matvec':
        self.assertEqual(argument[0].dimension, packed.num_slots)
      elif kind == 'add_plain':
        self.assertEqual(argument.values.shape, (packed.num_slots,))

  def test_every_constant_is_immutable_and_digested(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    for _, kind, _, argument in packed.operations:
      constant = argument[0] if kind == 'matvec' else argument
      if kind not in ('matvec', 'add_plain'):
        continue
      self.assertNotEmpty(constant.digest)
      if kind == 'add_plain':
        self.assertFalse(constant.values.flags.writeable)

  def test_bias_lands_in_the_slots_its_channel_occupies(self):
    model = _chain()
    packed = packing.pack(nn.vectorize(model, (1, 2, 2)))
    bias_op = next(op for op in packed.operations if op[1] == 'add_plain')
    values = bias_op[3].values
    expected = model.conv.bias.detach().numpy()
    # Channel-major over (2, 2, 2): four slots per channel.
    np.testing.assert_allclose(values[0:4], expected[0])
    np.testing.assert_allclose(values[4:8], expected[1])
    np.testing.assert_allclose(values[8:], 0.0)


class DeterminismTest(absltest.TestCase):

  def test_packing_the_same_program_twice_gives_one_fingerprint(self):
    program = nn.vectorize(_chain(), (1, 2, 2))
    self.assertEqual(
        packing.pack(program).fingerprint,
        packing.pack(program).fingerprint,
    )

  def test_packing_a_pickled_program_gives_the_same_fingerprint(self):
    program = nn.vectorize(_chain(), (1, 2, 2))
    restored = pickle.loads(pickle.dumps(program))
    self.assertEqual(
        packing.pack(restored).fingerprint,
        packing.pack(program).fingerprint,
    )

  def test_a_changed_weight_changes_the_fingerprint(self):
    model = _chain()
    before = packing.pack(nn.vectorize(model, (1, 2, 2))).fingerprint
    with torch.no_grad():
      model.fc.weight[0, 0] += 1.0
    self.assertNotEqual(
        packing.pack(nn.vectorize(model, (1, 2, 2))).fingerprint, before
    )


class FailureTest(absltest.TestCase):

  def test_a_program_deeper_than_any_secure_ring_is_refused(self):

    class TooDeep(tnn.Module):

      def __init__(self):
        super().__init__()
        self.layers = tnn.ModuleList(
            [tnn.Linear(4, 4) for _ in range(30)]
        )

      def forward(self, x):
        for layer in self.layers:
          x = layer(x)
          x = x * x
        return x

    program = nn.vectorize(TooDeep().double().eval(), (4,))
    self.assertGreater(program.critical_depth(), 56)
    with self.assertRaises(packing.PackingError) as caught:
      packing.pack(program)
    message = str(caught.exception)
    self.assertIn('no secure ring', message)
    self.assertIn('depth', message)
    self.assertIn('does not insert a bootstrap', message)
    self.assertIn(str(program.critical_depth()), message)

  def test_nothing_auto_inserts_a_bootstrap(self):
    for model, shape in ((_chain(), (1, 2, 2)), (_residual(), (2, 2, 2))):
      packed = packing.pack(nn.vectorize(model, shape))
      self.assertNotIn('bootstrap', packed.kinds())

  def test_the_policy_advertises_no_bootstrap_hook_it_does_not_honour(self):
    fields = {
        field.name for field in
        __import__('dataclasses').fields(packing.PackingPolicy)
    }
    self.assertNotIn('bootstrap_planner', fields)
    self.assertIn('lazy_constants', fields)


class PhysicalLayoutTest(absltest.TestCase):
  """A ciphertext has one physical layout; logical shapes are metadata.

  Mapping compares layouts for exact equality, so a matvec advertising its
  logical output shape would disagree with its own bias plaintext, and a
  flatten would disagree with everything after it.
  """

  def test_every_value_is_one_full_width_slot_vector(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    physical = (packed.num_slots,)
    self.assertEqual(packed.input_shape, physical)
    self.assertEqual(packed.input_packing, 'slots')
    for _, kind, _, argument in packed.operations:
      if kind == 'matvec':
        self.assertEqual(argument[0].input_shape, physical)
        self.assertEqual(argument[0].output_shape, physical)
        self.assertEqual(argument[0].packing, 'slots')
        self.assertEqual(argument[0].output_packing, 'slots')
      elif kind == 'add_plain':
        self.assertEqual(argument.shape, physical)
        self.assertEqual(argument.packing, 'slots')

  def test_a_matvec_and_its_bias_share_one_layout(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    matvec = next(op for op in packed.operations if op[1] == 'matvec')
    bias = next(op for op in packed.operations if op[1] == 'add_plain')
    self.assertEqual(
        (matvec[3][0].output_shape, matvec[3][0].output_packing),
        (bias[3].shape, bias[3].packing),
    )

  def test_logical_shapes_survive_as_metadata(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    self.assertEqual(packed.logical_input_shape, (1, 2, 2))
    self.assertEqual(packed.logical_output_shape, (3,))
    shapes = dict(packed.layer_shapes)
    self.assertNotEmpty(shapes)
    self.assertIn((1, 2, 2), [pair[0] for pair in shapes.values()])

  def test_the_reserved_input_value_id_is_used(self):
    """Mapping seeds 'input'; any other spelling is an unavailable value."""
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    produced = {value_id for value_id, _, _, _ in packed.operations}
    consumed = {
        name for _, _, inputs, _ in packed.operations for name in inputs
    }
    self.assertEqual(consumed - produced, {'input'})
    self.assertEqual(nn.PROGRAM_INPUT, 'input')


class MappingPlanningTest(absltest.TestCase):
  """A packed program survives the real scheduler's analysis pass."""

  def _context(self, ring_config):
    from ckks_ctx import CKKSContext
    return CKKSContext(
        {
            'degree': int(ring_config.degree),
            'num_slots': int(ring_config.num_slots),
            'q_towers': [int(q) for q in ring_config.q_towers],
            'p_towers': [int(q) for q in ring_config.p_towers],
            'composite_degree': int(ring_config.composite_degree),
            'scaling_factor': float(ring_config.scaling_factor),
            'output_scale': float(ring_config.scaling_factor),
            'degree_layout': (4, int(ring_config.degree) // 4),
        },
        batch=1,
        dnum=int(ring_config.dnum),
    )

  def _analyze(self, packed):
    import mapping as mapping_module
    mapping = mapping_module.Mapping.__new__(mapping_module.Mapping)
    context = self._context(packed.ring_config)
    mapping.packing = packed
    mapping.input_scale = None
    mapping.input_nsd = 1
    mapping.headroom = 0
    mapping.ctx = context
    mapping.global_batch = context.batch
    mapping.devices = ()
    mapping.device_count = 1
    mapping.per_device_batch = context.batch
    mapping.compile_mode = 'whole'
    mapping.bsgs_n_jobs = 1
    mapping.bsgs_streaming = False
    mapping.bindings = {}
    mapping._compiled = None
    mapping._compiled_regions = None
    mapping._templates = {}
    mapping_module._analyze_mapping(mapping, perf_test=True)
    return mapping

  def _tiny(self):

    class Tiny(tnn.Module):

      def __init__(self):
        super().__init__()
        self.fc = tnn.Linear(4, 4)

      def forward(self, x):
        return self.fc(x * x)

    return Tiny().double().eval()

  def test_a_packed_chain_analyzes_under_the_real_scheduler(self):
    packed = packing.pack(nn.vectorize(self._tiny(), (4,)))
    mapping = self._analyze(packed)
    output = mapping.value_specs[packed.output]
    self.assertEqual(output.shape, (packed.num_slots,))
    self.assertEqual(output.packing, 'slots')
    self.assertEqual(output.nsd, 1)
    # The chain was sized for exactly this program's depth.
    self.assertEqual(output.level, 0)

  def test_the_ring_the_packer_chose_leaves_no_level_unspent(self):
    packed = packing.pack(nn.vectorize(self._tiny(), (4,)))
    mapping = self._analyze(packed)
    self.assertEqual(
        mapping.value_specs['input'].level, packed.depth
    )

  def test_a_balanced_join_analyzes_under_the_real_scheduler(self):

    class Balanced(tnn.Module):

      def __init__(self):
        super().__init__()
        self.a = tnn.Linear(4, 4)
        self.b = tnn.Linear(4, 4)

      def forward(self, x):
        return self.a(x) + self.b(x)

    packed = packing.pack(nn.vectorize(Balanced().double().eval(), (4,)))
    self.assertIn('add', packed.kinds())
    mapping = self._analyze(packed)
    self.assertIn(packed.output, mapping.value_specs)

  def test_an_unbalanced_join_matches_level_scale_and_nsd_exactly(self):
    """The shallow arm must arrive indistinguishable from the deep one."""

    class Unbalanced(tnn.Module):

      def __init__(self):
        super().__init__()
        self.a = tnn.Linear(4, 4)
        self.b = tnn.Linear(4, 4)

      def forward(self, x):
        y = self.a(x)
        return self.b(y * y) + x

    packed = packing.pack(nn.vectorize(Unbalanced().double().eval(), (4,)))
    mapping = self._analyze(packed)
    join = next(op for op in packed.operations if op[1] == 'add')
    left, right = (mapping.value_specs[name] for name in join[2])
    self.assertEqual(left.level, right.level)
    self.assertEqual(left.nsd, right.nsd)
    self.assertEqual(left.nsd, 1)
    # Strict: these are the same float, not merely close.
    self.assertEqual(left.scale, right.scale)

  def test_alignment_uses_ones_multiplies_and_leaves_the_value_alone(self):

    class Unbalanced(tnn.Module):

      def __init__(self):
        super().__init__()
        self.a = tnn.Linear(4, 4)
        self.b = tnn.Linear(4, 4)

      def forward(self, x):
        y = self.a(x)
        return self.b(y * y) + x

    model = Unbalanced().double().eval()
    packed = packing.pack(nn.vectorize(model, (4,)))
    ones = [op for op in packed.operations if op[1] == 'mul_plain']
    self.assertNotEmpty(ones)
    for op in ones:
      np.testing.assert_array_equal(
          op[3].values, np.ones(packed.num_slots)
      )
      self.assertIsNone(op[3].scale)
    generator = np.random.default_rng(11)
    for _ in range(3):
      sample = generator.normal(size=(4,))
      with torch.no_grad():
        expected = model(
            torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
        ).numpy().ravel()
      np.testing.assert_allclose(
          run_packed(packed, sample)[:4], expected, atol=1e-9
      )


class LayoutReconciliationTest(absltest.TestCase):
  """A consumer reads slots through its own input_spec.

  Making every physical shape (num_slots,) does not make layouts
  interchangeable: a matrix built for a channel-major input would misread a
  stride-multiplexed one. Today's demos are uniformly channel-major, so these
  programs are built by hand to exercise the mismatch at all.
  """

  def _weights(self):
    builder = nn.WeightTableBuilder()
    return builder, builder.add(np.eye(8))

  def test_a_relabeling_between_equal_layouts_emits_nothing(self):
    builder, weight = self._weights()
    chw = packing.TensorSpec((2, 2, 2), packing.ChannelMajor())
    flat = packing.TensorSpec((8,), packing.ChannelMajor())
    producer = nn.VectorizedLayer(
        name='a', kind=nn.LINEAR_TRANSFORM, template_id='dense',
        input_spec=flat, output_spec=flat, weights={'weight': weight},
    )
    # Reads the same slots described as (2, 2, 2) channel-major.
    consumer = nn.VectorizedLayer(
        name='b', kind=nn.POLYNOMIAL_ACTIVATION, template_id='',
        input_spec=chw, output_spec=chw, inputs=('a',),
        activation_name='square',
    )
    program = nn.VectorizedProgram(
        layers=(producer, consumer), weights=builder,
        input_spec=flat, output='b',
    )
    packed = packing.pack(program)
    # Channel-major (2,2,2) and (8,) put the same element in the same slot.
    self.assertEqual(packed.kinds().count('matvec'), 1)
    self.assertEqual(packed.depth, 2)

  def test_a_genuine_layout_change_emits_a_repack_matrix(self):
    builder, weight = self._weights()
    chw = packing.TensorSpec((2, 2, 2), packing.ChannelMajor())
    muxed = packing.TensorSpec(
        (2, 2, 2), packing.StrideMultiplexed(stride=(2, 2))
    )
    self.assertNotEqual(
        tuple(chw._coordinate_map), tuple(muxed._coordinate_map)
    )
    flat = packing.TensorSpec((8,), packing.ChannelMajor())
    producer = nn.VectorizedLayer(
        name='a', kind=nn.LINEAR_TRANSFORM, template_id='dense',
        input_spec=flat, output_spec=flat, weights={'weight': weight},
    )
    consumer = nn.VectorizedLayer(
        name='b', kind=nn.POLYNOMIAL_ACTIVATION, template_id='',
        input_spec=muxed, output_spec=muxed, inputs=('a',),
        activation_name='square',
    )
    program = nn.VectorizedProgram(
        layers=(producer, consumer), weights=builder,
        input_spec=flat, output='b',
    )
    packed = packing.pack(program)
    del chw
    # The dense layer, plus a repack matrix for the layout change.
    self.assertEqual(packed.kinds().count('matvec'), 2)
    self.assertEqual(packed.depth, 3)
    self.assertTrue(
        any(value_id.startswith('repack')
            for value_id, _, _, _ in packed.operations)
    )

  def test_the_repack_matrix_is_the_permutation_between_the_layouts(self):
    builder, weight = self._weights()
    chw = packing.TensorSpec((2, 2, 2), packing.ChannelMajor())
    muxed = packing.TensorSpec(
        (2, 2, 2), packing.StrideMultiplexed(stride=(2, 2))
    )
    flat = packing.TensorSpec((8,), packing.ChannelMajor())
    program = nn.VectorizedProgram(
        layers=(
            nn.VectorizedLayer(
                name='a', kind=nn.LINEAR_TRANSFORM, template_id='dense',
                input_spec=flat, output_spec=flat, weights={'weight': weight},
            ),
            nn.VectorizedLayer(
                name='b', kind=nn.POLYNOMIAL_ACTIVATION, template_id='',
                input_spec=muxed, output_spec=muxed, inputs=('a',),
                activation_name='square',
            ),
        ),
        weights=builder, input_spec=flat, output='b',
    )
    packed = packing.pack(program)
    repack = next(
        op for op in packed.operations if op[0].startswith('repack')
    )
    matrix = np.zeros((packed.num_slots, packed.num_slots))
    for index, diagonal in repack[3][0].diagonals:
      for row in range(packed.num_slots):
        matrix[row, (row + index) % packed.num_slots] += diagonal[row]
    for source_slot, target_slot in zip(
        flat._coordinate_map, muxed._coordinate_map
    ):
      self.assertEqual(
          matrix[int(target_slot), int(source_slot)], 1.0,
          f'element at slot {source_slot} should move to {target_slot}',
      )
    self.assertEqual(int(matrix.sum()), 8)

  def test_an_impossible_repack_is_refused(self):
    builder = nn.WeightTableBuilder()
    weight = builder.add(np.eye(8))
    flat = packing.TensorSpec((8,), packing.ChannelMajor())
    wider = packing.TensorSpec((16,), packing.ChannelMajor())
    program = nn.VectorizedProgram(
        layers=(
            nn.VectorizedLayer(
                name='a', kind=nn.LINEAR_TRANSFORM, template_id='dense',
                input_spec=flat, output_spec=flat, weights={'weight': weight},
            ),
            nn.VectorizedLayer(
                name='b', kind=nn.POLYNOMIAL_ACTIVATION, template_id='',
                input_spec=wider, output_spec=wider, inputs=('a',),
                activation_name='square',
            ),
        ),
        weights=builder, input_spec=flat, output='b',
    )
    with self.assertRaises(packing.PackingError) as caught:
      packing.pack(program)
    self.assertIn('cannot repack', str(caught.exception))


class NonTerminalOutputTest(absltest.TestCase):
  """VectorizedProgram.output is authoritative, not the last layer.

  A layer the output does not depend on must not be emitted, and must not
  enlarge the depth or the slot demand: doing so would buy a larger, slower
  ring for a value nobody reads.
  """

  def _program(self, tail_depth=1, tail_slots=4):
    builder = nn.WeightTableBuilder()
    weight = builder.add(np.eye(4))
    flat = packing.TensorSpec((4,), packing.ChannelMajor())
    wide = packing.TensorSpec((tail_slots,), packing.ChannelMajor())
    tail_weight = builder.add(np.eye(tail_slots, 4))
    layers = [
        nn.VectorizedLayer(
            name='wanted', kind=nn.LINEAR_TRANSFORM, template_id='dense',
            input_spec=flat, output_spec=flat, weights={'weight': weight},
        ),
    ]
    previous = 'wanted'
    for index in range(tail_depth):
      layers.append(nn.VectorizedLayer(
          name=f'ignored{index}', kind=nn.LINEAR_TRANSFORM,
          template_id='dense', input_spec=flat, output_spec=wide,
          inputs=(previous,), weights={'weight': tail_weight},
      ))
      previous = f'ignored{index}'
    return nn.VectorizedProgram(
        layers=tuple(layers), weights=builder,
        input_spec=flat, output='wanted',
    )

  def test_a_layer_the_output_does_not_reach_is_not_emitted(self):
    packed = packing.pack(self._program())
    self.assertEqual(packed.kinds().count('matvec'), 1)
    self.assertFalse(
        any('ignored' in value_id for value_id, _, _, _ in packed.operations)
    )

  def test_an_unreached_layer_does_not_deepen_the_chain(self):
    shallow = packing.pack(self._program(tail_depth=1))
    deep = packing.pack(self._program(tail_depth=6))
    self.assertEqual(shallow.depth, 1)
    self.assertEqual(deep.depth, 1)
    self.assertEqual(
        int(deep.ring_config.logical_num_q),
        int(shallow.ring_config.logical_num_q),
    )

  def test_an_unreached_layer_does_not_widen_the_ring(self):
    narrow = packing.pack(self._program(tail_slots=4))
    # Chosen to exceed what the depth alone already buys: at depth 1 the
    # smallest secure degree is 8192, holding 4096 slots, so a tail below
    # that would not discriminate.
    self.assertLess(narrow.num_slots, 8192)
    wide = packing.pack(self._program(tail_slots=8192))
    self.assertEqual(wide.num_slots, narrow.num_slots)
    self.assertEqual(wide.ring_config.degree, narrow.ring_config.degree)

  def test_the_named_output_is_used_even_when_it_is_not_last(self):
    builder = nn.WeightTableBuilder()
    weight = builder.add(np.eye(4))
    flat = packing.TensorSpec((4,), packing.ChannelMajor())
    wide = packing.TensorSpec((4,), packing.ChannelMajor())
    program = nn.VectorizedProgram(
        layers=(
            nn.VectorizedLayer(
                name='wanted', kind=nn.LINEAR_TRANSFORM, template_id='dense',
                input_spec=flat, output_spec=wide, weights={'weight': weight},
            ),
            # A second consumer, topologically last but not the output.
            nn.VectorizedLayer(
                name='ignored', kind=nn.POLYNOMIAL_ACTIVATION,
                template_id='', input_spec=wide, output_spec=wide,
                inputs=('wanted',), activation_name='square',
            ),
        ),
        weights=builder, input_spec=flat, output='wanted',
    )
    packed = packing.pack(program)
    self.assertEqual(
        packed.logical_output_shape,
        program.layer('wanted').output_spec.shape,
    )
    self.assertTrue(packed.output.startswith('wanted'))


class SecurityPlanFingerprintTest(absltest.TestCase):
  """The fingerprint covers the whole plan, not just its shape."""

  def test_two_security_targets_do_not_alias(self):
    program = nn.vectorize(_chain(), (1, 2, 2))
    at128 = packing.pack(program, packing.PackingPolicy(security_bits=128))
    at192 = packing.pack(program, packing.PackingPolicy(security_bits=192))
    self.assertNotEqual(at128.fingerprint, at192.fingerprint)

  def test_configs_sharing_a_degree_still_differ(self):
    """Same degree and chain length, different moduli and scaling factor."""
    program = nn.vectorize(_chain(), (1, 2, 2))
    default = packing.pack(program)
    shifted = packing.pack(
        program, packing.PackingPolicy(scaling_mod_size=59, first_mod_size=60)
    )
    self.assertEqual(shifted.ring_config.degree, default.ring_config.degree)
    self.assertEqual(
        int(shifted.ring_config.logical_num_q),
        int(default.ring_config.logical_num_q),
    )
    self.assertNotEqual(
        list(shifted.ring_config.q_towers), list(default.ring_config.q_towers)
    )
    self.assertNotEqual(shifted.fingerprint, default.fingerprint)

  def test_the_fingerprint_payload_carries_the_runtime_fields(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    payload = packing._ring_config_payload(packed.ring_config)
    for field in ('security_bits', 'cost_model', 'q_towers', 'p_towers',
                  'dnum', 'composite_degree', 'scaling_factor', 'sigma',
                  'num_slots', 'degree'):
      self.assertIn(field, payload, f'{field} missing from the fingerprint')

  def test_the_same_policy_still_fingerprints_identically(self):
    program = nn.vectorize(_chain(), (1, 2, 2))
    self.assertEqual(
        packing.pack(program, packing.PackingPolicy()).fingerprint,
        packing.pack(program, packing.PackingPolicy()).fingerprint,
    )


class ProgramProtocolTest(absltest.TestCase):

  def test_a_non_program_is_refused_by_name(self):
    with self.assertRaises(TypeError) as caught:
      packing.pack(object())
    message = str(caught.exception)
    self.assertIn('VectorizedProgram', message)
    self.assertIn('nn.vectorize', message)

  def test_a_torch_model_passed_by_mistake_is_refused(self):
    with self.assertRaises(TypeError) as caught:
      packing.pack(_chain())
    self.assertIn('missing', str(caught.exception))

  def test_a_program_of_foreign_layers_is_refused(self):

    class NotALayer:
      name = 'x'

    class Fake:
      layers = (NotALayer(),)
      weights = None
      input_spec = None
      output = 'x'
      digest = 'd'

      def layer(self, name):
        return None

    with self.assertRaises(TypeError) as caught:
      packing.pack(Fake())
    self.assertIn('vectorized layers', str(caught.exception))


class ArityTest(absltest.TestCase):

  def test_every_emitted_kind_has_a_declared_arity(self):
    packed = packing.pack(nn.vectorize(_residual(), (2, 2, 2)))
    for value_id, kind, inputs, _ in packed.operations:
      self.assertIn(kind, packing._ARITY, f'{kind} has no declared arity')
      self.assertEqual(
          len(inputs), packing._ARITY[kind],
          f'{value_id} ({kind}) has {len(inputs)} inputs',
      )

  def test_level_reduce_is_a_declared_unary_kind(self):
    self.assertEqual(packing._ARITY['level_reduce'], 1)


class LazyConstantsTest(absltest.TestCase):
  """Materialized constants are the default; lazy is an explicit choice."""

  def test_the_default_materializes_sparse_diagonals(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    for _, kind, _, argument in packed.operations:
      if kind == 'matvec':
        self.assertIsInstance(argument[0], packing.SparseDiagonals)
        self.assertEqual(argument[0]._constant_kind, 'sparse')

  def test_lazy_constants_are_opt_in(self):
    self.assertFalse(packing.PackingPolicy().lazy_constants)
    packed = packing.pack(
        nn.vectorize(_chain(), (1, 2, 2)),
        packing.PackingPolicy(lazy_constants=True),
    )
    for _, kind, _, argument in packed.operations:
      if kind == 'matvec':
        self.assertEqual(argument[0]._constant_kind, 'lazy_sparse')

  def test_both_modes_compute_the_same_thing(self):
    program = nn.vectorize(_chain(), (1, 2, 2))
    eager = packing.pack(program)
    lazy = packing.pack(
        program, packing.PackingPolicy(lazy_constants=True)
    )
    generator = np.random.default_rng(5)
    for _ in range(3):
      sample = generator.normal(size=(1, 2, 2))
      np.testing.assert_allclose(
          run_packed(eager, sample)[:3],
          run_packed(lazy, sample)[:3],
          atol=1e-12,
      )

  def test_materialized_diagonals_are_immutable(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    matvec = next(op for op in packed.operations if op[1] == 'matvec')
    for _, diagonal in matvec[3][0].diagonals:
      self.assertFalse(diagonal.flags.writeable)


class FusedBiasTest(absltest.TestCase):
  """A bias added after a fused pool is corrected for padded windows."""

  def _padded(self, count_include_pad):

    class Padded(tnn.Module):

      def __init__(self):
        super().__init__()
        self.conv = tnn.Conv2d(1, 2, 3, padding=1)
        self.pool = tnn.AvgPool2d(
            3, 2, padding=1, count_include_pad=count_include_pad
        )
        self.fc = tnn.Linear(8, 2)

      def forward(self, x):
        x = self.pool(self.conv(x))
        x = x * x
        return self.fc(torch.flatten(x, 1))

    return Padded().double().eval()

  def _check(self, model):
    packed = packing.pack(nn.vectorize(model, (1, 4, 4)))
    self.assertEqual(packed.kinds().count('matvec'), 2)
    generator = np.random.default_rng(7)
    worst = 0.0
    for _ in range(4):
      sample = generator.normal(size=(1, 4, 4))
      with torch.no_grad():
        expected = model(
            torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
        ).numpy().ravel()
      worst = max(
          worst, float(np.max(np.abs(run_packed(packed, sample)[:2] - expected)))
      )
    self.assertLess(worst, 1e-9, f'max deviation {worst:.3e}')

  def test_a_count_include_pad_pool_fused_into_a_conv_matches_torch(self):
    self._check(self._padded(True))

  def test_a_count_exclude_pad_pool_fused_into_a_conv_matches_torch(self):
    self._check(self._padded(False))

  def test_the_correction_is_actually_applied(self):
    """Without it the padded edge biases would be wrong, not merely noisy."""
    model = self._padded(True)
    layer = nn.vectorize(model, (1, 4, 4)).layer('conv')
    factors = packing.layer_bias_factors(layer)
    self.assertIsNotNone(factors)
    self.assertLess(float(np.min(factors)), 1.0)
    self.assertAlmostEqual(float(np.max(factors)), 1.0)


class SlotRoundTripTest(absltest.TestCase):
  """Input and output travel through the layouts, not through a 0..n slice."""

  def _multiplexed_program(self):
    muxed = packing.TensorSpec(
        (2, 2, 2), packing.StrideMultiplexed(stride=(2, 2))
    )
    builder = nn.WeightTableBuilder()
    return nn.VectorizedProgram(
        layers=(nn.VectorizedLayer(
            name='sq', kind=nn.POLYNOMIAL_ACTIVATION, template_id='',
            input_spec=muxed, output_spec=muxed, activation_name='square',
        ),),
        weights=builder, input_spec=muxed, output='sq',
    ), muxed

  def test_the_maps_come_from_the_program_layouts(self):
    program, muxed = self._multiplexed_program()
    packed = packing.pack(program)
    self.assertEqual(
        packed.input_coordinate_map, tuple(muxed._coordinate_map)
    )
    self.assertEqual(
        packed.output_coordinate_map, tuple(muxed._coordinate_map)
    )
    self.assertNotEqual(
        packed.input_coordinate_map, tuple(range(8)),
        'the fixture must not be row-major or it proves nothing',
    )

  def test_a_non_row_major_input_is_scattered_not_sliced(self):
    program, muxed = self._multiplexed_program()
    packed = packing.pack(program)
    sample = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    slots = packed.pack(sample)
    for index, slot in enumerate(muxed._coordinate_map):
      self.assertEqual(slots[int(slot)], float(index))
    self.assertFalse(np.array_equal(slots[:8], sample.reshape(-1)))

  def test_a_non_row_major_round_trip_is_exact(self):
    program, _ = self._multiplexed_program()
    packed = packing.pack(program)
    sample = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    np.testing.assert_array_equal(packed.unpack(packed.pack(sample)), sample)

  def test_a_row_major_round_trip_is_exact(self):
    channel_major = packing.TensorSpec((2, 2, 2), packing.ChannelMajor())
    program = nn.VectorizedProgram(
        layers=(nn.VectorizedLayer(
            name='sq', kind=nn.POLYNOMIAL_ACTIVATION, template_id='',
            input_spec=channel_major, output_spec=channel_major,
            activation_name='square',
        ),),
        weights=nn.WeightTableBuilder(),
        input_spec=channel_major, output='sq',
    )
    packed = packing.pack(program)
    self.assertEqual(packed.input_coordinate_map, tuple(range(8)))
    sample = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    np.testing.assert_array_equal(packed.unpack(packed.pack(sample)), sample)

  def test_a_model_output_gathers_through_its_own_layout(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    slots = np.zeros(packed.num_slots, dtype=np.float64)
    for index, slot in enumerate(packed.output_coordinate_map):
      slots[int(slot)] = float(index) + 1.0
    np.testing.assert_array_equal(
        packed.unpack(slots), np.array([1.0, 2.0, 3.0])
    )

  def test_the_maps_are_in_the_fingerprint(self):
    program, muxed = self._multiplexed_program()
    packed = packing.pack(program)
    payload = packing._json_digest({'x': list(packed.input_coordinate_map)})
    del payload
    channel_major = packing.TensorSpec((2, 2, 2), packing.ChannelMajor())
    other = nn.VectorizedProgram(
        layers=(nn.VectorizedLayer(
            name='sq', kind=nn.POLYNOMIAL_ACTIVATION, template_id='',
            input_spec=channel_major, output_spec=channel_major,
            activation_name='square',
        ),),
        weights=nn.WeightTableBuilder(),
        input_spec=channel_major, output='sq',
    )
    self.assertNotEqual(
        packing.pack(other).fingerprint, packed.fingerprint
    )

  def test_a_small_imaginary_residual_is_accepted(self):
    """CKKS decoding of a real program never returns exactly real values."""
    program, _ = self._multiplexed_program()
    packed = packing.pack(program)
    sample = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    slots = packed.pack(sample).astype(np.complex128)
    slots += 1e-11j
    np.testing.assert_allclose(packed.unpack(slots), sample)

  def test_a_large_imaginary_residual_is_reported(self):
    program, _ = self._multiplexed_program()
    packed = packing.pack(program)
    slots = packed.pack(
        np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    ).astype(np.complex128)
    with self.assertRaises(ValueError) as caught:
      packed.unpack(slots + 5.0j)
    self.assertIn('imaginary residual', str(caught.exception))

  def test_the_tolerance_scales_with_the_magnitude(self):
    program, _ = self._multiplexed_program()
    packed = packing.pack(program)
    big = packed.pack(
        np.full((2, 2, 2), 1e6, dtype=np.float64)
    ).astype(np.complex128)
    # 0.1 absolute is negligible against 1e6 but not against 1.
    packed.unpack(big + 0.1j)
    small = packed.pack(
        np.full((2, 2, 2), 1.0, dtype=np.float64)
    ).astype(np.complex128)
    with self.assertRaises(ValueError):
      packed.unpack(small + 0.1j)

  def test_a_wrong_sized_input_is_refused(self):
    packed = packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    with self.assertRaisesRegex(ValueError, 'expected 4'):
      packed.pack(np.zeros(5))


# The numerical cases inherited from the retired lowering suite. Each is a
# real torch module now, so the reference is torch itself rather than a second
# numpy implementation of the same arithmetic. The awkward shapes are the
# point: asymmetric stride, one-sided padding, a pool that excludes padding
# from its denominator, an adaptive pool whose windows are uneven, and two
# zero-weight cases where the bias is all that survives.
_CASE_SEEDS = {
    'conv': 101, 'pool': 102, 'adaptive': 103,
    'dense': 104, 'zero_dense': 105, 'zero_conv': 106,
}


def _lowering_cases():

  def conv():
    module = tnn.Conv2d(2, 3, (3, 2), stride=(2, 1), padding=(1, 0))
    with torch.no_grad():
      module.weight.copy_(torch.tensor(
          (np.arange(3 * 2 * 3 * 2, dtype=np.float64).reshape(3, 2, 3, 2) - 13)
          / 17
      ))
      module.bias.copy_(torch.tensor([0.25, -0.5, 0.75], dtype=torch.float64))
    return module, (2, 5, 4)

  def pool():
    return tnn.AvgPool2d(
        (2, 3), stride=(2, 1), padding=(0, 1), count_include_pad=False
    ), (2, 5, 4)

  def adaptive():
    return tnn.AdaptiveAvgPool2d((3, 2)), (2, 5, 7)

  def dense():
    module = tnn.Linear(7, 5)
    with torch.no_grad():
      module.weight.copy_(torch.tensor(
          (np.arange(5 * 7, dtype=np.float64).reshape(5, 7) - 16) / 13
      ))
      module.bias.copy_(torch.tensor(
          np.linspace(-0.4, 0.6, 5), dtype=torch.float64
      ))
    return module, (7,)

  def zero_dense():
    """A zero matrix still has to deliver its bias."""
    module = tnn.Linear(4, 3)
    with torch.no_grad():
      module.weight.zero_()
      module.bias.copy_(torch.tensor([0.25, -0.5, 1.25], dtype=torch.float64))
    return module, (4,)

  def zero_conv():
    module = tnn.Conv2d(1, 2, (1, 2))
    with torch.no_grad():
      module.weight.zero_()
      module.bias.copy_(torch.tensor([0.75, -1.25], dtype=torch.float64))
    return module, (1, 2, 3)

  return {
      'conv': conv, 'pool': pool, 'adaptive': adaptive, 'dense': dense,
      'zero_dense': zero_dense, 'zero_conv': zero_conv,
  }


def _lowering_fixture(builder):
  """(model, packed program, input shape) for one lowering case."""

  class Single(tnn.Module):

    def __init__(self, inner):
      super().__init__()
      self.inner = inner

    def forward(self, x):
      return self.inner(x)

  module, shape = builder()
  model = Single(module).double().eval()
  program = nn.vectorize(model, shape)
  packed = packing.pack(program, packing.PackingPolicy(lazy_constants=True))
  return model, program, packed, shape


class SparseLoweringTest(parameterized.TestCase):
  """Numerical and allocation invariants inherited from the lowering suite.

  These are about the matrices packing emits: that they compute what torch
  computes on awkward geometry, that they stay sparse, and that generating
  them never materializes the slot square.
  """

  @parameterized.named_parameters(
      (name, name) for name in _lowering_cases()
  )
  def test_the_packed_matrix_matches_torch(self, case_name):
    model, _, packed, shape = _lowering_fixture(_lowering_cases()[case_name])
    generator = np.random.default_rng(_CASE_SEEDS[case_name])
    worst = 0.0
    for _ in range(4):
      sample = generator.normal(size=shape)
      with torch.no_grad():
        expected = model(
            torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
        ).numpy().reshape(-1)
      actual = packed.unpack(run_packed(packed, sample)).reshape(-1)
      self.assertEqual(actual.size, expected.size)
      worst = max(worst, float(np.max(np.abs(actual - expected))))
    self.assertLess(worst, 1e-12, f'{case_name} deviates by {worst:.3e}')

  @parameterized.named_parameters(
      (name, name) for name in _lowering_cases()
  )
  def test_the_matrix_stays_sparse(self, case_name):
    _, _, packed, _ = _lowering_fixture(_lowering_cases()[case_name])
    matrices = [
        argument[0] for _, kind, _, argument in packed.operations
        if kind == 'matvec'
    ]
    self.assertLen(matrices, 1)
    diagonals = matrices[0].source.as_dict()
    self.assertLess(
        len(diagonals), packed.num_slots,
        f'{case_name} emitted a dense diagonal set',
    )

  @parameterized.named_parameters(
      ('zero_dense', 'zero_dense'), ('zero_conv', 'zero_conv'),
  )
  def test_a_zero_weight_still_delivers_its_bias(self, case_name):
    """The matrix vanishes; the bias must not vanish with it."""
    model, _, packed, shape = _lowering_fixture(_lowering_cases()[case_name])
    matrices = [
        argument[0] for _, kind, _, argument in packed.operations
        if kind == 'matvec'
    ]
    self.assertEmpty(
        matrices[0].source.as_dict(), 'a zero matrix has no diagonals'
    )
    biases = [
        argument for _, kind, _, argument in packed.operations
        if kind == 'add_plain'
    ]
    self.assertLen(biases, 1)
    sample = np.zeros(shape)
    with torch.no_grad():
      expected = model(
          torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
      ).numpy().reshape(-1)
    np.testing.assert_allclose(
        packed.unpack(run_packed(packed, sample)).reshape(-1),
        expected,
        atol=1e-13,
    )
    self.assertGreater(float(np.max(np.abs(expected))), 0.0)

  @parameterized.named_parameters(
      (name, name) for name in _lowering_cases()
  )
  def test_generating_diagonals_never_allocates_the_slot_square(
      self, case_name
  ):
    _, _, packed, _ = _lowering_fixture(_lowering_cases()[case_name])
    square = packed.num_slots * packed.num_slots
    tripped = []
    originals = {
        name: getattr(np, name) for name in ('empty', 'full', 'ones', 'zeros')
    }

    def spy(name):
      original = originals[name]

      def wrapper(shape, *args, **keywords):
        size = shape if isinstance(shape, int) else int(np.prod(shape))
        if size >= square:
          tripped.append((name, size))
        return original(shape, *args, **keywords)

      return wrapper

    with mock.patch.object(np, 'zeros', spy('zeros')), \
         mock.patch.object(np, 'ones', spy('ones')), \
         mock.patch.object(np, 'full', spy('full')), \
         mock.patch.object(np, 'empty', spy('empty')):
      for _, kind, _, argument in packed.operations:
        if kind == 'matvec':
          argument[0].source.as_dict()
    self.assertEmpty(
        tripped,
        f'{case_name} allocated a {packed.num_slots}x{packed.num_slots} matrix',
    )

  def test_the_allocation_guard_detects_a_slot_square(self):
    """Otherwise the guard above could pass by never running."""
    _, _, packed, _ = _lowering_fixture(_lowering_cases()['conv'])
    square = packed.num_slots * packed.num_slots
    tripped = []
    original = np.zeros

    def wrapper(shape, *args, **keywords):
      size = shape if isinstance(shape, int) else int(np.prod(shape))
      if size >= square:
        tripped.append(size)
      return original(shape, *args, **keywords)

    with mock.patch.object(np, 'zeros', wrapper):
      np.zeros((packed.num_slots, packed.num_slots))
    self.assertLen(tripped, 1)

  @parameterized.named_parameters(
      (name, name) for name in _lowering_cases()
  )
  def test_diagonal_subset_and_maxima_agree_with_the_full_set(
      self, case_name
  ):
    _, _, packed, _ = _lowering_fixture(_lowering_cases()[case_name])
    source = next(
        argument[0].source for _, kind, _, argument in packed.operations
        if kind == 'matvec'
    )
    everything = source.as_dict()
    if not everything:
      self.skipTest('a zero matrix has no diagonals to subset')
    requested = tuple(sorted(everything)[:3])
    native = source.materialize_diagonals(requested)
    for index in requested:
      np.testing.assert_allclose(native[index], everything[index], atol=0)
    maxima = source.diagonal_maxima()
    for index, diagonal in everything.items():
      self.assertAlmostEqual(
          maxima[index], float(np.max(np.abs(diagonal))), places=12
      )


class UserTemplateExtensionTest(absltest.TestCase):
  """A user can add a template without editing either module.

  The frontend registration says when the template applies; the packing
  registration says what it means physically. Both are public, and neither
  requires touching shipped code -- which is the promise
  nn.register_layout_template makes.
  """

  TEMPLATE = 'user_wide_dense_test'

  def tearDown(self):
    super().tearDown()
    nn.unregister_layout_template(self.TEMPLATE)
    packing.unregister_template_lowering(self.TEMPLATE)

  def _model(self):

    class Wide(tnn.Module):

      def __init__(self):
        super().__init__()
        self.fc = tnn.Linear(16, 4)

      def forward(self, x):
        return self.fc(x * x)

    torch.manual_seed(3)
    return Wide().double().eval()

  def _register_frontend(self, threshold):
    return nn.register_layout_template(nn.LayoutTemplate(
        template_id=self.TEMPLATE,
        version='1',
        family='dense',
        kind=nn.LINEAR_TRANSFORM,
        depth_cost=1,
        targets=(('call_module', 'torch.nn.Linear'),),
        weight_roles=('weight', 'bias'),
        priority=100,
        condition=lambda parameters, input_spec: (
            input_spec is not None and input_spec.physical_size >= threshold
        ),
    ))

  def test_a_user_template_vectorizes_and_packs(self):
    self._register_frontend(threshold=16)
    packing.register_template_lowering(self.TEMPLATE, alias='dense', version='1')

    model = self._model()
    program = nn.vectorize(model, (16,))
    self.assertEqual(program.layer('fc').template_id, self.TEMPLATE)

    packed = packing.pack(program)
    self.assertEqual(packed.kinds().count('matvec'), 1)

    generator = np.random.default_rng(31)
    for _ in range(3):
      sample = generator.normal(size=(16,)) * 0.25
      with torch.no_grad():
        expected = model(
            torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
        ).numpy().reshape(-1)
      np.testing.assert_allclose(
          packed.unpack(run_packed(packed, sample)).reshape(-1),
          expected,
          atol=1e-12,
      )

  def test_the_condition_still_selects_between_templates(self):
    """Above the threshold the user template wins; below it, the default."""
    self._register_frontend(threshold=1024)
    packing.register_template_lowering(self.TEMPLATE, alias='dense', version='1')
    program = nn.vectorize(self._model(), (16,))
    self.assertEqual(program.layer('fc').template_id, 'dense')
    self.assertEqual(packing.pack(program).kinds().count('matvec'), 1)

  def test_a_template_without_a_lowering_fails_to_pack_by_name(self):
    """The gap this API closes: it must report, not crash obscurely."""
    self._register_frontend(threshold=16)
    program = nn.vectorize(self._model(), (16,))
    with self.assertRaises(packing.PackingError) as caught:
      packing.pack(program)
    self.assertIn(self.TEMPLATE, str(caught.exception))
    self.assertIn('no lowering', str(caught.exception))

  def test_a_lowering_version_mismatch_is_refused(self):
    self._register_frontend(threshold=16)
    packing.register_template_lowering(self.TEMPLATE, alias='dense', version='1')
    program = nn.vectorize(self._model(), (16,))
    packing.register_template_lowering(
        self.TEMPLATE, alias='dense', version='2', replace=True
    )
    with self.assertRaisesRegex(packing.PackingError, 'version'):
      packing.pack(program)

  def test_registering_twice_needs_replace_and_a_new_version(self):
    packing.register_template_lowering(self.TEMPLATE, alias='dense', version='1')
    with self.assertRaisesRegex(packing.PackingError, 'already registered'):
      packing.register_template_lowering(self.TEMPLATE, alias='dense', version='2')
    with self.assertRaisesRegex(packing.PackingError, 'new version'):
      packing.register_template_lowering(
          self.TEMPLATE, alias='dense', version='1', replace=True
      )
    packing.register_template_lowering(
        self.TEMPLATE, alias='dense', version='2', replace=True
    )
    self.assertEqual(packing.template_lowering(self.TEMPLATE).version, '2')

  def test_unregistering_restores_the_shipped_behaviour(self):
    self._register_frontend(threshold=16)
    packing.register_template_lowering(self.TEMPLATE, alias='dense', version='1')
    self.assertEqual(
        nn.vectorize(self._model(), (16,)).layer('fc').template_id,
        self.TEMPLATE,
    )
    nn.unregister_layout_template(self.TEMPLATE)
    packing.unregister_template_lowering(self.TEMPLATE)
    program = nn.vectorize(self._model(), (16,))
    self.assertEqual(program.layer('fc').template_id, 'dense')
    self.assertEqual(packing.pack(program).kinds().count('matvec'), 1)

  def test_a_callable_recipe_factory_is_an_extension_point(self):
    """A genuinely new realization, owing nothing to our recipe classes.

    The object below implements the recipe protocol from scratch -- digest,
    entries in non-decreasing row order, and diagonal_entries for a requested
    subset -- and computes a transposed dense product by hand. If packing
    accepts it and the packed circuit reproduces that arithmetic, the
    extension point is real rather than a re-entry into our own lowering.
    """
    self._register_frontend(threshold=16)

    class _TransposeRecipe:
      """y[j] = sum_i W[j, i] * x[i], enumerated directly."""

      def __init__(self, weight, input_spec, output_spec):
        self._weight = np.asarray(weight, dtype=np.float64)
        self._input_spec = input_spec
        self._output_spec = output_spec
        self.digest = 'transpose-recipe-' + nn.weight_digest(
            nn.normalize_weight(self._weight)
        )

      def _cell(self, row, column):
        return float(self._weight[row, column])

      def entries(self):
        outputs, inputs = self._weight.shape
        for row in range(outputs):  # non-decreasing row order
          slot_row = self._output_spec.coordinate_to_slot((row,))
          for column in range(inputs):
            value = self._cell(row, column)
            if value:
              yield (
                  slot_row,
                  self._input_spec.coordinate_to_slot((column,)),
                  value,
              )

      def diagonal_entries(self, indices, dimension):
        if not indices:
          return
        outputs, inputs = self._weight.shape
        for row in range(outputs):
          slot_row = self._output_spec.coordinate_to_slot((row,))
          for diagonal in indices:
            column = (slot_row + diagonal) % dimension
            if column >= inputs:
              continue
            value = self._cell(row, column)
            if value:
              yield diagonal, slot_row, value

    built = []

    def factory(layer, weights):
      built.append(layer.name)
      resolved = layer.resolve_weights(weights)
      return _TransposeRecipe(
          resolved['weight'], layer.input_spec, layer.output_spec
      )

    packing.register_template_lowering(
        self.TEMPLATE,
        packing.TemplateLowering(version='1', recipe_factory=factory),
    )
    model = self._model()
    packed = packing.pack(nn.vectorize(model, (16,)))
    self.assertEqual(built, ['fc'])
    self.assertEqual(packed.kinds().count('matvec'), 1)

    # The factory's own arithmetic is what runs: weight @ x, plus the bias
    # the frontend still supplies through its AddPlain.
    weight = model.fc.weight.detach().numpy().astype(np.float64)
    bias = model.fc.bias.detach().numpy().astype(np.float64)
    generator = np.random.default_rng(57)
    for _ in range(3):
      sample = generator.normal(size=(16,)) * 0.25
      np.testing.assert_allclose(
          packed.unpack(run_packed(packed, sample)).reshape(-1),
          weight @ (sample * sample) + bias,
          atol=1e-12,
      )

  def test_a_factory_recipe_must_expose_a_stable_digest(self):
    """The digest reaches the packed fingerprint, so it cannot be per-run."""
    self._register_frontend(threshold=16)

    class _NoDigest:

      def entries(self):
        return iter(())

      def diagonal_entries(self, indices, dimension):
        return iter(())

    packing.register_template_lowering(
        self.TEMPLATE,
        packing.TemplateLowering(
            version='1', recipe_factory=lambda layer, weights: _NoDigest()
        ),
    )
    with self.assertRaises((TypeError, AttributeError, ValueError)):
      packing.pack(nn.vectorize(self._model(), (16,)))

  def test_a_lowering_must_supply_exactly_one_realization(self):
    with self.assertRaisesRegex(packing.PackingError, 'exactly one'):
      packing.TemplateLowering(version='1')
    with self.assertRaisesRegex(packing.PackingError, 'exactly one'):
      packing.TemplateLowering(
          version='1', recipe_class=object, recipe_factory=lambda *_: None
      )
    with self.assertRaisesRegex(packing.PackingError, 'version'):
      packing.TemplateLowering(version='', recipe_class=object)

  def test_aliasing_an_unknown_lowering_is_refused(self):
    with self.assertRaisesRegex(packing.PackingError, 'cannot alias'):
      packing.register_template_lowering(self.TEMPLATE, alias='no_such_lowering')


class ImportIsolationTest(absltest.TestCase):

  def test_importing_packing_loads_neither_he_params_nor_jax(self):
    script = (
        'import sys\n'
        'import jaxite_word.packing as packing\n'
        'heavy = [name for name in ("he_params", "jax", "torch",'
        ' "jaxite_word.ckks_ctx") if name in sys.modules]\n'
        'assert not heavy, heavy\n'
        'assert hasattr(packing, "pack")\n'
        'print("clean")\n'
    )
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, cwd=root,
    )
    self.assertEqual(
        result.returncode, 0, f'{result.stdout}\n{result.stderr}'
    )
    self.assertIn('clean', result.stdout)

  def test_pack_loads_he_params_when_called(self):
    packing.pack(nn.vectorize(_chain(), (1, 2, 2)))
    self.assertIn('he_params', sys.modules)


class EquivalenceTest(absltest.TestCase):
  """The packed PP-ops compute what the torch model computes."""

  def _check(self, model, shape, samples=4, tolerance=1e-9):
    program = nn.vectorize(model, shape)
    packed = packing.pack(program)
    generator = np.random.default_rng(0)
    worst = 0.0
    outputs = int(np.prod(program.layers[-1].output_spec.shape))
    for _ in range(samples):
      sample = generator.normal(size=shape) * 0.5
      with torch.no_grad():
        expected = model(
            torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
        ).numpy().ravel()
      actual = run_packed(packed, sample)[:outputs]
      worst = max(worst, float(np.max(np.abs(actual - expected))))
    self.assertLess(worst, tolerance, f'max deviation {worst:.3e}')

  def test_a_chain_matches_torch(self):
    self._check(_chain(), (1, 2, 2))

  def test_a_residual_dag_matches_torch(self):
    self._check(_residual(), (2, 2, 2))

  def test_fused_pooling_matches_torch(self):
    self._check(_pooled(), (1, 4, 4))

  def test_unfused_pooling_matches_fused_pooling(self):
    model = _pooled()
    fused = packing.pack(nn.vectorize(model, (1, 4, 4)))
    separate = packing.pack(
        nn.vectorize(model, (1, 4, 4), fuse_pooling=False)
    )
    generator = np.random.default_rng(1)
    for _ in range(4):
      sample = generator.normal(size=(1, 4, 4)) * 0.5
      np.testing.assert_allclose(
          run_packed(fused, sample)[:2],
          run_packed(separate, sample)[:2],
          atol=1e-9,
      )



# =============================================================================
# Public packing API contracts
# =============================================================================

class PackingSurfaceTest(parameterized.TestCase):

  def test_internal_lowering_annotations_resolve(self):
    """Postponed annotations must still support runtime introspection."""
    argument_names = {
        '_conv_output_spec': 'layer',
        '_pool_output_spec': 'layer',
        '_adaptive_output_spec': 'layer',
        '_pool_window': 'layer',
        '_conv_entries': 'layer',
        '_pool_entries': 'layer',
        '_adaptive_entries': 'layer',
        '_dense_entries': 'layer',
        '_fused_conv_pool_entries': 'convolution',
    }
    for function_name, argument_name in argument_names.items():
      with self.subTest(function=function_name):
        hints = typing.get_type_hints(getattr(packing, function_name))
        self.assertIn(argument_name, hints)

  @parameterized.parameters(
      'Module', 'Conv2d', 'Dense', 'AvgPool2d', 'AdaptiveAvgPool2d',
      'Flatten', 'Sequential', 'Identity', 'Rotate', 'Rescale', 'MulPlain',
      'AddPlain', 'ParallelSum', 'Square', 'Bootstrap', 'Linear', 'Add',
      'Sub', 'Mul', 'PackedProgram', 'DEFAULT_BSGS_RATIO',
  )
  def test_a_retired_packing_name_is_absent(self, name):
    self.assertFalse(
        hasattr(packing, name), f'packing.{name} is still reachable'
    )

  def test_no_private_name_is_exported(self):
    """Whatever the recipes are built from stays an implementation detail.

    Which private helpers exist is packing's business and free to change;
    what matters is that none of them is public, so no second model-building
    API can appear by accident.
    """
    for name in packing.__all__:
      self.assertFalse(
          name.startswith('_'), f'packing.__all__ exports private {name}'
      )

  def test_one_canonical_result_type(self):
    self.assertTrue(hasattr(packing, 'Packing'))
    self.assertIn('Packing', packing.__all__)
    self.assertFalse(hasattr(packing, 'PackedProgram'))

  def test_every_exported_name_resolves(self):
    for name in packing.__all__:
      self.assertTrue(hasattr(packing, name), f'__all__ names missing {name}')

  def test_packing_freezes_no_global_bsgs_ratio(self):
    import dataclasses
    fields = {f.name for f in dataclasses.fields(packing.PackingPolicy)}
    self.assertNotIn('bsgs_ratio', fields)


if __name__ == '__main__':
  absltest.main()
