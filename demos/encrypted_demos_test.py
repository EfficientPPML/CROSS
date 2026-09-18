"""The demos go through the canonical pipeline, once each, and nothing else.

Constructing a real Mapping over a demo's own ring is not something a test
can afford. Key generation is not the reason -- that is fast, well under a
second even at degree 16384. The cost is Mapping initialization and the
physical constants: materializing BSGS diagonals for a matvec over 8192 slots
and building the per-level operator controls reached ~29 GB resident before
it was killed. So the call-count tests patch Mapping, and the one test that
exercises the real constructor uses a hand-built Packing sized for an
8-slot context, on a ring that is deliberately INSECURE.
"""

import unittest.mock as mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

# Runs from ``demos/`` with no PYTHONPATH, as the README documents: the
# library's flat modules (``nn``, ``packing``) live in ``../jaxite_word``.
import os
import sys
_JAXITE_WORD_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jaxite_word')
)
if _JAXITE_WORD_DIR not in sys.path:
  sys.path.insert(0, _JAXITE_WORD_DIR)

torch = None
nn = None
packing = None
encrypted_demos = None
canonical_demo = None


def setUpModule():
  global torch, nn, packing, encrypted_demos, canonical_demo
  try:
    import torch as torch_module
  except ImportError as error:  # pragma: no cover - environment dependent
    raise absltest.SkipTest(f'PyTorch unavailable: {error}')
  import nn as nn_module
  import packing as packing_module
  import canonical_demo as canonical_module
  import encrypted_demos as demos_module
  torch = torch_module
  nn = nn_module
  packing = packing_module
  canonical_demo = canonical_module
  encrypted_demos = demos_module


_EXPECTED = {
    'lenet': dict(depth=7, degree=32768, matvec=4, square=3, shape=(1, 28, 28)),
    'lola': dict(depth=5, degree=32768, matvec=3, square=2, shape=(1, 28, 28)),
    'alexnet-tiny': dict(
        depth=7, degree=32768, matvec=4, square=3, shape=(3, 16, 16)
    ),
    'alexnet-full': dict(
        depth=15, degree=65536, matvec=8, square=7, shape=(3, 16, 16)
    ),
}


class PlanTest(parameterized.TestCase):
  """build_plan is the cheap half: model -> vectorize -> pack."""

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_a_demo_plans_to_its_expected_ring(self, name):
    demo = encrypted_demos.DEMOS[name]()
    packed = demo.build_plan()
    expected = _EXPECTED[name]
    self.assertEqual(packed.depth, expected['depth'])
    self.assertEqual(int(packed.ring_config.degree), expected['degree'])
    self.assertEqual(packed.kinds().count('matvec'), expected['matvec'])
    self.assertEqual(packed.kinds().count('square'), expected['square'])
    self.assertEqual(demo.input_shape, expected['shape'])

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_a_demo_exposes_all_three_artifacts(self, name):
    demo = encrypted_demos.DEMOS[name]()
    self.assertIsInstance(demo.model, torch.nn.Module)
    self.assertIsInstance(demo.vectorized_program, nn.VectorizedProgram)
    self.assertIsInstance(demo.packed_program, packing.Packing)
    self.assertIs(demo.ring_config, demo.packed_program.ring_config)

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_planning_never_builds_a_mapping(self, name):
    """Inspecting a demo must not materialize constants or build a context."""
    demo = encrypted_demos.DEMOS[name]()
    with mock.patch('mapping.Mapping') as mapping_class:
      demo.build_plan()
      _ = demo.vectorized_program
      _ = demo.packed_program
      _ = demo.metadata()
      mapping_class.assert_not_called()
    self.assertFalse(demo.is_mapped())

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_the_plan_is_deterministic(self, name):
    first = encrypted_demos.DEMOS[name]().build_plan()
    second = encrypted_demos.DEMOS[name]().build_plan()
    self.assertEqual(first.fingerprint, second.fingerprint)

  def test_demos_plan_lazily_while_the_packer_stays_eager(self):
    demo = encrypted_demos.LeNetDemo()
    for _, kind, _, argument in demo.build_plan().operations:
      if kind == 'matvec':
        self.assertEqual(argument[0]._constant_kind, 'lazy_sparse')
    self.assertFalse(packing.PackingPolicy().lazy_constants)


class CallCountTest(parameterized.TestCase):
  """vectorize, pack and Mapping are each invoked exactly once."""

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_each_stage_runs_once_per_demo(self, name):
    demo = encrypted_demos.DEMOS[name]()
    with mock.patch.object(
        canonical_demo.nn, 'vectorize', wraps=nn.vectorize
    ) as vectorize, mock.patch.object(
        canonical_demo.packing, 'pack', wraps=packing.pack
    ) as pack, mock.patch('mapping.Mapping') as mapping_class:
      demo.build_plan()
      demo.build_plan()
      _ = demo.vectorized_program
      _ = demo.packed_program
      demo.materialize_mapping()
      demo.materialize_mapping()
      _ = demo.mapping
      self.assertEqual(vectorize.call_count, 1)
      self.assertEqual(pack.call_count, 1)
      self.assertEqual(mapping_class.call_count, 1)

  def test_the_mapping_is_constructed_from_the_packed_program(self):
    demo = encrypted_demos.LeNetDemo()
    with mock.patch('mapping.Mapping') as mapping_class:
      demo.materialize_mapping()
    (positional, _) = mapping_class.call_args
    self.assertIs(positional[0], demo.packed_program)
    self.assertLen(positional, 1, 'Mapping derives its ring from the program')

  def test_scheduling_knobs_reach_the_mapping_and_nothing_else(self):
    demo = encrypted_demos.LeNetDemo(global_batch=1, compile_mode='whole')
    before = demo.build_plan().fingerprint
    other = encrypted_demos.LeNetDemo(
        global_batch=1, compile_mode='operations', bsgs_n_jobs=2
    )
    # Execution policy does not change the program or the ring.
    self.assertEqual(other.build_plan().fingerprint, before)
    with mock.patch('mapping.Mapping') as mapping_class:
      other.materialize_mapping()
    _, keywords = mapping_class.call_args
    self.assertEqual(keywords['compile_mode'], 'operations')
    self.assertEqual(keywords['bsgs_n_jobs'], 2)


class LegacyBuilderTest(parameterized.TestCase):
  """No demo can construct a packing graph by hand: the names are gone.

  Spying on them is no longer possible, and that is the point -- a builder
  kept alive to be spied on would be a second implementation of what
  packing.pack derives from the traced model.
  """

  _RETIRED = ('Sequential', 'Rotate', 'MulPlain', 'ParallelSum', 'AddPlain',
              'Linear', 'Identity', 'Square', 'Rescale', 'Bootstrap',
              'Add', 'Sub', 'Mul')

  def test_the_retired_builders_are_absent_from_packing(self):
    for builder in self._RETIRED:
      self.assertFalse(
          hasattr(packing, builder), f'packing.{builder} came back'
      )

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_a_demo_plans_without_naming_a_retired_builder(self, name):
    import ast
    import pathlib as _pathlib
    demo = encrypted_demos.DEMOS[name]()
    demo.build_plan()
    root = _pathlib.Path(encrypted_demos.__file__).parent
    for module in ('encrypted_demos.py', 'canonical_demo.py'):
      tree = ast.parse((root / module).read_text())
      named = {
          node.attr for node in ast.walk(tree)
          if isinstance(node, ast.Attribute)
      } | {
          node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
      }
      for builder in self._RETIRED:
        self.assertNotIn(builder, named, f'{module} names {builder}')


class ProvenanceTest(parameterized.TestCase):
  """A demo says where its parameters came from."""

  def test_lola_reports_a_shipped_checkpoint(self):
    demo = encrypted_demos.LoLADemo()
    self.assertTrue(demo.provenance.is_checkpoint)
    self.assertNotEmpty(demo.provenance.files)
    self.assertIn('checkpoint', demo.metadata()['weights_source'])
    self.assertTrue(demo.metadata()['has_shipped_checkpoint'])

  @parameterized.named_parameters(
      ('lenet', 'lenet'),
      ('alexnet_tiny', 'alexnet-tiny'),
      ('alexnet_full', 'alexnet-full'),
  )
  def test_a_demo_without_a_checkpoint_says_so(self, name):
    demo = encrypted_demos.DEMOS[name]()
    self.assertEqual(demo.provenance.source, 'seeded-random')
    self.assertFalse(demo.metadata()['has_shipped_checkpoint'])
    described = demo.provenance.describe()
    self.assertIn('NO CHECKPOINT SHIPPED', described)
    self.assertIn('no accuracy meaning', described)

  def test_lola_validates_its_binary_shapes(self):
    demo = encrypted_demos.LoLADemo()
    state = demo.model.state_dict()
    self.assertEqual(tuple(state['conv1.weight'].shape), (5, 1, 2, 2))
    self.assertEqual(tuple(state['fc1.weight'].shape), (100, 980))
    self.assertEqual(tuple(state['fc2.weight'].shape), (10, 100))
    for tensor in state.values():
      self.assertEqual(tensor.dtype, torch.float64)
      self.assertTrue(bool(torch.isfinite(tensor).all()))

  def test_lola_rejects_a_mis_shaped_checkpoint(self):
    demo = encrypted_demos.LoLADemo()
    broken = demo.build_model()
    broken.fc1 = torch.nn.Linear(979, 100).double()
    with self.assertRaisesRegex(ValueError, 'fc1.weight'):
      demo._validate_shapes(broken)

  def test_seeded_parameters_are_reproducible(self):
    self.assertEqual(
        encrypted_demos.LeNetDemo().vectorized_program.digest,
        encrypted_demos.LeNetDemo().vectorized_program.digest,
    )


class CallerSuppliedWeightTest(parameterized.TestCase):
  """Caller-supplied arrays populate the model nn.vectorize actually traces."""

  def _random_for(self, demo, generator):
    state = demo.model.state_dict()
    return [
        generator.normal(size=tuple(state[parameter].shape)) * 0.05
        for _, parameter in demo.weight_order
    ]

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_supplied_weights_change_the_program(self, name):
    demo = encrypted_demos.DEMOS[name]()
    before = demo.build_plan().fingerprint
    generator = np.random.default_rng(11)
    demo.load_positional_weights(*self._random_for(demo, generator))
    self.assertNotEqual(demo.build_plan().fingerprint, before)

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_supplied_weights_reach_the_traced_model(self, name):
    demo = encrypted_demos.DEMOS[name]()
    generator = np.random.default_rng(12)
    arrays = self._random_for(demo, generator)
    demo.load_positional_weights(*arrays)
    state = demo.model.state_dict()
    for array, (_, parameter) in zip(arrays, demo.weight_order):
      np.testing.assert_allclose(
          state[parameter].detach().numpy(), array, atol=1e-12
      )

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_provenance_reports_caller_supplied_afterwards(self, name):
    demo = encrypted_demos.DEMOS[name]()
    original = demo.provenance.source
    generator = np.random.default_rng(13)
    demo.load_positional_weights(*self._random_for(demo, generator))
    self.assertNotEqual(demo.provenance.source, original)
    self.assertEqual(demo.provenance.source, 'caller-supplied')
    self.assertEqual(demo.metadata()['weights_source'], 'caller-supplied')
    self.assertFalse(demo.metadata()['has_shipped_checkpoint'])

  def test_lola_stops_claiming_a_shipped_checkpoint(self):
    """The most dangerous case: a real checkpoint replaced by other arrays."""
    demo = encrypted_demos.LoLADemo()
    self.assertTrue(demo.provenance.is_checkpoint)
    generator = np.random.default_rng(14)
    demo.load_positional_weights(*self._random_for(demo, generator))
    self.assertFalse(demo.provenance.is_checkpoint)
    self.assertEqual(demo.metadata()['weights_source'], 'caller-supplied')

  def test_a_none_bias_becomes_explicit_zeros(self):
    """Skipping it would silently keep the fallback or checkpoint bias."""
    import torch
    demo = encrypted_demos.LeNetDemo()
    with torch.no_grad():
      for parameter in ('conv1.bias', 'conv2.bias', 'fc1.bias', 'fc2.bias'):
        demo.model.state_dict()[parameter].fill_(7.0)
    generator = np.random.default_rng(15)
    weights = self._random_for(demo, generator)[:4]
    demo.load_positional_weights(*weights, None, None, None, None)
    state = demo.model.state_dict()
    for parameter in ('conv1.bias', 'conv2.bias', 'fc1.bias', 'fc2.bias'):
      np.testing.assert_array_equal(
          state[parameter].detach().numpy(),
          np.zeros(tuple(state[parameter].shape)),
      )

  def test_a_none_bias_is_zero_before_vectorize_sees_it(self):
    import torch
    demo = encrypted_demos.LeNetDemo()
    with torch.no_grad():
      demo.model.state_dict()['fc2.bias'].fill_(3.0)
    generator = np.random.default_rng(16)
    weights = self._random_for(demo, generator)[:4]
    demo.load_positional_weights(*weights, None, None, None, None)
    packed = demo.build_plan()
    bias_ops = [op for op in packed.operations if op[1] == 'add_plain']
    self.assertNotEmpty(bias_ops)
    np.testing.assert_array_equal(
        bias_ops[-1][3].values, np.zeros(packed.num_slots)
    )

  def test_a_wrong_arity_is_refused(self):
    demo = encrypted_demos.LeNetDemo()
    with self.assertRaisesRegex(ValueError, 'expected 8 arrays'):
      demo.load_positional_weights(np.zeros(1))

  def test_an_unknown_parameter_name_is_refused(self):
    demo = encrypted_demos.LeNetDemo()
    with self.assertRaisesRegex(ValueError, 'no parameter'):
      demo.load_weights({'conv9.weight': np.zeros(4)})

  def test_a_non_finite_array_is_refused(self):
    demo = encrypted_demos.LeNetDemo()
    values = np.zeros((4, 1, 5, 5))
    values[0, 0, 0, 0] = np.inf
    with self.assertRaisesRegex(ValueError, 'non-finite'):
      demo.load_weights({'conv1.weight': values})

  def test_loading_weights_discards_the_previous_plan(self):
    demo = encrypted_demos.LeNetDemo()
    demo.build_plan()
    with mock.patch('mapping.Mapping'):
      demo.materialize_mapping()
    self.assertTrue(demo.is_mapped())
    generator = np.random.default_rng(17)
    demo.load_positional_weights(*self._random_for(demo, generator))
    self.assertFalse(demo.is_mapped())


class RingPlanIntegrityTest(absltest.TestCase):
  """The derived ring's own parameters cannot be contradicted at execution."""

  def _tiny_packed(self):

    class Tiny(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

      def forward(self, x):
        return self.fc(x * x)

    torch.manual_seed(0)
    return packing.pack(
        nn.vectorize(Tiny().double().eval(), (4,)),
        packing.PackingPolicy(lazy_constants=True),
    )

  def test_a_contradicting_dnum_is_refused(self):
    import mapping as mapping_module
    packed = self._tiny_packed()
    planned = int(packed.ring_config.dnum)
    with self.assertRaises(ValueError) as caught:
      mapping_module.Mapping(packed, dnum=planned + 1, perf_test=True)
    message = str(caught.exception)
    self.assertIn(f'dnum={planned}', message)
    self.assertIn('validated', message)

  def test_restating_the_planned_dnum_is_allowed(self):
    """It must not be refused merely for being explicit."""
    import mapping as mapping_module
    packed = self._tiny_packed()
    planned = int(packed.ring_config.dnum)
    with mock.patch.object(
        mapping_module, 'ring_runtime_parameters'
    ) as derive:
      derive.side_effect = RuntimeError('reached parameter derivation')
      with self.assertRaisesRegex(RuntimeError, 'reached parameter'):
        mapping_module.Mapping(packed, dnum=planned, perf_test=True)

  def test_a_demo_forwards_no_legacy_dnum_default(self):
    """AlexNetFull must derive its own dnum, not inherit a legacy 5."""
    demo = encrypted_demos.AlexNetFullDemo()
    self.assertNotIn('dnum', demo._scheduling)
    with mock.patch('mapping.Mapping') as mapping_class:
      demo.materialize_mapping()
    _, keywords = mapping_class.call_args
    self.assertNotIn('dnum', keywords)

  def test_each_demo_plans_its_own_dnum(self):
    planned = {
        name: int(encrypted_demos.DEMOS[name]().ring_config.dnum)
        for name in _EXPECTED
    }
    # AlexNetFull is deeper, so its chain and digit count differ from the rest.
    self.assertNotEqual(planned['alexnet-full'], planned['lenet'])
    for name, value in planned.items():
      self.assertGreaterEqual(value, 1, name)


class BsgsRatioTest(absltest.TestCase):
  """A legacy global BSGS ratio is refused, never silently dropped.

  Packing no longer records one: the baby-step/giant-step split is chosen per
  matvec by Mapping, from the diagonals that operation actually has. A ratio
  fixed at packing time would decide it for every matvec in the program at
  once, so the argument has nowhere honest to go.
  """

  def test_packing_records_no_ratio(self):
    import dataclasses
    self.assertNotIn(
        'bsgs_ratio',
        {field.name for field in dataclasses.fields(packing.PackingPolicy)},
    )

  def test_a_packed_matvec_leaves_the_split_open(self):
    packed = encrypted_demos.LeNetDemo().build_plan()
    for _, kind, _, argument in packed.operations:
      if kind == 'matvec':
        self.assertEqual(argument[1:], (None, None, None, None))

  def test_a_nondefault_ratio_is_refused(self):
    with self.assertRaises(ValueError) as caught:
      canonical_demo.reject_legacy_bsgs_ratio(bsgs_ratio=4.0)
    message = str(caught.exception)
    self.assertIn('per matvec', message)
    self.assertIn('Mapping', message)

  def test_the_historic_default_is_accepted(self):
    canonical_demo.reject_legacy_bsgs_ratio(bsgs_ratio=2.0)
    canonical_demo.reject_legacy_bsgs_ratio(fc1_bsgs_ratio=2.0,
                                            fc2_bsgs_ratio=2.0)
    canonical_demo.reject_legacy_bsgs_ratio(bsgs_ratio=None)


class ClientPreprocessingTest(absltest.TestCase):
  """The 32->16 downsample stays outside encryption."""

  def test_the_encrypted_model_starts_at_16x16(self):
    for name in ('alexnet-tiny', 'alexnet-full'):
      demo = encrypted_demos.DEMOS[name]()
      self.assertEqual(demo.input_shape, (3, 16, 16))
      self.assertFalse(demo.model.client_downsample)

  def test_the_client_downsample_matches_torch_average_pooling(self):
    import torch.nn.functional as functional
    generator = np.random.default_rng(0)
    sample = generator.normal(size=(3, 32, 32))
    expected = functional.avg_pool2d(
        torch.tensor(sample, dtype=torch.float64).unsqueeze(0), 2, 2
    ).numpy().reshape(3, 16, 16)
    np.testing.assert_allclose(
        encrypted_demos.downsample_for_client(sample), expected, atol=1e-12
    )

  def test_downsampling_is_not_part_of_the_packed_program(self):
    demo = encrypted_demos.AlexNetTinyDemo()
    packed = demo.build_plan()
    # Four matvecs: three fused conv+pool and the final dense. A client-side
    # pool done under encryption would add a fifth and a level.
    self.assertEqual(packed.kinds().count('matvec'), 4)
    self.assertEqual(packed.depth, 7)


class CleartextTest(parameterized.TestCase):
  """The demo's cleartext path is the torch model itself."""

  @parameterized.named_parameters(
      (name.replace('-', '_'), name) for name in _EXPECTED
  )
  def test_cleartext_matches_the_module_forward(self, name):
    demo = encrypted_demos.DEMOS[name]()
    generator = np.random.default_rng(3)
    sample = generator.normal(size=demo.input_shape) * 0.25
    with torch.no_grad():
      expected = demo.model(
          torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
      ).numpy().reshape(-1)
    np.testing.assert_allclose(demo.cleartext(sample), expected, atol=1e-12)

  def test_lola_cleartext_matches_the_pre_migration_reference(self):
    """The migrated demo must compute what the old cleartext path computed."""
    import lola_he
    demo = encrypted_demos.LoLADemo()
    weights = lola_he.load_trained()
    generator = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5):
      sample = generator.random(784)
      reference = np.asarray(
          lola_he.lola_cleartext_inference(
              sample, *(weights[key] for key in
                        ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))
          ),
          dtype=np.float64,
      )
      migrated = demo.cleartext(sample.reshape(1, 28, 28))
      worst = max(worst, float(np.max(np.abs(migrated - reference))))
    self.assertLess(worst, 1e-9, f'max deviation {worst:.3e}')


class InsecureSmokeTest(absltest.TestCase):
  """One real Mapping, on a tiny hand-built program and an INSECURE ring.

  Deliberately not a packed demo. A Packing from packing.pack is at
  least 8192 slots wide, and materializing one matvec constant at that width
  costs tens of gigabytes -- the bottleneck is initialization and the physical
  constants, not key generation. So the fixture here is built by hand at the
  width the context can actually hold. It proves the constructor and binder
  boundary, and nothing about the cryptography: a degree-16 ring offers no
  security whatsoever.
  """

  _Q_TOWERS = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
  _P_TOWERS = [1073740609, 1073739937, 1073739649]
  _NUM_SLOTS = 8
  _DEGREE = 16

  def _tiny_packed(self):
    """A hand-built square-then-scale program over 8 slots."""
    scale = np.full(self._NUM_SLOTS, 0.5, dtype=np.float64)
    diagonals = ((0, np.full(self._NUM_SLOTS, 2.0, dtype=np.float64)),)
    physical = (self._NUM_SLOTS,)
    # Level-preserving on purpose. A multiplicative chain would need a
    # scale tuned against these moduli -- the tower ends in a ~2**19 prime,
    # so a rescale divides by far less than the scale -- and tuning a toy
    # chain proves nothing this test is for. What the packed arithmetic
    # computes is covered by the cleartext PP-op tests in packing_test; what is
    # covered here is that a Packing reaches the real constructor,
    # binder, executor and decoder intact.
    del diagonals
    operations = (
        (
            'out', 'add_plain', ('input',),
            packing.PlainSlots(
                values=scale, shape=physical, packing='slots'
            ),
        ),
    )

    class _InsecureRing:
      """Stands in for a RingConfig; this ring was never security-derived."""

      degree = self._DEGREE
      num_slots = self._NUM_SLOTS
      dnum = 3

    return packing.Packing(
        operations=operations,
        ring_config=_InsecureRing(),
        input_shape=physical,
        input_packing='slots',
        logical_input_shape=(self._NUM_SLOTS,),
        logical_output_shape=(self._NUM_SLOTS,),
        input_coordinate_map=tuple(range(self._NUM_SLOTS)),
        output_coordinate_map=tuple(range(self._NUM_SLOTS)),
        layer_shapes=(),
        output='out',
        num_slots=self._NUM_SLOTS,
        depth=0,
    )

  def _insecure_parameters(self):
    import key_gen
    keys = key_gen.gen_pke_pair(
        self._Q_TOWERS, self._P_TOWERS, self._DEGREE
    )
    return packing.test_only_parameters({
        'degree': self._DEGREE, 'num_slots': self._NUM_SLOTS,
        # ~2**30, matched to the moduli: CKKS wants the scale close to the
        # modulus it rescales by, or a squaring overflows the chain.
        'scaling_factor': float(1 << 30),
        'output_scale': float(1 << 30),
        'q_towers': self._Q_TOWERS, 'p_towers': self._P_TOWERS, 'p': 30,
        'CKKS_M_FACTOR': 1, 'max_bits_in_word': 61,
        'noise_scale_degree': 1, 'composite_degree': 1,
        'public_key': keys['public_key'], 'secret_key': keys['secret_key'],
        'degree_layout': (4, 4),
    })

  def test_release_evicts_codec_caches_and_refuses_execution(self):
    import ckks_ctx as cc
    import mapping as mapping_module

    packed = self._tiny_packed()
    mapping = mapping_module.Mapping(
        packed, self._insecure_parameters(), dnum=3, perf_test=True
    )
    sample = np.array([0.25, -0.5, 0.75, 1.0, 0.0, 0.125, -0.25, 0.5])
    np.asarray(mapping.infer(sample))
    context_id = id(mapping.ctx)
    self.assertTrue(
        any(key[0] == context_id for key in cc._encrypt_cache)
        or any(key[0] == context_id for key in cc._decrypt_cache),
        'inference should have populated a codec cache for this context',
    )

    mapping.release()

    self.assertFalse(any(key[0] == context_id for key in cc._encrypt_cache))
    self.assertFalse(any(key[0] == context_id for key in cc._decrypt_cache))
    self.assertIsNone(mapping.ctx)
    for call in (
        lambda: mapping.infer(sample),
        lambda: mapping.encrypt_input(sample),
        lambda: mapping.estimate_live_memory(),
        lambda: mapping.context,
    ):
      with self.assertRaisesRegex(RuntimeError, 'released'):
        call()
    mapping.release()  # idempotent

  def test_a_mapping_builds_on_a_test_only_ring(self):
    import mapping as mapping_module

    packed = self._tiny_packed()
    mapping = mapping_module.Mapping(
        packed, self._insecure_parameters(), dnum=3, perf_test=True
    )
    self.assertIs(mapping.packing, packed)
    self.assertEqual(mapping.ctx.degree, self._DEGREE)
    self.assertEqual(mapping.ctx.num_slots, self._NUM_SLOTS)
    self.assertIn('out', mapping.value_specs)

  def test_the_bound_program_computes_what_it_declares(self):
    import mapping as mapping_module

    packed = self._tiny_packed()
    mapping = mapping_module.Mapping(
        packed, self._insecure_parameters(), dnum=3, perf_test=True
    )
    sample = np.array([0.25, -0.5, 0.75, 1.0, 0.0, 0.125, -0.25, 0.5])
    recovered = np.asarray(mapping.infer(sample)).reshape(-1)
    np.testing.assert_allclose(recovered, sample + 0.5, atol=1e-3)

  def test_an_unmarked_parameter_override_is_refused(self):
    import mapping as mapping_module
    packed = self._tiny_packed()
    with self.assertRaisesRegex(ValueError, 'securely derived ring'):
      mapping_module.Mapping(packed, {'degree': 16}, perf_test=True)


class LegacyEntrypointTest(parameterized.TestCase):
  """The old public classes run the canonical pipeline and nothing else.

  These are the entrypoints callers actually have, so the guarantees have to
  hold here rather than only on the new registry.
  """

  def _cases(self):
    import lenet_he
    import lola_he
    import alexnet_he
    generator = np.random.default_rng(0)

    def draw(shapes):
      return [generator.normal(size=shape) * 0.05 for shape in shapes]

    lola_weights = lola_he.load_trained()
    return {
        'lenet': (
            lenet_he.LeNetHE,
            draw([(4, 1, 5, 5), (8, 4, 5, 5), (32, 392), (10, 32)])
            + [None] * 4,
            {'bsgs_ratio': 2.0},
        ),
        'lola': (
            lola_he.LoLAHE,
            [lola_weights[key]
             for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')],
            {'fc1_bsgs_ratio': 2.0, 'fc2_bsgs_ratio': 2.0},
        ),
        'alexnet_tiny': (
            alexnet_he.AlexNetTinyHE,
            draw([(4, 3, 3, 3), (4,), (8, 4, 3, 3), (8,),
                  (16, 8, 3, 3), (16,), (10, 64), (10,)]),
            {'bsgs_ratio': 2.0},
        ),
        'alexnet_full': (
            alexnet_he.AlexNetHE,
            draw([(8, 3, 3, 3), (8,), (16, 8, 3, 3), (16,),
                  (32, 16, 3, 3), (32,), (32, 32, 3, 3), (32,),
                  (32, 32, 3, 3), (32,), (64, 128), (64,),
                  (32, 64), (32,), (10, 32), (10,)]),
            {'bsgs_ratio': 2.0},
        ),
    }

  def _prepare(self, cls, arrays, options):
    import demo_test_utils
    with mock.patch.object(
        canonical_demo.nn, 'vectorize', wraps=nn.vectorize
    ) as vectorize, mock.patch.object(
        canonical_demo.packing, 'pack', wraps=packing.pack
    ) as pack, demo_test_utils.fake_mapping() as mapping_class:
      model = cls()
      model.precompute_plaintexts(*arrays, **options)
      return model, vectorize, pack, mapping_class

  @parameterized.named_parameters(
      ('lenet', 'lenet'), ('lola', 'lola'),
      ('alexnet_tiny', 'alexnet_tiny'), ('alexnet_full', 'alexnet_full'),
  )
  def test_each_stage_runs_exactly_once(self, name):
    cls, arrays, options = self._cases()[name]
    model, vectorize, pack, mapping_class = self._prepare(cls, arrays, options)
    self.assertEqual(vectorize.call_count, 1)
    self.assertEqual(pack.call_count, 1)
    self.assertEqual(mapping_class.call_count, 1)
    self.assertIs(mapping_class.call_args[0][0], model.packed_program)

  @parameterized.named_parameters(
      ('lenet', 'lenet'), ('lola', 'lola'),
      ('alexnet_tiny', 'alexnet_tiny'), ('alexnet_full', 'alexnet_full'),
  )
  def test_caller_weights_are_not_discarded(self, name):
    """The failure a call-count test cannot see: the fallback compiled."""
    cls, arrays, options = self._cases()[name]
    first, _, _, _ = self._prepare(cls, arrays, options)
    halved = [
        array * 0.5 if hasattr(array, 'shape') else array for array in arrays
    ]
    second, _, _, _ = self._prepare(cls, halved, options)
    self.assertNotEqual(
        second.packed_program.fingerprint, first.packed_program.fingerprint
    )

  @parameterized.named_parameters(
      ('lenet', 'lenet'), ('lola', 'lola'),
      ('alexnet_tiny', 'alexnet_tiny'), ('alexnet_full', 'alexnet_full'),
  )
  def test_a_legacy_bsgs_ratio_is_refused_not_dropped(self, name):
    cls, arrays, options = self._cases()[name]
    ratios = {key: 4.0 for key in options}
    with self.assertRaisesRegex(ValueError, 'per matvec'):
      self._prepare(cls, arrays, ratios)

  @parameterized.named_parameters(
      ('lenet', 'lenet'), ('lola', 'lola'),
      ('alexnet_tiny', 'alexnet_tiny'), ('alexnet_full', 'alexnet_full'),
  )
  def test_mapping_backed_access_is_refused_before_precompute(self, name):
    cls, _, _ = self._cases()[name]
    with self.assertRaises(RuntimeError):
      _ = cls().mapping

  @parameterized.named_parameters(
      ('lenet', 'lenet'), ('lola', 'lola'),
      ('alexnet_tiny', 'alexnet_tiny'), ('alexnet_full', 'alexnet_full'),
  )
  def test_no_legacy_builder_is_constructed(self, name):
    """The names are gone, so a legacy graph is unconstructable by any path.

    Stronger than a spy: a spy proves one run did not call them; absence
    proves no run can.
    """
    cls, arrays, options = self._cases()[name]
    self._prepare(cls, arrays, options)
    for builder in ('Sequential', 'Rotate', 'MulPlain', 'ParallelSum',
                    'AddPlain', 'Linear'):
      self.assertFalse(
          hasattr(packing, builder), f'packing.{builder} came back'
      )



# =============================================================================
# Retired performance-driver surface contracts
# =============================================================================

class ActivePerfDriverTest(parameterized.TestCase):
  """The perf drivers are off the cache, and the wrappers still refuse it.

  `save_cache` and `from_cache` raise: the cache existed to avoid rebuilding
  a hand-lowered graph and its keys, and the canonical pipeline re-derives the
  plan from the torch model instead. A driver that still reaches for them looks
  like coverage while skipping, which is how they survived the migration once.
  """

  _RETIRED = ('from_cache', 'save_cache')

  # The three drivers that used to build and reload the cache. They may not
  # mention it at all -- not a call, not a definition, not a getattr string.
  _PERF_DRIVERS = (
      'lenet_he_perf_test.py', 'lola_he_perf_test.py',
      'alexnet_he_perf_test.py',
  )
  # The wrappers keep the entry points so an old caller gets an explanation.
  # Defining one is the point; calling one is the bug.
  _WRAPPERS = (
      'lenet_he.py', 'lola_he.py', 'alexnet_he.py', 'canonical_demo.py',
      'encrypted_demos.py',
  )

  def _demos_dir(self):
    import pathlib
    return pathlib.Path(__file__).resolve().parent.parent / 'demos'

  def _tree(self, name):
    """Parse a demo file. A missing file fails; it never skips."""
    import ast
    path = self._demos_dir() / name
    self.assertTrue(path.exists(), f'demos/{name} is missing')
    return ast.parse(path.read_text(), filename=str(path))

  @parameterized.parameters(*_PERF_DRIVERS)
  def test_a_perf_driver_never_names_a_cache_api(self, name):
    """Every identifier occurrence counts -- no defined-method loophole."""
    import ast
    tree = self._tree(name)
    named = {}
    for node in ast.walk(tree):
      if isinstance(node, ast.Attribute):
        named.setdefault(node.attr, node.lineno)
      elif isinstance(node, ast.Name):
        named.setdefault(node.id, node.lineno)
      elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
        named.setdefault(node.name, node.lineno)
      elif isinstance(node, ast.keyword) and node.arg:
        named.setdefault(node.arg, node.lineno)
      elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        # Closes getattr(obj, 'save_cache'). Prose is a longer string, so
        # a docstring that discusses the cache is not a match.
        named.setdefault(node.value, node.lineno)
    for retired in self._RETIRED:
      self.assertNotIn(
          retired, named,
          f'demos/{name}:{named.get(retired)} names {retired}, which raises',
      )

  @parameterized.parameters(*_WRAPPERS)
  def test_a_wrapper_may_define_a_refusal_but_never_calls_one(self, name):
    import ast
    tree = self._tree(name)
    called = set()
    for node in ast.walk(tree):
      if not isinstance(node, ast.Call):
        continue
      if isinstance(node.func, ast.Attribute):
        called.add(node.func.attr)
      elif isinstance(node.func, ast.Name):
        called.add(node.func.id)
    for retired in self._RETIRED:
      self.assertNotIn(
          retired, called, f'demos/{name} calls {retired}, which raises'
      )

  @parameterized.parameters(*_PERF_DRIVERS)
  def test_a_perf_driver_names_no_cache_file(self, name):
    """Prose survives an identifier sweep; a `.pkl` path is still a lie."""
    path = self._demos_dir() / name
    text = path.read_text()
    for number, line in enumerate(text.splitlines(), start=1):
      self.assertNotIn(
          '.pkl', line, f'demos/{name}:{number} names a cache file'
      )

  def test_the_lenet_cli_offers_no_retired_option(self):
    """`--cache`, `--dnum` and `--mux` were accepted and then ignored."""
    import ast
    tree = self._tree('lenet_he_perf_test.py')
    flags = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.startswith('--')
    }
    for retired in ('--cache', '--dnum', '--mux', '--no-mux', '--reload'):
      self.assertNotIn(retired, flags, f'the LeNet CLI still takes {retired}')

  def test_the_wrapper_entry_points_still_refuse(self):
    """An old caller gets an explanation, not an AttributeError.

    This runs in a subprocess: importing a wrapper needs `demos/` on
    `sys.path`, and a test that mutates the shared path changes what every
    later test in this process imports.
    """
    import pathlib
    import subprocess
    import sys
    import textwrap
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(self._demos_dir())!r})
        sys.path.insert(0, {str(self._demos_dir().parent)!r})
        import lenet_he
        calls = (
            ('from_cache',
             lambda: lenet_he.LeNetHE.from_cache('/nonexistent/legacy.pkl')),
            ('save_cache',
             lambda: lenet_he.LeNetHE().save_cache('/nonexistent/out.pkl')),
        )
        for label, call in calls:
          try:
            call()
          except (ValueError, NotImplementedError):
            print(label + '=refused')
          else:
            print(label + '=ACCEPTED')
    """)
    result = subprocess.run(
        [sys.executable, '-B', '-c', script],
        capture_output=True, text=True, timeout=600, check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
    self.assertEqual(
        result.stdout.split(), ['from_cache=refused', 'save_cache=refused']
    )



if __name__ == '__main__':
  absltest.main()
