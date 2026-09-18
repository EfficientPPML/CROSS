"""Focused contracts for Packing analysis and Mapping-owned compilation."""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
import sys
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

# This file imports ``jaxite_word`` as a package, so the repository root must
# be importable even when it is launched the documented way from inside
# jaxite_word/ (``python3 mapping_compile_test.py``), where only this
# directory is on the path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from jaxite_word import CKKSContext, Mapping, Packing, nn
from jaxite_word import mapping as mapping_mod
from jaxite_word import ckks_ctx as cc
from jaxite_word import packing as packing_ir
import polynomial

try:
  import pytest
except ModuleNotFoundError:
  pytest = None


def _parameters(
    *,
    max_level=4,
    composite_degree=1,
    num_slots=8,
    scaling_factor=float(1 << 20),
):
  q_count = max_level * composite_degree + 1
  return {
      'degree': 2 * num_slots,
      'num_slots': num_slots,
      'q_towers': [1_000_003 + 2 * index for index in range(q_count)],
      'p_towers': [],
      'composite_degree': composite_degree,
      'scaling_factor': scaling_factor,
      'output_scale': scaling_factor,
      'degree_layout': (4, (2 * num_slots) // 4),
  }


def _context(*, batch=2, **parameter_kwargs):
  return CKKSContext(
      packing_ir.test_only_parameters(_parameters(**parameter_kwargs)),
      batch=batch,
      dnum=3,
  )


def _plain(
    value=1.0,
    *,
    shape=(8,),
    physical_slots=8,
    packing='slots',
    scale=None,
):
  return packing_ir.PlainSlots(
      np.full(physical_slots, value, dtype=np.float64),
      shape=shape,
      packing=packing,
      scale=scale,
  )


class RingRuntimeParametersTest(absltest.TestCase):

  def test_default_device_placement_uses_widest_divisible_prefix(self):
    devices = tuple(f'device-{index}' for index in range(8))

    self.assertEqual(
        mapping_mod._device_prefix_for_batch(32, devices), devices
    )
    self.assertEqual(
        mapping_mod._device_prefix_for_batch(6, devices), devices[:6]
    )
    self.assertEqual(
        mapping_mod._device_prefix_for_batch(10, devices), devices[:5]
    )

  def test_default_device_placement_avoids_empty_or_unequal_shards(self):
    devices = tuple(f'device-{index}' for index in range(8))

    self.assertEqual(
        mapping_mod._device_prefix_for_batch(3, devices), devices[:3]
    )
    self.assertEqual(
        mapping_mod._device_prefix_for_batch(11, devices), devices[:1]
    )
    self.assertEqual(
        mapping_mod._device_prefix_for_batch(1, devices), devices[:1]
    )

  def test_default_device_placement_requires_a_visible_device(self):
    with self.assertRaisesRegex(RuntimeError, 'no local devices'):
      mapping_mod._device_prefix_for_batch(8, ())

  def test_large_ring_uses_balanced_ntt_layout(self):
    ring = SimpleNamespace(
        runtime_kwargs=lambda: {
            'degree': 32768,
            'num_slots': 16384,
            'q_towers': (2147483489,),
            'p_towers': (2147483137,),
            'sigma': 3.19,
            'composite_degree': 1,
            'scaling_factor': 1 << 20,
        }
    )
    keys = {'public_key': object(), 'secret_key': object()}

    parameters = mapping_mod.ring_runtime_parameters(ring, keys=keys)

    self.assertEqual(parameters['degree_layout'], (128, 256))

  def test_multi_device_aot_input_crosses_an_explicit_host_boundary(self):
    payload = jnp.arange(16, dtype=jnp.uint32).reshape(8, 2)

    single_device = mapping_mod._aot_pmap_input(payload, 1)
    multi_device = mapping_mod._aot_pmap_input(payload, 8)

    self.assertIs(single_device, payload)
    self.assertIsInstance(multi_device, np.ndarray)
    np.testing.assert_array_equal(multi_device, np.asarray(payload))


class MappingMemoryEstimateTest(absltest.TestCase):

  def test_estimate_uses_mapping_layout_plan_and_global_batch(self):
    mapping = object.__new__(Mapping)
    mapping.global_batch = 2
    mapping.ctx = SimpleNamespace(
        q_towers=(17, 13, 11), p_towers=(7,), dnum=2
    )
    input_spec = mapping_mod._ValueSpec(
        level=2,
        batch=1,
        degree_layout=(4, 4),
        moduli=(17, 13, 11),
        shape=(2,),
        packing='slots',
        scale=16.0,
        nsd=1,
    )
    output_spec = mapping_mod._ValueSpec(
        level=1,
        batch=1,
        degree_layout=(4, 4),
        moduli=(17, 13),
        shape=(2,),
        packing='slots',
        scale=16.0,
        nsd=1,
    )
    mapping.value_specs = {
        'input': input_spec,
        'matvec': output_spec,
        'output': output_spec,
    }
    mapping.operation_plans = {
        'matvec': mapping_mod._MatVecPlan(
            n=16,
            n1=4,
            active_diagonal_indices=(0, 1, 5),
            pt_scale=1.0,
        ),
        'output': None,
    }
    mapping.required_rotation_indices = (1, 4, 5)
    mapping.packing = SimpleNamespace(
        output='output',
        operations=(
            ('matvec', 'matvec', ('input',), object()),
            ('output', 'add_plain', ('matvec',), object()),
        ),
    )

    estimate = mapping.estimate_live_memory()

    self.assertEqual(estimate['ciphertext_in_bytes'], 768)
    self.assertEqual(estimate['ciphertext_out_bytes'], 512)
    self.assertEqual(estimate['matvec_constants_bytes'], 576)
    self.assertEqual(estimate['plaintext_constants_bytes'], 128)
    self.assertEqual(estimate['evaluation_key_bytes'], 1024)
    self.assertEqual(estimate['rotation_keys_bytes'], 3072)
    self.assertEqual(estimate['rotation_key_count'], 3)
    self.assertEqual(estimate['total_bytes'], 6080)


def _sparse_matrix(
    indices=(0, 1, 5),
    *,
    input_shape=(8,),
    output_shape=(8,),
    output_packing=None,
    zero=False,
):
  return packing_ir.SparseDiagonals(
      dimension=8,
      diagonals=tuple(
          (index, np.full(8, 0.0 if zero else index + 1.0, dtype=np.float64))
          for index in indices
      ),
      input_shape=input_shape,
      output_shape=output_shape,
      packing='slots',
      output_packing=output_packing,
  )


# ---------------------------------------------------------------------------
# Fixtures: canonical packing.Packing values, built from explicit operations.
#
# The planner contract is what these tests are about -- what each PP-op does
# to level, scale and noise-scale degree; how regions fuse; when a buffer
# dies. That contract is unchanged by the restructure, so the assertions are
# unchanged too. What changed is how a fixture is written: a Packing is now
# built from the operations it contains, rather than lowered from a tree of
# declarative modules that no longer exists.
# ---------------------------------------------------------------------------


class _Step:
  """One operation in a fixture, before ids and inputs are assigned."""

  def __init__(self, kind, argument=None, name=None, branches=None):
    self.kind = kind
    self.argument = argument
    self.name = name
    self.branches = branches


def _identity(name=None):
  return _Step('identity', name=name)


def _square(name=None):
  return _Step('square', name=name)


def _rotate(index, name=None):
  return _Step('rotate', index, name)


def _rescale(steps=1, name=None):
  return _Step('rescale', int(steps), name)


def _add_plain(constant, name=None):
  return _Step('add_plain', constant, name)


def _mul_plain(constant, name=None):
  return _Step('mul_plain', constant, name)


def _bootstrap(level_budget=(4, 4), secret_key_dist='uniform_ternary',
               mode='single', precision=None, name=None):
  return _Step(
      'bootstrap', (level_budget, secret_key_dist, mode, precision), name
  )


def _linear(matrix, bias=None, n1=None, n2=None, bsgs_ratio=None,
            pt_scale=None, name=None):
  return _Step(
      'matvec', (matrix, n1, n2, bsgs_ratio, pt_scale), name,
      branches={'bias': bias},
  )


def _sequence(*steps, **_ignored):
  """Flatten nested sequences, mirroring how a tree used to compose."""
  flat = []
  for step in steps:
    flat.extend(step if isinstance(step, list) else [step])
  return flat


def _branch(kind, left, right, name=None):
  return _Step(kind, None, name, branches={'left': left, 'right': right})


def _add(left, right, name=None):
  return _branch('add', left, right, name)


def _sub(left, right, name=None):
  return _branch('sub', left, right, name)


def _mul(left, right, name=None):
  return _branch('mul', left, right, name)


def _parallel_sum(*branches, name=None):
  return _Step('parallel_sum', None, name, branches={'arms': list(branches)})


class _StubRing:
  """Stands in for a RingConfig: these fixtures bring their own context."""

  def __init__(self, num_slots, dnum=3):
    self.degree = 2 * num_slots
    self.num_slots = num_slots
    self.dnum = dnum


def _emit(steps, operations, source, counter):
  """Append ``steps`` to ``operations``, threading the value they consume."""
  current = source
  for step in steps if isinstance(steps, list) else [steps]:
    counter[0] += 1
    value_id = step.name or f'{step.kind}_{counter[0]}'
    if step.kind == 'identity':
      continue
    if step.kind == 'parallel_sum':
      arms = [
          _emit(arm, operations, current, counter)
          for arm in step.branches['arms']
      ]
      total = arms[0]
      for index, arm in enumerate(arms[1:], start=1):
        counter[0] += 1
        joined = value_id if index == len(arms) - 1 else f'{value_id}_add_{index}'
        operations.append((joined, 'add', (total, arm), None))
        total = joined
      current = total
      continue
    if step.branches and 'left' in step.branches:
      left = _emit(step.branches['left'], operations, current, counter)
      right = _emit(step.branches['right'], operations, current, counter)
      operations.append((value_id, step.kind, (left, right), None))
      current = value_id
      continue
    if step.kind == 'matvec' and step.name:
      value_id = f'{step.name}_matvec'
    operations.append((value_id, step.kind, (current,), step.argument))
    current = value_id
    bias = (step.branches or {}).get('bias')
    if bias is not None:
      counter[0] += 1
      bias_id = f'{step.name or value_id}_bias'
      operations.append((bias_id, 'add_plain', (current,), bias))
      current = bias_id
  return current


def _model(*layers, context_shape=(8,), num_slots=8, **kwargs):
  """A canonical Packing containing exactly the operations described.

  ``context_shape`` is the logical shape of one sample; ``num_slots`` is the
  ciphertext width, which matches the context these fixtures run against.
  """
  del kwargs
  operations = []
  output = _emit(_sequence(*layers), operations, 'input', [0])
  if not operations:
    # An identity-only fixture still needs a value to name as its output.
    operations.append(('identity', 'rotate', ('input',), 0))
    output = 'identity'
  return packing_ir.Packing(
      operations=tuple(operations),
      ring_config=_StubRing(num_slots),
      input_shape=(num_slots,),
      input_packing='slots',
      logical_input_shape=tuple(context_shape),
      logical_output_shape=tuple(context_shape),
      input_coordinate_map=tuple(range(int(math.prod(context_shape)))),
      output_coordinate_map=tuple(range(int(math.prod(context_shape)))),
      layer_shapes=(),
      output=output,
      num_slots=num_slots,
      depth=0,
  )


def _analyzed_mapping(
    model,
    ctx,
    *,
    input_packing='slots',
    input_scale=None,
    input_nsd=1,
    headroom=0,
    perf_test=False,
    global_batch=None,
    device_count=1,
    compile_mode='whole',
    bsgs_n_jobs=1,
    bsgs_streaming=False,
):
  """Analyze without key generation, constant encoding, or JAX lowering."""
  del input_packing
  return _analyzed_packing(
      model,
      ctx,
      input_scale=input_scale,
      input_nsd=input_nsd,
      headroom=headroom,
      perf_test=perf_test,
      global_batch=global_batch,
      device_count=device_count,
      compile_mode=compile_mode,
      bsgs_n_jobs=bsgs_n_jobs,
      bsgs_streaming=bsgs_streaming,
  )


def _analyzed_packing(
    packing,
    ctx,
    *,
    input_scale=None,
    input_nsd=1,
    headroom=0,
    perf_test=False,
    global_batch=None,
    device_count=1,
    compile_mode='whole',
    bsgs_n_jobs=1,
    bsgs_streaming=False,
):
  """Analyze an existing Packing without key generation or JAX lowering."""
  mapping = Mapping.__new__(Mapping)
  mapping.packing = packing
  mapping.input_scale = input_scale
  mapping.input_nsd = input_nsd
  mapping.headroom = headroom
  mapping.ctx = ctx
  mapping.global_batch = (
      ctx.batch if global_batch is None else int(global_batch)
  )
  mapping.devices = ()
  mapping.device_count = int(device_count)
  mapping.per_device_batch = ctx.batch
  mapping.compile_mode = compile_mode
  mapping.bsgs_n_jobs = bsgs_n_jobs
  mapping.bsgs_streaming = bsgs_streaming
  mapping.bindings = {}
  mapping._compiled = None
  mapping._compiled_regions = None
  mapping._templates = {}
  mapping_mod._analyze_mapping(mapping, perf_test=perf_test)
  return mapping


def _ciphertext(ctx, spec, *, scale=None, nsd=None):
  value = polynomial.Polynomial.from_array(
      jnp.zeros(spec.payload_shape, dtype=jnp.uint32),
      {
          'batch': spec.batch,
          'num_elements': spec.elements,
          'degree': ctx.degree,
          'num_moduli': spec.num_moduli,
          'precision': 32,
          'degree_layout': spec.degree_layout,
      },
      {
          'moduli': list(spec.moduli),
          'ntt_ctx': SimpleNamespace(ff_ctx=SimpleNamespace()),
      },
  )
  value._ckks_scale = spec.scale if scale is None else scale
  value._ckks_nsd = spec.nsd if nsd is None else nsd
  return value


class PlanningTest(absltest.TestCase):

  def test_recursive_scale_table_initializes_on_a_fresh_cache(self):
    cache = cc.HEParameterCache.__new__(cc.HEParameterCache)
    cache.q_towers = [101, 103, 107, 109, 113]
    cache.composite_degree = 2
    cache.num_q = len(cache.q_towers)
    cache.max_level = 2

    squared_scale = cache.scaling_factor_real_big(1)
    recursive_scale = cache.scaling_factor_recursive(1)

    self.assertTrue(math.isclose(
        squared_scale, recursive_scale**2, rel_tol=1e-12
    ))
    self.assertLen(cache._scaling_factors_real, len(cache.q_towers))
    self.assertLen(
        cache._scaling_factors_real_big, len(cache.q_towers) - 1
    )

  def test_preinit_recursive_scale_rejects_lossy_or_invalid_parameters(self):
    scale_for = cc.HEParameterCache.recursive_scaling_factor_for

    for q_towers in ([101.5, 103], [True, 103]):
      with self.subTest(q_towers=q_towers):
        with self.assertRaisesRegex(TypeError, 'positive ints'):
          scale_for(q_towers, 1, 0)
    with self.assertRaisesRegex(ValueError, 'positive ints'):
      scale_for([101, 0], 1, 0)
    for composite_degree, error in (
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (2, ValueError),
    ):
      with self.subTest(composite_degree=composite_degree):
        with self.assertRaises(error):
          scale_for([101, 103], composite_degree, 0)

  def test_arithmetic_propagates_level_moduli_scale_and_nsd(self):
    ctx = _context()
    max_level = ctx.max_level
    input_scale = float(ctx.scaling_factor)

    add_mapping = _analyzed_mapping(_model(
        _add_plain(_plain(), name='add_plain')
    ), ctx)
    self.assertEqual(add_mapping.output_spec, add_mapping.input_spec)

    mul_plain_mapping = _analyzed_mapping(_model(
        _mul_plain(_plain(scale=2.0), name='mul_plain')
    ), ctx)
    self.assertEqual(mul_plain_mapping.output_spec.level, max_level)
    self.assertEqual(mul_plain_mapping.output_spec.scale, input_scale * 2.0)
    self.assertEqual(mul_plain_mapping.output_spec.nsd, 2)

    rescale_mapping = _analyzed_mapping(_model(
        _mul_plain(_plain(scale=2.0), name='mul_plain'),
        _rescale(name='rescale'),
    ), ctx)
    self.assertEqual(rescale_mapping.output_spec.level, max_level - 1)
    self.assertEqual(
        rescale_mapping.output_spec.moduli, tuple(ctx.q_towers[:-1])
    )
    self.assertTrue(math.isclose(
        rescale_mapping.output_spec.scale,
        (input_scale * 2.0) / ctx.q_towers[-1],
    ))
    self.assertEqual(rescale_mapping.output_spec.nsd, 1)

    square_mapping = _analyzed_mapping(
        _model(_square(name='square')), ctx
    )
    divisor = ctx.q_towers[-1]
    self.assertEqual(square_mapping.output_spec.level, max_level - 1)
    self.assertTrue(math.isclose(
        square_mapping.output_spec.scale,
        (input_scale / divisor) ** 2,
    ))
    self.assertEqual(square_mapping.output_spec.nsd, 1)

    mul_mapping = _analyzed_mapping(_model(
        _mul(_identity(), _identity(), name='mul')
    ), ctx)
    self.assertEqual(mul_mapping.output_spec, square_mapping.output_spec)

  def test_composite_rescale_uses_the_complete_modulus_group(self):
    ctx = _context(max_level=3, composite_degree=2)
    mapping = _analyzed_mapping(_model(_rescale(name='rescale')), ctx)
    divisor = math.prod(ctx.q_towers[-2:])
    self.assertEqual(mapping.output_spec.level, 2)
    self.assertEqual(mapping.output_spec.moduli, tuple(ctx.q_towers[:-2]))
    self.assertTrue(math.isclose(
        mapping.output_spec.scale, ctx.scaling_factor / divisor
    ))

  def test_binary_alignment_underflow_and_headroom_are_explicit(self):
    ctx = _context(max_level=2)
    with self.assertRaisesRegex(ValueError, 'inputs disagree'):
      _analyzed_mapping(_model(_add(
          _square(name='square'),
          _identity(),
          name='misaligned',
      )), ctx)
    with self.assertRaisesRegex(ValueError, 'incompatible scales'):
      _analyzed_mapping(_model(_add(
          _mul_plain(_plain(scale=2), name='scaled'),
          _identity(),
          name='misaligned_scale',
      )), ctx)
    with self.assertRaisesRegex(ValueError, 'underflows'):
      _analyzed_mapping(_model(
          _square(name='square0'),
          _square(name='square1'),
          _square(name='square2'),
      ), ctx)
    with self.assertRaisesRegex(ValueError, 'headroom'):
      _analyzed_mapping(
          _model(_square(name='square')),
          ctx,
          headroom=ctx.max_level,
      )

  def test_multiply_allows_distinct_scales_but_requires_nsd_one(self):
    ctx = _context()
    left = _sequence(
        _mul_plain(_plain(scale=2), name='left_scale'),
        _rescale(name='left_rescale'),
    )
    right = _sequence(
        _mul_plain(_plain(scale=3), name='right_scale'),
        _rescale(name='right_rescale'),
    )
    mapping = _analyzed_mapping(
        _model(_mul(left, right, name='product')), ctx
    )
    self.assertEqual(mapping.output_spec.level, ctx.max_level - 2)
    self.assertNotEqual(
        mapping.value_specs['left_rescale'].scale,
        mapping.value_specs['right_rescale'].scale,
    )

    with self.assertRaisesRegex(ValueError, 'square requires input nsd=1'):
      _analyzed_mapping(_model(
          _mul_plain(_plain(scale=2), name='scaled'),
          _square(name='square'),
      ), ctx)

  def test_plaintext_layout_physical_slots_and_scale_are_validated(self):
    ctx = _context()
    with self.assertRaisesRegex(ValueError, 'plaintext layout'):
      _analyzed_mapping(_model(
          _add_plain(_plain(shape=(4, 2), scale=ctx.scaling_factor))
      ), ctx)
    # A constant that does not fill its declared layout is refused at
    # construction, so the planner's own slot-count check is unreachable
    # through a well-formed value. Enforced earlier, by a better owner.
    with self.assertRaisesRegex(ValueError, 'needs 8 slots'):
      _plain(shape=(8,), physical_slots=4)
    with self.assertRaisesRegex(ValueError, 'does not match ciphertext scale'):
      _analyzed_mapping(_model(
          _add_plain(_plain(scale=2.0))
      ), ctx)

  def test_rotations_are_canonicalized_and_deduplicated(self):
    ctx = _context()
    mapping = _analyzed_mapping(_model(_parallel_sum(
        _rotate(-1, name='negative'),
        _rotate(7, name='positive'),
        name='sum',
    )), ctx)
    self.assertEqual(mapping.required_rotation_indices, (7,))

  def test_matvec_freezes_bsgs_and_output_layout(self):
    ctx = _context()
    model = _model(_linear(
        _sparse_matrix(output_shape=(4,), output_packing='packed'),
        n1=2,
        n2=4,
        pt_scale=1024,
        name='linear',
    ))
    mapping = _analyzed_mapping(model, ctx)
    bsgs_plan = mapping.operation_plans['linear_matvec']

    self.assertEqual((bsgs_plan.n1, bsgs_plan.n2), (2, 4))
    self.assertEqual(bsgs_plan.active_diagonal_indices, (0, 1, 5))
    self.assertEqual(
        mapping.required_rotation_indices,
        tuple(sorted(set(bsgs_plan.required_rotation_indices))),
    )
    self.assertEqual(mapping.output_spec.level, ctx.max_level - 1)
    self.assertEqual(mapping.output_spec.shape, (4,))
    self.assertEqual(mapping.output_spec.packing, 'packed')

    with self.assertRaisesRegex(ValueError, 'do not match adaptive factors'):
      _analyzed_mapping(_model(_linear(
          _sparse_matrix(indices=(0,)),
          n1=2,
          n2=4,
          bsgs_ratio=2.0,
      )), ctx)

  def test_zero_weight_dense_and_conv_keep_matvec_and_bias_transitions(self):
    """A matrix with no surviving diagonal still has to carry its bias.

    The planner recognizes it as a zero matvec -- no rotations, no diagonals
    -- but the operation and its AddPlain successor stay in the graph, because
    the bias is the whole of the output.
    """
    for case_name in ('dense', 'conv2d'):
      with self.subTest(case=case_name):
        ctx = _context()
        mapping = _analyzed_packing(
            _model(
                _linear(
                    _sparse_matrix(indices=(0,), zero=True),
                    bias=_plain(0.75),
                    name=f'zero_{case_name}',
                )
            ),
            ctx,
        )
        matvec, bias = mapping.packing.operations
        matvec_id, matvec_kind, _, matvec_argument = matvec
        bias_id, bias_kind, bias_inputs, bias_constant = bias

        self.assertEqual(matvec_kind, 'matvec')
        self.assertEqual(bias_kind, 'add_plain')
        self.assertEqual(bias_inputs, (matvec_id,))
        self.assertTrue(np.any(bias_constant.values))
        self.assertFalse(
            any(matvec_argument[0].diagonal_maxima().values()),
            'a zero matrix must leave no surviving diagonal',
        )
        plan = mapping.operation_plans[matvec_id]
        self.assertIsInstance(plan, mapping_mod._ZeroMatVecPlan)
        self.assertEqual(plan.active_diagonal_indices, ())
        del bias_id

  def test_composite_matvec_defaults_to_recursive_scale(self):
    ctx = _context(max_level=4, composite_degree=2)
    base = float(math.prod(ctx.q_towers[-2:]))
    ctx.scaling_factor = base
    ctx.parameters['scaling_factor'] = base
    model = _model(
        _square(name='square1'),
        _square(name='square2'),
        _linear(_sparse_matrix(), name='linear'),
    )
    mapping = _analyzed_mapping(model, ctx)
    plan = mapping.operation_plans['linear_matvec']

    after_two_levels = base * base
    for modulus in ctx.q_towers[-4:-2]:
      after_two_levels /= float(modulus)
    source_spec = mapping.value_specs['square2']
    direct_drop_scale = float(math.prod(source_spec.moduli[-2:]))
    self.assertEqual(plan.pt_scale, source_spec.scale)
    self.assertTrue(math.isclose(
        plan.pt_scale, after_two_levels, rel_tol=1e-12
    ))
    self.assertNotEqual(plan.pt_scale, direct_drop_scale)

    explicit = _analyzed_mapping(
        _model(
            _square(name='square1'),
            _square(name='square2'),
            _linear(
                _sparse_matrix(), pt_scale=123456, name='linear'
            ),
        ),
        ctx,
    )
    self.assertEqual(
        explicit.operation_plans['linear_matvec'].pt_scale, 123456
    )

  def test_lazy_sparse_planning_reads_only_diagonal_maxima(self):
    source = SimpleNamespace(
        digest='lazy-source',
        diagonal_maxima=mock.Mock(return_value={0: 1.0, 3: 0.25}),
        as_dict=mock.Mock(
            side_effect=AssertionError('planning materialized diagonals')
        ),
        materialize_diagonals=mock.Mock(
            side_effect=AssertionError('planning materialized diagonals')
        ),
    )
    matrix = packing_ir._LazySparseDiagonals(
        dimension=8,
        source=source,
        input_shape=(8,),
        output_shape=(8,),
    )

    mapping = _analyzed_mapping(
        _model(_linear(matrix, name='linear')), _context()
    )

    self.assertEqual(
        mapping.operation_plans['linear_matvec'].active_diagonal_indices,
        (0, 3),
    )
    source.diagonal_maxima.assert_called_once_with()
    source.as_dict.assert_not_called()
    source.materialize_diagonals.assert_not_called()

  def test_lazy_sparse_dimension_must_match_context_slots(self):
    """A matrix that cannot hold its own layout is refused when it is built.

    Under the physical convention every value spans the ciphertext, so a
    matrix's diagonal dimension and its declared layout must agree. That is
    now enforced at construction, before any planner sees the value -- the
    planner's own check remains as a backstop but is unreachable through a
    well-formed constant.
    """
    source = SimpleNamespace(
        digest='mismatched-lazy-source',
        diagonal_maxima=mock.Mock(return_value={0: 1.0}),
        as_dict=mock.Mock(),
        materialize_diagonals=mock.Mock(),
    )
    with self.assertRaisesRegex(
        ValueError, 'does not fit the diagonal dimension'
    ):
      packing_ir._LazySparseDiagonals(
          dimension=4, source=source, input_shape=(8,), output_shape=(8,)
      )

  def test_dense_matvec_applies_the_plaintext_encode_floor(self):
    ctx = _context()
    retained = 0.1 * np.eye(8) + 0.01 * np.roll(np.eye(8), 1, axis=1)
    matrix = packing_ir.DenseMatrix(retained, (8,), (8,))
    with self.assertRaisesRegex(
        ValueError, 'Refusing to drop non-zero coefficients'
    ):
      _analyzed_mapping(_model(_linear(
          matrix, pt_scale=10, name='linear'
      )), ctx)

    too_small = packing_ir.DenseMatrix(0.01 * np.eye(8), (8,), (8,))
    with self.assertRaisesRegex(ValueError, 'below the plaintext encode floor'):
      _analyzed_mapping(_model(_linear(
          too_small, pt_scale=10, name='linear'
      )), ctx)

  def test_bootstrap_is_planned_as_a_fusion_barrier(self):
    ctx = _context(max_level=20)
    model = _model(
        _rotate(1, name='before'),
        _bootstrap(level_budget=(1, 1), name='refresh'),
        _rotate(2, name='after'),
    )
    mapping = _analyzed_mapping(model, ctx)
    bootstrap_plan = mapping.operation_plans['refresh']

    self.assertEqual(
        mapping.output_spec.level, bootstrap_plan.expected_output_level
    )
    bootstrap_scale = float(math.prod(
        ctx.q_towers[-ctx.composite_degree:]
    ))
    self.assertEqual(mapping.output_spec.scale, bootstrap_scale)
    self.assertNotEqual(mapping.output_spec.scale, ctx.output_scale)
    self.assertEqual(
        mapping.regions,
        (
            (True, ('before',)),
            (False, ('refresh',)),
            (True, ('after',)),
        ),
    )
    self.assertTrue(
        set(bootstrap_plan.required_rotation_indices)
        <= set(mapping.required_rotation_indices)
    )
    self.assertIn(2 * ctx.degree - 1, mapping.required_rotation_indices)

  def test_meta_bootstrap_requires_and_produces_headroom(self):
    ctx = _context(max_level=20)
    model = _model(
        _rescale(steps=18, name='consume'),
        _bootstrap(
            level_budget=(1, 1),
            mode='meta',
            precision=10,
            name='refresh',
        ),
    )
    mapping = _analyzed_mapping(model, ctx)
    self.assertGreater(
        mapping.output_spec.level, mapping.value_specs['consume'].level
    )
    bootstrap_scale = float(math.prod(
        ctx.q_towers[-ctx.composite_degree:]
    ))
    self.assertEqual(
        mapping.output_spec.scale, math.ldexp(bootstrap_scale, 10)
    )

  def test_bootstrap_contract_is_fully_known_during_planning(self):
    ctx = _context(max_level=40)
    with self.assertRaisesRegex(ValueError, 'input nsd=1'):
      _analyzed_mapping(
          _model(_bootstrap(level_budget=(1, 1), name='refresh')),
          ctx,
          input_nsd=None,
      )
    with self.assertRaisesRegex(ValueError, 'input nsd=1'):
      _analyzed_mapping(_model(
          _mul_plain(_plain(scale=2), name='scaled'),
          _bootstrap(level_budget=(1, 1), name='refresh'),
      ), ctx)
    with self.assertRaisesRegex(ValueError, 'multiple bootstrap configurations'):
      _analyzed_mapping(_model(
          _bootstrap(level_budget=(1, 1), name='first'),
          _bootstrap(level_budget=(2, 1), name='second'),
      ), ctx)

  def test_segment_liveness_preserves_values_across_one_branch_barrier(self):
    ctx = _context(max_level=20)
    bootstrap_scale = math.prod(ctx.q_towers[-ctx.composite_degree:])
    input_scale = bootstrap_scale * math.prod(ctx.q_towers[-16:])
    model = _model(_add(
        _sequence(
            _rotate(1, name='left_rotate'),
            _rescale(steps=16, name='left'),
        ),
        _sequence(
            _rotate(2, name='pre_refresh'),
            _bootstrap(level_budget=(1, 1), name='refresh'),
        ),
        name='join',
    ))
    mapping = _analyzed_mapping(model, ctx, input_scale=input_scale)
    _, inputs, outputs = mapping_mod._region_io(
        mapping, ('left_rotate', 'left', 'pre_refresh')
    )
    self.assertEqual(inputs, ('input',))
    self.assertEqual(outputs, ('left', 'pre_refresh'))

  def test_bootstrap_barrier_validates_engine_metadata_without_retagging(self):
    ctx = _context(max_level=20)
    mapping = _analyzed_mapping(_model(_bootstrap(
        level_budget=(1, 1), name='refresh'
    )), ctx)
    input_value = _ciphertext(ctx, mapping.input_spec)
    wrong_scale = mapping.output_spec.scale * 2
    wrong_output = _ciphertext(
        ctx, mapping.output_spec, scale=wrong_scale
    )
    mapping.bindings = {'refresh': lambda value: wrong_output}
    mapping._compiled_regions = ((
        ('refresh',), ('input',), ('refresh',), None
    ),)

    with self.assertRaisesRegex(ValueError, 'does not match planned scale'):
      mapping.execute(input_value)
    self.assertEqual(wrong_output._ckks_scale, wrong_scale)

  def test_fingerprint_is_content_addressed(self):
    ctx = _context()
    first = _model(_add_plain(_plain(1), name='bias'))
    same = _model(_add_plain(_plain(1), name='bias'))
    changed = _model(_add_plain(_plain(2), name='bias'))
    self.assertEqual(
        _analyzed_mapping(first, ctx).fingerprint,
        _analyzed_mapping(same, ctx).fingerprint,
    )
    self.assertNotEqual(
        _analyzed_mapping(first, ctx).fingerprint,
        _analyzed_mapping(changed, ctx).fingerprint,
    )

  def test_fingerprint_includes_global_batch_and_device_count(self):
    ctx = _context(batch=2)
    model = _model(_rotate(0, name='copy'))
    single_device = _analyzed_mapping(
        model, ctx, global_batch=2, device_count=1
    )
    two_devices = _analyzed_mapping(
        model, ctx, global_batch=4, device_count=2
    )
    self.assertNotEqual(single_device.fingerprint, two_devices.fingerprint)

  def test_operation_compile_mode_splits_regions_and_changes_fingerprint(self):
    ctx = _context()
    model = _model(
        _add_plain(_plain(), name='add'),
        _rotate(1, name='rotate'),
    )
    whole = _analyzed_mapping(model, ctx)
    operations = _analyzed_mapping(
        model, ctx, compile_mode='operations'
    )
    self.assertNotEqual(whole.fingerprint, operations.fingerprint)
    self.assertEqual(
        operations.regions,
        ((True, ('add',)), (True, ('rotate',))),
    )

  def test_bsgs_materialization_policy_changes_fingerprint(self):
    ctx = _context()
    model = _model(_linear(_sparse_matrix(), name='linear'))
    serial = _analyzed_mapping(
        model, ctx, compile_mode='operations'
    )
    parallel = _analyzed_mapping(
        model, ctx, compile_mode='operations', bsgs_n_jobs=2
    )
    streaming = _analyzed_mapping(
        model,
        ctx,
        compile_mode='operations',
        bsgs_streaming=True,
    )
    self.assertNotEqual(serial.fingerprint, parallel.fingerprint)
    self.assertNotEqual(serial.fingerprint, streaming.fingerprint)

  def test_the_kind_is_supported_and_maps_to_its_own_primitive(self):
    self.assertIn('level_reduce', mapping_mod._SUPPORTED_KINDS)
    self.assertEqual(
        mapping_mod._LOW_LEVEL_PRIMITIVE['level_reduce'], 'he_level_reduce'
    )
    self.assertNotEqual(
        mapping_mod._LOW_LEVEL_PRIMITIVE['level_reduce'],
        mapping_mod._LOW_LEVEL_PRIMITIVE['rescale'],
    )


class MappingInferenceFacadeTest(absltest.TestCase):

  @staticmethod
  def _mapping(*, global_batch):
    return _analyzed_packing(
        _model(
            _linear(_sparse_matrix(), name='classifier'),
            context_shape=(2,),
        ),
        _context(batch=global_batch),
        global_batch=global_batch,
    )

  def test_infer_owns_single_input_codec_and_logical_output_boundary(self):
    mapping = self._mapping(global_batch=1)
    codec = mock.Mock()
    encrypted = object()
    evaluated = object()
    codec.encrypt_slots.return_value = encrypted
    codec.decrypt_slots.return_value = np.asarray(
        [3.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    mapping.ctx = codec
    mapping.execute = mock.Mock(return_value=evaluated)
    mapping._validate_polynomial = mock.Mock()

    result = mapping.infer(np.asarray([3.0, -2.0]))

    np.testing.assert_array_equal(result, np.asarray([3.0, -2.0]))
    packed = codec.encrypt_slots.call_args.args[0]
    np.testing.assert_array_equal(
        packed, np.asarray([3.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    self.assertEqual(
        codec.encrypt_slots.call_args.kwargs,
        {'scale': mapping.input_spec.scale},
    )
    mapping.execute.assert_called_once_with(encrypted)
    mapping._validate_polynomial.assert_called_once_with(
        evaluated, mapping.output_spec, 'output'
    )
    codec.decrypt_slots.assert_called_once_with(
        evaluated,
        scale=mapping.output_spec.scale,
        validate_approximation=True,
    )

  def test_infer_owns_global_batch_codec_and_unpacks_each_output(self):
    mapping = self._mapping(global_batch=2)
    codec = mock.Mock()
    encrypted = object()
    evaluated = object()
    codec.encrypt_slots_batch.return_value = encrypted
    codec.decrypt_slots_batch.return_value = [
        np.asarray([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([-3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    mapping.ctx = codec
    mapping.execute = mock.Mock(return_value=evaluated)
    mapping._validate_polynomial = mock.Mock()

    result = mapping.infer(
        [np.asarray([1.0, 2.0]), np.asarray([-3.0, 4.0])],
        validate_approximation=False,
    )

    self.assertLen(result, 2)
    np.testing.assert_array_equal(result[0], np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(result[1], np.asarray([-3.0, 4.0]))
    packed_batch = codec.encrypt_slots_batch.call_args.args[0]
    self.assertLen(packed_batch, 2)
    np.testing.assert_array_equal(
        packed_batch[0],
        np.asarray([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        packed_batch[1],
        np.asarray([-3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    self.assertEqual(
        codec.encrypt_slots_batch.call_args.kwargs,
        {'scale': mapping.input_spec.scale},
    )
    mapping.execute.assert_called_once_with(encrypted)
    codec.decrypt_slots_batch.assert_called_once_with(
        evaluated,
        scale=mapping.output_spec.scale,
        validate_approximation=False,
    )

  def test_encrypt_input_rejects_incomplete_global_batch(self):
    mapping = self._mapping(global_batch=2)
    codec = mock.Mock()
    mapping.ctx = codec

    with self.assertRaisesRegex(ValueError, 'contains 1 values, expected 2'):
      mapping.encrypt_input([np.asarray([1.0, 2.0])])

    codec.encrypt_slots_batch.assert_not_called()


class MappingPublicSurfaceTest(absltest.TestCase):

  def test_mapping_has_no_model_entry_point(self):
    self.assertFalse(hasattr(mapping_mod.Mapping, 'from_model'))

  def test_mapping_does_not_import_the_frontend(self):
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(mapping_mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imported.update(alias.name for alias in node.names)
      elif isinstance(node, ast.ImportFrom):
        if node.module:
          imported.add(node.module)
        imported.update(alias.name for alias in node.names)
    self.assertNotIn('nn', imported)
    self.assertNotIn('jaxite_word.nn', imported)

  def test_the_error_names_the_one_producer(self):
    with self.assertRaises(TypeError) as caught:
      mapping_mod.Mapping(object())
    self.assertIn('packing.pack', str(caught.exception))


class MappingCompilationLifecycleTest(absltest.TestCase):

  @staticmethod
  def _fake_cache(ctx, **overrides):
    values = {
        'q_towers': list(ctx.q_towers),
        'p_towers': list(ctx.p_towers),
        'degree_layout': tuple(ctx.degree_layout),
        'dnum': ctx.dnum,
        'composite_degree': ctx.composite_degree,
        'batch': ctx.batch,
        'perf_test': False,
        'noise_std': cc._normalize_noise_std(
            ctx.parameters.get('sigma', cc.sigma)
        ),
        'noise_scale_degree': int(
            ctx.parameters.get('noise_scale_degree', 1)
        ),
        'key_generation_version': int(cc.kg.KEY_GENERATION_VERSION),
        'rot_indices': [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)

  def test_unspecified_devices_shard_the_global_batch_automatically(self):
    import jax

    devices = tuple(object() for _ in range(8))
    with (
        mock.patch.object(jax, 'local_devices', return_value=list(devices)),
        mock.patch.object(CKKSContext, 'program_initialization', autospec=True),
        mock.patch.object(mapping_mod, '_materialize_mapping'),
    ):
      mapping = Mapping(
          _model(),
          packing_ir.test_only_parameters(_parameters()),
          global_batch=32,
          dnum=3,
      )

    self.assertEqual(mapping.devices, devices)
    self.assertEqual(mapping.device_count, 8)
    self.assertEqual(mapping.per_device_batch, 4)
    self.assertEqual(mapping.ctx.batch, 4)

  def test_mapping_owns_context_trajectories_and_exact_rotation_setup(self):
    model = _model(_parallel_sum(
        _rotate(-1, name='left'),
        _rotate(1, name='right'),
        name='sum',
    ))
    initialized = []

    def initialize(context, **kwargs):
      initialized.append((context, kwargs))

    with (
        mock.patch.object(
            CKKSContext,
            'program_initialization',
            autospec=True,
            side_effect=initialize,
        ),
        mock.patch.object(mapping_mod, '_materialize_mapping') as materialize,
    ):
      mapping = Mapping(
          model, packing_ir.test_only_parameters(_parameters()),
          global_batch=2, dnum=3
      )

    self.assertLen(initialized, 1)
    self.assertIs(initialized[0][0], mapping.context)
    self.assertEqual(
        initialized[0][1]['total_rotation_indices'], [1, 7]
    )
    materialize.assert_called_once_with(mapping)
    self.assertLen(mapping.compute_trajectory, 3)
    self.assertLen(mapping.memory_trajectory, 4)
    self.assertFalse(hasattr(mapping.context, 'compile'))
    self.assertFalse(hasattr(mapping.context, 'execute'))



  def test_failed_materialization_restores_mapping_owned_context(self):
    model = _model(_rotate(1, name='rotate'))
    initialized_contexts = []

    def partially_initialize(context, **kwargs):
      del kwargs
      initialized_contexts.append(context)
      context._param_cache = object()
      context._he_add = object()

    with (
        mock.patch.object(
            CKKSContext,
            'program_initialization',
            autospec=True,
            side_effect=partially_initialize,
        ),
        mock.patch.object(
            mapping_mod,
            '_materialize_mapping',
            side_effect=RuntimeError('compile failed'),
        ),
    ):
      with self.assertRaisesRegex(RuntimeError, 'compile failed'):
        Mapping(model, packing_ir.test_only_parameters(_parameters()),
          global_batch=2, dnum=3)
    self.assertLen(initialized_contexts, 1)
    ctx = initialized_contexts[0]
    self.assertNotIn('_param_cache', ctx.__dict__)
    self.assertNotIn('_he_add', ctx.__dict__)

  def test_multi_device_bootstrap_fails_before_context_initialization(self):
    model = _model(_bootstrap(
        level_budget=(1, 1), name='refresh'
    ))
    with self.assertRaisesRegex(ValueError, 'multi-device.*bootstrap'):
      Mapping(
          model,
          packing_ir.test_only_parameters(_parameters(max_level=20)),
          global_batch=2,
          devices=(object(), object()),
          dnum=3,
      )

  def test_operation_compile_mode_rejects_multiple_devices(self):
    with self.assertRaisesRegex(ValueError, 'requires exactly one device'):
      Mapping(
          _model(),
          packing_ir.test_only_parameters(_parameters()),
          global_batch=2,
          devices=(object(), object()),
          dnum=3,
          compile_mode='operations',
      )

  def test_invalid_compile_mode_fails_before_context_initialization(self):
    with self.assertRaisesRegex(ValueError, 'compile_mode'):
      Mapping(
          _model(),
          packing_ir.test_only_parameters(_parameters()),
          global_batch=1,
          devices=(object(),),
          dnum=3,
          compile_mode='invalid',
      )

  def test_bsgs_materialization_policy_is_validated_before_context_setup(self):
    sparse = _model(_linear(_sparse_matrix(), name='linear'))
    for invalid in (True, 0, 1.5):
      with self.subTest(bsgs_n_jobs=invalid):
        with self.assertRaisesRegex(ValueError, 'bsgs_n_jobs'):
          Mapping(
              sparse,
              packing_ir.test_only_parameters(_parameters()),
              devices=(object(),),
              dnum=3,
              bsgs_n_jobs=invalid,
          )

    with self.assertRaisesRegex(
        ValueError, "requires compile_mode='operations'"
    ):
      Mapping(
          sparse,
          packing_ir.test_only_parameters(_parameters()),
          devices=(object(),),
          dnum=3,
          bsgs_streaming=True,
      )

    dense = packing_ir.DenseMatrix(np.eye(8), (8,), (8,))
    with self.assertRaisesRegex(ValueError, 'requires sparse'):
      Mapping(
          _model(_linear(dense, name='linear')),
          packing_ir.test_only_parameters(_parameters()),
          devices=(object(),),
          dnum=3,
          compile_mode='operations',
          bsgs_streaming=True,
      )

  def test_mapping_fingerprint_includes_material_performance_mode(self):
    ctx = _context()
    model = _model(_rotate(1, name='rotate'))
    self.assertNotEqual(
        _analyzed_mapping(model, ctx).fingerprint,
        _analyzed_mapping(model, ctx, perf_test=True).fingerprint,
    )

  def test_mapping_fingerprint_includes_key_generation_noise_contract(self):
    model = _model(_rotate(1, name='rotate'))
    baseline_ctx = _context()
    baseline = _analyzed_mapping(model, baseline_ctx).fingerprint

    sigma_ctx = _context()
    sigma_ctx.parameters['sigma'] = cc.sigma + 1.0
    self.assertNotEqual(
        baseline,
        _analyzed_mapping(model, sigma_ctx).fingerprint,
    )

    scale_ctx = _context()
    scale_ctx.parameters['noise_scale_degree'] = 2
    self.assertNotEqual(
        baseline,
        _analyzed_mapping(model, scale_ctx).fingerprint,
    )

    with mock.patch.object(
        cc.kg,
        'KEY_GENERATION_VERSION',
        cc.kg.KEY_GENERATION_VERSION + 1,
    ):
      versioned = _analyzed_mapping(model, _context()).fingerprint
    self.assertNotEqual(baseline, versioned)

  def test_preinitialized_cache_must_match_and_is_sealed_to_first_attempt(self):
    ctx = _context()
    mapping = _analyzed_mapping(_model(), ctx)
    ctx._param_cache = self._fake_cache(ctx, batch=ctx.batch + 1)
    with self.assertRaisesRegex(ValueError, 'parameter cache does not match'):
      mapping_mod._materialize_mapping(mapping)

    ctx.parameters['noise_scale_degree'] = 2
    changed_mapping = _analyzed_mapping(_model(), ctx)
    ctx._param_cache = self._fake_cache(ctx, noise_scale_degree=1)
    with self.assertRaisesRegex(ValueError, 'parameter cache does not match'):
      mapping_mod._materialize_mapping(changed_mapping)
    ctx.parameters['noise_scale_degree'] = 1

    ctx.parameters['sigma'] = cc.sigma + 1.0
    changed_mapping = _analyzed_mapping(_model(), ctx)
    ctx._param_cache = self._fake_cache(ctx, noise_std=cc.sigma)
    with self.assertRaisesRegex(ValueError, 'parameter cache does not match'):
      mapping_mod._materialize_mapping(changed_mapping)
    ctx.parameters.pop('sigma')

    ctx._param_cache = self._fake_cache(
        ctx,
        key_generation_version=cc.kg.KEY_GENERATION_VERSION - 1,
    )
    with self.assertRaisesRegex(ValueError, 'parameter cache does not match'):
      mapping_mod._materialize_mapping(mapping)

    ctx._param_cache = self._fake_cache(ctx, rot_indices=[7])
    with self.assertRaisesRegex(ValueError, 'unplanned rotation keys'):
      mapping_mod._materialize_mapping(mapping)

    ctx._param_cache = self._fake_cache(ctx)
    ctx._param_cache._static_model_fingerprint = 'different'
    with self.assertRaisesRegex(RuntimeError, 'different Mapping'):
      mapping_mod._materialize_mapping(mapping)

  def test_materialization_rejects_a_context_missing_exact_keys(self):
    ctx = _context()
    mapping = _analyzed_mapping(
        _model(_rotate(3, name='rotate')), ctx
    )
    ctx._param_cache = SimpleNamespace(rot_indices=[])
    with self.assertRaisesRegex(ValueError, 'missing planned rotation keys'):
      mapping_mod._materialize_mapping(mapping)


if pytest is not None:
  PlanningTest.pytestmark = [
      pytest.mark.correctness,
      pytest.mark.contract,
      pytest.mark.unit,
  ]
  MappingInferenceFacadeTest.pytestmark = [
      pytest.mark.correctness,
      pytest.mark.contract,
      pytest.mark.integration,
  ]
  MappingCompilationLifecycleTest.pytestmark = [
      pytest.mark.contract,
      pytest.mark.integration,
  ]


_NUM_SLOTS = 128
# The logical-workload lowering suite that stood here exercised
# nn.lower_model, which no longer exists. What outlived it moved to
# packing_test's SparseLoweringTest, rebuilt on real torch modules: the six
# numerical cases and their awkward geometry, the zero-weight/nonzero-bias
# behaviour, the diagonal subset and maxima invariants, and the guard that
# no slot-square is ever allocated -- along with the guard's own self-test.
# Only the lower_model plumbing itself was dropped.


if __name__ == '__main__':
  absltest.main()
