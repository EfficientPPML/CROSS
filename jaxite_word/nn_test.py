"""Tests for the complete ``nn.py`` frontend and vectorized-program IR.

The file is organized in four regions: package/lifecycle boundaries, the
declarative VectorizedProgram data model, real torch.fx vectorization, and
public frontend/documentation contracts.
"""

import ast
import dataclasses
import math
import os
import pathlib
from pathlib import Path
import pickle
import re
import subprocess
import textwrap
import sys
from unittest import mock, skipIf
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import bsgs
import finite_field
from he_params import HEParameterCache
import key_gen as kg

# This file imports ``jaxite_word`` as a package, so the repository root must
# be importable even when it is launched the documented way from inside
# jaxite_word/ (``python3 nn_test.py``), where only this directory is on the
# path. Without this the file fails at import before running a single case.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from jaxite_word import CKKSContext, Mapping, Packing, Polynomial, nn
from jaxite_word import packing
from jaxite_word import packing as packing_ir
from jaxite_word import mapping as mapping_mod

try:
  import pytest
except ModuleNotFoundError:
  pytest = None


# =============================================================================
# Region 1: package boundaries and Mapping lifecycle integration
# =============================================================================


def _plain(value=1.0, *, scale=None):
  return packing_ir.PlainSlots(
      np.full(8, value, dtype=np.float64),
      shape=(8,),
      packing='slots',
      scale=scale,
  )


def _matrix():
  return packing_ir.SparseDiagonals(
      dimension=8,
      diagonals=((0, np.ones(8, dtype=np.float64)),),
      input_shape=(8,),
      output_shape=(8,),
      packing='slots',
  )


class PackageBoundaryTest(absltest.TestCase):

  def test_canonical_imports_share_polynomial_and_context_class_identity(self):
    import ckks_ctx as flat_context
    import mapping as flat_mapping
    import nn as flat_nn
    import packing as flat_packing
    import polynomial as flat_polynomial
    from jaxite_word.ckks_ctx import CKKSContext as direct_context
    from jaxite_word.polynomial import Polynomial as direct_polynomial

    self.assertIs(Polynomial, direct_polynomial)
    self.assertIs(Polynomial, flat_polynomial.Polynomial)
    self.assertIs(CKKSContext, direct_context)
    self.assertIs(Mapping, flat_mapping.Mapping)
    # ``Packing`` is PP-op IR and belongs to ``packing``; it must no longer be
    # reachable as an ``nn`` frontend operator.
    self.assertIs(Packing, flat_packing.Packing)
    self.assertFalse(hasattr(nn, 'Packing'))
    for operator in ('Linear', 'PlainSlots', 'SparseDiagonals', 'Rotate',
                     'Rescale', 'Bootstrap', 'Sequential'):
      self.assertFalse(
          hasattr(nn, operator),
          msg=f'{operator} must not be exposed on the nn frontend',
      )
    self.assertIs(nn, flat_nn)
    # ``packing`` owns the slot-layout classes; ``nn`` re-exports them, and
    # both spellings of the module must yield one class identity.
    self.assertIs(nn.TensorSpec, flat_packing.TensorSpec)
    self.assertIs(nn.TensorLayout, flat_packing.TensorLayout)
    self.assertFalse(hasattr(nn, 'LazySparseDiagonals'))
    self.assertNotIn('_LazySparseDiagonals', nn.__all__)
    self.assertEqual(
        set(__import__('jaxite_word').__all__),
        {
            '__version__',
            'CKKSContext',
            'VectorizedProgram',
            'Mapping',
            'Packing',
            'Polynomial',
            'vectorize',
            'pack',
            'mapping',
            'nn',
            'packing',
        },
    )
    self.assertEqual(__import__('jaxite_word').__version__, '3.0.0')

  def test_canonical_nn_import_does_not_load_the_crypto_stack(self):
    """``import jaxite_word.nn`` must not drag in JAX or the HE modules.

    The frontend is pure NumPy so tooling can describe and vectorize a model
    without a ciphertext in sight. That only holds if the package facade keeps
    ``polynomial``, ``mapping`` and ``ckks_ctx`` lazy -- an eager export of any
    of them silently reintroduces the whole stack. Checked in a subprocess
    because ``sys.modules`` is process-global and this one already has
    everything loaded.
    """
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(root), str(root / 'jaxite_word')))
    # Import structure only -- no ciphertext math -- so pin the child to CPU.
    # An accelerator is exclusive to one process: inheriting the parent's env
    # makes the child abort with "TPU is already in use by process with pid
    # ..." whenever this suite runs on a real TPU host.
    env['JAX_PLATFORMS'] = 'cpu'
    code = textwrap.dedent(
        """
        import sys
        import jaxite_word.nn
        heavy = [
            name
            for name in (
                'jax',
                'jaxite_word.ckks_ctx',
                'jaxite_word.mapping',
                'jaxite_word.polynomial',
                'he_params',
                'bsgs',
                'bootstrapping',
            )
            if name in sys.modules
        ]
        assert not heavy, 'canonical nn import loaded: ' + repr(heavy)

        # The lazy names must still resolve, and to one identity per class.
        import jaxite_word
        import mapping as flat_mapping
        import polynomial as flat_polynomial
        assert jaxite_word.Mapping is flat_mapping.Mapping
        assert jaxite_word.Polynomial is flat_polynomial.Polynomial
        """
    )
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(
        completed.returncode,
        0,
        msg=completed.stdout + completed.stderr,
    )

  def test_class_identity_is_stable_in_both_import_orders(self):
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        (str(root), str(root / 'jaxite_word'))
    )
    # Class identity is a pure import property; keep the child off the
    # accelerator so it cannot collide with the parent's exclusive TPU claim.
    env['JAX_PLATFORMS'] = 'cpu'
    cases = (
        """
import polynomial, nn, packing, ckks_ctx, mapping
import jaxite_word
from jaxite_word import CKKSContext, Mapping, Packing, Polynomial
from jaxite_word import nn as package_nn
assert Polynomial is polynomial.Polynomial
assert CKKSContext is ckks_ctx.CKKSContext
assert Mapping is mapping.Mapping
assert Packing is packing.Packing
assert package_nn is nn
""",
        """
import jaxite_word
from jaxite_word import CKKSContext, Mapping, Packing, Polynomial, nn
import polynomial, packing, ckks_ctx, mapping
import nn as flat_nn
assert Polynomial is polynomial.Polynomial
assert CKKSContext is ckks_ctx.CKKSContext
assert Mapping is mapping.Mapping
assert Packing is packing.Packing
assert nn is flat_nn
""",
    )
    for index, code in enumerate(cases):
      with self.subTest(import_order=index):
        completed = subprocess.run(
            [sys.executable, '-c', code],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


class LayerTreeTest(absltest.TestCase):

  def test_constants_are_immutable_content_snapshots(self):
    source = np.arange(8, dtype=np.float64)
    plain = packing_ir.PlainSlots(source, shape=(8,))
    same = packing_ir.PlainSlots(np.arange(8, dtype=np.float64), shape=(8,))
    changed = packing_ir.PlainSlots(np.arange(8, dtype=np.float64) + 1, shape=(8,))
    source[:] = -1

    np.testing.assert_array_equal(plain.values, np.arange(8))
    self.assertFalse(plain.values.flags.writeable)
    self.assertEqual(plain.digest, same.digest)
    self.assertNotEqual(plain.digest, changed.digest)

    dense_source = np.eye(8)
    dense = packing_ir.DenseMatrix(dense_source, (8,), (8,))
    dense_source[:] = 0
    np.testing.assert_array_equal(dense.values, np.eye(8))
    self.assertFalse(dense.values.flags.writeable)

  def test_constants_validate_physical_and_logical_layouts(self):
    with self.assertRaisesRegex(ValueError, 'finite'):
      packing_ir.PlainSlots(np.array([np.nan]), shape=(1,))
    with self.assertRaisesRegex(ValueError, 'matrix shape'):
      packing_ir.DenseMatrix(np.eye(4), (8,), (8,))
    with self.assertRaisesRegex(ValueError, 'duplicate diagonal'):
      packing_ir.SparseDiagonals(
          8,
          ((1, np.ones(8)), (1, np.ones(8))),
          (8,),
          (8,),
      )
    with self.assertRaisesRegex(ValueError, 'outside'):
      packing_ir.SparseDiagonals(8, ((8, np.ones(8)),), (8,), (8,))
    with self.assertRaisesRegex(ValueError, 'at least one diagonal'):
      packing_ir.SparseDiagonals(8, (), (8,), (8,))


  def test_packing_rejects_nonzero_imaginary_slots(self):
    """A real-valued program cannot take a complex input."""
    values = np.zeros(8, dtype=np.complex128)
    values[0] = 1.0 + 2.0j
    with self.assertRaises((TypeError, ValueError)):
      packing_ir.PlainSlots(values=values, shape=(8,))


  def test_physical_constants_reject_unsupported_complex_values(self):
    with self.assertRaisesRegex(TypeError, 'PlainSlots.values must be real'):
      packing_ir.PlainSlots(
          np.ones(8, dtype=np.complex128),
          shape=(8,),
      )
    with self.assertRaisesRegex(TypeError, 'DenseMatrix.values must be real'):
      packing_ir.DenseMatrix(
          np.eye(2, dtype=np.complex128),
          input_shape=(2,),
          output_shape=(2,),
      )
    with self.assertRaisesRegex(
        TypeError, 'SparseDiagonals values must be real'
    ):
      packing_ir.SparseDiagonals(
          dimension=2,
          diagonals=((0, np.ones(2, dtype=np.complex128)),),
          input_shape=(2,),
          output_shape=(2,),
      )

  def test_constant_storage_cannot_be_made_writable_after_fingerprinting(self):
    constants = (
        packing_ir.PlainSlots(np.ones(2), shape=(2,)).values,
        packing_ir.DenseMatrix(
            np.eye(2), input_shape=(2,), output_shape=(2,)
        ).values,
        packing_ir.SparseDiagonals(
            dimension=2,
            diagonals=((0, np.ones(2)),),
            input_shape=(2,),
            output_shape=(2,),
        ).diagonals[0][1],
        packing_ir._Dense(np.eye(2)).weight,
    )

    for constant in constants:
      with self.subTest(shape=constant.shape):
        with self.assertRaises(ValueError):
          constant.setflags(write=True)

  def test_lazy_sparse_materializes_exact_real_subset(self):
    source = mock.Mock()
    source.digest = 'subset-source'
    source.diagonal_maxima.return_value = {1: 1.0, 5: 2.0}
    source.as_dict.side_effect = AssertionError('full materialization')
    source.materialize_diagonals.return_value = {
        1: np.ones(8),
        5: np.full(8, 2.0),
    }
    matrix = packing_ir._LazySparseDiagonals(
        dimension=8,
        source=source,
        input_shape=(8,),
        output_shape=(8,),
    )

    diagonals = matrix.materialize_diagonals((1, 5))

    self.assertEqual(tuple(diagonals), (1, 5))
    source.materialize_diagonals.assert_called_once_with((1, 5))
    source.as_dict.assert_not_called()
    self.assertFalse(diagonals[1].flags.writeable)
    with self.assertRaises(ValueError):
      diagonals[1].setflags(write=True)

    source.materialize_diagonals.return_value = {
        1: np.ones(8, dtype=np.complex128),
        5: np.full(8, 2.0),
    }
    with self.assertRaisesRegex(TypeError, 'values must be real'):
      matrix.materialize_diagonals((1, 5))

  def test_custom_layouts_are_injective_and_have_distinct_identities(self):
    class CustomLayout(nn.TensorLayout):

      def __init__(self, reverse=False, duplicate=False):
        self.reverse = reverse
        self.duplicate = duplicate

      def descriptor(self):
        return {'kind': 'test_layout'}

      @property
      def packing(self):
        return 'custom'

      def physical_size(self, shape):
        return math.prod(shape)

      def coordinate_to_slot(self, shape, coordinate):
        if self.duplicate:
          return 0
        index = int(np.ravel_multi_index(coordinate, shape))
        return math.prod(shape) - 1 - index if self.reverse else index

    with self.assertRaisesRegex(
        ValueError, 'multiple logical coordinates'
    ):
      nn.TensorSpec((2,), CustomLayout(duplicate=True))

    forward = nn.TensorSpec((2,), CustomLayout())
    mutable_layout = CustomLayout(reverse=True)
    reverse = nn.TensorSpec((2,), mutable_layout)
    self.assertNotEqual(
        forward.layout_fingerprint, reverse.layout_fingerprint
    )
    self.assertNotEqual(forward.packing, reverse.packing)
    mutable_layout.reverse = False
    np.testing.assert_array_equal(
      reverse.pack(np.asarray([1, 2]), 2),
      np.asarray([2, 1]),
    )

  def test_tensor_packing_rejects_same_size_inputs_with_the_wrong_shape(self):
    shape = (2, 3, 4)
    chw = np.arange(math.prod(shape)).reshape(shape)
    flat = chw.reshape(-1)
    hwc = np.transpose(chw, (1, 2, 0))
    layout = nn.ChannelMajor()
    spec = nn.TensorSpec(shape, layout)
    packers = (
        ('layout', lambda value: layout.pack(value, shape, 32)),
        ('spec', lambda value: spec.pack(value, 32)),
    )

    for name, pack in packers:
      with self.subTest(packer=name):
        expected = np.pad(flat, (0, 32 - flat.size))
        np.testing.assert_array_equal(pack(chw), expected)
        np.testing.assert_array_equal(pack(flat), expected)
        with self.assertRaisesRegex(
            ValueError, 'expected logical shape.*flat compatibility shape'
        ):
          pack(hwc)

  def test_custom_layout_requires_a_stable_descriptor(self):
    class MissingDescriptor(nn.TensorLayout):

      @property
      def packing(self):
        return 'missing_descriptor'

      def physical_size(self, shape):
        return math.prod(shape)

      def coordinate_to_slot(self, shape, coordinate):
        return int(np.ravel_multi_index(coordinate, shape))

    with self.assertRaisesRegex(NotImplementedError, 'descriptor'):
      nn.TensorSpec((2,), MissingDescriptor())



# ---------------------------------------------------------------------------
# Fixtures: canonical packing.Packing values built from explicit operations.
# The lifecycle contracts below are unchanged; only how a fixture is written
# changed, because the declarative module tree they used no longer exists.
# ---------------------------------------------------------------------------

_NUM_SLOTS = 8


class _Step:

  def __init__(self, kind, argument=None, name=None, branches=None):
    self.kind = kind
    self.argument = argument
    self.name = name
    self.branches = branches


def _identity(name=None):
  return _Step('identity', name=name)


def _rotate(index, name=None):
  return _Step('rotate', index, name)


def _square(name=None):
  return _Step('square', name=name)


def _rescale(steps=1, name=None):
  return _Step('rescale', int(steps), name)


def _add_plain(constant, name=None):
  return _Step('add_plain', constant, name)


def _mul_plain(constant, name=None):
  return _Step('mul_plain', constant, name)


def _linear(matrix, bias=None, n1=None, n2=None, bsgs_ratio=None,
            pt_scale=None, name=None):
  return _Step('matvec', (matrix, n1, n2, bsgs_ratio, pt_scale), name,
               branches={'bias': bias})


def _add(left, right, name=None):
  return _Step('add', None, name, branches={'left': left, 'right': right})


def _sub(left, right, name=None):
  return _Step('sub', None, name, branches={'left': left, 'right': right})


def _mul(left, right, name=None):
  return _Step('mul', None, name, branches={'left': left, 'right': right})


class _StubRing:

  def __init__(self, num_slots=_NUM_SLOTS, dnum=3):
    self.degree = 2 * num_slots
    self.num_slots = num_slots
    self.dnum = dnum


def _emit_steps(steps, operations, source, counter):
  current = source
  for step in steps if isinstance(steps, list) else [steps]:
    counter[0] += 1
    value_id = step.name or f'{step.kind}_{counter[0]}'
    if step.kind == 'identity':
      continue
    if step.branches and 'left' in step.branches:
      left = _emit_steps(step.branches['left'], operations, current, counter)
      right = _emit_steps(step.branches['right'], operations, current, counter)
      operations.append((value_id, step.kind, (left, right), None))
      current = value_id
      continue
    if step.kind == 'matvec' and step.name:
      value_id = f'{step.name}_matvec'
    operations.append((value_id, step.kind, (current,), step.argument))
    current = value_id
    bias = (step.branches or {}).get('bias')
    if bias is not None:
      operations.append(
          (f'{step.name or value_id}_bias', 'add_plain', (current,), bias)
      )
      current = f'{step.name or value_id}_bias'
  return current


def _sequence(*steps, **_ignored):
  flat = []
  for step in steps:
    flat.extend(step if isinstance(step, list) else [step])
  return flat


def _packed(*steps, num_slots=_NUM_SLOTS, **_ignored):
  operations = []
  output = _emit_steps(_sequence(*steps), operations, 'input', [0])
  if not operations:
    operations.append(('identity', 'rotate', ('input',), 0))
    output = 'identity'
  return packing_ir.Packing(
      operations=tuple(operations),
      ring_config=_StubRing(num_slots),
      input_shape=(num_slots,),
      input_packing='slots',
      logical_input_shape=(num_slots,),
      logical_output_shape=(num_slots,),
      input_coordinate_map=tuple(range(num_slots)),
      output_coordinate_map=tuple(range(num_slots)),
      layer_shapes=(),
      output=output,
      num_slots=num_slots,
      depth=0,
  )


class MappingLifecycleTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.degree = 16
    cls.num_slots = 8
    cls.r = cls.c = 4
    cls.dnum = 3
    cls.q_towers = [
        1073742881,
        1073742721,
        1073741441,
        1073741857,
        524353,
    ]
    cls.p_towers = [1073740609, 1073739937, 1073739649]
    cls.scaling_factor = 563019763943521
    cls.key_pair = kg.gen_pke_pair(
        cls.q_towers, cls.p_towers, cls.degree
    )

  def _parameters(self):
    return {
        'degree': self.degree,
        'num_slots': self.num_slots,
        'scaling_factor': self.scaling_factor,
        'output_scale': self.scaling_factor,
        'q_towers': self.q_towers,
        'p_towers': self.p_towers,
        'p': 30,
        'CKKS_M_FACTOR': 1,
        'max_bits_in_word': 61,
        'noise_scale_degree': 1,
        'secret_key': self.key_pair['secret_key'],
        'degree_layout': (self.r, self.c),
    }

  def _mapping(self, model, **mapping_options):
    return Mapping(
        model,
        packing_ir.test_only_parameters(self._parameters()),
        global_batch=1,
        dnum=self.dnum,
        **mapping_options,
    )

  def _ciphertext(self, ctx):
    cache = ctx._param_cache
    moduli = cache.q_moduli_at_level(ctx.max_level)
    shape = (ctx.batch, 2, self.r, self.c, len(moduli))
    values = np.arange(np.prod(shape), dtype=np.uint64).reshape(shape)
    payload = (values % np.asarray(moduli, dtype=np.uint64)).astype(np.uint32)
    result = Polynomial.from_array(
        jnp.asarray(payload),
        {
            'batch': ctx.batch,
            'num_elements': 2,
            'degree': self.degree,
            'num_moduli': len(moduli),
            'precision': 32,
            'degree_layout': (self.r, self.c),
        },
        {
            'moduli': moduli,
            'ntt_ctx': cache._level(ctx.max_level).sliced_ntt_q,
        },
    )
    result._ckks_scale = float(self.scaling_factor)
    result._ckks_nsd = 1
    return result

  def test_mapping_owns_the_complete_model_lifecycle(self):
    model = _packed(
        _rotate(0, name='copy'),
        name='Identity',
        input_shape=(8,),
    )
    packing = model
    mapping = Mapping(
        packing,
        packing_ir.test_only_parameters(self._parameters()),
        global_batch=1,
        dnum=self.dnum,
        input_scale=float(self.scaling_factor),
        input_nsd=1,
        headroom=1,
    )
    ctx = mapping.ctx

    self.assertIs(mapping.packing, packing)
    self.assertIs(mapping.context, ctx)
    self.assertIsInstance(ctx, CKKSContext)
    self.assertEqual(packing.input_packing, 'slots')
    self.assertEqual(mapping.input_scale, float(self.scaling_factor))
    self.assertEqual(mapping.input_nsd, 1)
    self.assertEqual(mapping.headroom, 1)
    self.assertFalse(hasattr(ctx, 'compile'))
    self.assertFalse(hasattr(ctx, 'execute'))
    self.assertEmpty(mapping.bindings)
    for accessor in (
        ctx.ptct_mul,
        ctx.he_mul,
        ctx.he_rot,
        ctx.he_rescale,
        ctx.he_level_reduce,
        ctx.he_add,
        ctx.he_sub,
    ):
      self.assertEmpty(accessor._instances)
    ciphertext = self._ciphertext(ctx)
    result = mapping.execute(ciphertext)
    np.testing.assert_array_equal(
        result.polynomial, ciphertext.polynomial
    )

    with self.assertRaisesRegex(TypeError, 'Polynomial'):
      mapping.execute({'input': ciphertext})
    with self.assertRaisesRegex(TypeError, 'Polynomial'):
      mapping.execute(ciphertext.polynomial)
    wrong_scale = self._ciphertext(ctx)
    wrong_scale._ckks_scale *= 2
    with self.assertRaisesRegex(ValueError, 'scale'):
      mapping.execute(wrong_scale)

  def test_public_residual_add_compiles_and_executes(self):
    model = _packed(
        _add(_identity(), _identity(), name='double'),
        name='ResidualAdd',
        input_shape=(8,),
    )
    mapping = self._mapping(model)
    ctx = mapping.ctx
    ciphertext = self._ciphertext(ctx)

    result = mapping.execute(ciphertext)

    moduli = np.asarray(ciphertext.moduli, dtype=np.uint64)
    expected = (
        2 * np.asarray(ciphertext.polynomial, dtype=np.uint64)
    ) % moduli
    np.testing.assert_array_equal(
        result.polynomial, expected.astype(np.uint32)
    )

  def test_operation_compile_mode_executes_multiple_regions(self):
    model = _packed(
        _add_plain(
            _plain(1, scale=self.scaling_factor), name='first_bias'
        ),
        _add_plain(
            _plain(2, scale=self.scaling_factor), name='second_bias'
        ),
        input_shape=(8,),
    )
    mapping = self._mapping(model, compile_mode='operations')
    self.assertIsNone(mapping._compiled)
    self.assertLen(mapping._compiled_regions, 2)
    self.assertEmpty(mapping.bindings)
    for accessor in (
        mapping.ctx.ptct_mul,
        mapping.ctx.he_mul,
        mapping.ctx.he_rot,
        mapping.ctx.he_rescale,
        mapping.ctx.he_add,
        mapping.ctx.he_sub,
    ):
      self.assertEmpty(accessor._instances)

    ciphertext = self._ciphertext(mapping.ctx)
    result = mapping.execute(ciphertext)
    self.assertEmpty(mapping.bindings)
    for accessor in (
        mapping.ctx.ptct_mul,
        mapping.ctx.he_mul,
        mapping.ctx.he_rot,
        mapping.ctx.he_rescale,
        mapping.ctx.he_add,
        mapping.ctx.he_sub,
    ):
      self.assertEmpty(accessor._instances)

    # Compare through the same HE addition facade rather than interpreting
    # encoded CKKS constants as coefficient-domain integers.
    add = mapping.ctx.he_add[mapping.ctx.max_level]
    expected = ciphertext
    for value in (1.0, 2.0):
      expected = add.add_plain(
          expected,
          mapping.ctx.encode_at_level(
              [value] * self.num_slots,
              mapping.ctx.max_level,
              scale=self.scaling_factor,
          ),
      )
    np.testing.assert_array_equal(result.polynomial, expected.polynomial)

  @skipIf(
      len(jax.local_devices()) < 2,
      'requires at least two local JAX devices',
  )
  def test_operation_compile_mode_honors_selected_device(self):
    device = jax.local_devices()[1]
    model = _packed(
        _add_plain(
            _plain(1, scale=self.scaling_factor), name='bias'
        ),
        input_shape=(8,),
    )
    mapping = self._mapping(
        model,
        devices=(device,),
        compile_mode='operations',
    )

    result = mapping.execute(self._ciphertext(mapping.ctx))

    self.assertEqual(result.polynomial.devices(), {device})

  def test_operation_compile_mode_streams_bsgs_dynamic_operands(self):
    model = _packed(
        _linear(
            _matrix(),
            n1=2,
            n2=4,
            pt_scale=self.scaling_factor,
            name='identity_matrix',
        ),
        input_shape=(8,),
    )
    preprocess_calls = []
    dynamic_operand_counts = []
    release_calls = []
    original_preprocess = bsgs._BSGSMatVecAtLevel.preprocess
    original_matvec = bsgs._BSGSMatVecAtLevel._matvec_array
    original_release = bsgs._BSGSMatVecAtLevel._release

    def counting_preprocess(evaluator, *args, **kwargs):
      preprocess_calls.append(evaluator)
      return original_preprocess(evaluator, *args, **kwargs)

    def counting_matvec(evaluator, ciphertext, *operands):
      dynamic_operand_counts.append(len(operands))
      return original_matvec(evaluator, ciphertext, *operands)

    def counting_release(evaluator):
      release_calls.append(evaluator)
      return original_release(evaluator)

    with (
        mock.patch.object(
            bsgs._BSGSMatVecAtLevel,
            'preprocess',
            new=counting_preprocess,
        ),
        mock.patch.object(
            bsgs._BSGSMatVecAtLevel,
            '_matvec_array',
            new=counting_matvec,
        ),
        mock.patch.object(
            bsgs._BSGSMatVecAtLevel,
            '_release',
            new=counting_release,
        ),
    ):
      mapping = self._mapping(
          model,
          compile_mode='operations',
          cache_rotation_keys=False,
      )
      self.assertEmpty(preprocess_calls)
      self.assertEmpty(mapping.bindings)
      self.assertEmpty(mapping.ctx._param_cache.raw_rotation_keys)
      self.assertEmpty(mapping.ctx._param_cache._formatted_rotation_keys)

      ciphertext = self._ciphertext(mapping.ctx)
      result = mapping.execute(ciphertext)
      repeated = mapping.execute(ciphertext)

    self.assertIsInstance(result, Polynomial)
    np.testing.assert_array_equal(result.polynomial, repeated.polynomial)
    self.assertLen(preprocess_calls, 2)
    self.assertTrue(dynamic_operand_counts)
    self.assertTrue(all(count == 4 for count in dynamic_operand_counts))
    self.assertLen(release_calls, 2)
    for evaluator in release_calls:
      self.assertIsNone(evaluator._groups)
      self.assertIsNone(evaluator._baby_eval_a)
    self.assertEmpty(mapping.bindings)
    self.assertEmpty(mapping.ctx._param_cache.raw_rotation_keys)
    self.assertEmpty(mapping.ctx._param_cache._formatted_rotation_keys)
    for accessor in (
        mapping.ctx.ptct_mul,
        mapping.ctx.he_mul,
        mapping.ctx.he_rot,
        mapping.ctx.he_rescale,
        mapping.ctx.he_add,
        mapping.ctx.he_sub,
    ):
      self.assertEmpty(accessor._instances)

  def test_operation_compile_mode_streams_sparse_bsgs_giant_steps(self):
    source = mock.Mock()
    source.digest = 'streaming-four-diagonals'
    diagonal_map = {
        index: np.full(8, 0.1 * (index + 1), dtype=np.float64)
        for index in (0, 1, 4, 5)
    }
    source.diagonal_maxima.return_value = {
        index: float(np.max(np.abs(diagonal)))
        for index, diagonal in diagonal_map.items()
    }
    source.as_dict.side_effect = AssertionError(
        'lazy streaming materialized the full diagonal map'
    )
    source.materialize_diagonals.side_effect = (
        lambda indices: {index: diagonal_map[index] for index in indices}
    )
    matrix = packing_ir._LazySparseDiagonals(
        dimension=8,
        source=source,
        input_shape=(8,),
        output_shape=(8,),
    )
    model = _packed(
        _linear(
            matrix,
            n1=2,
            n2=4,
            pt_scale=self.scaling_factor,
            name='identity_matrix',
        ),
        input_shape=(8,),
    )
    preprocess_calls = []
    matvec_calls = []
    release_calls = []
    original_preprocess = bsgs._BSGSMatVecAtLevel.preprocess
    original_matvec = bsgs._BSGSMatVecAtLevel.matvec
    original_release = bsgs._BSGSMatVecAtLevel._release

    def counting_preprocess(evaluator, *args, **kwargs):
      preprocess_calls.append(dict(kwargs))
      return original_preprocess(evaluator, *args, **kwargs)

    def counting_matvec(evaluator, ciphertext, *args, **kwargs):
      matvec_calls.append(dict(kwargs))
      return original_matvec(evaluator, ciphertext, *args, **kwargs)

    def counting_release(evaluator):
      release_calls.append(evaluator)
      return original_release(evaluator)

    with (
        mock.patch.object(
            bsgs._BSGSMatVecAtLevel,
            'preprocess',
            new=counting_preprocess,
        ),
        mock.patch.object(
            bsgs._BSGSMatVecAtLevel,
            'matvec',
            new=counting_matvec,
        ),
        mock.patch.object(
            bsgs._BSGSMatVecAtLevel,
            '_release',
            new=counting_release,
        ),
    ):
      mapping = self._mapping(
          model,
          compile_mode='operations',
          cache_rotation_keys=False,
          bsgs_n_jobs=1,
          bsgs_streaming=True,
      )
      source.as_dict.assert_not_called()
      result = mapping.execute(self._ciphertext(mapping.ctx))

    self.assertIsInstance(result, Polynomial)
    self.assertLen(preprocess_calls, 1)
    self.assertEqual(preprocess_calls[0]['n_jobs'], 1)
    self.assertTrue(preprocess_calls[0]['memory_bounded'])
    self.assertLen(matvec_calls, 1)
    self.assertEmpty(matvec_calls[0])
    source.diagonal_maxima.assert_called_once_with()
    source.as_dict.assert_not_called()
    self.assertEqual(
        source.materialize_diagonals.call_args_list,
        [mock.call((0, 1)), mock.call((4, 5))],
    )
    self.assertLen(release_calls, 1)
    self.assertIsNone(release_calls[0]._groups)
    self.assertIsNone(release_calls[0]._diagonal_map)
    self.assertIsNone(release_calls[0]._diagonal_loader)
    self.assertEmpty(mapping.bindings)
    for accessor in (
        mapping.ctx.ptct_mul,
        mapping.ctx.he_mul,
        mapping.ctx.he_rot,
        mapping.ctx.he_rescale,
        mapping.ctx.he_add,
        mapping.ctx.he_sub,
    ):
      self.assertEmpty(accessor._instances)

  def test_binary_schedule_is_fully_materialized_before_execute(self):
    model = _packed(
        _mul(
            _identity(),
            _rotate(1, name='factor_rotation'),
            name='branch_product',
        ),
        _sub(
            _identity(),
            _rotate(0, name='same_value'),
            name='cancel',
        ),
        name='BinarySchedule',
        input_shape=(8,),
    )
    mapping = self._mapping(model)
    ctx = mapping.ctx
    materialized = {
        'mul': set(ctx.he_mul._instances),
        'rot': set(ctx.he_rot._instances),
        'sub': set(ctx.he_sub._instances),
    }

    result = mapping.execute(self._ciphertext(ctx))

    self.assertIsInstance(result, Polynomial)
    np.testing.assert_array_equal(
        result.polynomial, np.zeros_like(result.polynomial)
    )
    self.assertEqual(materialized['mul'], set(ctx.he_mul._instances))
    self.assertEqual(materialized['rot'], set(ctx.he_rot._instances))
    self.assertEqual(materialized['sub'], set(ctx.he_sub._instances))

  def test_plain_constants_are_encoded_only_during_compile(self):
    model = _packed(
        _add_plain(
            _plain(1, scale=self.scaling_factor), name='first_bias'
        ),
        _add_plain(
            _plain(2, scale=self.scaling_factor), name='second_bias'
        ),
        input_shape=(8,),
    )
    original_encode = CKKSContext.encode_at_level
    encode_calls = []

    def counting_encode(context, *args, **kwargs):
      encode_calls.append((args, kwargs))
      return original_encode(context, *args, **kwargs)

    with mock.patch.object(
        CKKSContext, 'encode_at_level', new=counting_encode
    ):
      mapping = self._mapping(model)
      self.assertLen(encode_calls, 2)
      ciphertext = self._ciphertext(mapping.ctx)
      mapping.execute(ciphertext)
      mapping.execute(ciphertext)
      self.assertLen(encode_calls, 2)

  def test_montgomery_mapping_plaintext_bindings_round_trip(self):
    model = _packed(
        _add_plain(
            _plain(1, scale=self.scaling_factor), name='add_bias'
        ),
        _mul_plain(
            _plain(2, scale=self.scaling_factor), name='multiply_weight'
        ),
        input_shape=(8,),
    )
    original_initialize = CKKSContext.program_initialization
    original_prepare = HEParameterCache._prepare_plaintext_payload
    prepare_calls = []

    def initialize_montgomery(context, *args, **kwargs):
      kwargs['finite_field_context'] = finite_field.MontgomeryContext
      return original_initialize(context, *args, **kwargs)

    def counting_prepare(cache, plaintext_data, level):
      prepare_calls.append(level)
      return original_prepare(cache, plaintext_data, level)

    with (
        mock.patch.object(
            CKKSContext,
            'program_initialization',
            new=initialize_montgomery,
        ),
        mock.patch.object(
            HEParameterCache,
            '_prepare_plaintext_payload',
            new=counting_prepare,
        ),
    ):
      mapping = self._mapping(model)
      self.assertLen(prepare_calls, 2)

    ctx = mapping.ctx
    self.assertIsInstance(
        ctx._param_cache.ff_q_max, finite_field.MontgomeryContext
    )
    ctx.public_key = self.key_pair['public_key']
    slots = np.asarray(
        [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=np.float64,
    )
    ciphertext = ctx.encrypt_slots(slots)

    result = mapping.execute(ciphertext)

    np.testing.assert_allclose(
        np.asarray(ctx.decrypt_slots(result)),
        (slots + 1.0) * 2.0,
        rtol=0,
        atol=1e-3,
    )

  def test_mapping_requires_a_complete_packing_boundary(self):
    """Only the value packing.pack produces may enter the scheduler.

    Checked by type rather than by shape: an object carrying the right
    attribute names would otherwise be a second, unvalidated producer.
    """
    parameters = packing_ir.test_only_parameters(self._parameters())

    class LooksComplete:
      operations = ()
      fingerprint = 'x'
      ring_config = None

    for candidate in (object(), LooksComplete()):
      with self.assertRaisesRegex(TypeError, 'packing.pack'):
        Mapping(candidate, parameters, global_batch=1, dnum=self.dnum)

  def test_mapping_maps_global_batch_across_devices_behind_polynomial(self):
    model = _packed(
        _rescale(name='drop_level'),
        input_shape=(8,),
    )
    observed = {}

    def executable(payload):
      observed['payload'] = payload
      return payload[..., :-1]

    class FakeMapped:

      def lower(self, input_struct):
        observed['shape'] = input_struct.shape
        observed['dtype'] = input_struct.dtype
        return self

      def compile(self):
        return executable

    import jax
    devices = (object(), object())
    with mock.patch.object(
        jax, 'pmap', return_value=FakeMapped()
    ) as pmap:
      mapping = Mapping(
          model,
          packing_ir.test_only_parameters(self._parameters()),
          global_batch=2,
          devices=devices,
          dnum=self.dnum,
      )

    pmap.assert_called_once()
    self.assertTrue(callable(pmap.call_args.args[0]))
    self.assertEqual(pmap.call_args.kwargs['devices'], devices)
    self.assertEqual(mapping.global_batch, 2)
    self.assertEqual(mapping.device_count, 2)
    self.assertEqual(mapping.per_device_batch, 1)
    self.assertEqual(mapping.ctx.batch, 1)
    self.assertEqual(mapping.input_spec.batch, 2)
    self.assertEqual(mapping.output_spec.batch, 2)
    self.assertEqual(mapping.value_specs['input'].batch, 1)
    self.assertEqual(mapping.value_specs[mapping.packing.output].batch, 1)
    local_shape = mapping.value_specs['input'].payload_shape
    self.assertEqual(observed['shape'], (2,) + local_shape)
    self.assertEqual(str(observed['dtype']), 'uint32')

    local_value = self._ciphertext(mapping.ctx)
    value = local_value._clone_with_payload(
        jnp.concatenate(
            (local_value.polynomial, local_value.polynomial), axis=0
        ),
        batch=2,
    )
    result = mapping.execute(value)

    self.assertIsInstance(result, Polynomial)
    self.assertEqual(result.batch, 2)
    self.assertEqual(tuple(result.moduli), mapping.output_spec.moduli)
    self.assertEqual(result._ckks_scale, mapping.output_spec.scale)
    self.assertEqual(result._ckks_nsd, mapping.output_spec.nsd)
    self.assertEqual(
        observed['payload'].shape, (2,) + local_shape
    )
    np.testing.assert_array_equal(
        result.polynomial, value.polynomial[..., :-1]
    )

  def test_mapping_rejects_global_batch_not_divisible_by_devices(self):
    model = _packed(_identity(), input_shape=(8,))
    with self.assertRaisesRegex(ValueError, 'must be divisible'):
      Mapping(
          model,
          packing_ir.test_only_parameters(self._parameters()),
          global_batch=3,
          devices=(object(), object()),
          dnum=self.dnum,
      )

  def test_mapping_rejects_empty_and_duplicate_device_sets(self):
    model = _packed(_identity(), input_shape=(8,))
    packing = model
    with self.assertRaisesRegex(ValueError, 'at least one'):
      Mapping(
          packing,
          packing_ir.test_only_parameters(self._parameters()),
          global_batch=1,
          devices=(),
          dnum=self.dnum,
      )
    device = object()
    with self.assertRaisesRegex(ValueError, 'duplicates'):
      Mapping(
          packing,
          packing_ir.test_only_parameters(self._parameters()),
          global_batch=2,
          devices=(device, device),
          dnum=self.dnum,
      )

  def test_one_device_executes_the_complete_global_batch(self):
    import jax

    model = _packed(
        _rotate(0, name='copy'), input_shape=(8,)
    )
    mapping = Mapping(
        model,
        packing_ir.test_only_parameters(self._parameters()),
        global_batch=2,
        devices=(jax.devices()[0],),
        dnum=self.dnum,
    )
    value = self._ciphertext(mapping.ctx)

    result = mapping.execute(value)

    self.assertEqual(mapping.device_count, 1)
    self.assertEqual(mapping.per_device_batch, 2)
    self.assertEqual(result.batch, 2)
    np.testing.assert_array_equal(result.polynomial, value.polynomial)


if pytest is not None:
  PackageBoundaryTest.pytestmark = [pytest.mark.contract, pytest.mark.unit]
  LayerTreeTest.pytestmark = [pytest.mark.contract, pytest.mark.unit]
  MappingLifecycleTest.pytestmark = [
      pytest.mark.correctness,
      pytest.mark.integration,
  ]

# =============================================================================
# Region 2: declarative VectorizedProgram and registry contracts
# =============================================================================

def _spec(shape):
  return packing.TensorSpec(shape, packing.ChannelMajor())


def _conv_program(seed=0):
  """A small conv -> square -> dense program with real weights."""
  rng = np.random.default_rng(seed)
  weights = nn.WeightTableBuilder()
  conv_weight = weights.add(rng.normal(size=(2, 1, 3, 3)))
  conv_bias = weights.add(rng.normal(size=(2,)))
  dense_weight = weights.add(rng.normal(size=(4, 8)))

  conv = nn.VectorizedLayer(
      name='conv',
      kind=nn.LINEAR_TRANSFORM,
      template_id='conv2d',
      input_spec=_spec((1, 4, 4)),
      output_spec=_spec((2, 2, 2)),
      params={'stride': (2, 2), 'padding': (1, 1)},
      weights={'weight': conv_weight, 'bias': conv_bias},
  )
  square = nn.VectorizedLayer(
      name='square',
      kind=nn.POLYNOMIAL_ACTIVATION,
      template_id='',
      input_spec=_spec((2, 2, 2)),
      output_spec=_spec((2, 2, 2)),
      inputs=('conv',),
      activation_name='square',
  )
  flat = nn.VectorizedLayer(
      name='flat',
      kind=nn.LAYOUT,
      template_id='flatten',
      input_spec=_spec((2, 2, 2)),
      output_spec=_spec((8,)),
      inputs=('square',),
  )
  dense = nn.VectorizedLayer(
      name='dense',
      kind=nn.LINEAR_TRANSFORM,
      template_id='dense',
      input_spec=_spec((8,)),
      output_spec=_spec((4,)),
      inputs=('flat',),
      weights={'weight': dense_weight},
  )
  return nn.VectorizedProgram(
      layers=(conv, square, flat, dense),
      weights=weights,
      input_spec=_spec((1, 4, 4)),
      output='dense',
  )


class KindBoundaryTest(absltest.TestCase):
  """Gate: vectorized layers carry application semantics only."""

  def test_vectorized_kinds_name_no_he_realization(self):
    self.assertEqual(
        nn.VECTORIZED_KINDS,
        frozenset({'linear_transform', 'polynomial_activation',
                   'add', 'layout'}),
    )
    for forbidden in ('rescale', 'bootstrap', 'matvec', 'mul_plain',
                      'add_plain', 'rotate'):
      self.assertNotIn(forbidden, nn.VECTORIZED_KINDS)

  def test_a_program_never_carries_a_rescale_or_bootstrap_layer(self):
    program = _conv_program()
    for layer in program.layers:
      self.assertIn(layer.kind, nn.VECTORIZED_KINDS)

  def test_constructing_an_he_kind_layer_is_refused(self):
    for forbidden in ('rescale', 'bootstrap'):
      with self.assertRaisesRegex(ValueError, 'expected one of'):
        nn.VectorizedLayer(
            name='x',
            kind=forbidden,
            template_id='',
            input_spec=_spec((4,)),
            output_spec=_spec((4,)),
        )

  def test_registering_a_template_of_an_he_kind_is_refused(self):
    with self.assertRaisesRegex(ValueError, 'expected one of'):
      nn.register_layout_template(nn.LayoutTemplate(
          template_id='bogus_rescale',
          version='1',
          family='dense',
          kind='rescale',
      ))


class DeclarativeLayerTest(absltest.TestCase):
  """Gate: no closures, no arrays, deterministic identity."""

  def test_no_field_holds_an_array_or_a_callable(self):
    program = _conv_program()
    for layer in program.layers:
      for name in ('params', 'weights', 'inputs', 'template_id',
                   'activation_name'):
        value = getattr(layer, name)
        self.assertNotIsInstance(value, np.ndarray)
        self.assertFalse(callable(value), f'{name} is callable')
      for _, item in layer.params:
        self.assertNotIsInstance(item, np.ndarray)

  def test_an_array_valued_parameter_is_refused(self):
    with self.assertRaisesRegex(TypeError, 'weight table'):
      nn.VectorizedLayer(
          name='x',
          kind=nn.LINEAR_TRANSFORM,
          template_id='dense',
          input_spec=_spec((4,)),
          output_spec=_spec((4,)),
          params={'weight': np.zeros((4, 4))},
      )

  def test_layers_hash_and_compare_by_digest(self):
    left = _conv_program().layer('conv')
    right = _conv_program().layer('conv')
    self.assertEqual(left, right)
    self.assertEqual(hash(left), hash(right))
    self.assertEqual(len({left, right}), 1)

  def test_digests_are_deterministic_across_processes(self):
    """A digest built from ``hash()`` would vary with the per-process salt.

    Two subprocesses run with different PYTHONHASHSEED values, so any use of
    Python's salted hash anywhere under the digest shows up as a mismatch.
    """
    script = (
        'import sys\n'
        f'sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n'
        'import numpy as np, nn, packing\n'
        'table = nn.WeightTableBuilder()\n'
        'weight = table.add(np.arange(12, dtype=np.float64).reshape(4, 3))\n'
        'layer = nn.VectorizedLayer(\n'
        '    name="dense", kind=nn.LINEAR_TRANSFORM, template_id="dense",\n'
        '    input_spec=packing.TensorSpec((3,), packing.ChannelMajor()),\n'
        '    output_spec=packing.TensorSpec((4,), packing.ChannelMajor()),\n'
        '    params={"alpha": 0.5, "mode": "wide", "window": (2, 2)},\n'
        '    weights={"weight": weight})\n'
        'program = nn.VectorizedProgram(\n'
        '    layers=(layer,), weights=table,\n'
        '    input_spec=packing.TensorSpec((3,), packing.ChannelMajor()),\n'
        '    output="dense")\n'
        'print(layer.digest, program.digest, weight)\n'
    )
    digests = set()
    for seed in ('0', '1', '12345'):
      environment = dict(os.environ, PYTHONHASHSEED=seed)
      result = subprocess.run(
          [sys.executable, '-c', script],
          capture_output=True, text=True, check=True, env=environment,
      )
      digests.add(result.stdout.strip())
    self.assertLen(digests, 1, f'digest varied with hash seed: {digests}')

  def test_an_int_weight_digests_as_its_float_equivalent(self):
    table = nn.WeightTableBuilder()
    integral = table.add(np.arange(6, dtype=np.int32).reshape(2, 3))
    floating = table.add(np.arange(6, dtype=np.float64).reshape(2, 3))
    self.assertEqual(integral, floating)
    self.assertLen(table, 1)

  def test_a_fortran_ordered_weight_digests_as_its_c_ordered_equivalent(self):
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    table = nn.WeightTableBuilder()
    self.assertEqual(table.add(values), table.add(np.asfortranarray(values)))

  def test_reshaping_a_weight_changes_its_digest(self):
    """Shape must be digested: these two differ only in shape, not in bytes."""
    values = np.arange(6, dtype=np.float64)
    table = nn.WeightTableBuilder()
    wide = table.add(values.reshape(2, 3))
    tall = table.add(values.reshape(3, 2))
    flat = table.add(values)
    self.assertLen({wide, tall, flat}, 3)
    self.assertLen(table, 3)

  def test_dtype_is_digested_for_arrays_that_skip_normalization(self):
    """``weight_digest`` is public, so it cannot lean on a prior cast."""
    values = np.arange(6).reshape(2, 3)
    self.assertNotEqual(
        nn.weight_digest(values.astype(np.float32)),
        nn.weight_digest(values.astype(np.float64)),
    )

  def test_normalized_weights_are_immutable(self):
    array = nn.normalize_weight(np.arange(4, dtype=np.float64))
    with self.assertRaises(ValueError):
      array[0] = 1.0

  def test_a_non_finite_weight_is_refused(self):
    with self.assertRaisesRegex(ValueError, 'finite'):
      nn.normalize_weight(np.array([1.0, np.inf]))


class WeightTableImmutabilityTest(absltest.TestCase):
  """Gate: a program's weights are a frozen, content-addressed snapshot."""

  def test_normalized_weights_never_share_storage_with_the_caller(self):
    source = np.arange(6, dtype=np.float64)
    normalized = nn.normalize_weight(source)
    self.assertTrue(source.flags.writeable, 'caller array was made read-only')
    self.assertFalse(np.shares_memory(normalized, source))
    source[0] = 999.0
    self.assertEqual(normalized[0], 0.0)

  def test_writing_through_a_base_array_cannot_change_a_stored_weight(self):
    base = np.arange(12, dtype=np.float64)
    builder = nn.WeightTableBuilder()
    digest = builder.add(base[:6])
    table = builder.freeze()
    base[0] = 999.0
    self.assertEqual(table[digest][0], 0.0)
    self.assertEqual(nn.weight_digest(table[digest]), digest)

  def test_a_torch_style_tensor_is_detached_before_storage(self):

    class FakeTensor:
      """Stands in for torch.Tensor without importing torch."""

      def __init__(self, array):
        self._array = array

      def detach(self):
        return self

      def cpu(self):
        return self

      def numpy(self):
        return self._array

    shared = np.arange(4, dtype=np.float64)
    normalized = nn.normalize_weight(FakeTensor(shared))
    self.assertFalse(np.shares_memory(normalized, shared))
    shared[0] = 5.0
    self.assertEqual(normalized[0], 0.0)

  def test_a_key_that_does_not_name_its_content_is_refused(self):
    with self.assertRaisesRegex(ValueError, 'does not match the digest'):
      nn.WeightTable({'0' * 64: np.arange(4, dtype=np.float64)})

  def test_a_frozen_table_exposes_no_mutator(self):
    table = nn.WeightTableBuilder().freeze()
    self.assertFalse(hasattr(table, 'add'))

  def test_a_program_does_not_alias_the_builder_it_was_given(self):
    builder = nn.WeightTableBuilder()
    weight = builder.add(np.eye(4))
    layer = nn.VectorizedLayer(
        name='dense',
        kind=nn.LINEAR_TRANSFORM,
        template_id='dense',
        input_spec=_spec((4,)),
        output_spec=_spec((4,)),
        weights={'weight': weight},
    )
    program = nn.VectorizedProgram(
        layers=(layer,), weights=builder,
        input_spec=_spec((4,)), output='dense',
    )
    before = program.digest
    builder.add(np.ones((3, 3)))
    self.assertEqual(program.weights.digests(), (weight,))
    self.assertEqual(program.digest, before)


class RegistryDurabilityTest(absltest.TestCase):
  """Gate: mutating a registry cannot silently redefine a built program.

  ``template_id`` and ``activation_name`` are keys into mutable tables. A
  layer that resolved them lazily would change meaning -- its depth, its
  matrix entries -- whenever someone replaced a registration, while its
  digest stayed identical. So a layer captures what it needs at construction,
  and packing rechecks the template version it was built against.
  """

  def test_a_layer_captures_its_activation_coefficients(self):
    layer = _conv_program().layer('square')
    self.assertEqual(layer.activation_coefficients, (0.0, 0.0, 1.0))
    self.assertEqual(layer.depth_cost, 1)
    self.assertEqual(layer.polynomial().degree, 2)

  def test_a_layer_captures_its_template_version(self):
    layer = _conv_program().layer('conv')
    self.assertEqual(
        layer.template_version, nn.layout_template('conv2d').version
    )
    self.assertNotEmpty(layer.template_version)

  def test_replacing_an_activation_leaves_built_programs_unchanged(self):
    program = _conv_program()
    before_digest = program.digest
    before_depth = program.critical_depth()
    nn.register_activation(
        nn.ActivationPolynomial(
            name='square', coefficients=(0.0,) * 8 + (1.0,)
        ),
        replace=True,
    )
    try:
      # The built program still means what it meant: depth is computed from
      # the coefficients the layer captured, not from the registry.
      self.assertEqual(program.digest, before_digest)
      self.assertEqual(program.critical_depth(), before_depth)
      self.assertEqual(
          program.layer('square').activation_coefficients, (0.0, 0.0, 1.0)
      )
      # A newly built program picks up the replacement and so digests apart.
      rebuilt = _conv_program()
      self.assertNotEqual(rebuilt.digest, before_digest)
      self.assertEqual(rebuilt.layer('square').depth_cost, 3)
      self.assertEqual(rebuilt.critical_depth(), before_depth + 2)
    finally:
      nn.register_activation(
          nn.ActivationPolynomial(name='square', coefficients=(0.0, 0.0, 1.0)),
          substitutes=('relu', 'silu', 'square'),
          replace=True,
      )
    self.assertEqual(_conv_program().digest, before_digest)

  def test_replacing_a_template_without_a_new_version_is_refused(self):
    current = nn.layout_template('dense')
    with self.assertRaisesRegex(ValueError, 'new version'):
      nn.register_layout_template(
          dataclasses.replace(current, depth_cost=7), replace=True
      )

  def test_a_template_must_declare_a_version(self):
    with self.assertRaisesRegex(ValueError, 'must declare a version'):
      nn.register_layout_template(nn.LayoutTemplate(
          template_id='unversioned_test',
          family='dense',
          kind=nn.LINEAR_TRANSFORM,
          version='',
      ))

  def test_packing_a_program_built_against_an_old_template_is_refused(self):
    """A layer captures the version it was built against.

    The implementation now lives in packing, so that is what the captured
    version guards: if the packer implements a different version of the same
    template, the program it was digested as is not the program that would
    run, and packing refuses rather than quietly computing the other one.
    """
    program = _conv_program()
    before_digest = program.digest
    original = packing.template_lowering('conv2d')
    packing.register_template_lowering(
        'conv2d', dataclasses.replace(original, version='2'), replace=True
    )
    try:
      # The digest cannot notice -- it was sealed before the replacement.
      self.assertEqual(program.digest, before_digest)
      with self.assertRaisesRegex(packing.PackingError, 'version'):
        packing.layer_recipe(program.layer('conv'), program.weights)
    finally:
      packing.register_template_lowering('conv2d', original, replace=True)
    packing.layer_recipe(program.layer('conv'), program.weights)

  def test_a_frontend_template_version_change_redigests_new_programs(self):
    """The other half: a new version means a different program identity."""
    import dataclasses
    before = _conv_program().digest
    current = nn.layout_template('conv2d')
    nn.register_layout_template(
        dataclasses.replace(current, version='2'), replace=True
    )
    try:
      self.assertNotEqual(_conv_program().digest, before)
    finally:
      nn.register_layout_template(current, replace=True)
    self.assertEqual(_conv_program().digest, before)


class SerializationTest(absltest.TestCase):
  """Gate: a real weighted layer still packs after a round trip."""

  def test_a_pickled_program_still_yields_identical_matrix_entries(self):
    program = _conv_program()
    restored = pickle.loads(pickle.dumps(program))

    self.assertEqual(restored.digest, program.digest)
    self.assertEqual(restored, program)

    # Integrity before equivalence. numpy pickles array data but not the
    # writeable flag, so a restored table can look correct while its weights
    # have become mutable -- at which point the digest stops describing the
    # content and every downstream guarantee derived from it is void.
    for digest in restored.weights.digests():
      weight = restored.weights[digest]
      self.assertFalse(
          weight.flags.writeable, f'{digest[:16]} restored writable'
      )
      self.assertIsNotNone(weight.base, 'restored weight has no buffer behind it')
      with self.assertRaises(ValueError):
        weight[(0,) * weight.ndim] = 1.0
      self.assertEqual(nn.weight_digest(weight), digest)

    for name in ('conv', 'dense', 'flat'):
      original = list(
          packing.layer_recipe(program.layer(name), program.weights).entries()
      )
      round_tripped = list(
          packing.layer_recipe(restored.layer(name), restored.weights).entries()
      )
      self.assertEqual(round_tripped, original, f'{name} entries diverged')
      self.assertNotEmpty(original)

  def test_a_pickled_weight_table_survives_a_second_round_trip(self):
    """Restoring must re-normalize, not merely copy, or integrity decays."""
    program = _conv_program()
    restored = pickle.loads(pickle.dumps(
        pickle.loads(pickle.dumps(program))
    ))
    self.assertEqual(restored.digest, program.digest)
    for digest in restored.weights.digests():
      self.assertFalse(restored.weights[digest].flags.writeable)
      self.assertEqual(nn.weight_digest(restored.weights[digest]), digest)

  def test_a_pickled_builder_restores_immutable_weights(self):
    builder = nn.WeightTableBuilder()
    digest = builder.add(np.arange(6, dtype=np.float64).reshape(2, 3))
    restored = pickle.loads(pickle.dumps(builder)).freeze()
    self.assertEqual(restored.digests(), (digest,))
    self.assertFalse(restored[digest].flags.writeable)
    self.assertEqual(nn.weight_digest(restored[digest]), digest)

  def test_a_pickled_weight_table_preserves_bytes_exactly(self):
    program = _conv_program()
    restored = pickle.loads(pickle.dumps(program))
    for digest in program.weights.digests():
      np.testing.assert_array_equal(
          restored.weights[digest], program.weights[digest]
      )
      self.assertEqual(nn.weight_digest(restored.weights[digest]), digest)


class CriticalDepthTest(absltest.TestCase):
  """Gate: depth is the branch maximum, never the branch sum."""

  def _linear_chain(self, name, source, length, weights, dense_weight):
    layers = []
    previous = source
    for index in range(length):
      layer = nn.VectorizedLayer(
          name=f'{name}{index}',
          kind=nn.LINEAR_TRANSFORM,
          template_id='dense',
          input_spec=_spec((4,)),
          output_spec=_spec((4,)),
          inputs=(previous,),
          weights={'weight': dense_weight},
      )
      layers.append(layer)
      previous = layer.name
    return layers, previous

  def test_a_diamond_costs_its_deeper_arm_not_the_sum(self):
    weights = nn.WeightTableBuilder()
    dense_weight = weights.add(np.eye(4))
    stem = nn.VectorizedLayer(
        name='stem',
        kind=nn.LINEAR_TRANSFORM,
        template_id='dense',
        input_spec=_spec((4,)),
        output_spec=_spec((4,)),
        weights={'weight': dense_weight},
    )
    short, short_end = self._linear_chain(
        'short', 'stem', 2, weights, dense_weight
    )
    long, long_end = self._linear_chain(
        'long', 'stem', 4, weights, dense_weight
    )
    join = nn.VectorizedLayer(
        name='join',
        kind=nn.ADD,
        template_id='',
        input_spec=_spec((4,)),
        output_spec=_spec((4,)),
        inputs=(short_end, long_end),
    )
    program = nn.VectorizedProgram(
        layers=(stem, *short, *long, join),
        weights=weights,
        input_spec=_spec((4,)),
        output='join',
    )
    # stem(1) + long arm(4) + join(0). The sum over both arms would be 7.
    self.assertEqual(program.critical_depth(), 5)

  def test_a_wide_fan_in_of_equal_arms_costs_one_arm(self):
    """The shape of LoLA's 8-branch conv: a sum is not eight multiplies deep."""
    weights = nn.WeightTableBuilder()
    dense_weight = weights.add(np.eye(4))
    branches = [
        nn.VectorizedLayer(
            name=f'branch{index}',
            kind=nn.LINEAR_TRANSFORM,
            template_id='dense',
            input_spec=_spec((4,)),
            output_spec=_spec((4,)),
            weights={'weight': dense_weight},
        )
        for index in range(8)
    ]
    joins = []
    previous = branches[0].name
    for index in range(1, 8):
      join = nn.VectorizedLayer(
          name=f'join{index}',
          kind=nn.ADD,
          template_id='',
          input_spec=_spec((4,)),
          output_spec=_spec((4,)),
          inputs=(previous, branches[index].name),
      )
      joins.append(join)
      previous = join.name
    program = nn.VectorizedProgram(
        layers=(*branches, *joins),
        weights=weights,
        input_spec=_spec((4,)),
        output=previous,
    )
    self.assertEqual(program.critical_depth(), 1)

  def test_a_chain_costs_the_sum_of_its_layers(self):
    program = _conv_program()
    # conv(1) + square(1) + flatten(0) + dense(1). A flatten relabels slots;
    # only packing charges a level, and only if it materializes a repack.
    self.assertEqual(program.critical_depth(), 3)

  def test_max_live_slots_covers_the_input_and_every_output(self):
    program = _conv_program()
    self.assertEqual(program.max_live_slots(), 16)


class ActivationRegistryTest(absltest.TestCase):
  """Gate: activation substitution is a registry, extensible by users."""

  def test_relu_substitutes_to_square_by_default(self):
    self.assertEqual(nn.default_activation_for('relu').name, 'square')
    self.assertEqual(nn.activation('square').coefficients, (0.0, 0.0, 1.0))

  def test_square_costs_one_level(self):
    self.assertEqual(nn.activation('square').degree, 2)
    self.assertEqual(nn.activation('square').depth_cost, 1)

  def test_polynomial_depth_follows_the_paterson_stockmeyer_bound(self):
    for degree in (2, 3, 4, 5, 7, 8, 9, 16):
      polynomial = nn.ActivationPolynomial(
          name=f'degree{degree}', coefficients=(0.0,) * degree + (1.0,)
      )
      self.assertEqual(polynomial.degree, degree)
      self.assertEqual(polynomial.depth_cost, math.ceil(math.log2(degree)))

  def test_a_user_can_register_a_new_approximation(self):
    polynomial = nn.register_activation(
        nn.ActivationPolynomial(
            name='relu_deg4_test', coefficients=(0.0, 0.0, 0.0, 0.0, 1.0)
        ),
    )
    try:
      self.assertEqual(nn.activation('relu_deg4_test'), polynomial)
      self.assertEqual(polynomial.degree, 4)
      self.assertEqual(polynomial.depth_cost, 2)
      layer = nn.VectorizedLayer(
          name='act',
          kind=nn.POLYNOMIAL_ACTIVATION,
          template_id='',
          input_spec=_spec((4,)),
          output_spec=_spec((4,)),
          activation_name='relu_deg4_test',
      )
      self.assertEqual(layer.depth_cost, 2)
    finally:
      nn._ACTIVATIONS.pop('relu_deg4_test', None)

  def test_a_polynomial_packing_cannot_evaluate_is_refused(self):
    """Refused where it is introduced, not later when a model fails to pack."""
    for coefficients, why in (
        ((0.0, 0.0, 0.0, 1.0), 'x**3 is not a squaring chain'),
        ((0.1, 0.5, 0.3), 'several terms'),
        ((0.0, 0.0, 2.0), 'a non-unit coefficient'),
    ):
      with self.assertRaises(ValueError, msg=why) as caught:
        nn.register_activation(
            nn.ActivationPolynomial(name='bad_test', coefficients=coefficients)
        )
      self.assertIn('x**(2**k)', str(caught.exception))
      self.assertNotIn('bad_test', nn._ACTIVATIONS)

  def test_registering_over_a_name_needs_replace(self):
    with self.assertRaisesRegex(ValueError, 'already registered'):
      nn.register_activation(
          nn.ActivationPolynomial(name='square', coefficients=(0.0,) * 4 + (1.0,))
      )

  def test_an_unknown_activation_names_the_registered_ones(self):
    with self.assertRaisesRegex(KeyError, 'square'):
      nn.activation('gelu_exact')


class LayoutTemplateRegistryTest(absltest.TestCase):
  """Gate: per-layer vectorization is a registry of templates."""

  def test_every_shipped_template_is_registered_under_a_vector_kind(self):
    expected = {'conv2d', 'avg_pool2d', 'adaptive_avg_pool2d', 'dense',
                'flatten', 'conv_pool'}
    self.assertContainsSubset(expected, set(nn._LAYOUT_TEMPLATES))
    for template_id in expected:
      template = nn.layout_template(template_id)
      self.assertIn(template.kind, nn.VECTORIZED_KINDS)

  def test_every_template_is_reachable_from_an_operator_target(self):
    """A template no target names is a template the frontend can never pick."""
    reachable = {
        template.template_id
        for candidates in nn._TEMPLATES_BY_TARGET.values()
        for template in [nn.layout_template(name) for name in candidates]
    }
    self.assertContainsSubset(
        {'conv2d', 'avg_pool2d', 'adaptive_avg_pool2d', 'dense', 'flatten',
         'conv_pool'},
        reachable,
    )

  def test_targets_cover_all_three_fx_node_kinds(self):
    kinds = {
        node_op
        for template in nn._LAYOUT_TEMPLATES.values()
        for node_op, _ in template.targets
    }
    self.assertContainsSubset(
        {'call_module', 'call_function', 'call_method'}, kinds
    )

  def test_functional_pooling_and_the_module_resolve_to_one_template(self):
    for target in (('call_module', 'torch.nn.AvgPool2d'),
                   ('call_function', 'torch.nn.functional.avg_pool2d')):
      self.assertEqual(
          nn.select_layout_template(target).template_id, 'avg_pool2d'
      )

  def test_tensor_methods_resolve_to_the_flatten_template(self):
    for method in ('view', 'reshape', 'flatten'):
      self.assertEqual(
          nn.select_layout_template(('call_method', method)).template_id,
          'flatten',
      )

  def test_fusing_a_pool_wins_only_when_a_pool_was_paired(self):
    target = ('call_module', 'torch.nn.Conv2d')
    self.assertGreater(
        nn.layout_template('conv_pool').priority,
        nn.layout_template('conv2d').priority,
    )
    # Highest priority first, so conv_pool is considered before conv2d...
    self.assertEqual(
        [template.template_id
         for template in nn.layout_template_candidates(target)],
        ['conv_pool', 'conv2d'],
    )
    # ...but its condition fails without a paired pool, so conv2d is chosen.
    self.assertEqual(
        nn.select_layout_template(target, {'stride': (1, 1)}).template_id,
        'conv2d',
    )
    self.assertEqual(
        nn.select_layout_template(
            target, {'stride': (1, 1), 'pool_kind': 'avg_pool2d'}
        ).template_id,
        'conv_pool',
    )

  def test_a_user_override_wins_only_where_its_condition_matches(self):
    target = ('call_module', 'torch.nn.Linear')
    nn.register_layout_template(nn.LayoutTemplate(
        template_id='dense_wide_test',
        version='1',
        family='dense',
        kind=nn.LINEAR_TRANSFORM,
        depth_cost=1,
        targets=(target,),
        priority=100,
        condition=lambda parameters, input_spec: (
            input_spec is not None and input_spec.physical_size >= 512
        ),
        weight_roles=('weight', 'bias'),
    ))
    try:
      self.assertEqual(
          [template.template_id
           for template in nn.layout_template_candidates(target)],
          ['dense_wide_test', 'dense'],
      )
      self.assertEqual(
          nn.select_layout_template(target, {}, _spec((1024,))).template_id,
          'dense_wide_test',
      )
      self.assertEqual(
          nn.select_layout_template(target, {}, _spec((8,))).template_id,
          'dense',
      )
    finally:
      nn.unregister_layout_template('dense_wide_test')
    self.assertEqual(
        nn.select_layout_template(target, {}, _spec((1024,))).template_id,
        'dense',
    )

  def test_an_unclaimed_target_names_the_claimed_ones(self):
    with self.assertRaisesRegex(KeyError, 'no layout template vectorizes'):
      nn.select_layout_template(('call_function', 'torch.erf'))

  def test_a_template_instantiates_a_recipe_holding_the_real_weights(self):
    program = _conv_program()
    layer = program.layer('conv')
    recipe = packing.layer_recipe(layer, program.weights)
    self.assertIsInstance(recipe, packing.Conv2dLowering)
    np.testing.assert_array_equal(
        recipe.layer.weight, program.weights[dict(layer.weights)['weight']]
    )
    self.assertEqual(recipe.layer.stride, (2, 2))
    self.assertEqual(recipe.layer.padding, (1, 1))

  def test_recipe_entries_match_a_directly_built_lowering(self):
    program = _conv_program()
    layer = program.layer('conv')
    weights = layer.resolve_weights(program.weights)
    direct = packing.Conv2dLowering(
        input_spec=layer.input_spec,
        output_spec=layer.output_spec,
        layer=packing._Conv2d(
            name='conv',
            weight=weights['weight'],
            bias=weights['bias'],
            stride=(2, 2),
            padding=(1, 1),
        ),
    )
    self.assertEqual(
        list(packing.layer_recipe(layer, program.weights).entries()), list(direct.entries())
    )

  def test_an_unknown_template_names_the_registered_ones(self):
    with self.assertRaisesRegex(KeyError, 'conv2d'):
      nn.layout_template('conv3d')


class ProgramValidationTest(absltest.TestCase):

  def test_a_layer_naming_an_absent_weight_is_refused(self):
    weights = nn.WeightTableBuilder()
    layer = nn.VectorizedLayer(
        name='dense',
        kind=nn.LINEAR_TRANSFORM,
        template_id='dense',
        input_spec=_spec((4,)),
        output_spec=_spec((4,)),
        weights={'weight': 'deadbeef' * 8},
    )
    with self.assertRaisesRegex(ValueError, 'weight table'):
      nn.VectorizedProgram(
          layers=(layer,), weights=weights,
          input_spec=_spec((4,)), output='dense',
      )

  def test_an_out_of_order_layer_is_refused(self):
    weights = nn.WeightTableBuilder()
    dense_weight = weights.add(np.eye(4))
    layer = nn.VectorizedLayer(
        name='second',
        kind=nn.LINEAR_TRANSFORM,
        template_id='dense',
        input_spec=_spec((4,)),
        output_spec=_spec((4,)),
        inputs=('first',),
        weights={'weight': dense_weight},
    )
    with self.assertRaisesRegex(ValueError, 'topological order'):
      nn.VectorizedProgram(
          layers=(layer,), weights=weights,
          input_spec=_spec((4,)), output='second',
      )

  def test_a_polynomial_activation_must_name_its_polynomial(self):
    with self.assertRaisesRegex(ValueError, 'names no polynomial'):
      nn.VectorizedLayer(
          name='act',
          kind=nn.POLYNOMIAL_ACTIVATION,
          template_id='',
          input_spec=_spec((4,)),
          output_spec=_spec((4,)),
      )

# =============================================================================
# Region 3: real torch.fx workload vectorization
# =============================================================================

torch = None
tnn = None
F = None


def _load_torch_frontend():
  global torch, tnn, F
  if torch is not None:
    return
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


class _TorchFrontendTest(absltest.TestCase):
  """Load PyTorch only for tests that exercise ``nn.vectorize``."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    _load_torch_frontend()


# --------------------------------------------------------------------------
# Models. Each is a real torch.nn.Module written the way a user would.
# --------------------------------------------------------------------------


def _models():
  """Build the model classes once torch is known to be importable."""

  class Chain(tnn.Module):
    """Conv -> square -> conv -> square -> flatten -> fc -> square -> fc."""

    def __init__(self):
      super().__init__()
      self.conv1 = tnn.Conv2d(1, 4, 5, stride=2, padding=2)
      self.conv2 = tnn.Conv2d(4, 8, 5, stride=2, padding=2)
      self.fc1 = tnn.Linear(392, 32)
      self.fc2 = tnn.Linear(32, 10)

    def forward(self, x):
      x = self.conv1(x)
      x = x * x
      x = self.conv2(x)
      x = x * x
      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      x = x * x
      return self.fc2(x)

  class Residual(tnn.Module):
    """A skip connection joining the model input with a two-layer branch."""

    def __init__(self):
      super().__init__()
      self.a = tnn.Conv2d(2, 2, 3, padding=1)
      self.b = tnn.Conv2d(2, 2, 3, padding=1)
      self.fc = tnn.Linear(8, 3)

    def forward(self, x):
      y = self.a(x)
      y = y * y
      y = self.b(y)
      z = x + y
      return self.fc(torch.flatten(z, 1))

  class Pooled(tnn.Module):
    """Functional pooling plus the reshape idiom the demos use."""

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 4, 3, padding=1)
      self.fc = tnn.Linear(16, 2)

    def forward(self, x):
      x = F.avg_pool2d(self.conv(x), 2, 2)
      x = x * x
      return self.fc(x.reshape(x.size(0), -1))

  class ModulePooled(tnn.Module):
    """The same shape, built from layer modules and nn.Flatten."""

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 4, 3, padding=1)
      self.pool = tnn.AvgPool2d(2, 2)
      self.flat = tnn.Flatten()
      self.fc = tnn.Linear(16, 2)

    def forward(self, x):
      x = self.pool(self.conv(x))
      x = x * x
      return self.fc(self.flat(x))

  class AdaptivePooled(tnn.Module):

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 4, 3, padding=1)
      self.fc = tnn.Linear(16, 2)

    def forward(self, x):
      x = F.adaptive_avg_pool2d(self.conv(x), (2, 2))
      x = x * x
      return self.fc(torch.flatten(x, 1))

  class ReluNet(tnn.Module):

    def __init__(self):
      super().__init__()
      self.fc1 = tnn.Linear(4, 4)
      self.act = tnn.ReLU()
      self.fc2 = tnn.Linear(4, 2)

    def forward(self, x):
      return self.fc2(self.act(self.fc1(x)))

  class FunctionalRelu(tnn.Module):

    def __init__(self):
      super().__init__()
      self.fc1 = tnn.Linear(4, 4)
      self.fc2 = tnn.Linear(4, 2)

    def forward(self, x):
      return self.fc2(F.relu(self.fc1(x)))

  class ShapeGuard(tnn.Module):
    """The AlexNet idiom: data-dependent control flow on a shape."""

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 2, 3, padding=1)
      self.fc = tnn.Linear(8, 2)

    def forward(self, x):
      if x.shape[-1] == 32:
        x = F.avg_pool2d(x, 2, 2)
      x = self.conv(x)
      x = x * x
      return self.fc(torch.flatten(x, 1))

  class BatchNormed(tnn.Module):

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 2, 3, padding=1)
      self.bn = tnn.BatchNorm2d(2)
      self.fc = tnn.Linear(8, 2)

    def forward(self, x):
      x = self.bn(self.conv(x))
      x = x * x
      return self.fc(torch.flatten(x, 1))

  class CrossMultiply(tnn.Module):

    def __init__(self):
      super().__init__()
      self.a = tnn.Linear(4, 4)
      self.b = tnn.Linear(4, 4)

    def forward(self, x):
      return self.a(x) * self.b(x)

  class MaxPooled(tnn.Module):

    def __init__(self):
      super().__init__()
      self.conv = tnn.Conv2d(1, 2, 3, padding=1)
      self.fc = tnn.Linear(8, 2)

    def forward(self, x):
      x = F.max_pool2d(self.conv(x), 2, 2)
      return self.fc(torch.flatten(x, 1))

  return locals()


# --------------------------------------------------------------------------
# A cleartext evaluator for a VectorizedProgram, so the frontend graph can be
# checked against the torch model it came from.
# --------------------------------------------------------------------------


def _dense_matrix(layer, weights):
  recipe = packing.layer_recipe(layer, weights)
  matrix = np.zeros(
      (layer.output_spec.physical_size, layer.input_spec.physical_size),
      dtype=np.float64,
  )
  for row, column, value in recipe.entries():
    matrix[row, column] += value
  return matrix, recipe


def _bias_vector(layer, recipe):
  logical = getattr(recipe, 'layer', None)
  bias = getattr(logical, 'bias', None) if logical is not None else None
  if bias is None:
    return None
  shape = layer.output_spec.shape
  slots = np.zeros(layer.output_spec.physical_size, dtype=np.float64)
  for coordinate in np.ndindex(shape):
    slots[layer.output_spec.coordinate_to_slot(coordinate)] = bias[
        coordinate[0]
    ]
  return slots


def evaluate(program, sample):
  """Run a VectorizedProgram in cleartext over flat slot vectors."""
  values = {nn.PROGRAM_INPUT: np.asarray(sample, dtype=np.float64).ravel()}
  for layer in program.layers:
    sources = [
        values[name] for name in (layer.inputs or (nn.PROGRAM_INPUT,))
    ]
    if layer.kind == nn.LINEAR_TRANSFORM:
      matrix, recipe = _dense_matrix(layer, program.weights)
      result = matrix @ sources[0]
      bias = _bias_vector(layer, recipe)
      if bias is not None:
        result = result + bias
    elif layer.kind == nn.POLYNOMIAL_ACTIVATION:
      result = sum(
          coefficient * sources[0] ** power
          for power, coefficient in enumerate(layer.activation_coefficients)
      )
    elif layer.kind == nn.ADD:
      result = sources[0] + sources[1]
    elif layer.kind == nn.LAYOUT:
      matrix, _ = _dense_matrix(layer, program.weights)
      result = matrix @ sources[0]
    else:
      raise AssertionError(f'unhandled kind {layer.kind}')
    values[layer.name] = result
  return values[program.output]


def vectorize_quietly(model, shape, **options):
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', nn.ActivationSubstitutionWarning)
    return nn.vectorize(model, shape, **options)


class NodeCoverageTest(_TorchFrontendTest):
  """Every fx node kind the demos emit maps to something explicit."""

  def setUp(self):
    super().setUp()
    self.models = _models()

  def test_a_chain_of_modules_and_tensor_squares(self):
    program = nn.vectorize(self.models['Chain'](), (1, 28, 28))
    self.assertEqual(
        [(layer.name, layer.kind, layer.template_id or layer.activation_name)
         for layer in program.layers],
        [('conv1', nn.LINEAR_TRANSFORM, 'conv2d'),
         ('mul', nn.POLYNOMIAL_ACTIVATION, 'square'),
         ('conv2', nn.LINEAR_TRANSFORM, 'conv2d'),
         ('mul_1', nn.POLYNOMIAL_ACTIVATION, 'square'),
         ('fc1', nn.LINEAR_TRANSFORM, 'dense'),
         ('mul_2', nn.POLYNOMIAL_ACTIVATION, 'square'),
         ('fc2', nn.LINEAR_TRANSFORM, 'dense')],
    )
    # The depth and slot high-water mark the LeNet demo was measured at.
    self.assertEqual(program.critical_depth(), 7)
    self.assertEqual(program.max_live_slots(), 784)

  def test_size_and_view_are_metadata_and_emit_no_layer(self):
    program = nn.vectorize(self.models['Chain'](), (1, 28, 28))
    for layer in program.layers:
      self.assertNotEqual(layer.kind, nn.LAYOUT)
    # The reshape still took effect: fc1 consumes the flattened shape.
    self.assertEqual(program.layer('fc1').input_spec.shape, (392,))
    self.assertEqual(program.layer('mul_1').output_spec.shape, (8, 7, 7))

  def test_a_residual_add_keeps_both_dag_edges(self):
    program = nn.vectorize(self.models['Residual'](), (2, 2, 2))
    add = program.layer('add')
    self.assertEqual(add.kind, nn.ADD)
    self.assertEqual(add.inputs, (nn.PROGRAM_INPUT, 'b'))
    self.assertEqual(add.depth_cost, 0)
    # Branch max, not sum: a(1) + square(1) + b(1) = 3, then fc(1).
    self.assertEqual(program.critical_depth(), 4)

  def test_functional_and_module_pooling_agree(self):
    functional = vectorize_quietly(self.models['Pooled'](), (1, 4, 4))
    modular = vectorize_quietly(self.models['ModulePooled'](), (1, 4, 4))
    self.assertEqual(
        [layer.template_id for layer in functional.layers],
        [layer.template_id for layer in modular.layers],
    )
    self.assertEqual(
        [layer.output_spec.shape for layer in functional.layers],
        [layer.output_spec.shape for layer in modular.layers],
    )

  def test_adaptive_pooling_is_recognized(self):
    program = nn.vectorize(self.models['AdaptivePooled'](), (1, 4, 4))
    self.assertEqual(program.layer('conv').template_id, 'conv_pool')
    self.assertEqual(
        dict(program.layer('conv').params)['pool_kind'],
        'adaptive_avg_pool2d',
    )

  def test_pool_fusion_saves_a_level(self):
    model = self.models['Pooled']()
    fused = nn.vectorize(model, (1, 4, 4))
    separate = nn.vectorize(model, (1, 4, 4), fuse_pooling=False)
    self.assertEqual(fused.layer('conv').template_id, 'conv_pool')
    self.assertEqual(separate.layer('conv').template_id, 'conv2d')
    self.assertEqual(fused.critical_depth(), separate.critical_depth() - 1)

  def test_torch_flatten_is_metadata(self):
    program = nn.vectorize(self.models['Residual'](), (2, 2, 2))
    self.assertEqual(program.layer('fc').input_spec.shape, (8,))


class ActivationSubstitutionTest(_TorchFrontendTest):

  def setUp(self):
    super().setUp()
    self.models = _models()

  def test_a_relu_module_warns_and_names_the_polynomial(self):
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      program = nn.vectorize(self.models['ReluNet'](), (4,))
    messages = [
        str(warning.message) for warning in caught
        if issubclass(warning.category, nn.ActivationSubstitutionWarning)
    ]
    self.assertLen(messages, 1)
    self.assertIn("'square'", messages[0])
    self.assertIn('degree 2', messages[0])
    self.assertIn('relu', messages[0])
    self.assertIn('register_activation', messages[0])
    self.assertEqual(program.layer('act').kind, nn.POLYNOMIAL_ACTIVATION)
    self.assertEqual(
        program.layer('act').activation_coefficients, (0.0, 0.0, 1.0)
    )

  def test_a_functional_relu_warns_too(self):
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      nn.vectorize(self.models['FunctionalRelu'](), (4,))
    self.assertTrue(any(
        issubclass(warning.category, nn.ActivationSubstitutionWarning)
        for warning in caught
    ))

  def test_a_tensor_square_is_not_a_substitution_and_does_not_warn(self):
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      nn.vectorize(self.models['Chain'](), (1, 28, 28))
    self.assertEmpty([
        warning for warning in caught
        if issubclass(warning.category, nn.ActivationSubstitutionWarning)
    ])

  def test_a_user_override_changes_the_substituted_polynomial(self):
    nn.register_activation(
        nn.ActivationPolynomial(
            name='relu_quartic_test',
            coefficients=(0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        substitutes=('relu',),
    )
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        program = nn.vectorize(self.models['ReluNet'](), (4,))
      self.assertEqual(
          program.layer('act').activation_name, 'relu_quartic_test'
      )
      self.assertEqual(program.layer('act').depth_cost, 2)
      self.assertTrue(any(
          'relu_quartic_test' in str(warning.message) for warning in caught
      ))
    finally:
      nn._ACTIVATIONS.pop('relu_quartic_test', None)
      nn._ACTIVATION_DEFAULTS['relu'] = 'square'

  def test_warnings_can_be_suppressed_without_changing_the_program(self):
    model = self.models['ReluNet']()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      loud = nn.vectorize(model, (4,))
      quiet = nn.vectorize(model, (4,), warn=False)
    self.assertEqual(loud.digest, quiet.digest)
    self.assertLen([
        warning for warning in caught
        if issubclass(warning.category, nn.ActivationSubstitutionWarning)
    ], 1)


class RejectionTest(_TorchFrontendTest):
  """Unsupported nodes raise something a user can act on -- never skipped."""

  def setUp(self):
    super().setUp()
    self.models = _models()

  def test_data_dependent_control_flow_is_rejected(self):
    with self.assertRaises(nn.VectorizeError) as caught:
      nn.vectorize(self.models['ShapeGuard'](), (1, 4, 4))
    message = str(caught.exception)
    self.assertIn('control flow', message)
    self.assertIn('__init__', message)

  def test_batch_norm_is_rejected_with_folding_guidance(self):
    with self.assertRaises(nn.VectorizeError) as caught:
      nn.vectorize(self.models['BatchNormed'](), (1, 2, 2))
    message = str(caught.exception)
    self.assertIn('BatchNorm', message)
    self.assertIn('folded', message)

  def test_multiplying_two_distinct_values_is_rejected(self):
    with self.assertRaises(nn.VectorizeError) as caught:
      nn.vectorize(self.models['CrossMultiply'](), (4,))
    self.assertIn('two different values', str(caught.exception))

  def test_max_pooling_is_rejected_by_name(self):
    with self.assertRaises(nn.VectorizeError) as caught:
      nn.vectorize(self.models['MaxPooled'](), (1, 4, 4))
    message = str(caught.exception)
    self.assertIn('max_pool2d', message)
    self.assertIn('register_layout_template', message)

  def test_a_non_module_is_rejected(self):
    with self.assertRaises(nn.VectorizeError):
      nn.vectorize(lambda x: x, (4,))

  def test_grouped_convolution_is_rejected(self):

    class Grouped(tnn.Module):

      def __init__(self):
        super().__init__()
        self.conv = tnn.Conv2d(2, 2, 3, padding=1, groups=2)
        self.fc = tnn.Linear(8, 2)

      def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))

    with self.assertRaises(nn.VectorizeError) as caught:
      nn.vectorize(Grouped(), (2, 2, 2))
    self.assertIn('groups', str(caught.exception))

  def test_no_node_is_silently_dropped(self):
    """Every traced node is either a layer, metadata, fused, or an error."""
    model = self.models['Chain']()
    program = nn.vectorize(model, (1, 28, 28))
    graph = torch.fx.symbolic_trace(model).graph
    computational = [
        node for node in graph.nodes
        if node.op in ('call_module', 'call_function')
    ]
    # Every computational node became exactly one layer here (no fusion in
    # this model, and size/view are call_method metadata).
    self.assertLen(program.layers, len(computational))


class CallerStateTest(_TorchFrontendTest):
  """vectorize() must leave the caller's model exactly as it found it.

  Shape inference executes the graph, and modules with buffers update them in
  place. A BatchNorm left in training mode would fold vectorize()'s synthetic
  zero example into the caller's running statistics.
  """

  def setUp(self):
    super().setUp()
    self.models = _models()

  def _snapshot(self, model):
    state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    flags = {name: child.training
             for name, child in model.named_modules()}
    return state, flags

  def _assert_unchanged(self, model, snapshot):
    state, flags = snapshot
    current = model.state_dict()
    self.assertEqual(set(current), set(state))
    for name, tensor in state.items():
      self.assertTrue(
          torch.equal(current[name], tensor),
          f'{name} changed: {tensor} -> {current[name]}',
      )
    self.assertEqual(
        {name: child.training for name, child in model.named_modules()}, flags
    )

  def test_a_training_mode_model_is_not_mutated(self):
    model = self.models['Chain']()
    model.train()
    snapshot = self._snapshot(model)
    nn.vectorize(model, (1, 28, 28))
    self._assert_unchanged(model, snapshot)
    self.assertTrue(model.training)

  def test_batch_norm_running_statistics_survive_a_rejection(self):
    """The rejecting path runs shape inference first, so it must be clean."""
    model = self.models['BatchNormed']()
    model.train()
    snapshot = self._snapshot(model)
    with self.assertRaises(nn.VectorizeError):
      nn.vectorize(model, (1, 2, 2))
    self._assert_unchanged(model, snapshot)
    self.assertTrue(model.bn.training)
    self.assertEqual(int(model.bn.num_batches_tracked), 0)

  def test_a_control_flow_rejection_leaves_the_model_untouched(self):
    model = self.models['ShapeGuard']()
    model.train()
    snapshot = self._snapshot(model)
    with self.assertRaises(nn.VectorizeError):
      nn.vectorize(model, (1, 4, 4))
    self._assert_unchanged(model, snapshot)

  def test_running_a_batch_norm_model_directly_would_change_it(self):
    """Confirms the guard above is not vacuous."""
    model = self.models['BatchNormed']()
    model.train()
    before = model.bn.running_mean.detach().clone()
    model(torch.zeros(1, 1, 2, 2))
    self.assertFalse(torch.equal(model.bn.running_mean, before))

  def test_the_program_holds_the_caller_weights_by_value(self):
    model = self.models['Chain']().eval()
    program = nn.vectorize(model, (1, 28, 28))
    expected = model.fc2.weight.detach().numpy().astype(np.float64)
    digest = nn.weight_digest(nn.normalize_weight(expected))
    self.assertIn(digest, program.weights.digests())
    with torch.no_grad():
      model.fc2.weight.mul_(2.0)
    np.testing.assert_allclose(program.weights[digest], expected)


class TemplateSelectionTest(_TorchFrontendTest):
  """Template conditions are live in the parser, with the real input spec."""

  def setUp(self):
    super().setUp()
    self.models = _models()

  def tearDown(self):
    super().tearDown()
    for template_id in ('conv_wide_test', 'conv_narrow_test',
                        'dense_any_test', 'softmax_test'):
      nn.unregister_layout_template(template_id)

  def _register_conv_override(self, template_id, threshold, priority=50):
    return nn.register_layout_template(nn.LayoutTemplate(
        template_id=template_id,
        version='1',
        family='conv2d',
        kind=nn.LINEAR_TRANSFORM,
        depth_cost=1,
        targets=(('call_module', 'torch.nn.Conv2d'),),
        weight_roles=('weight', 'bias'),
        priority=priority,
        condition=lambda parameters, input_spec: (
            input_spec is not None
            and input_spec.physical_size >= threshold
        ),
    ))

  def test_a_condition_true_for_the_real_input_spec_wins(self):
    # Chain's first conv sees 1x28x28 = 784 slots.
    self._register_conv_override('conv_wide_test', threshold=784)
    program = nn.vectorize(self.models['Chain'](), (1, 28, 28))
    self.assertEqual(program.layer('conv1').template_id, 'conv_wide_test')

  def test_a_condition_false_for_the_real_input_spec_falls_back(self):
    self._register_conv_override('conv_narrow_test', threshold=10**6)
    program = nn.vectorize(self.models['Chain'](), (1, 28, 28))
    self.assertEqual(program.layer('conv1').template_id, 'conv2d')

  def test_one_override_can_apply_to_some_layers_and_not_others(self):
    """conv1 sees 784 slots, conv2 sees 4x14x14 = 784 too; use a tighter cut."""
    self._register_conv_override('conv_wide_test', threshold=785)
    program = nn.vectorize(self.models['Chain'](), (1, 28, 28))
    self.assertEqual(program.layer('conv1').template_id, 'conv2d')

  def test_a_condition_sees_weight_and_output_shape(self):
    seen = []

    nn.register_layout_template(nn.LayoutTemplate(
        template_id='conv_wide_test',
        version='1',
        family='conv2d',
        kind=nn.LINEAR_TRANSFORM,
        depth_cost=1,
        targets=(('call_module', 'torch.nn.Conv2d'),),
        weight_roles=('weight', 'bias'),
        priority=50,
        condition=lambda parameters, input_spec: (
            seen.append(dict(parameters)) or False
        ),
    ))
    nn.vectorize(self.models['Chain'](), (1, 28, 28))
    self.assertNotEmpty(seen)
    self.assertEqual(seen[0]['weight_shape'], (4, 1, 5, 5))
    self.assertEqual(seen[0]['output_shape'], (4, 14, 14))
    self.assertTrue(seen[0]['has_bias'])
    self.assertEqual(seen[0]['stride'], (2, 2))

  def test_a_subclass_of_a_supported_layer_resolves_through_the_mro(self):

    class MyConv(tnn.Conv2d):
      pass

    class Wrapped(tnn.Module):

      def __init__(self):
        super().__init__()
        self.conv = MyConv(1, 2, 3, padding=1)
        self.fc = tnn.Linear(8, 2)

      def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))

    program = nn.vectorize(Wrapped(), (1, 2, 2))
    self.assertEqual(program.layer('conv').template_id, 'conv2d')

  def test_a_template_for_an_unsupported_family_is_refused(self):
    """Extensibility covers better templates for known families only.

    A template decides how a family is vectorized; the parser still has to
    read the torch layer to fill it in. Claiming an unrelated module type
    would leave the parser guessing, so it is refused at registration.
    """
    with self.assertRaises(ValueError) as caught:
      nn.register_layout_template(nn.LayoutTemplate(
          template_id='softmax_test',
          version='1',
          family='softmax',
          kind=nn.LINEAR_TRANSFORM,
          depth_cost=1,
          targets=(('call_module', 'torch.nn.Softmax'),),
      ))
    message = str(caught.exception)
    self.assertIn('family', message)
    self.assertIn('parameter extractor', message)
    self.assertNotIn('softmax_test', nn._LAYOUT_TEMPLATES)

  # Removed: test_a_user_can_register_the_parser_for_a_new_family.
  # It exercised nn.register_vectorization_family / nn.VectorizationFamily /
  # nn.unregister_vectorization_family, added in 8bb666a and deleted from nn.py
  # by 405cac2 (PR #25) when SUPPORTED_FAMILIES became a frozenset. Closing the
  # family set was deliberate, so the test -- not the API -- was the stale side.
  # The remaining template-registration tests still cover the extension point
  # that survived: register_layout_template / unregister_layout_template.

  def test_every_shipped_template_declares_a_supported_family(self):
    for template_id, template in nn._LAYOUT_TEMPLATES.items():
      self.assertIn(
          template.family, nn.SUPPORTED_FAMILIES,
          f'{template_id} declares family {template.family!r}',
      )

  def test_declining_a_fusion_leaves_the_pool_as_its_own_layer(self):
    nn.register_layout_template(nn.LayoutTemplate(
        template_id='conv_narrow_test',
        version='1',
        family='conv2d',
        kind=nn.LINEAR_TRANSFORM,
        depth_cost=1,
        targets=(('call_module', 'torch.nn.Conv2d'),),
        weight_roles=('weight', 'bias'),
        priority=999,
    ))
    program = vectorize_quietly(self.models['Pooled'](), (1, 4, 4))
    self.assertEqual(program.layer('conv').template_id, 'conv_narrow_test')
    self.assertEqual(program.layer('avg_pool2d').template_id, 'avg_pool2d')
    self.assertEqual(program.layer('conv').output_spec.shape, (4, 4, 4))


class BroadcastTest(_TorchFrontendTest):
  """Ciphertext addition is slotwise, so a broadcast add cannot be an ADD."""

  def test_a_broadcasting_add_is_rejected(self):

    class Broadcasting(tnn.Module):

      def __init__(self):
        super().__init__()
        self.a = tnn.Conv2d(1, 2, 3, padding=1)
        self.b = tnn.Conv2d(1, 2, 2)
        self.fc = tnn.Linear(8, 2)

      def forward(self, x):
        # (2, 2, 2) + (2, 1, 1) broadcasts in torch.
        z = self.a(x) + self.b(x)
        return self.fc(torch.flatten(z, 1))

    with self.assertRaises(nn.VectorizeError) as caught:
      nn.vectorize(Broadcasting(), (1, 2, 2))
    message = str(caught.exception)
    self.assertIn('broadcast', message)
    self.assertIn('slotwise', message)

  def test_a_matching_add_is_accepted(self):
    program = nn.vectorize(_models()['Residual'](), (2, 2, 2))
    add = program.layer('add')
    self.assertEqual(add.input_spec.shape, add.output_spec.shape)


class EquivalenceTest(_TorchFrontendTest):
  """The vectorized graph computes what the torch model computes."""

  def setUp(self):
    super().setUp()
    self.models = _models()

  def _check(self, model, shape, tolerance=1e-9, samples=5):
    model = model.double().eval()
    program = vectorize_quietly(model, shape)
    generator = np.random.default_rng(0)
    worst = 0.0
    for _ in range(samples):
      sample = generator.normal(size=shape) * 0.5
      with torch.no_grad():
        expected = model(
            torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
        ).numpy().ravel()
      actual = evaluate(program, sample)
      self.assertEqual(actual.shape, expected.shape)
      worst = max(worst, float(np.max(np.abs(actual - expected))))
    self.assertLess(worst, tolerance, f'max deviation {worst:.3e}')
    return worst

  def test_a_conv_and_dense_chain_matches_torch(self):
    self._check(self.models['Chain'](), (1, 28, 28))

  def test_a_residual_matches_torch(self):
    self._check(self.models['Residual'](), (2, 2, 2))

  def test_fused_pooling_matches_torch(self):
    self._check(self.models['Pooled'](), (1, 4, 4))

  def test_unfused_pooling_matches_fused_pooling(self):
    model = self.models['Pooled']().double().eval()
    fused = vectorize_quietly(model, (1, 4, 4))
    separate = vectorize_quietly(model, (1, 4, 4), fuse_pooling=False)
    generator = np.random.default_rng(1)
    for _ in range(5):
      sample = generator.normal(size=(1, 4, 4)) * 0.5
      np.testing.assert_allclose(
          evaluate(fused, sample), evaluate(separate, sample), atol=1e-9
      )

  def test_adaptive_pooling_matches_torch(self):
    self._check(self.models['AdaptivePooled'](), (1, 4, 4))

  def test_module_pooling_matches_torch(self):
    self._check(self.models['ModulePooled'](), (1, 4, 4))

  def test_a_substituted_relu_matches_the_squared_model(self):
    """Substitution changes the function; it must change it the stated way."""
    model = self.models['ReluNet']().double().eval()
    program = vectorize_quietly(model, (4,))
    generator = np.random.default_rng(2)
    for _ in range(5):
      sample = generator.normal(size=(4,))
      with torch.no_grad():
        hidden = model.fc1(torch.tensor(sample, dtype=torch.float64))
        expected = model.fc2(hidden * hidden).numpy()
      np.testing.assert_allclose(
          evaluate(program, sample), expected, atol=1e-9
      )


class ProgramIdentityTest(_TorchFrontendTest):

  def setUp(self):
    super().setUp()
    self.models = _models()

  def test_vectorizing_the_same_model_twice_gives_one_digest(self):
    model = self.models['Chain']()
    self.assertEqual(
        nn.vectorize(model, (1, 28, 28)).digest,
        nn.vectorize(model, (1, 28, 28)).digest,
    )

  def test_a_changed_weight_changes_the_digest(self):
    model = self.models['Chain']()
    before = nn.vectorize(model, (1, 28, 28)).digest
    with torch.no_grad():
      model.fc2.weight[0, 0] += 1.0
    self.assertNotEqual(nn.vectorize(model, (1, 28, 28)).digest, before)

  def test_a_vectorized_program_survives_pickling(self):
    program = nn.vectorize(self.models['Residual'](), (2, 2, 2))
    restored = pickle.loads(pickle.dumps(program))
    self.assertEqual(restored.digest, program.digest)
    generator = np.random.default_rng(3)
    sample = generator.normal(size=(2, 2, 2))
    np.testing.assert_allclose(
        evaluate(restored, sample), evaluate(program, sample), atol=1e-12
    )

  def test_weights_are_deduplicated_by_content(self):
    program = nn.vectorize(self.models['Chain'](), (1, 28, 28))
    # 4 layers x (weight + bias)
    self.assertLen(program.weights.digests(), 8)

  def test_the_program_names_no_he_concept(self):
    program = nn.vectorize(self.models['Residual'](), (2, 2, 2))
    for layer in program.layers:
      self.assertIn(layer.kind, nn.VECTORIZED_KINDS)
      for name, _ in layer.params:
        for forbidden in ('level', 'modulus', 'scale', 'nsd', 'ring',
                          'rescale', 'bootstrap'):
          self.assertNotIn(forbidden, name.lower())


class ImportIsolationTest(absltest.TestCase):
  """Importing the frontend must not drag in torch, JAX, or the crypto stack."""

  def test_importing_jaxite_word_nn_loads_neither_torch_nor_jax(self):
    script = (
        'import sys\n'
        'import jaxite_word.nn\n'
        'heavy = [name for name in ("torch", "jax", "jaxite_word.ckks_ctx",'
        ' "jaxite_word.mapping", "jaxite_word.polynomial", "he_params")'
        ' if name in sys.modules]\n'
        'assert not heavy, heavy\n'
        'assert hasattr(jaxite_word.nn, "vectorize")\n'
        'print("clean")\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, cwd=_repository_root(),
    )
    self.assertEqual(
        result.returncode, 0,
        f'stdout={result.stdout}\nstderr={result.stderr}',
    )
    self.assertIn('clean', result.stdout)


def _repository_root():
  import os
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Region 4: public frontend and documentation contracts
# =============================================================================

class FrontendSurfaceTest(parameterized.TestCase):

  @parameterized.parameters(
      'Conv2d', 'Dense', 'AvgPool2d', 'AdaptiveAvgPool2d', 'Flatten',
      'LoweredModel', 'LoweringPolicy', 'lower_model', 'infer_input_spec',
      'DEFAULT_BSGS_RATIO', '_LINEAR_OPTION_NAMES',
      '_normalize_linear_options', '_TransformRecipe', 'Conv2dLowering',
      '_MatrixFreeSource', '_matrix_constant', '_bias_slots',
  )
  def test_a_retired_frontend_name_is_absent(self, name):
    self.assertFalse(hasattr(nn, name), f'nn.{name} is still reachable')

  def test_every_exported_name_resolves(self):
    for name in nn.__all__:
      self.assertTrue(hasattr(nn, name), f'nn.__all__ names missing {name}')

  def test_the_frontend_builds_no_physical_constant(self):
    """Emitting slot constants is the packing phase's job."""
    import ast
    import pathlib
    source = pathlib.Path(nn.__file__).read_text()
    called = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for physical in ('PlainSlots', 'SparseDiagonals', '_LazySparseDiagonals',
                     'DenseMatrix'):
      self.assertNotIn(physical, called)



_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DOCS = (
    _ROOT / 'jaxite_word' / 'API_REFERENCE.md',
    _ROOT / 'demos' / 'README.md',
    _ROOT / 'jaxite_word' / 'README.md',
    _ROOT / 'README.md',
)

# Names the restructure removed. A doc that still uses one is telling a reader
# to write code that cannot run.
_RETIRED = (
    'nn.Sequential', 'nn.Identity', 'nn.Linear', 'nn.Square', 'nn.Rotate',
    'nn.Rescale', 'nn.Bootstrap', 'nn.AddPlain', 'nn.MulPlain',
    'nn.ParallelSum', 'nn.PlainSlots', 'nn.DenseMatrix',
    'nn.SparseDiagonals', 'nn.Conv2d', 'nn.Dense', 'nn.AvgPool2d',
    'nn.Flatten', 'nn.lower_model', 'nn.LoweredModel', 'nn.LoweringPolicy',
    'Packing(network', 'Mapping.from_model', 'PackedProgram',
    'packing.Sequential', 'packing.Linear', 'packing.Bootstrap',
)

# Ownership claims that were true before the restructure and are not now.
_FALSE_CLAIMS = (
    (r'ckks_ctx\.py[^|\n]*\|[^|\n]*`Mapping`',
     'ckks_ctx.py no longer contains Mapping; mapping.py does'),
    (r'Mapping analyzer[^.]*in a marked\s+implementation block in\s+`?ckks_ctx',
     'the Mapping analyzer lives in mapping.py'),
    (r'frontend contains HE primitives',
     'the frontend contains semantics, not HE primitives'),
)


def _docs():
  return [path for path in _DOCS if path.exists()]


class RetiredNameTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (path.name.replace('.', '_') + '_' + str(index), path)
      for index, path in enumerate(_docs())
  )
  def test_a_doc_uses_no_retired_name(self, path):
    text = path.read_text()
    # torch.nn.Linear and friends are correct; only our own nn is retired.
    text = text.replace('torch.nn.', 'torch_nn_')
    offenders = sorted({name for name in _RETIRED if name in text})
    self.assertEmpty(
        offenders,
        f'{path.name} advertises retired API: {offenders}',
    )

  @parameterized.named_parameters(
      (path.name.replace('.', '_') + '_' + str(index), path)
      for index, path in enumerate(_docs())
  )
  def test_a_doc_makes_no_false_ownership_claim(self, path):
    text = path.read_text()
    for pattern, why in _FALSE_CLAIMS:
      self.assertIsNone(
          re.search(pattern, text), f'{path.name}: {why}'
      )


# Patterns built from live names that nonetheless describe the retired flow.
# A name-only check cannot see these: ``Mapping``, ``packing`` and ``pack`` all
# still exist -- it is the shape of the call that is dead.
_RETIRED_PATTERNS = (
    (
        r'Mapping\(\s*\n?\s*packing\s*,',
        'Mapping takes the Packing that packing.pack returns, not a '
        'Packing-plus-parameters pair',
    ),
    (
        r'Mapping\([^)]*\bparams\b',
        'a normal Mapping takes no parameters dict; only a '
        'packing.test_only_parameters(...) override is accepted',
    ),
    (
        r'Mapping\([^)]*\bdnum\s*=\s*dnum\b',
        'dnum comes from the ring; a normal caller passes none',
    ),
    (
        r'\bpacking\.pack\(\s*(?:example|value|model_input|inputs)\b',
        'packing.pack takes a VectorizedProgram; the old object-style '
        'packing.pack(sample) was the input packer, now mapping.encrypt_input',
    ),
    (
        r'\bpacking\.pack\(\s*\)',
        'packing.pack needs a VectorizedProgram argument',
    ),
)

# A fenced block introduced by one of these is a record of what was measured,
# not an instruction to copy. Historical facts are preserved deliberately.
_HISTORICAL_MARKERS = (
    'historical', 'Historical', 'benchmark', 'Benchmark', 'measured',
    'as of', 'previously', 'legacy run', 'reference numbers',
)


def _current_use_blocks(text):
    """Fenced blocks that read as instructions, not as recorded history."""
    blocks = []
    for match in re.finditer(r'```[a-z]*\n(.*?)```', text, re.S):
        prologue = text[max(0, match.start() - 400):match.start()]
        if any(marker in prologue for marker in _HISTORICAL_MARKERS):
            continue
        blocks.append((match.start(), match.group(1)))
    return blocks


class CurrentUsePatternTest(parameterized.TestCase):
    """Live names can still spell a dead call; catch the shape."""

    @parameterized.named_parameters(
        (path.name.replace('.', '_') + '_' + str(index), path)
        for index, path in enumerate(_docs())
    )
    def test_no_current_use_block_shows_a_retired_call(self, path):
        text = path.read_text()
        offenders = []
        for offset, block in _current_use_blocks(text):
            line = text[:offset].count('\n') + 1
            for pattern, why in _RETIRED_PATTERNS:
                if re.search(pattern, block):
                    offenders.append(f'{path.name}:~{line}: {why}')
        self.assertEmpty(sorted(set(offenders)))

    @parameterized.named_parameters(
        (path.name.replace('.', '_') + '_' + str(index), path)
        for index, path in enumerate(_docs())
    )
    def test_prose_makes_no_retired_call_claim(self, path):
        """Outside fenced blocks too -- inline code spreads just as far."""
        text = re.sub(r'```[a-z]*\n.*?```', '', path.read_text(), flags=re.S)
        for pattern, why in _RETIRED_PATTERNS[:1]:
            self.assertIsNone(
                re.search(pattern, text), f'{path.name}: {why}'
            )


class DocumentedSurfaceTest(absltest.TestCase):
  """Every jaxite_word symbol a doc shows in code must actually exist."""

  _MODULES = {'nn': nn, 'packing': packing}

  def test_documented_symbols_resolve(self):
    missing = []
    for path in _docs():
      text = path.read_text().replace('torch.nn.', 'torch_nn_')
      for module_name, module in self._MODULES.items():
        for symbol in re.findall(rf'\b{module_name}\.([A-Za-z_][A-Za-z0-9_]*)',
                                 text):
          if symbol in ('py',):  # "nn.py" is a filename, not a symbol
            continue
          if not hasattr(module, symbol):
            missing.append(f'{path.name}: {module_name}.{symbol}')
    self.assertEmpty(sorted(set(missing)))


class ExampleIsExecutableTest(absltest.TestCase):
  """The reference example must parse and name only live API."""

  def test_the_api_reference_example_parses(self):
    reference = _ROOT / 'jaxite_word' / 'API_REFERENCE.md'
    blocks = re.findall(r'```python\n(.*?)```', reference.read_text(), re.S)
    self.assertNotEmpty(blocks)
    for block in blocks:
      # Examples elide bindings such as `model_input`; parsing is what proves
      # the shown API spelling is real.
      ast.parse(block)

  def test_the_documented_pipeline_names_the_real_functions(self):
    reference = (_ROOT / 'jaxite_word' / 'API_REFERENCE.md').read_text()
    for call in ('nn.vectorize(', 'packing.pack(', 'Mapping('):
      self.assertIn(call, reference)
    self.assertTrue(callable(nn.vectorize))
    self.assertTrue(callable(packing.pack))



class DocumentedCommandTest(absltest.TestCase):
  """A documented command line must be one the script actually accepts.

  The README told readers to `build --batch 1 --mux` long after `--mux` stopped
  selecting anything, because nothing compared the prose against the parser.
  This reads the flags straight out of each driver's `add_argument` calls.
  """

  def _accepted(self, path):
    """(flags, positional words) the script's parser really takes."""
    tree = ast.parse(path.read_text(), filename=str(path))
    flags, words = set(), set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
          and node.func.attr in ('add_argument', 'add_parser'):
        for argument in node.args:
          if isinstance(argument, ast.Constant) and isinstance(
              argument.value, str
          ):
            (flags if argument.value.startswith('-') else words).add(
                argument.value
            )
      elif isinstance(node, ast.ClassDef):
        words.add(node.name)  # absltest takes a test class name
      elif isinstance(node, ast.FunctionDef) and node.name.startswith('cmd_'):
        words.add(node.name[len('cmd_'):])
    return flags, words

  def test_every_documented_driver_command_is_accepted(self):
    offenders = []
    for doc in _docs():
      text = doc.read_text()
      for offset, block in _current_use_blocks(text):
        for line in block.splitlines():
          match = re.search(r'python3?\s+(\w+_he_perf_test\.py)\s*(.*)', line)
          if not match:
            continue
          script = _ROOT / 'demos' / match.group(1)
          if not script.exists():
            offenders.append(f'{doc.name}: no such script {match.group(1)}')
            continue
          flags, words = self._accepted(script)
          line_number = text[:offset].count('\n') + 1
          tokens = match.group(2).split()
          for index, token in enumerate(tokens):
            if token.startswith('-'):
              name = token.split('=')[0]
              if name not in flags:
                offenders.append(
                    f'{doc.name}:~{line_number}: {match.group(1)} takes no '
                    f'{name}'
                )
            elif index == 0 and token not in words:
              offenders.append(
                  f'{doc.name}:~{line_number}: {match.group(1)} has no '
                  f'{token!r} subcommand or test'
              )
    self.assertEmpty(sorted(set(offenders)))


if __name__ == '__main__':
  absltest.main()
