"""Architecture gates for the encrypted-demo API contract.

These tests intentionally avoid building a model or Mapping. They enforce the
cross-demo structure documented in ``jaxite_word/API_REFERENCE.md``: one
vectorize/pack/map implementation and one Mapping-owned serving boundary.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from unittest import mock

from absl.testing import absltest


_DEMOS = pathlib.Path(__file__).resolve().parent
_JAXITE = _DEMOS.parent / 'jaxite_word'
for _path in (_JAXITE, _DEMOS):
  if str(_path) not in sys.path:
    sys.path.insert(0, str(_path))

import canonical_demo  # noqa: E402


_MODEL_FILES = ('lenet_he.py', 'lola_he.py', 'alexnet_he.py')
_PERF_FILES = (
    'lenet_he_perf_test.py', 'lola_he_perf_test.py',
    'alexnet_he_perf_test.py',
)
_SHARED_SERVING_METHODS = frozenset((
    'infer', 'encrypt', 'encrypt_batch', 'decrypt', 'decrypt_batch',
    'metadata',
))
_DIRECT_CONTEXT_CODECS = frozenset((
    'encrypt_slots', 'encrypt_slots_batch',
    'decrypt_slots', 'decrypt_slots_batch',
))
_PRIVATE_DEPLOYMENT_STATE = frozenset((
    '_param_cache', 'operation_plans', 'value_specs',
    'required_rotation_indices',
))


def _tree(name: str) -> ast.Module:
  path = _DEMOS / name
  return ast.parse(path.read_text(), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
  return next(
      node for node in tree.body
      if isinstance(node, ast.ClassDef) and node.name == name
  )


class DemoSourceContractTest(absltest.TestCase):

  def test_model_entrypoints_inherit_one_serving_adapter(self):
    expected = {
        'lenet_he.py': ('LeNetHE', 'CanonicalDemoAdapter'),
        'lola_he.py': ('LoLAHE', 'CanonicalDemoAdapter'),
        'alexnet_he.py': ('_AlexNetHEEntrypoint', 'CanonicalDemoAdapter'),
    }
    for filename, (class_name, base_name) in expected.items():
      with self.subTest(filename=filename):
        definition = _class(_tree(filename), class_name)
        bases = {
            base.attr if isinstance(base, ast.Attribute) else base.id
            for base in definition.bases
            if isinstance(base, (ast.Attribute, ast.Name))
        }
        self.assertIn(base_name, bases)
        defined = {
            node.name for node in definition.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEmpty(
            defined & _SHARED_SERVING_METHODS,
            f'{class_name} reimplemented the shared Mapping boundary',
        )

    alexnet_tree = _tree('alexnet_he.py')
    for class_name in ('AlexNetTinyHE', 'AlexNetHE'):
      definition = _class(alexnet_tree, class_name)
      defined = {
          node.name for node in definition.body
          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
      }
      self.assertEmpty(
          defined & _SHARED_SERVING_METHODS,
          f'{class_name} reimplemented the shared Mapping boundary',
      )

  def test_model_files_do_not_call_context_codec_methods_directly(self):
    for filename in _MODEL_FILES + _PERF_FILES:
      with self.subTest(filename=filename):
        calls = {
            node.func.attr
            for node in ast.walk(_tree(filename))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertEmpty(calls & _DIRECT_CONTEXT_CODECS)

  def test_model_and_perf_files_do_not_read_private_deployment_state(self):
    for filename in _MODEL_FILES + _PERF_FILES:
      with self.subTest(filename=filename):
        attributes = {
            node.attr
            for node in ast.walk(_tree(filename))
            if isinstance(node, ast.Attribute)
        }
        self.assertEmpty(attributes & _PRIVATE_DEPLOYMENT_STATE)

  def test_registry_contains_every_encrypted_demo_once(self):
    tree = _tree('encrypted_demos.py')
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == 'DEMOS'
            for target in node.targets
        )
    )
    self.assertIsInstance(assignment.value, ast.Dict)
    registered = {
        key.value: value.id
        for key, value in zip(assignment.value.keys, assignment.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Name)
    }
    self.assertEqual(
        registered,
        {
            'lenet': 'LeNetDemo',
            'lola': 'LoLADemo',
            'alexnet-tiny': 'AlexNetTinyDemo',
            'alexnet-full': 'AlexNetFullDemo',
        },
    )

  def test_only_canonical_demo_implements_the_compiler_pipeline(self):
    owners = {
        'vectorize': set(),
        'pack': set(),
        'Mapping': set(),
    }
    runtime_files = ('canonical_demo.py', 'encrypted_demos.py') + _MODEL_FILES
    for filename in runtime_files:
      for node in ast.walk(_tree(filename)):
        if not isinstance(node, ast.Call):
          continue
        if isinstance(node.func, ast.Attribute):
          name = node.func.attr
        elif isinstance(node.func, ast.Name):
          name = node.func.id
        else:
          continue
        if name in owners:
          owners[name].add(filename)
    self.assertEqual(owners['vectorize'], {'canonical_demo.py'})
    self.assertEqual(owners['pack'], {'canonical_demo.py'})
    self.assertEqual(owners['Mapping'], {'canonical_demo.py'})


class _FakeDemo(canonical_demo.CanonicalDemo):
  input_shape = (2,)

  def __init__(self, **scheduling):
    super().__init__(**scheduling)
    self.fake_mapping = mock.Mock()

  def build_model(self):
    raise AssertionError('the adapter serving test must not build a model')

  @property
  def default_provenance(self):
    return canonical_demo.WeightProvenance('seeded-random', 'test')

  def materialize_mapping(self):
    return self.fake_mapping

  @property
  def mapping(self):
    return self.fake_mapping

  def infer(self, sample, **options):
    return self.fake_mapping.infer(sample, **options)


class SharedAdapterContractTest(absltest.TestCase):

  def test_all_serving_calls_use_mapping_logical_boundaries(self):
    adapter = canonical_demo.CanonicalDemoAdapter(_FakeDemo, batch=2)
    adapter._prepared = True
    values = [object(), object()]
    ciphertext = object()
    encrypted = object()
    decrypted = object()
    inferred = object()
    adapter.demo.fake_mapping.encrypt_input.return_value = encrypted
    adapter.demo.fake_mapping.decrypt_output.return_value = decrypted
    adapter.demo.fake_mapping.infer.return_value = inferred

    self.assertIs(adapter.encrypt(values), encrypted)
    adapter.demo.fake_mapping.encrypt_input.assert_called_once_with(values)
    self.assertIs(adapter.decrypt(ciphertext), decrypted)
    adapter.demo.fake_mapping.decrypt_output.assert_called_once_with(ciphertext)
    self.assertIs(adapter.infer(values), inferred)
    adapter.demo.fake_mapping.infer.assert_called_once_with(
        values, trace_dir=None
    )

  def test_demo_rejects_non_scheduling_mapping_state(self):
    with self.assertRaisesRegex(TypeError, 'scheduling-only'):
      _FakeDemo(parameters={'degree': 16})
    with self.assertRaisesRegex(TypeError, 'scheduling-only'):
      _FakeDemo(keys={'secret_key': object()})

  def test_batch_shape_is_validated_once_by_the_shared_adapter(self):
    with self.assertRaisesRegex(ValueError, 'batch must be'):
      canonical_demo.CanonicalDemoAdapter(_FakeDemo, batch=0)
    adapter = canonical_demo.CanonicalDemoAdapter(_FakeDemo, batch=2)
    adapter._prepared = True
    with self.assertRaisesRegex(ValueError, 'contains 1 samples, expected 2'):
      adapter.infer([object()])


if __name__ == '__main__':
  absltest.main()
