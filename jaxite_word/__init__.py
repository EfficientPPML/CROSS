"""CROSS CKKS implementation.

``nn`` and ``packing`` describe and vectorize a workload; ``mapping`` schedules
it onto a ``CKKSContext``, which owns the codec, primitives and operators.
Public ciphertext boundaries use canonical rank-5 ``Polynomial`` values.

Import cost: ``nn`` and ``packing`` are imported eagerly because they are pure
NumPy. ``polynomial``, ``mapping`` and ``ckks_ctx`` pull in JAX and the whole
crypto stack, so they load on first access instead -- ``import jaxite_word.nn``
must stay cheap enough to use from tooling that never touches a ciphertext.
"""

from __future__ import annotations

import importlib
import os
import sys

_PACKAGE_DIR = os.path.dirname(__file__)
if _PACKAGE_DIR not in sys.path:
  # Internal modules still use the repository's historical flat sibling
  # imports. Keep that compatibility detail at this one package boundary.
  sys.path.insert(0, _PACKAGE_DIR)

# Modules reachable under both the flat (``polynomial``) and package
# (``jaxite_word.polynomial``) spelling, mapped to the public names they back.
# Both spellings must resolve to ONE module object or the wrapper classes get
# two identities in one process.
_DUAL_SPELLED = {
    'polynomial': ('Polynomial',),
    'ckks_ctx': ('CKKSContext',),
    'mapping': ('Mapping',),
}

# If a flat spelling was imported before this package, alias it now. This costs
# no import and keeps identity stable regardless of import order.
for _module_name in _DUAL_SPELLED:
  _flat = sys.modules.get(_module_name)
  if _flat is not None:
    sys.modules.setdefault(f'{__name__}.{_module_name}', _flat)
del _module_name, _flat

nn = importlib.import_module(f'{__name__}.nn')
packing = importlib.import_module(f'{__name__}.packing')
Packing = packing.Packing
pack = packing.pack
vectorize = nn.vectorize
VectorizedProgram = nn.VectorizedProgram


def _load_dual_spelled(module_name):
  """Import one dual-spelled module, binding both spellings to one object."""
  package_name = f'{__name__}.{module_name}'
  flat_module = sys.modules.get(module_name)
  if flat_module is not None:
    sys.modules.setdefault(package_name, flat_module)
    module = flat_module
  else:
    module = importlib.import_module(package_name)
  # Internal code still uses the historical flat name; point it at the same
  # module so no second class identity can appear.
  sys.modules.setdefault(module_name, module)
  globals()[module_name] = module
  for attribute in _DUAL_SPELLED[module_name]:
    globals()[attribute] = getattr(module, attribute)
  return module


def __getattr__(name):
  for module_name, attributes in _DUAL_SPELLED.items():
    if name == module_name:
      return _load_dual_spelled(module_name)
    if name in attributes:
      return getattr(_load_dual_spelled(module_name), name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# Release version. Kept here rather than in packaging metadata because the
# repository is used in place (`cd jaxite_word && python3 <item>_test.py`) and
# has no setup.py/pyproject.toml, so this is the only place a consumer can ask
# what they are running. Bump together with the git tag.
__version__ = '3.0.0'

__all__ = [
    'CKKSContext', 'Mapping', 'Packing', 'Polynomial', 'VectorizedProgram',
    '__version__',
    'mapping', 'nn', 'pack', 'packing', 'vectorize',
]
