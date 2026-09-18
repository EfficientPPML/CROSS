"""Test-only helpers for the demo suites.

Nothing here belongs in a demo's own module: it exists so plan-level tests can
observe the pipeline without paying for the crypto stack.
"""

import sys
import types
import unittest.mock as mock

import numpy as np


class StubRingConfig:
  """The ring fields displayed by demo performance drivers."""

  degree = 32768
  logical_num_q = 8
  dnum = 4


class RecordingPerfModel:
  """Record driver setup and inference without materializing HE state."""

  instances: list['RecordingPerfModel'] = []
  output_size = 10

  def __init__(self, batch: int = 1, devices=None, **kwargs):
    self.batch = batch
    self.devices = devices
    self.kwargs = kwargs
    self.weights = None
    self.inferred = []
    self._batch = batch
    self.ring_config = StubRingConfig()
    type(self).instances.append(self)

  def precompute_plaintexts(self, *args, **kwargs):
    self.weights = args
    self.kwargs.update(kwargs)

  def infer(self, value, **_):
    self.inferred.append(value)
    return np.zeros(self.output_size, dtype=np.float64)

  @classmethod
  def reset(cls):
    cls.instances = []


def fake_mapping():
  """Patch a stand-in ``mapping`` module into ``sys.modules``.

  ``CanonicalDemo.materialize_mapping`` imports ``mapping`` inside the method,
  so replacing the module is enough to keep the real method running -- the
  Packing it passes and the scheduling keywords it forwards stay
  observable -- while JAX and XLA are never imported. Patching
  ``mapping.Mapping`` directly would not do: that imports the real module
  first, hundreds of megabytes before the mock exists. Patching the method
  itself would be worse, because the kwargs these tests are checking would
  never be forwarded at all.

  Returns a context manager whose value is the ``Mapping`` mock.
  """

  module = types.ModuleType('mapping')
  module.Mapping = mock.MagicMock(name='Mapping')

  class _Patch:

    def __enter__(self):
      self._patch = mock.patch.dict(sys.modules, {'mapping': module})
      self._patch.start()
      return module.Mapping

    def __exit__(self, *exception):
      self._patch.stop()
      return False

  return _Patch()
