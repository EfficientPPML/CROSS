"""The shipped demo models pack onto the rings their HE graphs were built for.

These figures are cross-checks, not aspirations: the depths were measured
independently from the hand-built HE graphs in ``lenet_he``/``lola_he``/
``alexnet_he``, and the degrees are the smallest ``he_params`` admits at
128-bit classical security for those depths.
"""

import os
import sys

from absl.testing import absltest

# Runs from ``demos/`` with no PYTHONPATH, as the README documents: the
# library's flat modules (``nn``, ``packing``) live in ``../jaxite_word``.
_JAXITE_WORD_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jaxite_word')
)
if _JAXITE_WORD_DIR not in sys.path:
  sys.path.insert(0, _JAXITE_WORD_DIR)

torch = None
nn = None
packing = None
lenet_train = None
lola_train = None
alexnet_train = None
alexnet_infer = None


def setUpModule():
  global torch, nn, packing
  global lenet_train, lola_train, alexnet_train, alexnet_infer
  try:
    import torch as torch_module
  except ImportError as error:  # pragma: no cover - environment dependent
    raise absltest.SkipTest(f'PyTorch unavailable: {error}')
  import nn as nn_module
  import packing as packing_module
  import lenet_train as lenet
  import lola_train as lola
  import alexnet_train as alexnet
  import alexnet_infer as infer
  torch = torch_module
  nn = nn_module
  packing = packing_module
  lenet_train, lola_train = lenet, lola
  alexnet_train, alexnet_infer = alexnet, infer
  torch.manual_seed(0)


class DemoPackingTest(absltest.TestCase):

  def _pack(self, model, shape):
    program = nn.vectorize(model, shape)
    # Planning mode. These checks are about the ring and the emitted op
    # sequence, and eager materialization at demo scale is prohibitive:
    # AlexNetFull's 3711 non-zero diagonals over 32768 slots would be ~1 GB
    # of float64. Constants are eager by default; packing_test proves that.
    packed = packing.pack(
        program, packing.PackingPolicy(lazy_constants=True)
    )
    # The frontend's declaration and packing's recomputation must agree.
    self.assertEqual(packed.depth, program.critical_depth())
    self.assertEqual(packed.num_slots, packed.ring_config.degree // 2)
    self.assertEqual(
        int(packed.ring_config.logical_num_q), packed.depth + 1
    )
    self.assertNotIn('bootstrap', packed.kinds())
    return packed

  def test_lenet_packs_at_degree_32768(self):
    packed = self._pack(lenet_train.QuadLeNet().double().eval(), (1, 28, 28))
    self.assertEqual(packed.depth, 7)
    self.assertEqual(packed.ring_config.degree, 32768)
    self.assertEqual(packed.kinds().count('matvec'), 4)
    self.assertEqual(packed.kinds().count('square'), 3)

  def test_lola_packs_at_degree_32768(self):
    packed = self._pack(lola_train.QuadLoLA().double().eval(), (1, 28, 28))
    self.assertEqual(packed.depth, 5)
    self.assertEqual(packed.ring_config.degree, 32768)
    self.assertEqual(packed.kinds().count('matvec'), 3)
    self.assertEqual(packed.kinds().count('square'), 2)

  def test_alexnet_tiny_packs_at_degree_32768(self):
    trained = alexnet_train.QuadAlexNetTiny().double().eval()
    packed = self._pack(
        alexnet_infer.to_inference_model(trained), (3, 16, 16)
    )
    self.assertEqual(packed.depth, 7)
    self.assertEqual(packed.ring_config.degree, 32768)

  def test_alexnet_full_needs_the_next_degree_up(self):
    """Depth 15 exceeds what N=32768 admits, so the ring doubles."""
    trained = alexnet_train.QuadAlexNetFull().double().eval()
    packed = self._pack(
        alexnet_infer.to_inference_model(trained), (3, 16, 16)
    )
    self.assertEqual(packed.depth, 15)
    self.assertEqual(packed.ring_config.degree, 65536)
    self.assertEqual(packed.kinds().count('square'), 7)
    self.assertEqual(packed.kinds().count('matvec'), 8)

  def test_depth_not_slot_demand_drives_the_degree(self):
    """Every demo uses a small fraction of the slots the chain forces."""
    packed = self._pack(lenet_train.QuadLeNet().double().eval(), (1, 28, 28))
    self.assertLess(784, packed.num_slots // 8)


if __name__ == '__main__':
  absltest.main()
