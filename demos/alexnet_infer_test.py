"""The inference-shaped AlexNets match the trained models and vectorize."""

import copy
import sys

from absl.testing import absltest
import numpy as np

# Runs from ``demos/`` with no PYTHONPATH, as the README documents: the
# library's flat modules (``nn``, ``packing``) live in ``../jaxite_word``.
import os
_JAXITE_WORD_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jaxite_word')
)
if _JAXITE_WORD_DIR not in sys.path:
  sys.path.insert(0, _JAXITE_WORD_DIR)

torch = None
alexnet_train = None
alexnet_infer = None
nn = None


def setUpModule():
  global torch, alexnet_train, alexnet_infer, nn
  try:
    import torch as torch_module
  except ImportError as error:  # pragma: no cover - environment dependent
    raise absltest.SkipTest(f'PyTorch unavailable: {error}')
  import alexnet_train as train_module
  import alexnet_infer as infer_module
  import nn as nn_module
  torch = torch_module
  alexnet_train = train_module
  alexnet_infer = infer_module
  nn = nn_module
  torch.manual_seed(0)


def _trained(cls, shape, steps=3):
  """A model whose BatchNorm statistics have actually moved off their init."""
  model = cls().double()
  model.train()
  with torch.no_grad():
    for _ in range(steps):
      model(torch.randn(8, *shape, dtype=torch.float64))
  return model


class FoldingTest(absltest.TestCase):

  def _check(self, cls, shape, samples=10):
    trained = _trained(cls, shape)
    trained.eval()
    inference = alexnet_infer.to_inference_model(trained)
    worst = 0.0
    with torch.no_grad():
      for _ in range(samples):
        sample = torch.randn(1, *shape, dtype=torch.float64)
        worst = max(
            worst, float((trained(sample) - inference(sample)).abs().max())
        )
    self.assertLess(worst, 1e-12, f'folded output deviates by {worst:.3e}')
    return trained, inference

  def test_tiny_folded_matches_the_trained_model_in_eval(self):
    self._check(alexnet_train.QuadAlexNetTiny, (3, 16, 16))

  def test_full_folded_matches_the_trained_model_in_eval(self):
    self._check(alexnet_train.QuadAlexNetFull, (3, 16, 16))

  def test_batch_norm_statistics_actually_moved(self):
    """Otherwise folding an identity BatchNorm would prove nothing."""
    trained = _trained(alexnet_train.QuadAlexNetTiny, (3, 16, 16))
    fresh = alexnet_train.QuadAlexNetTiny().double()
    self.assertFalse(
        torch.equal(trained.bn1.running_var, fresh.bn1.running_var)
    )
    self.assertGreater(int(trained.bn1.num_batches_tracked), 0)

  def test_conversion_leaves_the_trained_model_untouched(self):
    trained = _trained(alexnet_train.QuadAlexNetTiny, (3, 16, 16))
    before = {
        name: tensor.detach().clone()
        for name, tensor in trained.state_dict().items()
    }
    flags = {name: child.training
             for name, child in trained.named_modules()}
    alexnet_infer.to_inference_model(trained)
    for name, tensor in before.items():
      self.assertTrue(
          torch.equal(trained.state_dict()[name], tensor), f'{name} changed'
      )
    self.assertEqual(
        {name: child.training for name, child in trained.named_modules()},
        flags,
    )
    self.assertTrue(trained.training, 'training flag was cleared')

  def test_the_inference_model_carries_no_batch_norm(self):
    trained = _trained(alexnet_train.QuadAlexNetFull, (3, 16, 16))
    inference = alexnet_infer.to_inference_model(trained)
    for child in inference.modules():
      self.assertNotIsInstance(
          child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
      )

  def test_conversion_preserves_dtype(self):
    trained = _trained(alexnet_train.QuadAlexNetTiny, (3, 16, 16))
    inference = alexnet_infer.to_inference_model(trained)
    self.assertEqual(next(inference.parameters()).dtype, torch.float64)

  def test_an_unrelated_model_is_refused(self):
    with self.assertRaises(TypeError):
      alexnet_infer.to_inference_model(torch.nn.Linear(2, 2))


class VectorizationTest(absltest.TestCase):
  """Both variants vectorize at their intended HE input shape."""

  def test_tiny_vectorizes_at_its_he_input_shape(self):
    trained = _trained(alexnet_train.QuadAlexNetTiny, (3, 16, 16))
    program = nn.vectorize(
        alexnet_infer.to_inference_model(trained), (3, 16, 16)
    )
    # The depth measured from the hand-built AlexNetTiny HE graph.
    self.assertEqual(program.critical_depth(), 7)
    self.assertEqual(program.max_live_slots(), 768)
    kinds = [layer.template_id or layer.activation_name
             for layer in program.layers]
    self.assertEqual(kinds.count('conv_pool'), 3)
    self.assertEqual(kinds.count('square'), 3)
    self.assertEqual(kinds.count('dense'), 1)

  def test_full_vectorizes_at_its_he_input_shape(self):
    trained = _trained(alexnet_train.QuadAlexNetFull, (3, 16, 16))
    program = nn.vectorize(
        alexnet_infer.to_inference_model(trained), (3, 16, 16)
    )
    # The depth measured from the hand-built AlexNetHE graph.
    self.assertEqual(program.critical_depth(), 15)
    self.assertEqual(program.max_live_slots(), 768)
    kinds = [layer.template_id or layer.activation_name
             for layer in program.layers]
    self.assertEqual(kinds.count('square'), 7)
    self.assertEqual(kinds.count('dense'), 3)

  def test_the_client_downsample_flag_is_static_and_traceable(self):
    trained = _trained(alexnet_train.QuadAlexNetTiny, (3, 32, 32))
    inference = alexnet_infer.to_inference_model(
        trained, client_downsample=True
    )
    program = nn.vectorize(inference, (3, 32, 32))
    # Downsampling in the graph costs one more level than doing it on the
    # client, which is why the demo does it before encryption.
    self.assertEqual(program.critical_depth(), 8)
    self.assertEqual(program.max_live_slots(), 3072)

  def test_the_downsampling_model_matches_the_trained_model_on_32x32(self):
    trained = _trained(alexnet_train.QuadAlexNetTiny, (3, 32, 32))
    trained.eval()
    inference = alexnet_infer.to_inference_model(
        trained, client_downsample=True
    )
    with torch.no_grad():
      for _ in range(5):
        sample = torch.randn(1, 3, 32, 32, dtype=torch.float64)
        self.assertLess(
            float((trained(sample) - inference(sample)).abs().max()), 1e-12
        )

  def test_the_default_model_does_not_downsample(self):
    trained = _trained(alexnet_train.QuadAlexNetTiny, (3, 16, 16))
    inference = alexnet_infer.to_inference_model(trained)
    self.assertFalse(inference.client_downsample)
    program = nn.vectorize(inference, (3, 16, 16))
    self.assertEqual(program.layers[0].input_spec.shape, (3, 16, 16))


if __name__ == '__main__':
  absltest.main()
