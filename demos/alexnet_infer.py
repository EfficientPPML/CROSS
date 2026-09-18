"""Inference-shaped AlexNet models for encrypted evaluation.

The trained models in ``alexnet_train`` carry BatchNorm and decide the
client-side 2x downsample with ``if x.shape[-1] == 32`` inside ``forward``.
Both are right for training and wrong for a compiler: BatchNorm is a separate
layer only while its statistics are still moving, and a shape test is control
flow that ``torch.fx`` cannot trace.

So this module holds the same networks written for inference. BatchNorm is
folded into the preceding Conv2d or Linear -- exactly the arithmetic
``export_weights`` already performs -- and the downsample becomes a
constructor flag, fixed before the graph exists. The result is an ordinary
``torch.nn.Module`` with no BatchNorm and no branch, which
``jaxite_word.nn.vectorize`` accepts unchanged.

Training semantics are untouched: ``to_inference_model`` deep-copies, so the
trained model keeps its own parameters, buffers and training flags.
"""

import copy

import torch
from torch import nn
import torch.nn.functional as F

import alexnet_train


class QuadAlexNetTinyInfer(nn.Module):
  """``QuadAlexNetTiny`` with BatchNorm folded and the branch resolved."""

  def __init__(self, num_classes=alexnet_train.TINY_FC_OUT,
               client_downsample=False):
    super().__init__()
    # Fixed here rather than tested in forward(): the input resolution is a
    # property of the deployment, and the graph must not depend on the value
    # of a tensor.
    self.client_downsample = bool(client_downsample)
    self.conv1 = nn.Conv2d(
        alexnet_train.TINY_CI, alexnet_train.TINY_CO1,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.conv2 = nn.Conv2d(
        alexnet_train.TINY_CO1, alexnet_train.TINY_CO2,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.conv3 = nn.Conv2d(
        alexnet_train.TINY_CO2, alexnet_train.TINY_CO3,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.fc = nn.Linear(alexnet_train.TINY_FC_IN, num_classes, bias=True)

  def forward(self, x):
    if self.client_downsample:
      x = F.avg_pool2d(x, 2, 2)
    x = F.avg_pool2d(self.conv1(x), 2, 2)
    x = x * x
    x = F.avg_pool2d(self.conv2(x), 2, 2)
    x = x * x
    x = F.adaptive_avg_pool2d(self.conv3(x), (2, 2))
    x = x * x
    return self.fc(x.reshape(x.size(0), -1))


class QuadAlexNetFullInfer(nn.Module):
  """``QuadAlexNetFull`` with BatchNorm folded and the branch resolved."""

  def __init__(self, num_classes=alexnet_train.FULL_FC3_OUT,
               client_downsample=False):
    super().__init__()
    self.client_downsample = bool(client_downsample)
    self.conv1 = nn.Conv2d(
        alexnet_train.FULL_CI, alexnet_train.FULL_CO1,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.conv2 = nn.Conv2d(
        alexnet_train.FULL_CO1, alexnet_train.FULL_CO2,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.conv3 = nn.Conv2d(
        alexnet_train.FULL_CO2, alexnet_train.FULL_CO3,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.conv4 = nn.Conv2d(
        alexnet_train.FULL_CO3, alexnet_train.FULL_CO4,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.conv5 = nn.Conv2d(
        alexnet_train.FULL_CO4, alexnet_train.FULL_CO5,
        alexnet_train.KH, padding=alexnet_train.PAD, bias=True)
    self.fc1 = nn.Linear(
        alexnet_train.FULL_FC1_IN, alexnet_train.FULL_FC1_OUT, bias=True)
    self.fc2 = nn.Linear(
        alexnet_train.FULL_FC1_OUT, alexnet_train.FULL_FC2_OUT, bias=True)
    self.fc3 = nn.Linear(
        alexnet_train.FULL_FC2_OUT, num_classes, bias=True)

  def forward(self, x):
    if self.client_downsample:
      x = F.avg_pool2d(x, 2, 2)
    x = F.avg_pool2d(self.conv1(x), 2, 2)
    x = x * x
    x = F.avg_pool2d(self.conv2(x), 2, 2)
    x = x * x
    x = self.conv3(x)
    x = x * x
    x = self.conv4(x)
    x = x * x
    x = F.adaptive_avg_pool2d(self.conv5(x), (2, 2))
    x = x * x
    x = x.reshape(x.size(0), -1)
    x = self.fc1(x)
    x = x * x
    x = self.fc2(x)
    x = x * x
    return self.fc3(x)


_TINY_FOLDS = (('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3'))
_FULL_CONV_FOLDS = (
    ('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3'),
    ('conv4', 'bn4'), ('conv5', 'bn5'),
)
_FULL_LINEAR_FOLDS = (('fc1', 'bnf1'), ('fc2', 'bnf2'))


def _assign(target, weight, bias):
  with torch.no_grad():
    target.weight.copy_(weight)
    target.bias.copy_(bias)


def to_inference_model(trained, client_downsample=False):
  """Return an inference model equivalent to ``trained`` in eval mode.

  ``trained`` is deep-copied and left exactly as it was found, including its
  training flags and BatchNorm running statistics.
  """
  source = copy.deepcopy(trained).eval()
  if isinstance(trained, alexnet_train.QuadAlexNetTiny):
    model = QuadAlexNetTinyInfer(
        num_classes=source.fc.out_features,
        client_downsample=client_downsample,
    )
    conv_folds, linear_folds = _TINY_FOLDS, ()
    plain = ('fc',)
  elif isinstance(trained, alexnet_train.QuadAlexNetFull):
    model = QuadAlexNetFullInfer(
        num_classes=source.fc3.out_features,
        client_downsample=client_downsample,
    )
    conv_folds, linear_folds = _FULL_CONV_FOLDS, _FULL_LINEAR_FOLDS
    plain = ('fc3',)
  else:
    raise TypeError(
        f'to_inference_model expects a QuadAlexNetTiny or QuadAlexNetFull, '
        f'got {type(trained).__name__}.'
    )

  # Adopt the trained model's dtype and device, so a float64 reference model
  # stays float64 and the two can be compared directly.
  reference = next(source.parameters())
  model = model.to(dtype=reference.dtype, device=reference.device)

  for convolution, batch_norm in conv_folds:
    weight, bias = alexnet_train._fold_bn_into_conv(
        getattr(source, convolution), getattr(source, batch_norm)
    )
    _assign(getattr(model, convolution), weight, bias)
  for linear, batch_norm in linear_folds:
    weight, bias = alexnet_train._fold_bn_into_linear(
        getattr(source, linear), getattr(source, batch_norm)
    )
    _assign(getattr(model, linear), weight, bias)
  for name in plain:
    layer = getattr(source, name)
    _assign(getattr(model, name), layer.weight.detach(), layer.bias.detach())
  return model.eval()
