"""LeNet, LoLA and AlexNet on the canonical pipeline.

Each demo's source of model truth is an ordinary ``torch.nn.Module``. Nothing
here builds a packing graph by hand: the module is traced by ``nn.vectorize``,
packed by ``packing.pack``, and scheduled by ``Mapping``.

Only LoLA ships trained binaries. LeNet and AlexNet have no checkpoint in the
repository, so they run on a deterministic seeded draw -- reported as such by
every demo's provenance, metadata and log line, and never presented as an
accuracy result.
"""

import os

import numpy as np

# The demos use the library's flat sibling imports; make ``jaxite_word/``
# importable from ``demos/`` so this module works with no PYTHONPATH.
import sys
_JAXITE_WORD_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jaxite_word')
)
if _JAXITE_WORD_DIR not in sys.path:
  sys.path.insert(0, _JAXITE_WORD_DIR)

import alexnet_infer
import alexnet_train
import canonical_demo
import lenet_train
import lola_train
from canonical_demo import CanonicalDemo, WeightProvenance, seeded_parameters

# Deterministic, so a demo's program digest is reproducible across runs.
LENET_SEED = 20240517
ALEXNET_TINY_SEED = 20240518
ALEXNET_FULL_SEED = 20240519

# Client-side preprocessing. CIFAR arrives at 32x32 and the encrypted model
# starts at 16x16, so this average-pool runs on the client, before encryption,
# exactly as it does today. Doing it under encryption would cost a level and
# a ring four times wider, for a step the client can do for free.
ALEXNET_HE_INPUT = (3, 16, 16)
ALEXNET_CLIENT_INPUT = (3, 32, 32)


def downsample_for_client(sample) -> np.ndarray:
  """Average-pool a 3x32x32 sample to 3x16x16, in the clear."""
  array = np.asarray(sample, dtype=np.float64).reshape(ALEXNET_CLIENT_INPUT)
  channels, height, width = array.shape
  return array.reshape(
      channels, height // 2, 2, width // 2, 2
  ).mean(axis=(2, 4))


class LeNetDemo(CanonicalDemo):
  """QuadLeNet: conv, square, conv, square, flatten, fc, square, fc."""

  input_shape = (1, 28, 28)
  weight_order = (
      ('W1w', 'conv1.weight'), ('W2w', 'conv2.weight'),
      ('W3w', 'fc1.weight'), ('W4w', 'fc2.weight'),
      ('b1', 'conv1.bias'), ('b2', 'conv2.bias'),
      ('b3', 'fc1.bias'), ('b4', 'fc2.bias'),
  )

  def build_model(self):
    import torch

    model = lenet_train.QuadLeNet().double().eval()
    return seeded_parameters(model, LENET_SEED)

  @property
  def default_provenance(self) -> WeightProvenance:
    return WeightProvenance(
        source='seeded-random',
        detail=(
            f'no LeNet checkpoint ships with this repository; parameters are '
            f'a deterministic draw at seed {LENET_SEED}'
        ),
    )


class LoLADemo(CanonicalDemo):
  """QuadLoLA, loaded from the trained binaries this repository ships."""

  input_shape = (1, 28, 28)
  weight_order = (
      ('W1', 'conv1.weight'), ('b1', 'conv1.bias'),
      ('W2', 'fc1.weight'), ('b2', 'fc1.bias'),
      ('W3', 'fc2.weight'), ('b3', 'fc2.bias'),
  )

  def build_model(self):
    model = lola_train.load_pretrained_lola()
    self._validate_shapes(model)
    return model.double().eval()

  def _validate_shapes(self, model):
    """Check the loaded binaries against the architecture they must fill."""
    expected = {
        'conv1.weight': (5, 1, 2, 2),
        'conv1.bias': (5,),
        'fc1.weight': (100, 980),
        'fc1.bias': (100,),
        'fc2.weight': (10, 100),
        'fc2.bias': (10,),
    }
    state = model.state_dict()
    for name, shape in expected.items():
      if name not in state:
        raise ValueError(f'LoLA checkpoint is missing {name!r}.')
      actual = tuple(state[name].shape)
      if actual != shape:
        raise ValueError(
            f'LoLA {name} has shape {actual}, expected {shape}.'
        )
      values = state[name].detach().numpy()
      if values.dtype.kind != 'f':
        raise ValueError(
            f'LoLA {name} has dtype {values.dtype}, expected floating point.'
        )
      if not np.all(np.isfinite(values)):
        raise ValueError(f'LoLA {name} contains non-finite values.')

  @property
  def default_provenance(self) -> WeightProvenance:
    return WeightProvenance(
        source='checkpoint',
        detail='loaded from the shipped LoLA binaries',
        files=tuple(name for name, _, _ in lola_train.WEIGHT_FILES),
    )


class _AlexNetDemo(CanonicalDemo):
  """Shared AlexNet plumbing.

  The exported AlexNet binaries are already BatchNorm-folded, so they load
  straight into the BatchNorm-free inference modules. Nothing here tries to
  reconstruct BatchNorm state the repository does not ship.
  """

  input_shape = ALEXNET_HE_INPUT
  _seed = 0
  _module = None

  def build_model(self):
    model = self._module(client_downsample=False).double().eval()
    return seeded_parameters(model, self._seed)

  @property
  def default_provenance(self) -> WeightProvenance:
    return WeightProvenance(
        source='seeded-random',
        detail=(
            f'no AlexNet checkpoint ships with this repository; parameters '
            f'are a deterministic draw at seed {self._seed}, loaded directly '
            'into the BatchNorm-free inference module'
        ),
    )

  def preprocess(self, sample) -> np.ndarray:
    """Client-side 32x32 -> 16x16 average pool, outside encryption."""
    return downsample_for_client(sample)


class AlexNetTinyDemo(_AlexNetDemo):
  _seed = ALEXNET_TINY_SEED
  _module = staticmethod(alexnet_infer.QuadAlexNetTinyInfer)
  weight_order = (
      ('W1', 'conv1.weight'), ('b1', 'conv1.bias'),
      ('W2', 'conv2.weight'), ('b2', 'conv2.bias'),
      ('W3', 'conv3.weight'), ('b3', 'conv3.bias'),
      ('Wf', 'fc.weight'), ('bf', 'fc.bias'),
  )


class AlexNetFullDemo(_AlexNetDemo):
  _seed = ALEXNET_FULL_SEED
  _module = staticmethod(alexnet_infer.QuadAlexNetFullInfer)
  weight_order = (
      ('W1', 'conv1.weight'), ('b1', 'conv1.bias'),
      ('W2', 'conv2.weight'), ('b2', 'conv2.bias'),
      ('W3', 'conv3.weight'), ('b3', 'conv3.bias'),
      ('W4', 'conv4.weight'), ('b4', 'conv4.bias'),
      ('W5', 'conv5.weight'), ('b5', 'conv5.bias'),
      ('Wf1', 'fc1.weight'), ('bf1', 'fc1.bias'),
      ('Wf2', 'fc2.weight'), ('bf2', 'fc2.bias'),
      ('Wf3', 'fc3.weight'), ('bf3', 'fc3.bias'),
  )


DEMOS = {
    'lenet': LeNetDemo,
    'lola': LoLADemo,
    'alexnet-tiny': AlexNetTinyDemo,
    'alexnet-full': AlexNetFullDemo,
}


def main(argv=None):
  import argparse
  import json
  import logging

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('demo', choices=sorted(DEMOS))
  parser.add_argument(
      '--json', action='store_true', help='print metadata as JSON'
  )
  arguments = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format='%(message)s')

  demo = DEMOS[arguments.demo]()
  metadata = demo.metadata()
  if arguments.json:
    print(json.dumps(metadata, indent=2, sort_keys=True))
  else:
    for key, value in sorted(metadata.items()):
      print(f'{key}: {value}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
