"""Correctness tests for standalone demo models and asset tooling."""

from __future__ import annotations

import importlib
import os
import pathlib
import struct
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


_DEMOS = pathlib.Path(__file__).resolve().parent
if str(_DEMOS) not in sys.path:
  sys.path.insert(0, str(_DEMOS))

import lola_train  # pylint: disable=wrong-import-position
import alexnet_train  # pylint: disable=wrong-import-position
import lenet_train  # pylint: disable=wrong-import-position
import prepare_demo_assets  # pylint: disable=wrong-import-position
import resnet20_cifar  # pylint: disable=wrong-import-position


def _write_idx_images(path: pathlib.Path, images: np.ndarray) -> None:
  images = np.asarray(images, dtype=np.uint8)
  header = struct.pack(">IIII", 2051, len(images), images.shape[1], images.shape[2])
  path.write_bytes(header + images.tobytes())


def _write_idx_labels(path: pathlib.Path, labels: np.ndarray,
                      *, declared_count: int | None = None) -> None:
  labels = np.asarray(labels, dtype=np.uint8)
  count = len(labels) if declared_count is None else declared_count
  path.write_bytes(struct.pack(">II", 2049, count) + labels.tobytes())


class LoLADataAndExportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.mnist_dir = pathlib.Path(self._tmp.name) / "mnist"
    self.mnist_dir.mkdir()
    train = np.stack([
        np.zeros((28, 28), dtype=np.uint8),
        np.full((28, 28), 255, dtype=np.uint8),
    ])
    test = np.stack([
        np.full((28, 28), 17, dtype=np.uint8),
        np.full((28, 28), 201, dtype=np.uint8),
        np.full((28, 28), 89, dtype=np.uint8),
    ])
    _write_idx_images(self.mnist_dir / "train-images-idx3-ubyte", train)
    _write_idx_labels(self.mnist_dir / "train-labels-idx1-ubyte", [1, 2])
    _write_idx_images(self.mnist_dir / "t10k-images-idx3-ubyte", test)
    _write_idx_labels(self.mnist_dir / "t10k-labels-idx1-ubyte", [3, 4, 5])

  def test_loader_preserves_lola_zero_to_one_input_convention(self):
    train_x, train_y, test_x, test_y = lola_train.load_mnist(
        str(self.mnist_dir)
    )
    self.assertEqual(train_x.shape, (2, 784))
    self.assertEqual(test_x.shape, (3, 784))
    self.assertEqual(train_x.dtype, np.float32)
    self.assertEqual(float(train_x[0, 0]), 0.0)
    self.assertEqual(float(train_x[1, 0]), 1.0)
    np.testing.assert_array_equal(train_y, [1, 2])
    np.testing.assert_array_equal(test_y, [3, 4, 5])

  def test_lenet_loader_validates_and_normalizes_the_same_idx_assets(self):
    train_x, train_y, test_x, test_y = lenet_train.load_mnist(
        str(self.mnist_dir)
    )
    self.assertEqual(train_x.shape, (2, 28, 28))
    self.assertEqual(test_x.shape, (3, 28, 28))
    self.assertEqual(train_x.dtype, np.float32)
    self.assertAlmostEqual(
        float(train_x[0, 0, 0]),
        -lenet_train.MNIST_MEAN / lenet_train.MNIST_STD,
        places=6,
    )
    np.testing.assert_array_equal(train_y, [1, 2])
    np.testing.assert_array_equal(test_y, [3, 4, 5])

  def test_lenet_loader_rejects_image_label_count_mismatch(self):
    _write_idx_labels(
        self.mnist_dir / "train-labels-idx1-ubyte", [1], declared_count=1
    )
    with self.assertRaisesRegex(ValueError, "image/label count mismatch"):
      lenet_train.load_mnist(str(self.mnist_dir))

  def test_truncated_label_file_is_rejected(self):
    _write_idx_labels(
        self.mnist_dir / "train-labels-idx1-ubyte", [1], declared_count=2
    )
    with self.assertRaisesRegex(ValueError, "expected 2 labels, got 1"):
      lola_train.load_mnist(str(self.mnist_dir))

  def test_truncated_image_file_is_rejected(self):
    path = self.mnist_dir / "train-images-idx3-ubyte"
    path.write_bytes(path.read_bytes()[:-1])
    with self.assertRaisesRegex(ValueError, "expected 1568 pixels, got 1567"):
      lola_train.load_mnist(str(self.mnist_dir))

  def test_wrong_image_geometry_is_rejected_by_the_fixed_shape_model(self):
    bad = np.zeros((2, 27, 28), dtype=np.uint8)
    _write_idx_images(self.mnist_dir / "train-images-idx3-ubyte", bad)
    with self.assertRaisesRegex(ValueError, "must be 28x28"):
      lola_train.load_mnist(str(self.mnist_dir))

  def test_export_round_trips_weights_without_mutating_caller_dtype(self):
    torch.manual_seed(7)
    model = lola_train.QuadLoLA().to(torch.float32)
    original = {
        key: value.detach().clone() for key, value in model.named_parameters()
    }
    out_dir = pathlib.Path(self._tmp.name) / "weights"

    lola_train.export_weights(
        model, out_dir=str(out_dir), mnist_dir=str(self.mnist_dir), n_test=2
    )

    self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
    loaded = lola_train.load_pretrained_lola(str(out_dir), torch.float64)
    for key, value in loaded.named_parameters():
      np.testing.assert_array_equal(
          value.detach().numpy(), original[key].numpy().astype(np.float64)
      )
    images, labels, predictions = lola_train.load_mnist_test_200(str(out_dir))
    self.assertEqual(images.shape, (2, 784))
    self.assertEqual(labels.shape, (2,))
    self.assertEqual(predictions.shape, (2,))

  def test_export_supports_namespace_package_import(self):
    packaged = importlib.import_module("demos.lola_train")
    out_dir = pathlib.Path(self._tmp.name) / "packaged_weights"
    packaged.export_weights(
        packaged.QuadLoLA(), out_dir=str(out_dir),
        mnist_dir=str(self.mnist_dir), n_test=1,
    )
    self.assertTrue((out_dir / "lola_plaintext_pred_200.bin").is_file())

  def test_export_rejects_a_slice_larger_than_the_test_set(self):
    out_dir = os.path.join(self._tmp.name, "weights")
    with self.assertRaisesRegex(ValueError, "exceeds the available test set"):
      lola_train.export_weights(
          lola_train.QuadLoLA(), out_dir=out_dir,
          mnist_dir=str(self.mnist_dir), n_test=4
      )
    self.assertFalse(os.path.exists(out_dir))

  def test_test_asset_loader_rejects_mismatched_counts(self):
    out_dir = pathlib.Path(self._tmp.name) / "weights"
    lola_train.export_weights(
        lola_train.QuadLoLA(), out_dir=str(out_dir),
        mnist_dir=str(self.mnist_dir), n_test=2
    )
    (out_dir / "lola_plaintext_pred_200.bin").write_bytes(
        np.array([0], dtype=np.int32).tobytes()
    )
    with self.assertRaisesRegex(ValueError, "asset count mismatch"):
      lola_train.load_mnist_test_200(str(out_dir))

  def test_test_asset_loader_rejects_an_empty_slice(self):
    out_dir = pathlib.Path(self._tmp.name) / "empty_weights"
    out_dir.mkdir()
    for name in (
        "mnist_test_200.bin",
        "mnist_test_200_labels.bin",
        "lola_plaintext_pred_200.bin",
    ):
      (out_dir / name).write_bytes(b"")
    with self.assertRaisesRegex(ValueError, "at least one image"):
      lola_train.load_mnist_test_200(str(out_dir))


class TrainerInputContractTest(absltest.TestCase):

  def test_every_standalone_trainer_rejects_zero_epochs_before_data_loading(self):
    calls = (
        lambda: alexnet_train.train(0, 128, 1e-3, 1e-4, 42),
        lambda: lenet_train.train(0, 128, 1e-3, 1e-4, 42),
        lambda: lola_train.train(epochs=0),
        lambda: resnet20_cifar.train(torch.nn.Linear(1, 1), epochs=0),
    )
    for call in calls:
      with self.subTest(call=call), self.assertRaisesRegex(
          ValueError, "epochs must be"
      ):
        call()

  def test_lola_evaluation_rejects_a_nonpositive_slice_before_loading(self):
    with mock.patch.object(lola_train, "load_pretrained_lola") as loader:
      with self.assertRaises(SystemExit) as raised:
        lola_train.main(["--n", "0"])
    self.assertEqual(raised.exception.code, 2)
    loader.assert_not_called()


class PrepareAssetsTest(absltest.TestCase):

  def test_download_guard_rejects_existing_and_dangling_symlinks(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      target = root / "target"
      target.mkdir()
      for name, destination in (
          ("existing", target), ("dangling", root / "missing")
      ):
        link = root / name
        link.symlink_to(destination, target_is_directory=True)
        with self.subTest(name=name), self.assertRaisesRegex(
            SystemExit, "is a symlink"
        ):
          prepare_demo_assets._guard_not_symlink(str(link))

  def test_data_only_honours_the_selected_model(self):
    with mock.patch.object(prepare_demo_assets, "fetch_cifar10") as cifar, \
         mock.patch.object(prepare_demo_assets, "fetch_mnist") as mnist:
      self.assertEqual(
          prepare_demo_assets.main(["--models", "lola", "--data-only"]), 0
      )
    cifar.assert_not_called()
    mnist.assert_called_once_with()

  def test_contradictory_data_flags_are_rejected(self):
    with self.assertRaises(SystemExit) as raised:
      prepare_demo_assets.main(["--data-only", "--skip-data"])
    self.assertEqual(raised.exception.code, 2)

  def test_nonpositive_epoch_override_is_rejected(self):
    with self.assertRaises(SystemExit) as raised:
      prepare_demo_assets.main(["--skip-data", "--epochs", "0"])
    self.assertEqual(raised.exception.code, 2)

  def test_default_training_recipe_uses_the_documented_thirty_epochs(self):
    with mock.patch.object(
        prepare_demo_assets, "train_lenet", return_value=0.9
    ) as train_lenet:
      self.assertEqual(
          prepare_demo_assets.main(["--models", "lenet", "--skip-data"]), 0
      )
    train_lenet.assert_called_once_with(30, "cpu", 42)


class ResNet20Test(absltest.TestCase):

  def test_both_supported_models_produce_ten_logits(self):
    x = torch.randn(2, 3, 32, 32)
    for activation in ("relu", "silu"):
      with self.subTest(activation=activation):
        model = resnet20_cifar.ResNet20(act=activation).eval()
        with torch.no_grad():
          self.assertEqual(tuple(model(x).shape), (2, 10))

  def test_unknown_activation_is_rejected_by_name(self):
    with self.assertRaisesRegex(ValueError, "unknown activation"):
      resnet20_cifar.ResNet20(act="quadratic")

  def test_training_saves_a_best_checkpoint_even_at_zero_accuracy(self):
    x = torch.zeros(4, 3, 2, 2)
    y = torch.ones(4, dtype=torch.int64)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10))
    with torch.no_grad():
      model[1].weight.zero_()
      model[1].bias.zero_()
    with tempfile.TemporaryDirectory() as tmp:
      out = os.path.join(tmp, "best.pt")
      with mock.patch.object(
          resnet20_cifar, "_loaders", return_value=(loader, loader)
      ):
        best, final = resnet20_cifar.train(
            model, epochs=1, batch_size=2, lr=0.0, out_path=out
        )
      self.assertEqual(best, 0.0)
      self.assertEqual(final, 0.0)
      self.assertTrue(os.path.isfile(out))

  def test_training_rejects_nonpositive_epochs_before_loading_data(self):
    with mock.patch.object(resnet20_cifar, "_loaders") as loaders:
      with self.assertRaisesRegex(ValueError, "epochs must be"):
        resnet20_cifar.train(torch.nn.Linear(1, 1), epochs=0)
    loaders.assert_not_called()


if __name__ == "__main__":
  absltest.main()
