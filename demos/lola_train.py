"""PyTorch definition of the LoLA-MNIST network that `lola_he.py` evaluates
under CKKS, plus a loader that populates it from the pretrained float64
binaries in `$CROSS_DATA_ROOT/pretrained_weights/lola`.

Architecture (exactly mirrors `lola_cleartext_inference`, lola_he.py:230):

    Input X: (1, 28, 28)
    Conv1: 1->5, 2x2, stride=2, pad=0, +bias  -> (5, 14, 14)
    Quad:  y = x**2
    Flatten (channel-major CHW)               -> 980
    FC1:   980 -> 100 +bias
    Quad
    FC2:   100 -> 10  +bias

Only two ciphertext multiplications (the two squares), so the encrypted graph
has multiplicative depth 5.

Ordering note. `lola_he.py` packs the encrypted conv1 output as eight
plaintext-multiply/rotate branches over a stride-multiplexed input, and the
resulting 980 slots come out channel-major -- slot `co * 196 + h * 14 + w`.
That is exactly `torch.Tensor.reshape(N, -1)` on an `(N, 5, 14, 14)` tensor,
so `fc1` consumes the plain flatten with no permutation.

Weight-file layout is float64, C-order, and matches PyTorch's own parameter
layout, so `load_pretrained_lola` is a straight reshape-and-copy.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get(
    "CROSS_DATA_ROOT", os.path.join(os.path.dirname(_HERE), "data")
)
# Same resolution order as lola_he.load_trained (lola_he.py:60).
TRAINED_DIR = os.environ.get(
    "LOLA_WEIGHTS_DIR",
    os.path.join(_DATA_ROOT, "pretrained_weights", "lola"),
)

# Architectural constants -- must match lola_he.py:171 exactly.
CI, H_IN, W_IN = 1, 28, 28
CO, KH, KW = 5, 2, 2
STRIDE, PAD = 2, 0
H_OUT, W_OUT = 14, 14
FC1_IN, FC1_OUT = CO * H_OUT * W_OUT, 100
FC2_IN, FC2_OUT = FC1_OUT, 10

# Size of the test slice shipped alongside the weights. Mirrors
# lola_he._N_TEST_IMAGES (lola_he.py:65); duplicated rather than imported so
# that exporting weights does not require importing the encrypted demo.
_N_TEST_IMAGES = 200

# Output filename <-> parameter name <-> flat-array shape.  The same six files
# lola_he.load_trained reads at lola_he.py:99-104.
WEIGHT_FILES = [
    ("lola_conv1_W.bin", "conv1.weight", (CO, CI, KH, KW)),
    ("lola_conv1_b.bin", "conv1.bias", (CO,)),
    ("lola_fc1_W.bin", "fc1.weight", (FC1_OUT, FC1_IN)),
    ("lola_fc1_b.bin", "fc1.bias", (FC1_OUT,)),
    ("lola_fc2_W.bin", "fc2.weight", (FC2_OUT, FC2_IN)),
    ("lola_fc2_b.bin", "fc2.bias", (FC2_OUT,)),
]


# ---------------------------------------------------------------------------
# Model.  Pure quadratic activations -- matches the HE pipeline exactly.
# ---------------------------------------------------------------------------
class QuadLoLA(nn.Module):
  """LoLA-MNIST: one strided 2x2 conv and two dense layers, x**2 between."""

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(CI, CO, (KH, KW), stride=STRIDE, padding=PAD,
                           bias=True)
    self.fc1 = nn.Linear(FC1_IN, FC1_OUT, bias=True)
    self.fc2 = nn.Linear(FC2_IN, FC2_OUT, bias=True)
    self._init_small()

  def _init_small(self):
    # Smaller-than-Kaiming init keeps activations bounded under x**2.
    nn.init.kaiming_normal_(self.conv1.weight, a=0, mode="fan_in",
                            nonlinearity="linear")
    with torch.no_grad():
      self.conv1.weight.mul_(0.5)
      self.conv1.bias.zero_()
    for m in (self.fc1, self.fc2):
      nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in",
                              nonlinearity="linear")
      with torch.no_grad():
        m.weight.mul_(0.3)
        m.bias.zero_()

  def forward(self, x):
    x = self.conv1(x)
    x = x * x
    # Channel-major flatten: matches the HE slot order co*196 + h*14 + w.
    x = x.reshape(x.size(0), -1)
    x = self.fc1(x)
    x = x * x
    return self.fc2(x)


# ---------------------------------------------------------------------------
# Load the pretrained float64 binaries into a QuadLoLA.
# ---------------------------------------------------------------------------
def _load_floats(path: str) -> np.ndarray:
  with open(path, "rb") as f:
    return np.frombuffer(f.read(), dtype=np.float64)


def load_pretrained_lola(
    weights_dir: str | None = None,
    dtype: torch.dtype = torch.float64,
) -> QuadLoLA:
  """Build a QuadLoLA holding the BN-folded weights `lola_he.py` encrypts.

  Args:
    weights_dir: directory holding the six `lola_*.bin` files; defaults to
      `TRAINED_DIR` (`LOLA_WEIGHTS_DIR`, else
      `$CROSS_DATA_ROOT/pretrained_weights/lola`).
    dtype: parameter dtype.  float64 reproduces the NumPy reference in
      `lola_cleartext_inference` to roundoff; float32 is enough for training.

  Returns:
    A QuadLoLA in eval mode with every parameter populated.
  """
  src = TRAINED_DIR if weights_dir is None else weights_dir
  if not os.path.isdir(src):
    raise FileNotFoundError(
        f"Trained weights not found at {src}. Populate it with "
        "`python3 demos/lola_train.py --train` (or "
        "`python3 demos/prepare_demo_assets.py` to fetch data and train "
        "every demo model)."
    )
  model = QuadLoLA().to(dtype)
  params = dict(model.named_parameters())
  with torch.no_grad():
    for fname, key, shape in WEIGHT_FILES:
      flat = _load_floats(os.path.join(src, fname))
      expected = int(np.prod(shape))
      if flat.size != expected:
        raise ValueError(
            f"{fname}: expected {expected} float64 values for shape {shape}, "
            f"got {flat.size}"
        )
      params[key].copy_(
          torch.from_numpy(np.array(flat, copy=True).reshape(shape)).to(dtype)
      )
  return model.eval()


def load_mnist_test_200(weights_dir: str | None = None):
  """Return the demo's 200-image MNIST slice as (images, labels, pt_pred)."""
  src = TRAINED_DIR if weights_dir is None else weights_dir
  if not os.path.isdir(src):
    raise FileNotFoundError(f"LoLA test assets not found at {src}")
  flat_images = _load_floats(os.path.join(src, "mnist_test_200.bin"))
  if flat_images.size % (H_IN * W_IN):
    raise ValueError(
        "mnist_test_200.bin must contain a whole number of 28x28 images; "
        f"got {flat_images.size} float64 values"
    )
  imgs = flat_images.reshape(-1, H_IN * W_IN)
  with open(os.path.join(src, "mnist_test_200_labels.bin"), "rb") as f:
    labels = np.frombuffer(f.read(), dtype=np.int32)
  with open(os.path.join(src, "lola_plaintext_pred_200.bin"), "rb") as f:
    pt_pred = np.frombuffer(f.read(), dtype=np.int32)
  if not len(imgs):
    raise ValueError("LoLA test assets must contain at least one image")
  if not (len(imgs) == len(labels) == len(pt_pred)):
    raise ValueError(
        "LoLA test asset count mismatch: "
        f"{len(imgs)} images, {len(labels)} labels, "
        f"{len(pt_pred)} plaintext predictions"
    )
  return imgs, labels, pt_pred


# ---------------------------------------------------------------------------
# MNIST.  Raw IDX, scaled to [0, 1] and nothing else.
#
# `lola_cleartext_inference` (lola_he.py:241) feeds its `img_flat` straight into
# conv1 with no normalisation step, and the random fallback in
# `_generate_random_fallback` (lola_he.py:155) draws images from `rng.rand`,
# i.e. [0, 1).  So [0, 1] is the convention the encrypted path already assumes.
# LeNet's mean/std standardisation deliberately does NOT apply here.
#
# The IDX readers are kept local rather than imported from `lenet_train`, whose
# `load_mnist` bakes in the mean/std step LoLA must not have.
# ---------------------------------------------------------------------------
_MNIST_DIR = os.path.join(os.path.dirname(_HERE), "mnist", "data")

_IDX_CANDIDATES = {
    "train_x": ["train-images-idx3-ubyte", "train-images.idx3-ubyte"],
    "train_y": ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"],
    "test_x": ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"],
    "test_y": ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"],
}


def _resolve_idx(mnist_dir: str) -> dict[str, str]:
  paths = {}
  for key, names in _IDX_CANDIDATES.items():
    for name in names:
      candidate = os.path.join(mnist_dir, name)
      if os.path.isfile(candidate):
        paths[key] = candidate
        break
    else:
      raise FileNotFoundError(
          f"MNIST {key} not found under {mnist_dir}. Run "
          "`python3 demos/prepare_demo_assets.py --data-only` to download it."
      )
  return paths


def _load_idx_images(path: str) -> np.ndarray:
  with open(path, "rb") as f:
    magic = int.from_bytes(f.read(4), "big")
    if magic != 2051:
      raise ValueError(f"{path}: bad IDX image magic {magic}")
    count = int.from_bytes(f.read(4), "big")
    rows = int.from_bytes(f.read(4), "big")
    cols = int.from_bytes(f.read(4), "big")
    buf = np.frombuffer(f.read(), dtype=np.uint8)
  expected = count * rows * cols
  if buf.size != expected:
    raise ValueError(f"{path}: expected {expected} pixels, got {buf.size}")
  return buf.reshape(count, rows * cols)


def _load_idx_labels(path: str) -> np.ndarray:
  with open(path, "rb") as f:
    magic = int.from_bytes(f.read(4), "big")
    if magic != 2049:
      raise ValueError(f"{path}: bad IDX label magic {magic}")
    count = int.from_bytes(f.read(4), "big")
    buf = np.frombuffer(f.read(), dtype=np.uint8)
  if buf.size != count:
    raise ValueError(f"{path}: expected {count} labels, got {buf.size}")
  return buf.astype(np.int64)


def load_mnist(mnist_dir: str = _MNIST_DIR):
  """Return (train_x, train_y, test_x, test_y); images float32 in [0, 1]."""
  paths = _resolve_idx(mnist_dir)
  train_x = _load_idx_images(paths["train_x"]).astype(np.float32) / 255.0
  test_x = _load_idx_images(paths["test_x"]).astype(np.float32) / 255.0
  train_y = _load_idx_labels(paths["train_y"])
  test_y = _load_idx_labels(paths["test_y"])
  for split, images, labels in (
      ("train", train_x, train_y), ("test", test_x, test_y)
  ):
    if images.shape[1:] != (H_IN * W_IN,):
      raise ValueError(
          f"MNIST {split} images must be {H_IN}x{W_IN}; "
          f"got flattened shape {images.shape[1:]}"
      )
    if not len(images):
      raise ValueError(f"MNIST {split} set must not be empty")
    if len(images) != len(labels):
      raise ValueError(
          f"MNIST {split} image/label count mismatch: "
          f"{len(images)} images, {len(labels)} labels"
      )
  return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
              batch_size: int = 512) -> float:
  if len(x) == 0:
    raise ValueError("cannot evaluate an empty data set")
  model.eval()
  correct = 0
  with torch.no_grad():
    for start in range(0, len(x), batch_size):
      logits = model(x[start:start + batch_size])
      correct += int((logits.argmax(dim=1) == y[start:start + batch_size]).sum())
  return correct / len(x)


def train(epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
          weight_decay: float = 1e-4, seed: int = 0, device: str = "cpu",
          mnist_dir: str = _MNIST_DIR):
  """Train QuadLoLA on MNIST and return (model, history).

  Two stacked squarings make this net far more init- and LR-sensitive than a
  ReLU one: activations are raised to the fourth power end to end, so a step
  that is merely large enough to be noisy under ReLU can send this to NaN.
  `QuadLoLA._init_small` shrinks the init for that reason; gradient-norm
  clipping below is the matching runtime guard.
  """
  if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
    raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
  if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
      or batch_size <= 0):
    raise ValueError(
        f"batch_size must be a positive integer, got {batch_size!r}"
    )
  torch.manual_seed(seed)
  np.random.seed(seed)

  train_x, train_y, test_x, test_y = load_mnist(mnist_dir)
  xtr = torch.from_numpy(train_x).reshape(-1, CI, H_IN, W_IN).to(device)
  ytr = torch.from_numpy(train_y).to(device)
  xte = torch.from_numpy(test_x).reshape(-1, CI, H_IN, W_IN).to(device)
  yte = torch.from_numpy(test_y).to(device)

  model = QuadLoLA().to(torch.float32).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  loss_fn = nn.CrossEntropyLoss()

  history = {"train_acc": [], "test_acc": [], "loss": []}
  n = len(xtr)
  for epoch in range(epochs):
    model.train()
    order = torch.randperm(n, device=device)
    running, seen, correct = 0.0, 0, 0
    for start in range(0, n, batch_size):
      idx = order[start:start + batch_size]
      xb, yb = xtr[idx], ytr[idx]
      optimizer.zero_grad(set_to_none=True)
      logits = model(xb)
      loss = loss_fn(logits, yb)
      if not torch.isfinite(loss):
        raise RuntimeError(
            f"loss diverged at epoch {epoch + 1}; lower --lr (quadratic "
            "activations raise activations to the 4th power end to end)."
        )
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      running += float(loss.detach()) * len(idx)
      seen += len(idx)
      correct += int((logits.argmax(dim=1) == yb).sum())
    scheduler.step()
    test_acc = _evaluate(model, xte, yte)
    history["loss"].append(running / seen)
    history["train_acc"].append(correct / seen)
    history["test_acc"].append(test_acc)
    print(f"  epoch {epoch + 1:2d}/{epochs}  loss={running / seen:.4f}  "
          f"train_acc={correct / seen * 100:.2f}%  "
          f"test_acc={test_acc * 100:.2f}%", flush=True)
  return model.eval(), history


# ---------------------------------------------------------------------------
# Export.  Writes the same nine files `lola_he.load_trained` reads.
# ---------------------------------------------------------------------------
def _save_floats(path: str, arr: np.ndarray) -> None:
  with open(path, "wb") as f:
    f.write(np.asarray(arr, dtype=np.float64).ravel().tobytes())


def _save_ints(path: str, arr: np.ndarray) -> None:
  with open(path, "wb") as f:
    f.write(np.asarray(arr, dtype=np.int32).ravel().tobytes())


def export_weights(model: nn.Module, out_dir: str = TRAINED_DIR,
                   mnist_dir: str = _MNIST_DIR,
                   n_test: int = _N_TEST_IMAGES) -> None:
  """Write the six weight binaries plus the demo's 200-image test slice.

  `lola_plaintext_pred_200.bin` is produced by `lola_cleartext_inference` --
  the same NumPy reference the encrypted path is compared against -- rather
  than by the torch forward, so that a pt/HE disagreement always means the
  ciphertext path drifted, never that the two references disagree.
  """
  # Local import avoids the encrypted-demo dependency during ordinary trainer
  # import. Support both ``python demos/lola_train.py`` and namespace-package
  # use via ``import demos.lola_train``.
  if __package__:
    from .lola_he import lola_cleartext_inference
  else:
    from lola_he import lola_cleartext_inference

  model.eval()
  params = dict(model.named_parameters())
  weight_arrays = []
  for fname, key, shape in WEIGHT_FILES:
    arr = params[key].detach().cpu().numpy().astype(np.float64)
    if arr.shape != shape:
      raise ValueError(f"{key}: expected {shape}, got {arr.shape}")
    weight_arrays.append((fname, arr))

  _, _, test_x, test_y = load_mnist(mnist_dir)
  if not isinstance(n_test, int) or isinstance(n_test, bool) or n_test <= 0:
    raise ValueError(f"n_test must be a positive integer, got {n_test!r}")
  if n_test > len(test_x):
    raise ValueError(
        f"n_test={n_test} exceeds the available test set ({len(test_x)})"
    )
  imgs = test_x[:n_test].astype(np.float64)
  labels = test_y[:n_test].astype(np.int32)

  W1 = params["conv1.weight"].detach().cpu().numpy().astype(np.float64).ravel()
  b1 = params["conv1.bias"].detach().cpu().numpy().astype(np.float64)
  W2 = params["fc1.weight"].detach().cpu().numpy().astype(np.float64)
  b2 = params["fc1.bias"].detach().cpu().numpy().astype(np.float64)
  W3 = params["fc2.weight"].detach().cpu().numpy().astype(np.float64)
  b3 = params["fc2.bias"].detach().cpu().numpy().astype(np.float64)
  pt_pred = np.zeros(n_test, dtype=np.int32)
  for i in range(n_test):
    pt_pred[i] = int(np.argmax(
        lola_cleartext_inference(imgs[i], W1, b1, W2, b2, W3, b3)))

  # Only create or update the asset directory once all input validation and
  # reference inference have succeeded, avoiding misleading partial exports.
  os.makedirs(out_dir, exist_ok=True)
  for fname, arr in weight_arrays:
    _save_floats(os.path.join(out_dir, fname), arr)
  _save_floats(os.path.join(out_dir, "mnist_test_200.bin"), imgs)
  _save_ints(os.path.join(out_dir, "mnist_test_200_labels.bin"), labels)
  _save_ints(os.path.join(out_dir, "lola_plaintext_pred_200.bin"), pt_pred)

  agree = int((pt_pred == labels).sum())
  print(f"[export] wrote 9 files to {out_dir}")
  print(f"[export] cleartext reference accuracy on the slice: "
        f"{agree}/{n_test} = {agree / n_test * 100:.1f}%")


def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--train", action="store_true",
                  help="Train on MNIST and export weights instead of "
                       "evaluating an existing set.")
  ap.add_argument("--epochs", type=int, default=30)
  ap.add_argument("--batch-size", type=int, default=128)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--weight-decay", type=float, default=1e-4)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--mnist-dir", default=_MNIST_DIR)
  ap.add_argument("--out-dir", default=TRAINED_DIR)
  ap.add_argument("--weights-dir", default=None)
  ap.add_argument("--n", type=int, default=200, help="Images to score.")
  args = ap.parse_args(argv)

  if not args.train and args.n <= 0:
    ap.error("--n must be positive")

  if args.train:
    print("=" * 64)
    print("LoLA-MNIST (quadratic activations) — training")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  weight_decay={args.weight_decay}  device={args.device}")
    print("=" * 64)
    model, history = train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device,
        mnist_dir=args.mnist_dir)
    export_weights(model, out_dir=args.out_dir, mnist_dir=args.mnist_dir)
    print(f"\n[done] final test acc = {history['test_acc'][-1] * 100:.2f}%")
    return

  model = load_pretrained_lola(args.weights_dir)
  imgs, labels, pt_pred = load_mnist_test_200(args.weights_dir)
  n = min(args.n, len(imgs))
  x = torch.from_numpy(
      np.array(imgs[:n], copy=True).reshape(n, CI, H_IN, W_IN)
  ).to(torch.float64)
  with torch.no_grad():
    pred = model(x).argmax(dim=1).numpy()
  print(f"[data] {TRAINED_DIR if args.weights_dir is None else args.weights_dir}")
  print(f"torch acc vs labels:   {int((pred == labels[:n]).sum())}/{n}")
  print(f"torch == cleartext pt: {int((pred == pt_pred[:n]).sum())}/{n}")


if __name__ == "__main__":
  main()
