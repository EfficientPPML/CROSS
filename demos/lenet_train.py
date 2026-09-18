"""Train the scaled LeNet (matches `lenet_he.py`) on MNIST and dump float64
weight binaries for the encrypted demo to consume.

Architecture (exactly mirrors the depth-7 LeNetHE pipeline):

    Input X: (1, 28, 28)
    Conv1: 1->4, 5x5, stride=2, pad=2, +bias  -> (4, 14, 14)
    Quad:  y = x²
    Conv2: 4->8, 5x5, stride=2, pad=2, +bias  -> (8, 7, 7)
    Quad
    FC1:   392 -> 32  +bias
    Quad
    FC2:   32  -> 10  +bias

x² activations make this network finicky to train (no negative-region
gradient masking, activations grow rapidly). We use:
  * MNIST canonical normalisation (mean=0.1307, std=0.3081)
  * small init (Kaiming with gain 0.1)
  * low LR Adam + heavy weight decay
  * gradient clipping at norm 1.0

This is *not* a competitive MNIST classifier; the goal is to land somewhere
clearly above chance (target: ≥80% test accuracy at this scale) so the HE
demo's argmaxes track the true labels often enough to be visually convincing.
"""
from __future__ import annotations

import argparse
import os
import struct
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


_HERE = os.path.dirname(os.path.abspath(__file__))
_MNIST_DIR = os.path.join(_HERE, "..", "mnist", "data")
_OUT_DIR = os.path.join(_HERE, "maple_data")

# Architectural constants — must match lenet_he.py exactly.
CI, H_IN, W_IN = 1, 28, 28
CO1, H1, W1 = 4, 14, 14
CO2, H2, W2 = 8, 7, 7
KH, KW = 5, 5
STRIDE, PAD = 2, 2
FC1_IN, FC1_OUT = CO2 * H2 * W2, 32
FC2_OUT = 10

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# Output filename ↔ flat-array shape, must match lenet_he._TRAINED_FILES.
OUTPUT_FILES = [
    ("lenet_conv1_W.bin", "conv1.weight", (CO1, CI, KH, KW)),
    ("lenet_conv1_b.bin", "conv1.bias",   (CO1,)),
    ("lenet_conv2_W.bin", "conv2.weight", (CO2, CO1, KH, KW)),
    ("lenet_conv2_b.bin", "conv2.bias",   (CO2,)),
    ("lenet_fc1_W.bin",   "fc1.weight",   (FC1_OUT, FC1_IN)),
    ("lenet_fc1_b.bin",   "fc1.bias",     (FC1_OUT,)),
    ("lenet_fc2_W.bin",   "fc2.weight",   (FC2_OUT, FC1_OUT)),
    ("lenet_fc2_b.bin",   "fc2.bias",     (FC2_OUT,)),
]


# ---------------------------------------------------------------------------
# Raw MNIST IDX loader (no torchvision dependency).
# ---------------------------------------------------------------------------
def _load_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"{path}: truncated IDX image header")
        magic, n, h, w = struct.unpack(">IIII", header)
        if magic != 0x803:
            raise ValueError(f"{path}: bad magic {magic:#x}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    expected = n * h * w
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} pixels, got {data.size}")
    return data.reshape(n, h, w)


def _load_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"{path}: truncated IDX label header")
        magic, n = struct.unpack(">II", header)
        if magic != 0x801:
            raise ValueError(f"{path}: bad magic {magic:#x}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n:
        raise ValueError(f"{path}: expected {n} labels, got {data.size}")
    return data


def load_mnist(mnist_dir: str = _MNIST_DIR) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """Return (train_X, train_y, test_X, test_y) as numpy arrays.

    Images are normalised: float32 in [0,1], then `(x - mean) / std`.
    """
    cands = {
        "train_x": ["train-images-idx3-ubyte", "train-images.idx3-ubyte"],
        "train_y": ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"],
        "test_x":  ["t10k-images-idx3-ubyte",  "t10k-images.idx3-ubyte"],
        "test_y":  ["t10k-labels-idx1-ubyte",  "t10k-labels.idx1-ubyte"],
    }
    paths: dict[str, str] = {}
    for k, names in cands.items():
        for n in names:
            p = os.path.join(mnist_dir, n)
            if os.path.isfile(p):
                paths[k] = p
                break
        else:
            raise FileNotFoundError(
                f"{k}: none of {names} found in {mnist_dir}")
    train_x = _load_idx_images(paths["train_x"])
    test_x = _load_idx_images(paths["test_x"])
    train_y = _load_idx_labels(paths["train_y"]).astype(np.int64)
    test_y = _load_idx_labels(paths["test_y"]).astype(np.int64)
    for split, images, labels in (
        ("train", train_x, train_y), ("test", test_x, test_y)
    ):
        if images.shape[1:] != (H_IN, W_IN):
            raise ValueError(
                f"MNIST {split} images must be {H_IN}x{W_IN}; "
                f"got shape {images.shape[1:]}"
            )
        if not len(images):
            raise ValueError(f"MNIST {split} set must not be empty")
        if len(images) != len(labels):
            raise ValueError(
                f"MNIST {split} image/label count mismatch: "
                f"{len(images)} images, {len(labels)} labels"
            )
    train_x = train_x.astype(np.float32) / 255.0
    test_x = test_x.astype(np.float32) / 255.0
    train_x = (train_x - MNIST_MEAN) / MNIST_STD
    test_x  = (test_x  - MNIST_MEAN) / MNIST_STD
    return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------------------
# Model.  Pure quadratic activations — matches the HE pipeline exactly.
# ---------------------------------------------------------------------------
class QuadLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CI,  CO1, KH, stride=STRIDE, padding=PAD,
                               bias=True)
        self.conv2 = nn.Conv2d(CO1, CO2, KH, stride=STRIDE, padding=PAD,
                               bias=True)
        self.fc1   = nn.Linear(FC1_IN, FC1_OUT, bias=True)
        self.fc2   = nn.Linear(FC1_OUT, FC2_OUT, bias=True)
        self._init_small()

    def _init_small(self):
        # Smaller-than-Kaiming init keeps activations bounded under x².
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in",
                                    nonlinearity="linear")
            with torch.no_grad():
                m.weight.mul_(0.5)
                m.bias.zero_()
        for m in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in",
                                    nonlinearity="linear")
            with torch.no_grad():
                m.weight.mul_(0.3)
                m.bias.zero_()

    def forward(self, x):
        x = self.conv1(x); x = x * x
        x = self.conv2(x); x = x * x
        x = x.view(x.size(0), -1)
        x = self.fc1(x);   x = x * x
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Train / eval loops.
# ---------------------------------------------------------------------------
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.size(0))
    if total == 0:
        raise ValueError("cannot evaluate an empty data loader")
    return correct / total


def train(epochs: int, batch_size: int, lr: float, wd: float,
          seed: int, device: str = "cpu") -> Tuple[QuadLeNet, dict]:
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or batch_size <= 0):
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[data] loading MNIST from {_MNIST_DIR} ...")
    train_x, train_y, test_x, test_y = load_mnist()
    print(f"  train: {train_x.shape}, test: {test_x.shape}")

    # Pad with channel dim.
    train_t = torch.from_numpy(train_x).unsqueeze(1)        # (N, 1, 28, 28)
    test_t  = torch.from_numpy(test_x).unsqueeze(1)
    train_y_t = torch.from_numpy(train_y)
    test_y_t  = torch.from_numpy(test_y)
    train_loader = DataLoader(
        TensorDataset(train_t, train_y_t),
        batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(
        TensorDataset(test_t,  test_y_t),
        batch_size=512, shuffle=False)

    model = QuadLeNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params = {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        t_ep = time.perf_counter()
        model.train()
        run_loss = 0.0
        run_correct = run_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            run_loss += float(loss.item()) * xb.size(0)
            run_correct += int((logits.argmax(dim=1) == yb).sum().item())
            run_total += int(yb.size(0))
        train_loss = run_loss / max(run_total, 1)
        train_acc  = run_correct / max(run_total, 1)
        test_acc   = _evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"  epoch {ep+1:>2d}/{epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc*100:.2f}%  test_acc={test_acc*100:.2f}%  "
              f"({time.perf_counter()-t_ep:.1f}s)")
    return model, history


# ---------------------------------------------------------------------------
# Export to flat float64 binaries (matches lenet_he._TRAINED_FILES).
# ---------------------------------------------------------------------------
def export_weights(model: nn.Module, out_dir: str = _OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    print(f"\n[export] writing trained weights to {out_dir}/ ...")
    for fname, key, expected_shape in OUTPUT_FILES:
        if key not in state:
            raise KeyError(f"state_dict missing '{key}'")
        arr = state[key]
        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"{key}: expected {expected_shape}, got {arr.shape}")
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            f.write(arr.astype(np.float64).ravel().tobytes())
        print(f"  {fname}  shape={arr.shape}  bytes={arr.size*8}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=_OUT_DIR)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    print("=" * 64)
    print("LeNet (scaled, quad activations) — MNIST training")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  weight_decay={args.weight_decay}  device={args.device}")
    print("=" * 64)

    model, history = train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        wd=args.weight_decay, seed=args.seed, device=args.device)
    export_weights(model, out_dir=args.out_dir)

    final_test_acc = history["test_acc"][-1] * 100
    print(f"\n[done] final test acc = {final_test_acc:.2f}%")


if __name__ == "__main__":
    main()
