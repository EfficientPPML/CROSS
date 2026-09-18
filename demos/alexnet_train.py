"""HE-friendly AlexNet for CIFAR-10 — training script.

Trains either:
  * AlexNetTiny (default) — depth-7 variant, 3 conv + 1 FC, ~13K params.
    Matches `AlexNetTinyHE` in `alexnet_he.py`.
  * AlexNet (full, --full) — depth-15 variant, 5 conv + 3 FC, ~50K params.
    Matches `AlexNetHE` in `alexnet_he.py`. Architecture follows the
    OpenFHE reference (`openfhe_ref_code/models/alexnet.py`) but with
    channels scaled DOWN to fit NUM_SLOTS=1024 at every layer.

In both cases BN is included during training (right before each Quad,
to keep activations bounded under x²) and **folded into the preceding
conv/linear weights at export time** — saved binaries describe a pure
Conv→Pool→Quad→…→Linear network with no BN at inference. Pool layers
stay explicit at export; they're folded into BSGS matrices in alexnet_he.py.

Architecture details:

  AlexNetTiny (depth-7):
    Input X: client-side avg-pool 2x2 from CIFAR-10 (3, 32, 32) → (3, 16, 16)
    Conv1(3→4,  3×3/s=1/p=1) + AvgPool 2×2 + Quad   →  4× 8× 8 = 256 slots
    Conv2(4→8,  3×3/s=1/p=1) + AvgPool 2×2 + Quad   →  8× 4× 4 = 128 slots
    Conv3(8→16, 3×3/s=1/p=1) + AdaptivePool 2×2 + Quad → 16× 2× 2 = 64 slots
    Linear(64→10)

  AlexNet (full, depth-15):
    Input X: client-side avg-pool 2x2 from CIFAR-10 (3, 32, 32) → (3, 16, 16)
    Conv1(3→8,   3×3/s=1/p=1) + AvgPool 2×2 + Quad      →  8× 8× 8 = 512 slots
    Conv2(8→16,  3×3/s=1/p=1) + AvgPool 2×2 + Quad      → 16× 4× 4 = 256 slots
    Conv3(16→32, 3×3/s=1/p=1)              + Quad      → 32× 4× 4 = 512 slots
    Conv4(32→32, 3×3/s=1/p=1)              + Quad      → 32× 4× 4 = 512 slots
    Conv5(32→32, 3×3/s=1/p=1) + AdaptivePool 2×2 + Quad → 32× 2× 2 = 128 slots
    Linear(128→64)  + Quad
    Linear(64→32)   + Quad
    Linear(32→10)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, "maple_data")
_CIFAR_DIR = os.path.join(_HERE, "cifar10_data")

# Architecture — must match TINY_* constants in alexnet_he.py.
TINY_CI = 3
TINY_H_IN, TINY_W_IN = 16, 16
TINY_CO1, TINY_CO2, TINY_CO3 = 4, 8, 16
TINY_FC_IN  = 64
TINY_FC_OUT = 10
KH, KW, PAD = 3, 3, 1

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

OUTPUT_FILES = [
    ("alexnet_tiny_conv1_W.bin", "conv1.weight",  (TINY_CO1, TINY_CI, KH, KW)),
    ("alexnet_tiny_conv1_b.bin", "conv1.bias",    (TINY_CO1,)),
    ("alexnet_tiny_conv2_W.bin", "conv2.weight",  (TINY_CO2, TINY_CO1, KH, KW)),
    ("alexnet_tiny_conv2_b.bin", "conv2.bias",    (TINY_CO2,)),
    ("alexnet_tiny_conv3_W.bin", "conv3.weight",  (TINY_CO3, TINY_CO2, KH, KW)),
    ("alexnet_tiny_conv3_b.bin", "conv3.bias",    (TINY_CO3,)),
    ("alexnet_tiny_fc_W.bin",    "fc.weight",     (TINY_FC_OUT, TINY_FC_IN)),
    ("alexnet_tiny_fc_b.bin",    "fc.bias",       (TINY_FC_OUT,)),
]


# ---------------------------------------------------------------------------
# Full AlexNet (depth-15) constants — must match FULL_* in alexnet_he.py.
# ---------------------------------------------------------------------------
FULL_CI = 3
FULL_H_IN, FULL_W_IN = 16, 16
FULL_CO1, FULL_CO2, FULL_CO3, FULL_CO4, FULL_CO5 = 8, 16, 32, 32, 32
FULL_FC1_IN  = FULL_CO5 * 2 * 2     # 32·2·2 = 128
FULL_FC1_OUT = 64
FULL_FC2_OUT = 32
FULL_FC3_OUT = 10

OUTPUT_FILES_FULL = [
    ("alexnet_full_conv1_W.bin", "conv1.weight",  (FULL_CO1, FULL_CI,  KH, KW)),
    ("alexnet_full_conv1_b.bin", "conv1.bias",    (FULL_CO1,)),
    ("alexnet_full_conv2_W.bin", "conv2.weight",  (FULL_CO2, FULL_CO1, KH, KW)),
    ("alexnet_full_conv2_b.bin", "conv2.bias",    (FULL_CO2,)),
    ("alexnet_full_conv3_W.bin", "conv3.weight",  (FULL_CO3, FULL_CO2, KH, KW)),
    ("alexnet_full_conv3_b.bin", "conv3.bias",    (FULL_CO3,)),
    ("alexnet_full_conv4_W.bin", "conv4.weight",  (FULL_CO4, FULL_CO3, KH, KW)),
    ("alexnet_full_conv4_b.bin", "conv4.bias",    (FULL_CO4,)),
    ("alexnet_full_conv5_W.bin", "conv5.weight",  (FULL_CO5, FULL_CO4, KH, KW)),
    ("alexnet_full_conv5_b.bin", "conv5.bias",    (FULL_CO5,)),
    ("alexnet_full_fc1_W.bin",   "fc1.weight",    (FULL_FC1_OUT, FULL_FC1_IN)),
    ("alexnet_full_fc1_b.bin",   "fc1.bias",      (FULL_FC1_OUT,)),
    ("alexnet_full_fc2_W.bin",   "fc2.weight",    (FULL_FC2_OUT, FULL_FC1_OUT)),
    ("alexnet_full_fc2_b.bin",   "fc2.bias",      (FULL_FC2_OUT,)),
    ("alexnet_full_fc3_W.bin",   "fc3.weight",    (FULL_FC3_OUT, FULL_FC2_OUT)),
    ("alexnet_full_fc3_b.bin",   "fc3.bias",      (FULL_FC3_OUT,)),
]


# ---------------------------------------------------------------------------
# Model: depth-7 HE-friendly AlexNetTiny.  3 conv blocks + 1 FC.
# ---------------------------------------------------------------------------
class QuadAlexNetTiny(nn.Module):
    def __init__(self, num_classes: int = TINY_FC_OUT):
        super().__init__()
        # Initial 2×2 client-side avg-pool is part of the data path (applied
        # in forward() on (3, 32, 32) inputs); the model proper starts at
        # (3, 16, 16).
        self.conv1 = nn.Conv2d(TINY_CI, TINY_CO1, KH, padding=PAD, bias=True)
        self.bn1   = nn.BatchNorm2d(TINY_CO1)
        self.conv2 = nn.Conv2d(TINY_CO1, TINY_CO2, KH, padding=PAD, bias=True)
        self.bn2   = nn.BatchNorm2d(TINY_CO2)
        self.conv3 = nn.Conv2d(TINY_CO2, TINY_CO3, KH, padding=PAD, bias=True)
        self.bn3   = nn.BatchNorm2d(TINY_CO3)
        self.fc    = nn.Linear(TINY_FC_IN, num_classes, bias=True)
        self._init_small()

    def _init_small(self):
        for m in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in",
                                    nonlinearity="linear")
            with torch.no_grad():
                m.weight.mul_(0.5)
                m.bias.zero_()
        nn.init.kaiming_normal_(self.fc.weight, a=0, mode="fan_in",
                                nonlinearity="linear")
        with torch.no_grad():
            self.fc.weight.mul_(0.3)
            self.fc.bias.zero_()

    def forward(self, x):
        # Client-side initial avg-pool (matches alexnet_he.downsample_cifar10_2x).
        if x.shape[-1] == 32:
            x = F.avg_pool2d(x, 2, 2)        # (3, 32, 32) → (3, 16, 16)
        # Conv1 + BN + Pool + Quad
        x = self.bn1(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)            # 16→8
        x = x * x
        # Conv2 + BN + Pool + Quad
        x = self.bn2(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)            # 8→4
        x = x * x
        # Conv3 + BN + AdaptivePool + Quad
        x = self.bn3(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (2, 2)) # 4→2
        x = x * x
        # FC
        x = x.reshape(x.size(0), -1)  # reshape (not view): adaptive_avg_pool2d
                                      # output is non-contiguous on newer torch
        x = self.fc(x)
        return x


class QuadAlexNetFull(nn.Module):
    """Depth-15 HE-friendly AlexNet (5 conv + 3 FC + 7 quads)."""

    def __init__(self, num_classes: int = FULL_FC3_OUT):
        super().__init__()
        self.conv1 = nn.Conv2d(FULL_CI,  FULL_CO1, KH, padding=PAD, bias=True)
        self.bn1   = nn.BatchNorm2d(FULL_CO1)
        self.conv2 = nn.Conv2d(FULL_CO1, FULL_CO2, KH, padding=PAD, bias=True)
        self.bn2   = nn.BatchNorm2d(FULL_CO2)
        self.conv3 = nn.Conv2d(FULL_CO2, FULL_CO3, KH, padding=PAD, bias=True)
        self.bn3   = nn.BatchNorm2d(FULL_CO3)
        self.conv4 = nn.Conv2d(FULL_CO3, FULL_CO4, KH, padding=PAD, bias=True)
        self.bn4   = nn.BatchNorm2d(FULL_CO4)
        self.conv5 = nn.Conv2d(FULL_CO4, FULL_CO5, KH, padding=PAD, bias=True)
        self.bn5   = nn.BatchNorm2d(FULL_CO5)
        self.fc1   = nn.Linear(FULL_FC1_IN,  FULL_FC1_OUT, bias=True)
        self.bnf1  = nn.BatchNorm1d(FULL_FC1_OUT)
        self.fc2   = nn.Linear(FULL_FC1_OUT, FULL_FC2_OUT, bias=True)
        self.bnf2  = nn.BatchNorm1d(FULL_FC2_OUT)
        self.fc3   = nn.Linear(FULL_FC2_OUT, num_classes, bias=True)
        self._init_small()

    def _init_small(self):
        for m in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in",
                                    nonlinearity="linear")
            with torch.no_grad():
                # Smaller-than-Tiny init since we have 7 squarings (vs 3)
                # — values blow up super-exponentially under x², so we
                # squash the conv weights more aggressively.
                m.weight.mul_(0.3)
                m.bias.zero_()
        for m in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in",
                                    nonlinearity="linear")
            with torch.no_grad():
                m.weight.mul_(0.2)
                m.bias.zero_()
        nn.init.kaiming_normal_(self.fc3.weight, a=0, mode="fan_in",
                                nonlinearity="linear")
        with torch.no_grad():
            self.fc3.weight.mul_(0.2)
            self.fc3.bias.zero_()

    def forward(self, x):
        if x.shape[-1] == 32:
            x = F.avg_pool2d(x, 2, 2)        # (3, 32, 32) → (3, 16, 16)
        # Conv1 + BN + AvgPool 2x2 + Quad
        x = self.bn1(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)            # 16→8
        x = x * x
        # Conv2 + BN + AvgPool 2x2 + Quad
        x = self.bn2(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)            # 8→4
        x = x * x
        # Conv3 + BN + Quad (no pool)
        x = self.bn3(self.conv3(x))
        x = x * x
        # Conv4 + BN + Quad (no pool)
        x = self.bn4(self.conv4(x))
        x = x * x
        # Conv5 + BN + AdaptivePool + Quad
        x = self.bn5(self.conv5(x))
        x = F.adaptive_avg_pool2d(x, (2, 2)) # 4→2
        x = x * x
        # FC1 + BN + Quad
        x = x.reshape(x.size(0), -1)  # reshape (not view): adaptive_avg_pool2d
                                      # output is non-contiguous on newer torch
        x = self.bnf1(self.fc1(x))
        x = x * x
        # FC2 + BN + Quad
        x = self.bnf2(self.fc2(x))
        x = x * x
        # FC3
        x = self.fc3(x)
        return x


def _fold_bn_into_linear(linear, bn):
    W = linear.weight.detach()
    b = linear.bias.detach() if linear.bias is not None else torch.zeros(
        W.shape[0], device=W.device)
    gamma = bn.weight.detach()
    beta  = bn.bias.detach()
    mu    = bn.running_mean.detach()
    var   = bn.running_var.detach()
    eps   = bn.eps
    sigma = torch.sqrt(var + eps)
    scale = gamma / sigma
    W_new = W * scale.view(-1, 1)
    b_new = (b - mu) * scale + beta
    return W_new, b_new


def _fold_bn_into_conv(conv, bn):
    W = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(
        W.shape[0], device=W.device)
    gamma = bn.weight.detach()
    beta  = bn.bias.detach()
    mu    = bn.running_mean.detach()
    var   = bn.running_var.detach()
    eps   = bn.eps
    sigma = torch.sqrt(var + eps)
    scale = gamma / sigma
    W_new = W * scale.view(-1, 1, 1, 1)
    b_new = (b - mu) * scale + beta
    return W_new, b_new


def _evaluate(model, loader, device):
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


def train(epochs, batch_size, lr, weight_decay, seed, device="cpu",
          full: bool = False):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or batch_size <= 0):
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise RuntimeError("torchvision required for CIFAR-10 download.")

    print(f"[data] preparing CIFAR-10 in {_CIFAR_DIR} ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10(_CIFAR_DIR, train=True, download=True,
                                  transform=transform)
    test_set  = datasets.CIFAR10(_CIFAR_DIR, train=False, download=True,
                                  transform=transform)
    print(f"  train: {len(train_set)} images, test: {len(test_set)} images")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, drop_last=True, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=512,
                               shuffle=False, num_workers=2)

    if full:
        model = QuadAlexNetFull(num_classes=FULL_FC3_OUT).to(device)
        label = "AlexNet (full, depth-15)"
    else:
        model = QuadAlexNetTiny(num_classes=TINY_FC_OUT).to(device)
        label = "AlexNetTiny (depth-7)"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {label} params = {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        t0 = time.perf_counter()
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
              f"({time.perf_counter()-t0:.1f}s)")
    return model, history


def export_weights(model, out_dir=_OUT_DIR):
    """BN-fold + write weight binaries. Auto-detects Tiny vs Full from
    the layer attributes present on the model."""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    is_full = isinstance(model, QuadAlexNetFull)
    print(f"\n[export] BN-folding + writing trained "
          f"{'AlexNet (full)' if is_full else 'AlexNetTiny'} "
          f"weights to {out_dir}/ ...")

    folded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if is_full:
        folded["conv1"] = _fold_bn_into_conv(model.conv1, model.bn1)
        folded["conv2"] = _fold_bn_into_conv(model.conv2, model.bn2)
        folded["conv3"] = _fold_bn_into_conv(model.conv3, model.bn3)
        folded["conv4"] = _fold_bn_into_conv(model.conv4, model.bn4)
        folded["conv5"] = _fold_bn_into_conv(model.conv5, model.bn5)
        folded["fc1"]   = _fold_bn_into_linear(model.fc1, model.bnf1)
        folded["fc2"]   = _fold_bn_into_linear(model.fc2, model.bnf2)
        folded["fc3"]   = (model.fc3.weight.detach(), model.fc3.bias.detach())
        files = OUTPUT_FILES_FULL
    else:
        folded["conv1"] = _fold_bn_into_conv(model.conv1, model.bn1)
        folded["conv2"] = _fold_bn_into_conv(model.conv2, model.bn2)
        folded["conv3"] = _fold_bn_into_conv(model.conv3, model.bn3)
        folded["fc"]    = (model.fc.weight.detach(), model.fc.bias.detach())
        files = OUTPUT_FILES

    state_flat: dict[str, np.ndarray] = {}
    for layer, (W, b) in folded.items():
        state_flat[f"{layer}.weight"] = W.cpu().numpy()
        state_flat[f"{layer}.bias"]   = b.cpu().numpy()

    for fname, key, expected_shape in files:
        if key not in state_flat:
            raise KeyError(f"missing '{key}'")
        arr = state_flat[key]
        if tuple(arr.shape) != expected_shape:
            raise ValueError(f"{key}: expected {expected_shape}, got {arr.shape}")
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            f.write(arr.astype(np.float64).ravel().tobytes())
        print(f"  {fname:<28}  shape={arr.shape}  bytes={arr.size*8}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="Train the depth-15 AlexNet (full) variant. "
                          "Default trains AlexNetTiny (depth-7).")
    ap.add_argument("--epochs",       type=int,   default=20)
    ap.add_argument("--batch-size",   type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out-dir",      default=_OUT_DIR)
    ap.add_argument("--device",       default=("cuda" if torch.cuda.is_available()
                                                else "cpu"))
    args = ap.parse_args(argv)

    label = ("AlexNet (full, depth-15)" if args.full
              else "AlexNetTiny (depth-7)")
    print("=" * 64)
    print(f"{label} HE-friendly — CIFAR-10 training")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  weight_decay={args.weight_decay}  device={args.device}")
    print("=" * 64)

    model, history = train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device,
        full=args.full)
    export_weights(model, out_dir=args.out_dir)

    final_test_acc = history["test_acc"][-1] * 100
    print(f"\n[done] final test acc = {final_test_acc:.2f}%")


if __name__ == "__main__":
    main()
