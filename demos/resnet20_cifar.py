"""Standard ResNet-20 for CIFAR-10 (He et al. variant), trained with ORION's
recipe — the SotA-following counterpart to this folder's HE demo nets.

Run: `python3 resnet20_cifar.py --act relu` (or `--act silu`). See the demos
README section "Reaching CIFAR-10 SotA (>90%): ResNet-20" for the accuracy
gap-diagnosis, results, training recipe, and the encrypted-inference caveat.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

# CIFAR-10 stats used by ORION (github.com/baahl-nyu/orion).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

_ACTS = {"relu": nn.ReLU, "silu": nn.SiLU}


# ---------------------------------------------------------------------------
# Standard ResNet-20 (He et al. CIFAR variant) — matches ORION's ResNet20:
# conv1(3->16), three [3,3,3] BasicBlock stages with channels [16,32,64] and
# strides [1,2,2], global avg-pool, Linear(64->10). ~272K params.
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, ci, co, stride=1, act=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(co)
        self.act1 = act()
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.act2 = act()
        self.short = nn.Sequential()
        if stride != 1 or ci != co:
            self.short = nn.Sequential(
                nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return self.act2(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes: int = 10, act: str = "relu"):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(
                f"unknown activation {act!r}; expected one of {tuple(_ACTS)}"
            )
        a = _ACTS[act]
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = a()
        self.layer1 = self._make(16, 16, 3, 1, a)
        self.layer2 = self._make(16, 32, 3, 2, a)
        self.layer3 = self._make(32, 64, 3, 2, a)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make(ci, co, n, stride, a):
        blocks = [BasicBlock(ci, co, stride, a)]
        blocks += [BasicBlock(co, co, 1, a) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training — ORION's exact CIFAR-10 recipe.
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch (host + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, deterministic seed derived from
    torch's per-epoch base seed (so augmentation RNG is reproducible)."""
    # DataLoader has already incorporated worker_id into initial_seed().
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _loaders(data_dir, batch_size, num_workers=8, seed=42):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tr = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf_train)
    te = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_test)
    # Seed the shuffle order via a dedicated generator; seed workers via
    # worker_init_fn so a fixed --seed makes the whole run reproducible.
    gen = torch.Generator()
    gen.manual_seed(seed)
    test_workers = max(1, num_workers // 2) if num_workers > 0 else 0
    return (DataLoader(tr, batch_size, shuffle=True, num_workers=num_workers,
                       pin_memory=True, drop_last=True,
                       generator=gen, worker_init_fn=_worker_init_fn),
            DataLoader(te, 512, shuffle=False, num_workers=test_workers,
                       pin_memory=True, worker_init_fn=_worker_init_fn))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += int((model(xb).argmax(1) == yb).sum()); total += len(yb)
    if total == 0:
        raise ValueError("cannot evaluate an empty data loader")
    return correct / total


def train(model, data_dir="./cifar10_data", epochs=200, batch_size=128,
          lr=0.1, momentum=0.9, weight_decay=5e-4, device="cpu",
          num_workers=8, seed=42, out_path=None):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or batch_size <= 0):
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )
    train_loader, test_loader = _loaders(data_dir, batch_size, num_workers, seed)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    best_state = None
    acc = 0.0
    for ep in range(epochs):
        t0 = time.perf_counter(); model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        sched.step()
        acc = evaluate(model, test_loader, device)
        if best_state is None or acc > best:
            best = acc
            # Snapshot the best weights on CPU so a later crash/interrupt still
            # leaves us the best checkpoint, not just the final epoch.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"  epoch {ep+1:>3}/{epochs}  test_acc={acc*100:5.2f}%  "
              f"best={best*100:5.2f}%  ({time.perf_counter()-t0:.1f}s)", flush=True)
    final = acc
    if out_path is not None and best_state is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(best_state, out_path)
        print(f"[ckpt] saved best state_dict (test_acc={best*100:.2f}%) -> "
              f"{out_path}", flush=True)
    return best, final


def main(argv=None):
    _here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--act", choices=list(_ACTS), default="relu")
    ap.add_argument("--data-dir", default=os.path.join(_here, "cifar10_data"))
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="path for the best-epoch state_dict checkpoint "
                         "(default demos/maple_data/resnet20_<act>.pt)")
    args = ap.parse_args(argv)

    out_path = args.out or os.path.join(
        _here, "maple_data", f"resnet20_{args.act}.pt")
    seed_everything(args.seed)
    if args.device == "cpu" and args.epochs > 5:
        print(f"[warn] device=cpu with --epochs {args.epochs}: a full "
              f"200-epoch ResNet-20 run takes DAYS on CPU (the ~6 min figure "
              f"assumes a high-end GPU). Use --device cuda, or drop --epochs "
              f"for a smoke test.", flush=True)

    net = ResNet20(act=args.act)
    n = sum(p.numel() for p in net.parameters())
    print(f"ResNet-20 ({args.act}) CIFAR-10 — params={n:,}, "
          f"device={args.device}, seed={args.seed}, "
          f"batch={args.batch_size}, lr={args.lr}")
    best, final = train(
        net, data_dir=args.data_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
        num_workers=args.num_workers, seed=args.seed, out_path=out_path)
    print(f"\n[done] best test acc = {best*100:.2f}%  |  "
          f"final-epoch test acc = {final*100:.2f}%")


if __name__ == "__main__":
    main()
