"""Fetch the datasets the demos need and train the weights they load.

A clean clone ships no datasets and no weights, so every demo model either
skips its accuracy gate or raises `FileNotFoundError`. This script is the one
command that fixes that:

    python3 demos/prepare_demo_assets.py              # data + train everything
    python3 demos/prepare_demo_assets.py --data-only  # just download
    python3 demos/prepare_demo_assets.py --models lola
    python3 demos/prepare_demo_assets.py --epochs 3   # fast smoke run

What it writes
--------------
    demos/cifar10_data/          CIFAR-10 (torchvision layout) -- AlexNet
    mnist/data/                  MNIST IDX files              -- LeNet, LoLA
    demos/maple_data/            AlexNet + LeNet weight binaries
    data/pretrained_weights/lola LoLA weight binaries + 200-image test slice

`demos/cifar10_data` used to be a committed symlink into a developer's
container (`/workspace/CROSS_online/...`). It is a plain directory now, created
here on demand -- a dangling symlink made `os.makedirs(..., exist_ok=True)`
raise `FileExistsError` and took AlexNet training down on every other machine.

MNIST is fetched through torchvision purely for its mirror list, then the raw
IDX files are copied to `mnist/data`, which is where `lenet_train.load_mnist`
and `lola_train.load_mnist` look for them.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "jaxite_word"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CIFAR_DIR = os.path.join(_HERE, "cifar10_data")
MNIST_DIR = os.path.join(_REPO, "mnist", "data")
MAPLE_DIR = os.path.join(_HERE, "maple_data")

MODELS = ("alexnet_tiny", "alexnet_full", "lenet", "lola")

_IDX_NAMES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _banner(text: str) -> None:
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)
    sys.stdout.flush()


def _guard_not_symlink(path: str) -> None:
    """Keep downloads inside the documented repository-owned directory."""
    if os.path.islink(path):
        target = os.readlink(path)
        raise SystemExit(
            f"{path} is a symlink -> {target}\n"
            f"Remove it before continuing:  rm {path}"
        )


def fetch_cifar10() -> None:
    _banner("CIFAR-10 (AlexNet)")
    _guard_not_symlink(CIFAR_DIR)
    from torchvision import datasets

    os.makedirs(CIFAR_DIR, exist_ok=True)
    datasets.CIFAR10(CIFAR_DIR, train=True, download=True)
    datasets.CIFAR10(CIFAR_DIR, train=False, download=True)
    print(f"[cifar] ready under {CIFAR_DIR}")


def fetch_mnist() -> None:
    _banner("MNIST (LeNet, LoLA)")
    _guard_not_symlink(MNIST_DIR)
    os.makedirs(MNIST_DIR, exist_ok=True)
    if all(os.path.isfile(os.path.join(MNIST_DIR, n)) for n in _IDX_NAMES):
        print(f"[mnist] IDX files already present under {MNIST_DIR}")
        return

    from torchvision import datasets

    staging = os.path.join(MNIST_DIR, "_torchvision")
    datasets.MNIST(staging, train=True, download=True)
    datasets.MNIST(staging, train=False, download=True)
    raw = os.path.join(staging, "MNIST", "raw")
    copied = 0
    for name in _IDX_NAMES:
        src = os.path.join(raw, name)
        if not os.path.isfile(src):
            raise SystemExit(f"[mnist] torchvision did not produce {src}")
        shutil.copy2(src, os.path.join(MNIST_DIR, name))
        copied += 1
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[mnist] copied {copied} IDX files to {MNIST_DIR}")


def train_alexnet(full: bool, epochs: int, device: str, seed: int) -> float:
    label = "AlexNet (full, depth-15)" if full else "AlexNetTiny (depth-7)"
    _banner(f"{label} — training")
    import alexnet_train

    model, history = alexnet_train.train(
        epochs=epochs, batch_size=128, lr=1e-3, weight_decay=1e-4,
        seed=seed, device=device, full=full)
    alexnet_train.export_weights(model, out_dir=MAPLE_DIR)
    return history["test_acc"][-1]


def train_lenet(epochs: int, device: str, seed: int) -> float:
    _banner("LeNet (quad activations) — training")
    import lenet_train

    model, history = lenet_train.train(
        epochs=epochs, batch_size=128, lr=1e-3, wd=1e-4,
        seed=seed, device=device)
    lenet_train.export_weights(model, out_dir=MAPLE_DIR)
    return history["test_acc"][-1]


def train_lola(epochs: int, device: str, seed: int) -> float:
    _banner("LoLA-MNIST (quad activations) — training")
    import lola_train

    model, history = lola_train.train(
        epochs=epochs, batch_size=128, lr=1e-3, weight_decay=1e-4,
        seed=seed, device=device, mnist_dir=MNIST_DIR)
    lola_train.export_weights(model, out_dir=lola_train.TRAINED_DIR,
                              mnist_dir=MNIST_DIR)
    return history["test_acc"][-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=MODELS + ("all",),
                    default=["all"],
                    help="Which models to train (default: all).")
    ap.add_argument("--data-only", action="store_true",
                    help="Download datasets and stop.")
    ap.add_argument("--skip-data", action="store_true",
                    help="Assume datasets are already present.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override every model's epoch count (default: 30, "
                         "matching the published accuracy matrix).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.data_only and args.skip_data:
        ap.error("--data-only and --skip-data are mutually exclusive")
    if args.epochs is not None and args.epochs <= 0:
        ap.error("--epochs must be positive")

    wanted = set(MODELS) if "all" in args.models else set(args.models)

    if not args.skip_data:
        if wanted & {"alexnet_tiny", "alexnet_full"}:
            fetch_cifar10()
        if wanted & {"lenet", "lola"}:
            fetch_mnist()
    if args.data_only:
        print("\n[done] datasets ready; --data-only requested, not training.")
        return 0

    # The published accuracy matrix uses one consistent 30-epoch recipe.  Keep
    # the all-in-one command reproducible even though the standalone trainers
    # retain their shorter historical defaults for quick local iteration.
    defaults = dict.fromkeys(MODELS, 30)
    results: list[tuple[str, float, float]] = []
    for name in MODELS:
        if name not in wanted:
            continue
        epochs = args.epochs if args.epochs is not None else defaults[name]
        t0 = time.time()
        if name == "alexnet_tiny":
            acc = train_alexnet(False, epochs, args.device, args.seed)
        elif name == "alexnet_full":
            acc = train_alexnet(True, epochs, args.device, args.seed)
        elif name == "lenet":
            acc = train_lenet(epochs, args.device, args.seed)
        else:
            acc = train_lola(epochs, args.device, args.seed)
        results.append((name, acc, time.time() - t0))

    _banner("Summary")
    for name, acc, secs in results:
        print(f"  {name:<14} test accuracy {acc * 100:5.2f}%   ({secs:6.1f}s)")
    print(f"\n  AlexNet/LeNet weights -> {MAPLE_DIR}")
    if "lola" in wanted:
        import lola_train
        print(f"  LoLA weights          -> {lola_train.TRAINED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
