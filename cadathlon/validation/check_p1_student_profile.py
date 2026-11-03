"""Check the opt-in smaller P1 ring against the archived trained-model images.

This organizer-only check is not a harness/scoring run. Run from any working
directory; choose JAX_PLATFORMS and XLA_FLAGS before invoking Python.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cadathlon/non_agent_pset/p1_benchmark/src"
sys.path[:0] = [str(ROOT), str(SRC), str(SRC / "suite"),
               str(ROOT / "cadathlon/common")]


def run(devices):
    import jax
    import torch
    from baseline_student import build_service
    import stage_lib as lib
    from support import metrics, require_secure_execution, synchronize

    jax.config.update("jax_enable_x64", True)
    torch.set_num_threads(1)
    available = jax.devices()
    if devices > len(available):
        raise ValueError(f"need {devices} devices, found {len(available)}")
    started = time.perf_counter()
    model = lib.load_model()
    evidence = SRC / "suite/results/C3_baseline/evidence"
    pixels_path = evidence / "test_pixels.txt"
    labels_path = evidence / "test_labels.txt"
    images = (lib.read_pixels(pixels_path) - lib.MNIST_MEAN) / lib.MNIST_STD
    labels = np.loadtxt(labels_path, dtype=np.int64).reshape(-1)
    with torch.no_grad():
        expected = model(torch.from_numpy(images)).numpy()
    service = build_service(model, (784,), batch=len(images),
                            devices=tuple(available[:devices]))
    try:
        warm = service.warmup_input()
        require_secure_execution(service, warm)
        synchronize(service.execute(warm))
        setup = time.perf_counter() - started
        request = time.perf_counter()
        encrypted = service.encrypt(images)
        require_secure_execution(service, encrypted)
        synchronize(encrypted)
        server_started = time.perf_counter()
        result = service.execute(encrypted)
        synchronize(result)
        server_s = time.perf_counter() - server_started
        rings = require_secure_execution(service, result)
        actual = service.decrypt(result)
        end_to_end_s = time.perf_counter() - request
        errors = metrics(actual, expected)
        if not np.all(np.abs(actual - expected) <= 0.02 + 0.01 * np.abs(expected)):
            raise ValueError(f"trained-model logit gate failed: {errors}")
        if errors["top1_agreement"] < 0.99:
            raise ValueError(f"trained-model argmax gate failed: {errors}")
        report = {
            "passed": True, "mode": "trained-model-arithmetic-check",
            "profile": "student", "images": len(images),
            "devices": [str(device) for device in available[:devices]],
            "setup_including_warmup_s": setup,
            "server_s": server_s, "end_to_end_s": end_to_end_s,
            "cleartext_correct": int(np.sum(expected.argmax(-1) == labels)),
            "encrypted_correct": int(np.sum(actual.argmax(-1) == labels)),
            "execution_rings": rings,
            "fixture_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (pixels_path, labels_path)},
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("jax", "jaxlib", "numpy", "torch")},
            **errors,
        }
    finally:
        service.release()
    report["total_s"] = time.perf_counter() - started
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()
    if args.devices < 1:
        parser.error("devices must be positive")
    with contextlib.redirect_stdout(sys.stderr):
        report = run(args.devices)
    print(json.dumps(report, allow_nan=False))
