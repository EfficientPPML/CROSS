"""Shared local smoke-test utilities; not a trusted contest judge."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


def cross_root():
    root = Path(os.environ.get("CROSS_ROOT", Path(__file__).resolve().parents[2]))
    if not (root / "jaxite_word" / "API_REFERENCE.md").is_file():
        raise FileNotFoundError("Set CROSS_ROOT to the CROSS checkout")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "demos"))
    return root


def load_module(path):
    loader = importlib.machinery.SourceFileLoader("contest_implementation", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text())


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Invalid output: {actual.shape}; expected {expected.shape}, finite")
    error = actual - expected
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "nrmse": float(np.sqrt(np.mean(error * error)) /
                       max(np.sqrt(np.mean(expected * expected)), 0.01)),
        "top1_agreement": float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
    }


def require_ciphertexts(value):
    from jaxite_word import Polynomial
    if isinstance(value, Polynomial):
        value.validate()
        if len(value.shape) != 5:
            raise TypeError("Ciphertext must have rank 5")
        return
    if isinstance(value, (tuple, list)) and value:
        for child in value:
            require_ciphertexts(child)
        return
    raise TypeError("Expected a Polynomial or a nonempty list/tuple of ciphertexts")


def synchronize(value):
    import jax
    jax.block_until_ready(value)
