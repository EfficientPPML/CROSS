"""Shared local smoke-test utilities; not a trusted contest judge."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
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
        if len(value.shape) != 5 or value.num_elements < 2:
            raise TypeError("Ciphertext must have rank 5 and at least two components")
        return
    if isinstance(value, (tuple, list)) and value:
        for child in value:
            require_ciphertexts(child)
        return
    raise TypeError("Expected a Polynomial or a nonempty list/tuple of ciphertexts")


def require_secure_execution(owner, bundle):
    """Check live evaluator Q*P rings and bind every ciphertext to one of them.

    Services using custom wrappers must expose ``security_contexts()`` returning
    their live CKKSContext objects. Standard ``ctx``, ``_ctx``, ``mapping`` and
    ``_mapping`` wrappers are recognized directly. A declared Packing, profile,
    or ring manifest is never accepted in place of an actual context. Run this
    after both encryption and evaluation so lazily constructed server contexts
    are checked too. This is a local regression gate, not a sandbox against
    malicious Python; the organizer still audits the evaluator independently.
    """
    from jaxite_word import CKKSContext, Polynomial
    from jaxite_word import he_params

    contexts, visited = [], set()

    def collect(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, CKKSContext):
            contexts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                collect(child)
        elif value is not None:
            export = getattr(value, "security_contexts", None)
            if callable(export):
                collect(export())
            # Inspect stored state only: do not call lazy setup methods.
            state = getattr(value, "__dict__", {})
            for name in ("ctx", "_ctx", "context", "_context", "mapping",
                         "_mapping", "mappings", "_mappings", "codec", "_codec"):
                if name in state:
                    collect(state[name])

    collect(owner)
    if not contexts:
        raise ValueError("Cannot verify the actual ring: expose live CKKSContext "
                         "objects through security_contexts()")
    target = he_params.SecurityTarget(128, he_params.CostModel.CLASSICAL)
    rings, manifests = [], []
    for ctx in contexts:
        # Read execution fields, not a separately declared ring_config.
        q, p, degree = tuple(ctx.q_towers), tuple(ctx.p_towers), ctx.degree
        if not he_params.qp_is_secure(q, p, degree, target):
            raise ValueError(
                f"Actual ring fails 128-bit security: N={degree}, "
                f"Q*P bits={math.prod(int(v) for v in (*q, *p)).bit_length()}, "
                f"ceiling={he_params.security_ceiling(degree, target)}")
        q, p = tuple(int(v) for v in q), tuple(int(v) for v in p)
        noise = float(ctx.parameters.get("sigma", he_params.DEFAULT_SIGMA))
        if not math.isfinite(noise) or noise < he_params.DEFAULT_SIGMA:
            raise ValueError("Actual context noise sigma is below the security table assumption")
        distribution = ctx.parameters.get("secret_distribution", he_params.SECRET_DISTRIBUTION)
        if distribution != he_params.SECRET_DISTRIBUTION:
            raise ValueError("Security table requires uniform_ternary secrets")
        rings.append((int(degree), tuple(int(v) for v in q)))
        manifests.append({"degree": int(degree), "num_slots": int(ctx.num_slots),
                          "num_q": len(q), "num_p": len(p),
                          "dnum": ctx.dnum,
                          "max_level": (len(q) - 1) // ctx.composite_degree,
                          "q_towers": list(q), "p_towers": list(p),
                          "log2_Q": math.log2(math.prod(q)),
                          "log2_QP": math.log2(math.prod((*q, *p))),
                          "security_bits": 128, "security_model": "classical"})

    count = 0

    def check(value):
        nonlocal count
        if isinstance(value, Polynomial):
            require_ciphertexts(value)
            moduli = tuple(int(v) for v in value.moduli)
            if not any(value.degree == degree and moduli == q[:len(moduli)]
                       and 0 < len(moduli) <= len(q) for degree, q in rings):
                raise ValueError("Ciphertext ring does not match a verified execution context")
            count += 1
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                check(child)
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return  # P1 permits lane-count bookkeeping in a bundle.
        else:
            raise TypeError("Expected ciphertexts and integer bookkeeping in bundle")

    check(bundle)
    if not count:
        raise ValueError("Ciphertext bundle must contain at least one ciphertext")
    return manifests


def synchronize(value):
    import jax
    jax.block_until_ready(value)
