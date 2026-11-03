"""Serialize an opaque ciphertext bundle across the harness's process split.

The contestant's `encrypt` may return a single `Polynomial`, a list, a dict or
any nesting of those plus integer bookkeeping -- that freedom is the point of
the problem, because it is what lets a team change how images map onto
ciphertexts. The harness runs encryption, evaluation and decryption as three
separate processes, so something has to move those bundles through files
without constraining their shape. That is this module: it walks the structure,
writes each ciphertext payload with its own metadata, and records the shape so
the far side can rebuild it exactly.

Contestants do not implement any of this.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _is_ciphertext(value):
    try:
        from jaxite_word import Polynomial
    except Exception:
        return False
    return isinstance(value, Polynomial)


def _encode(value, out, store):
    if _is_ciphertext(value):
        index = len(store)
        store.append(value)
        return {"__ct__": index}
    if isinstance(value, (bool, int, np.integer)):
        return {"__int__": int(value)}
    if isinstance(value, float):
        return {"__float__": float(value)}
    if value is None:
        return {"__none__": True}
    if isinstance(value, (list, tuple)):
        return {"__seq__": [_encode(v, out, store) for v in value],
                "tuple": isinstance(value, tuple)}
    if isinstance(value, dict):
        return {"__map__": {str(k): _encode(v, out, store) for k, v in value.items()}}
    raise TypeError(
        f"bundle contains {type(value).__name__}; only ciphertexts, ints, "
        "floats, None and nestings of list/tuple/dict are transportable")


def _decode(node, cts):
    if "__ct__" in node:
        return cts[node["__ct__"]]
    if "__int__" in node:
        return node["__int__"]
    if "__float__" in node:
        return node["__float__"]
    if "__none__" in node:
        return None
    if "__seq__" in node:
        items = [_decode(v, cts) for v in node["__seq__"]]
        return tuple(items) if node.get("tuple") else items
    if "__map__" in node:
        return {k: _decode(v, cts) for k, v in node["__map__"].items()}
    raise ValueError(f"unrecognized bundle node: {sorted(node)}")


def save_bundle(bundle, directory) -> int:
    """Write `bundle` under `directory`; return the ciphertext count."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("ct_*.npz"):
        stale.unlink()
    store: list = []
    shape = _encode(bundle, directory, store)
    for index, ct in enumerate(store):
        payload = np.asarray(ct.to_array())
        scale = getattr(ct, "_ckks_scale", None)
        nsd = getattr(ct, "_ckks_nsd", None)
        np.savez(directory / f"ct_{index}.npz", payload=payload,
                 moduli=np.asarray(ct.get_moduli(), dtype=np.int64),
                 degree_layout=np.asarray(ct.degree_layout, dtype=np.int64),
                 precision=np.int64(ct.precision),
                 scale=np.float64(np.nan if scale is None else scale),
                 nsd=np.int64(-1 if nsd is None else nsd))
    (directory / "bundle.json").write_text(
        json.dumps({"shape": shape, "count": len(store)}, indent=2))
    return len(store)


def load_bundle(directory):
    """Rebuild the bundle written by `save_bundle`."""
    import jax.numpy as jnp
    from polynomial import Polynomial

    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text())
    cts = []
    for index in range(manifest["count"]):
        path = directory / f"ct_{index}.npz"
        with np.load(path) as data:
            payload = np.asarray(data["payload"])
            moduli = [int(v) for v in data["moduli"]]
            rows, columns = (int(v) for v in data["degree_layout"])
            precision = int(data["precision"])
            scale = float(data["scale"])
            nsd = int(data["nsd"])
        if payload.ndim != 5:
            raise ValueError(f"{path}: expected a rank-5 payload, got {payload.shape}")
        dtype = jnp.uint32 if precision <= 32 else jnp.uint64
        ct = Polynomial.from_array(
            jnp.asarray(payload, dtype=dtype),
            {"batch": payload.shape[0], "num_elements": payload.shape[1],
             "degree": rows * columns, "num_moduli": payload.shape[4],
             "precision": precision, "degree_layout": (rows, columns)},
            {"moduli": moduli})
        if not np.isnan(scale):
            setattr(ct, "_ckks_scale", scale)
        if nsd >= 0:
            setattr(ct, "_ckks_nsd", nsd)
        cts.append(ct)
    return _decode(manifest["shape"], cts)
