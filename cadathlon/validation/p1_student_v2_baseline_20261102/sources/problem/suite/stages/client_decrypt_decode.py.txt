"""Suite stage; installed into the ml-inference checkout by install_suite.py."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["P3_SUITE_DIR"])
import stage_lib as lib

import time

import numpy as np

sys.path.insert(0, os.environ["P3_SUITE_DIR"])
import bundle_io


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    size = int(sys.argv[1])
    paths = lib.Paths(size)
    normalized = np.load(paths.interm / "normalized.npy")
    batch = len(normalized)
    started = time.perf_counter()
    import jax
    keys = lib.load_keys(paths)
    service = lib.build_service(batch, tuple(jax.devices()[:1]), keys)
    setup_s = time.perf_counter() - started
    try:
        decrypt_started = time.perf_counter()
        logits = np.asarray(service.decrypt(bundle_io.load_bundle(paths.download)),
                            dtype=np.float64)
        decrypt_s = time.perf_counter() - decrypt_started
    finally:
        service.release()
    if logits.shape != (batch, 10):
        raise SystemExit(f"decrypt returned {logits.shape}, expected {(batch, 10)}")
    np.save(paths.interm / "logits.npy", logits)
    lib.report_steps(paths, {"client_decrypt_s": decrypt_s,
                             "client_decrypt_setup_s": setup_s})


if __name__ == "__main__":
    main()
