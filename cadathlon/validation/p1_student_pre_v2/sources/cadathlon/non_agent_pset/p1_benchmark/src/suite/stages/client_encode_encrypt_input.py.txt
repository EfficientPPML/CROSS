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
        encrypt_started = time.perf_counter()
        bundle = service.encrypt(normalized)
        count = bundle_io.save_bundle(bundle, paths.upload)
        encrypt_s = time.perf_counter() - encrypt_started
    finally:
        service.release()
    print(f"         [p3] encrypted {batch} image(s) into {count} ciphertext(s)")
    lib.report_steps(paths, {"client_encrypt_s": encrypt_s,
                             "client_encrypt_setup_s": setup_s,
                             "client_input_ciphertexts": count})


if __name__ == "__main__":
    main()
