"""Suite stage; installed into the ml-inference checkout by install_suite.py."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["P3_SUITE_DIR"])
import stage_lib as lib

import time

import numpy as np


def main():
    size = int(sys.argv[1])
    paths = lib.Paths(size)
    started = time.perf_counter()
    if not paths.test_pixels.exists():
        raise SystemExit(f"{paths.test_pixels} missing; harness stage 4 must run first")
    pixels = lib.read_pixels(paths.test_pixels)
    normalized = (pixels - lib.MNIST_MEAN) / lib.MNIST_STD
    paths.interm.mkdir(parents=True, exist_ok=True)
    np.save(paths.interm / "normalized.npy", normalized)
    lib.report_steps(paths, {"client_preprocess_s": time.perf_counter() - started})


if __name__ == "__main__":
    main()
