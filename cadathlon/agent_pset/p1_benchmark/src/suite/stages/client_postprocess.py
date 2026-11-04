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
    logits = np.real(np.load(paths.interm / "logits.npy"))
    if not np.isfinite(logits).all():
        raise SystemExit("decrypted logits contain non-finite values")
    labels = np.argmax(logits, axis=1)
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    paths.predictions.write_text("".join(f"{int(v)}\n" for v in labels))
    lib.report_steps(paths, {"client_postprocess_s": time.perf_counter() - started})


if __name__ == "__main__":
    main()
