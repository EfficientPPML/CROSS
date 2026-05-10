"""CI gate for the vectorized encrypt/decrypt fast paths.

Runs the `FastEncryptCorrectness` and `FastDecryptCorrectness` classes from
`ckks_ctx_test` (where the bit-exact tests now live, after both the old
`encrypt_fast.py` and `decrypt_fast.py` were inlined into `ckks_ctx.py`) in
one process and prints a single PASS/FAIL summary. Returns exit code 0 if
all green, nonzero otherwise.

Run before any change to the embedded fast-encrypt or fast-decrypt section
of `ckks_ctx.py`, or the `LoLAHE.encrypt` / `LoLAHE.decrypt` wrappers.

Usage:
    python3 jaxite_word/fast_encryption_decryption_test.py
"""
from __future__ import annotations
import os
import sys
import time
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "demos"))
for p in (THIS_DIR, DEMO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    import jax
    jax.config.update("jax_enable_x64", True)

    import ckks_ctx_test
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(
        ckks_ctx_test.FastEncryptCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(
        ckks_ctx_test.FastDecryptCorrectness))

    print(f"[fast-path CI] running {suite.countTestCases()} tests "
          f"from ckks_ctx_test::FastEncryptCorrectness + "
          f"FastDecryptCorrectness ...")
    t0 = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 68)
    if result.wasSuccessful():
        print(f"PASS  ({result.testsRun} tests in {elapsed:.0f}s)")
    else:
        print(f"FAIL  ({len(result.failures)} failed, "
              f"{len(result.errors)} errored, of {result.testsRun}, "
              f"in {elapsed:.0f}s)")
    print("=" * 68)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
