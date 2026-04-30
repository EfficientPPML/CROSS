"""CI gate for the vectorized encrypt/decrypt fast paths.

Runs `decrypt_fast_test` and `encrypt_fast_test` in one process and prints a
single PASS/FAIL summary. Returns exit code 0 if all green, nonzero otherwise.

Run before any change to `decrypt_fast.py`, `encrypt_fast.py`, or the
`LoLAHE.encrypt` / `LoLAHE.decrypt` wrappers.

Usage:
    python3 jaxite_word/run_fast_path_tests.py
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

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module_name in ("decrypt_fast_test", "encrypt_fast_test"):
        suite.addTests(loader.loadTestsFromName(module_name))

    print(f"[fast-path CI] running {suite.countTestCases()} tests "
          f"from decrypt_fast_test + encrypt_fast_test ...")
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
