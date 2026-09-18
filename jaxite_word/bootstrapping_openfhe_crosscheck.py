#!/usr/bin/env python3
"""Cross-validate CROSS bootstrapping against OpenFHE.

For each ring dimension, runs independently randomized OpenFHE reference
bootstraps, takes their median precision, then runs CROSS on the same plaintext
vector.  CROSS must recover the plaintext and its minimum precision must be
within the configured tolerance of the OpenFHE median.

OpenFHE reference binaries (built from
openfhe-development/src/pke/examples/bootstrap-stage-compare*.cpp):
  N=256  -> build/bin/examples/pke/bootstrap-stage-compare-param 8 1 1 5 56 61 31
  N=512  -> build/bin/examples/pke/bootstrap-stage-compare-param 9 2 2 10 56 61 31
  N=4096 -> build/bin/examples/pke/bootstrap-stage-compare-param 12 4 4 6 56 61 31

All three share CROSS's config (firstMod=61, scalingMod=56, compositeDegree=2,
registerWordSize=31, UNIFORM_TERNARY, levelBudget per degree).

Run:
    python bootstrapping_openfhe_crosscheck.py            # N=256 + N=512
    BOOTSTRAP_TEST_N4096=1 python bootstrapping_openfhe_crosscheck.py  # + N=4096
    # Run each large degree in a fresh process; N>=16384 needs low-memory mode.
    BOOTSTRAP_LOW_MEM=1 BOOTSTRAP_TEST_LARGEN=1 \
        CROSS_LAZY_TOPLEVEL_ROTKEYS=1 \
        BOOTSTRAP_MIN_DEGREE=16384 BOOTSTRAP_MAX_DEGREE=16384 \
        python bootstrapping_openfhe_crosscheck.py
"""
from __future__ import annotations

import math
import os
import re
import statistics
import subprocess

import jax
jax.config.update("jax_enable_x64", True)

import bootstrapping
import ckks_ctx
import key_gen as kg
import util
from bootstrapping import decrypt_decode
from composite_prime_gen import composite_prime_gen

# Standalone cross-check script: bootstrap intermediates exceed the decode
# limit by design, and the process exits when it finishes.
ckks_ctx.bypass_decode_stddev_check().__enter__()

_OPENFHE_DIR = os.environ.get("OPENFHE_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "openfhe-development"))
_BIN_DIR = os.path.join(_OPENFHE_DIR, "build", "bin", "examples", "pke")

# (label, openfhe_binary_or_(binary, args), degree, nq, level_budget, mode,
#  precision[, num_slots])
# mode: "single" | "meta" | "dc-single" | "dc-meta" (dc-* = DC-template
# subtraction active).  num_slots defaults to degree//2 (fully packed);
# smaller = sparse packing (must match the OpenFHE binary's numSlots arg).
CONFIGS = [
    (
        "N=256",
        ("bootstrap-stage-compare-param", ["8", "1", "1", "5", "56", "61", "31"]),
        256, 44, [1, 1], "single", 8,
    ),
    (
        "N=512",
        ("bootstrap-stage-compare-param", ["9", "2", "2", "10", "56", "61", "31"]),
        512, 58, [2, 2], "single", 9,
    ),
    (
        "N=4096",
        ("bootstrap-stage-compare-param", ["12", "4", "4", "6", "56", "61", "31"]),
        4096, 58, [4, 4], "single", 9,
    ),
]

# Full-packed large-ring sweep (enable with BOOTSTRAP_TEST_LARGEN=1).  Every
# entry keeps q58, p19, scalingMod=56, [4,4], and six output levels.
LARGEN_CONFIGS = [
    ("N=8192",
     ("bootstrap-stage-compare-param", ["13", "4", "4", "6", "56", "61", "31"]),
     8192, 58, [4, 4], "single", 9),
    ("N=16384",
     ("bootstrap-stage-compare-param", ["14", "4", "4", "6", "56", "61", "31"]),
     16384, 58, [4, 4], "single", 9),
    ("N=32768",
     ("bootstrap-stage-compare-param", ["15", "4", "4", "6", "56", "61", "31"]),
     32768, 58, [4, 4], "single", 9),
    ("N=65536",
     ("bootstrap-stage-compare-param", ["16", "4", "4", "6", "56", "61", "31"]),
     65536, 58, [4, 4], "single", 9),
]

_SLOT_RE = re.compile(
    r"slot\[(\d+)\]\s*=\s*([-\d.eE+]+)\s+expected=([-\d.eE+]+).*?bits=([-\d.eE+]+)")
_MIN_RE = re.compile(r"Min precision:\s*([-\d.eE+]+)")


def run_openfhe(binary):
  """Run an OpenFHE reference binary; return (expected[], per_slot_bits[], min_bits).

  `binary` is either a name or a (name, [args...]) tuple.
  """
  args = []
  if isinstance(binary, tuple):
    binary, args = binary
  path = os.path.join(_BIN_DIR, binary)
  if not os.path.exists(path):
    raise FileNotFoundError(f"OpenFHE binary not found: {path}")
  proc = subprocess.run([path] + list(args), capture_output=True, text=True,
                        timeout=3600)
  if proc.returncode != 0:
    detail = (proc.stderr.strip() or proc.stdout.strip())[-2000:]
    raise RuntimeError(
        f"OpenFHE reference exited with status {proc.returncode}: {detail}"
    )

  actual, expected, bits = {}, {}, {}
  for line in proc.stdout.splitlines():
    m = _SLOT_RE.search(line)
    if m:
      i = int(m.group(1))
      actual[i] = float(m.group(2))
      expected[i] = float(m.group(3))
      bits[i] = float(m.group(4))
  if not expected:
    raise ValueError("OpenFHE reference produced no parseable slot records.")
  indices = list(range(max(expected) + 1))
  if (sorted(expected) != indices
      or sorted(actual) != indices
      or sorted(bits) != indices):
    raise ValueError(
        "OpenFHE reference produced incomplete or non-contiguous slot records."
    )
  for label, values in (("actual", actual), ("expected", expected),
                        ("precision", bits)):
    if not all(math.isfinite(values[i]) for i in indices):
      raise ValueError(f"OpenFHE reference produced non-finite {label} values.")
  mins = _MIN_RE.findall(proc.stdout)
  min_bits = float(mins[-1]) if mins else min(bits.values())
  if not math.isfinite(min_bits):
    raise ValueError("OpenFHE reference produced a non-finite minimum precision.")
  return ([expected[i] for i in indices],
          [bits[i] for i in indices], min_bits)


def run_openfhe_samples(binary, samples):
  """Return per-slot and minimum medians from independent reference runs."""
  if samples < 1:
    raise ValueError("OpenFHE reference sample count must be positive")
  runs = [run_openfhe(binary) for _ in range(samples)]
  expected = runs[0][0]
  if any(run[0] != expected for run in runs[1:]):
    raise ValueError("OpenFHE reference runs used different plaintext vectors")
  per_slot = [
      statistics.median(run[1][i] for run in runs)
      for i in range(len(expected))
  ]
  minimums = [run[2] for run in runs]
  return expected, per_slot, statistics.median(minimums), minimums


def _discover_rotation_indices(degree, num_slots, level_budget):
  """Compute initially materialized keys without allocating a context.

  Full-packed collapsed transforms use only their exact hoisted baby/giant
  rotations.  Legacy diagonal maps remain registered later by control_gen(),
  so changing the diagnostic opt-out can still generate fallback keys lazily;
  avoiding their eager materialization saves about 12 GiB at N=65536.
  """
  m = 2 * degree
  sparse = num_slots < degree // 2
  key_mod = 2 * num_slots if sparse else None
  rotations = set()
  for inverse, levels in ((False, level_budget[0]),
                          (True, level_budget[1])):
    diag_levels = bootstrapping._decompose_dft_into_levels(
        num_slots, 4 * num_slots, levels, inverse=inverse,
        key_mod=key_mod,
    )
    use_exact_hoisted = (
        not sparse
        and num_slots > 1
        and levels > 1
        and os.environ.get("BOOTSTRAP_DISABLE_HOISTED_LT") != "1"
    )
    if use_exact_hoisted:
      groups = bootstrapping._decompose_dft_into_hoisted_groups(
          num_slots, 4 * num_slots, levels, inverse=inverse
      )
      for level_groups in groups or []:
        for group in level_groups:
          giant = int(group['giant'])
          if giant:
            rotations.add(giant)
          rotations.update(
              int(baby) for baby, _ in group['entries'] if int(baby) != 0
          )
    else:
      for diags in diag_levels:
        rotations.update(int(k) for k in diags if int(k) != 0)

  if sparse:
    gap = (degree // 2) // num_slots
    j = 1
    while j < gap:
      rotations.add(j * num_slots)
      j <<= 1
    rotations.add(num_slots)
  rotations.add(m - 1)
  return sorted(rotations)


def run_cross(degree, nq, cd, level_budget, x, mode, precision, dnum=3,
              num_slots=None, scaling_mod_size=56):
  """Run a CROSS bootstrap on input x; return (per_slot_values, per_slot_bits, min_bits)."""
  if num_slots is None:
    num_slots = degree // 2
  cycl_order = 2 * degree
  r = 16 if degree <= 256 else 64
  c = degree // r
  q_towers = composite_prime_gen(
      cd, nq, 61, scaling_mod_size, cycl_order, 31)
  p_towers = util.generate_p_towers(q_towers, dnum=dnum, degree=degree)
  key_pair = kg.gen_pke_pair(q_towers, [], degree)
  sf0 = float(q_towers[-2]) * float(q_towers[-1])
  params = dict(
      degree=degree, num_slots=num_slots, scaling_factor=sf0, output_scale=sf0,
      q_towers=q_towers, p_towers=p_towers, p=round(math.log2(sf0)),
      CKKS_M_FACTOR=1, max_bits_in_word=61, noise_scale_degree=1,
      composite_degree=cd, public_key=key_pair["public_key"],
      secret_key=key_pair["secret_key"])

  ri = _discover_rotation_indices(
      degree, num_slots, level_budget
  )

  # At very large N, defer full-chain rotation-key generation until an exact
  # hoisted level first touches each key.  Keys are still generated at max
  # level and retained, so decomposition partitions remain unchanged and each
  # generated key is reused thereafter.  This changes RNG draw order, but not
  # the key distribution, and avoids materializing later S2C keys during C2S.
  lazy_toplevel_keys = os.environ.get("CROSS_LAZY_TOPLEVEL_ROTKEYS") == "1"
  if lazy_toplevel_keys and os.environ.get("CROSS_SKIP_TOPLEVEL_ROTKEYS") == "1":
    raise ValueError(
        "CROSS_LAZY_TOPLEVEL_ROTKEYS and CROSS_SKIP_TOPLEVEL_ROTKEYS "
        "cannot both be enabled"
    )

  ctx = ckks_ctx.CKKSContext(params)
  ctx.program_initialization(
      total_rotation_indices=[] if lazy_toplevel_keys else ri,
      dnum=dnum, r=r, c=c, batch=1)
  bs = bootstrapping.Bootstrap(ctx)
  bs.control_gen(level_budget=level_budget)
  bs.setup_key()

  slots = [complex(v, 0) for v in x] + [complex(0, 0)] * (num_slots - len(x))
  ct = ctx.encrypt(ctx.encode(slots))
  ct.validate()
  bs._set_scale(ct, sf0); bs._set_nsd(ct, 1)
  ct_dep, _ = bs._level_reduce(ct, bs._infer_level(ct), 1)

  if mode.startswith("dc-"):
    bs._dc_subtract = True
    bs.precompute_dc_template()
  if mode.endswith("meta"):
    out = bs.meta_bootstrap(ct_dep, precision=precision)
  else:
    out = bs.bootstrap(ct_dep)
  decoded = decrypt_decode(ctx, bs, out, min(num_slots, len(x)))

  vals, bits = [], []
  mn = 99.0
  for i, expected in enumerate(x):
    v = decoded[i].real
    err = abs(v - expected)
    b = -math.log2(err / abs(expected)) if err > 0 and expected != 0 else 99.0
    vals.append(v); bits.append(b); mn = min(mn, b)
  return vals, bits, mn


def main():
  run_n4096 = os.environ.get("BOOTSTRAP_TEST_N4096") == "1"
  run_largen = os.environ.get("BOOTSTRAP_TEST_LARGEN") == "1"
  min_degree = int(os.environ.get("BOOTSTRAP_MIN_DEGREE", "0"))
  max_degree = int(os.environ.get("BOOTSTRAP_MAX_DEGREE", "0"))
  reference_samples = int(os.environ.get("BOOTSTRAP_OPENFHE_SAMPLES", "3"))
  overall_ok = True
  selected = 0
  executed = 0
  configs = list(CONFIGS)
  if run_largen:
    configs += list(LARGEN_CONFIGS)
  configs = [c + (None,) if len(c) == 7 else c for c in configs]
  for label, binary, degree, nq, lb, mode, precision, num_slots in configs:
    if degree < min_degree:
      print(f"\n=== {label}: SKIPPED (below BOOTSTRAP_MIN_DEGREE) ===")
      continue
    if max_degree and degree > max_degree:
      print(f"\n=== {label}: SKIPPED (above BOOTSTRAP_MAX_DEGREE) ===")
      continue
    if degree >= 4096 and degree < 8192 and not (run_n4096 or run_largen):
      print(f"\n=== {label}: SKIPPED (set BOOTSTRAP_TEST_N4096=1) ===")
      continue
    selected += 1
    print(f"\n{'='*64}\n {label}  (CROSS {mode} vs OpenFHE)\n{'='*64}")

    try:
      exp, ofhe_bits, ofhe_min, ofhe_mins = run_openfhe_samples(
          binary, reference_samples
      )
    except Exception as e:
      overall_ok = False
      print(f"  OpenFHE reference failed: {e}")
      continue
    executed += 1
    print(
        f"  OpenFHE: median min precision = {ofhe_min:.1f} bits "
        f"over {reference_samples} run(s), range "
        f"[{min(ofhe_mins):.1f}, {max(ofhe_mins):.1f}]"
    )

    x = exp  # bootstrap the SAME input OpenFHE used
    vals, cross_bits, cross_min = run_cross(degree, nq, 2, lb, x, mode,
                                            precision, num_slots=num_slots,
                                            scaling_mod_size=56)

    print(f"\n  {'slot':>4} {'expected':>10} {'CROSS out':>13} "
          f"{'CROSS bits':>10} {'OpenFHE bits':>12}")
    for i in range(len(x)):
      ob = ofhe_bits[i] if i < len(ofhe_bits) else float("nan")
      print(f"  {i:>4} {x[i]:>10.4f} {vals[i]:>13.8f} "
            f"{cross_bits[i]:>10.1f} {ob:>12.1f}")
    print(f"\n  CROSS   min = {cross_min:.1f} bits")
    print(f"  OpenFHE median min = {ofhe_min:.1f} bits")

    # The two implementations sample independent encryption/evaluation-key
    # noise.  Compare against a multi-run OpenFHE median rather than an
    # absolute 15-bit gate (OpenFHE itself is below 15 bits at larger matched
    # dimensions); the tolerance accommodates the remaining CROSS sample
    # variance while still detecting a sustained precision gap.
    parity_tolerance = float(os.environ.get("BOOTSTRAP_PARITY_TOLERANCE", "2.0"))
    parity_ok = cross_min >= ofhe_min - parity_tolerance
    # Functional correctness: every slot recovered the expected value.
    values_ok = all(abs(vals[i] - x[i]) < 0.05 * max(abs(x[i]), 1.0)
                    for i in range(len(x)))
    status = "PASS" if (parity_ok and values_ok) else "FAIL"
    overall_ok = overall_ok and parity_ok and values_ok
    print(f"  -> {label}: values_recovered={values_ok}  "
          f"parity(OpenFHE-{parity_tolerance:g}b)={parity_ok}  [{status}]")

  if selected == 0 or executed != selected:
    overall_ok = False
  print(f"\n{'='*64}")
  print(f"References executed: {executed}/{selected}")
  print(f"CROSSCHECK: {'PASS' if overall_ok else 'FAIL'}")
  print(f"{'='*64}")
  return 0 if overall_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
