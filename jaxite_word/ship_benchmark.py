"""Validate and optionally trace full SHIP bootstrapping through CROSS.

Run from jaxite_word. No wall-clock result is labeled as device latency.
Profiles include only the warmed bootstrap, excluding setup and decryption.
"""

import argparse
import json
from pathlib import Path

import ckks_ctx
import jax
import key_gen
import numpy as np
from ship import SHIPConfig
from ship import math as ship_math


def parameters(degree, weight):
  used = set()

  def primes(bits, count):
    value = ((1 << bits) - 2) // (2 * degree) * (2 * degree) + 1
    result = []
    while len(result) < count:
      if value not in used and ship_math.is_prime(value):
        result.append(value)
        used.add(value)
      value -= 2 * degree
    return result

  depth = (weight - 1).bit_length()
  return primes(31, 1) + primes(30, depth + 2), primes(30, depth + 4)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--degree", type=int, default=16384)
  parser.add_argument("--weight", type=int, default=31)
  parser.add_argument("--theta", type=int, default=9)
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Plaintext fixture seed; keys use OS randomness",
  )
  parser.add_argument("--profile-dir", type=Path)
  parser.add_argument("--iterations", type=int, default=3)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  n = args.degree
  if n < 8 or n & (n - 1) or not 1 <= args.weight <= n:
    parser.error("degree must be a power of two >= 8 and 1 <= weight <= degree")
  if not 1 <= args.theta <= n // 2 or args.iterations < 1:
    parser.error("theta must be in [1,N/2] and iterations must be positive")
  q, p = parameters(n, args.weight)
  r = 1 << ((n.bit_length() - 1) // 2)
  print("Preparing CROSS context and SHIP evaluation keys", flush=True)
  keys = key_gen.gen_pke_pair(q, p, n)
  ctx = ckks_ctx.CKKSContext(
    dict(
      degree=n,
      num_slots=n // 2,
      q_towers=q,
      p_towers=p,
      scaling_factor=q[-1],
      output_scale=q[-1],
      p=30,
      CKKS_M_FACTOR=1,
      max_bits_in_word=31,
      composite_degree=1,
      noise_scale_degree=1,
      **keys,
    )
  )
  ctx.program_initialization(total_rotation_indices=[], dnum=2, r=r, c=n // r, batch=1)
  config = SHIPConfig(weight=args.weight, theta=args.theta)
  plan = config.plan(ctx)
  bs = ctx.he_bootstrap.configure(scheme="ship", ship_config=config).setup()
  fixtures = {
    "ramp": np.linspace(-0.125, 0.125, n // 2),
    "random": np.random.default_rng(args.seed).uniform(-0.125, 0.125, n // 2),
  }
  errors = {}
  for name, values in fixtures.items():
    print(f"Checking encrypted {name} fixture", flush=True)
    ct = ctx.encrypt(ctx.encode_at_level(list(values), 1, scale=plan.input_scale))
    out = bs.bootstrap(ct)
    jax.block_until_ready(out.polynomial)
    # CROSS's decoder validates imaginary leakage before returning reals.
    decoded = ctx.decrypt_slots(out)
    error = float(np.max(np.abs(decoded - values)))
    if not np.isfinite(error) or error > 2**-10:
      raise RuntimeError(f"{name} accuracy gate failed: {error}")
    errors[name] = error
    print(f"{name}: max absolute error {error:.9g}", flush=True)
  report = {
    "degree": n,
    "slots": n // 2,
    "weight": args.weight,
    "theta": args.theta,
    "q": q,
    "p": p,
    "gamma": config.gamma,
    "input_scale": plan.input_scale,
    "output_scale": plan.output_scale,
    "output_level": plan.expected_output_level,
    "max_error": errors,
    "device_kind": jax.devices()[0].device_kind,
    "jax_version": jax.__version__,
    "ciphertext_dtype": str(out.polynomial.dtype),
    "security_qualified": False,
    "profiles": [],
  }
  if args.profile_dir:
    if jax.default_backend() != "tpu":
      raise RuntimeError("Device profiling requires a TPU backend")
    for index in range(args.iterations):
      directory = args.profile_dir / str(index)
      options = jax.profiler.ProfileOptions()
      options.host_tracer_level = 0
      options.python_tracer_level = 0
      options.device_tracer_level = 1
      options.advanced_configuration = {"tpu_trace_mode": "TRACE_ONLY_XLA"}
      jax.profiler.start_trace(str(directory), profiler_options=options)
      try:
        out = bs.bootstrap(ct)
        jax.block_until_ready(out.polynomial)
      finally:
        jax.profiler.stop_trace()
      report["profiles"].append(str(directory))
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
