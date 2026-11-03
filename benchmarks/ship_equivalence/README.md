# Pristine SHIP compatibility checks

The requested comparison holds ring degree, slot count, Q/P prime lists,
input/output active Q subsets, input/output scales, and retained application
capacity constant across unoptimized Jaxite SHIP, optimized Jaxite SHIP,
optimized CROSS SHIP, and native CROSS bootstrap. Historical timings do not
satisfy this contract and must not fill missing benchmark cells.

## Baseline

Unoptimized means pristine `google/jaxite` remote `main`, resolved 2026-11-03:
`1c35ddd29179e946ab7d08515053cefd7e76c46e`.
The probe checks that exact HEAD and rejects tracked modifications. It does
not monkeypatch or modify upstream code. The locally modified PR147 checkout
is not this baseline. The historical 1724.291 ms result is not a measurement
of this baseline.

Upstream exposes a coefficient-to-slot half-bootstrap. A full slot-to-slot
benchmark would additionally need an encrypted input transform and appropriate
key switching, all included in timing. Upstream setup takes one key, assumes
positive sparse coefficients when creating selector keys (`s_j=1`), and
retains the sparse support. It has no independent output-key setup argument.
These differences must be disclosed and cannot be silently repaired in the
pristine variant.

## Executed checks — 2026-11-03

CPU: Python 3.11.16, JAX/JAXlib 0.9.1, NumPy 2.4.6. These are compatibility
checks at small ring sizes, **not performance measurements or security
qualifications**. No new TPU latency is claimed.

| Check | Result |
| --- | --- |
| Upstream `test_half_bootstrap_hw1` and `test_half_bootstrap_hw1_second_half` | 2 passed, 6 deselected |
| Upstream hw1 fixture, N=16, five Q primes, one P prime, zero noise | Passed twice; maximum errors against sine reference 9.23e-10 and 6.56e-9 |
| Same hw1 fixture, two P primes | Failed at final addition: shapes `(2,16,4)` and `(2,16,5)` |
| Noisy sparse-binary probe, N=16, five Q primes, one P prime, gamma=8, ramp | Failed 2^-10 gate; error approximately 3.28e27 |
| Same noisy small profile with two P primes | Failed with the same final-addition shape mismatch |
| Longer candidate: N=32, 18 Q primes, 19 P primes, weight=5, theta=4 | Setup failed: `OverflowError: Python int too large to convert to C long` |

The passing control uses the original upstream test primes and its sine
reference with a 1e-3 error gate; it is not proof that noisy full bootstrapping
meets the application error gate. The noisy checks compare to original
plaintext at 2^-10. Different checks must not be conflated.

The two-P failure is explained by upstream `bootstrapping.py`: the conjugation
rescale kernel is configured with `num_rescales=1`, although it must remove the
P basis. With two P primes, one auxiliary limb remains, causing the final
addition mismatch. The longer-chain overflow occurs in upstream
`key_gen.gen_hmuxrot_key`, at `(p_val * beta) % all_moduli_u64`.

CROSS SHIP's current planner requires at least two P primes. Therefore the
successful one-P upstream control is not a common configuration for the
current implementations. Moving to two P primes exposes an upstream failure.
Keeping the pristine baseline means reporting **unsupported/failed**, not
substituting patched code or timing an incorrect result. This is evidence for
the tested configurations and source-level constraints, not an exhaustive
search of all possible cryptographic parameters.

The current native CROSS uniform-ternary schedule separately needs at least
18 Q primes at composite degree 1, transform budgets [1,1], and one retained
level. That schedule calculation does not establish accuracy. The historical
eight-Q SHIP profile cannot simply be assigned to native CROSS bootstrap.

## Reproduce

Use an isolated checkout and environment. Install `jax[cpu]==0.9.1`,
`numpy==2.4.6`, `gmpy2`, `pytest`, `hypothesis`, and `absl-py`.

```sh
git clone https://github.com/google/jaxite.git /tmp/cross-equivalence-jaxite
git -C /tmp/cross-equivalence-jaxite checkout --detach \
  1c35ddd29179e946ab7d08515053cefd7e76c46e
export PYTHONPATH=/tmp/cross-equivalence-jaxite
export JAX_PLATFORMS=cpu

python -m pytest /tmp/cross-equivalence-jaxite/jaxite/jaxite_ckks/bootstrapping_test.py \
  -q -k 'test_half_bootstrap_hw1 and not nonzero'

# Positive control; this should exit 0.
python benchmarks/ship_equivalence/probe_upstream.py \
  --upstream-fixture --output /tmp/ship-pristine-control-p1.json

# Compatibility failure; this should exit 1 and save its traceback.
python benchmarks/ship_equivalence/probe_upstream.py \
  --upstream-fixture --p-count 2 --output /tmp/ship-pristine-control-p2.json

# Larger modulus-chain candidate; this should exit 1 during setup.
python benchmarks/ship_equivalence/probe_upstream.py \
  --degree 32 --q-count 18 --p-count 19 --weight 5 --theta 4 \
  --output /tmp/ship-pristine-common-shape.json

# Noisy single-P probe; this should exit 1 at the accuracy gate.
python benchmarks/ship_equivalence/probe_upstream.py \
  --output /tmp/ship-pristine-p1.json
```

Every JSON report includes exact Q/P primes, source commit/hash, backend,
runtime version, scales, fixture/noise information, stage, and pass/fail.
Failed operations include a traceback. One process per invocation avoids
cross-case reuse of upstream mutable kernel defaults.

## Remaining benchmark work

The pristine row currently has no valid common-profile latency. Correctness
repairs, if made, belong to a separately named patched/optimized Jaxite
variant. Pin its patch/source hash and distinguish correctness repairs from
performance optimizations. The other implementations still need a validated
common profile, equal output capacity verified with a subsequent multiplication,
and fresh complete TPU traces before publishing comparative latency numbers.
