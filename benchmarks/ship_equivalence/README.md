# Matched bootstrap comparison and compatibility checks

The requested comparison holds ring degree, slot count, Q/P prime lists,
input/output active Q subsets, input/output scales, and retained application
capacity constant across unoptimized Jaxite SHIP, optimized Jaxite SHIP,
optimized CROSS SHIP, and native CROSS bootstrap. Historical timings do not
satisfy this contract and must not fill missing benchmark cells.

Current review-fix validation and the matched TPU comparison are recorded in
[RESULTS.md](RESULTS.md). The earlier checks below explain the baseline patches.

## Baseline

Unoptimized means pristine `google/jaxite` remote `main`, resolved 2026-11-03:
`1c35ddd29179e946ab7d08515053cefd7e76c46e`.
The probe checks that exact HEAD and rejects tracked modifications unless
`--expected-patch` names the exact reviewed diff. The locally modified PR147 checkout
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
qualifications**. This historical section contains no TPU latency measurements.

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

## Authorized lightweight repairs and full-bootstrap checks

`jaxite-correctness.patch` applies to the pinned upstream main revision. It
removes the complete P basis during conjugation, reduces large key constants
before NumPy conversion, supports signed sparse keys and a separate dense output
key, includes a partial final rotation block, and reduces modular products before
accumulation. This variant is **correctness-patched upstream main**, not pristine
main and not the optimized PR147 variant. `test_jaxite_patch.py` covers signed
secret-key extension and accumulation near the supported modulus boundary.

`cross-cd1-correctness.patch` is an **experimental**
native CROSS patch. It repairs degree-1 multiplication level selection,
sub-resolution scalar handling, and degree-1 transform scaling constants.
It has only been applied in an isolated checkout; the production kernel has
not been changed. Passing execution alone is insufficient for publication.

`run_common.py` runs complete encrypted slot-to-slot bootstraps. The Jaxite
variants use the same CROSS encrypted input adapter as CROSS SHIP and their
respective Jaxite half-bootstrap kernels. Adapter work and output normalization
are included in timing; setup and compilation are excluded. This must be
described as Jaxite SHIP **with a common CROSS input adapter**, not an entirely
upstream end-to-end implementation.

`common-n32.json` is the exact **CPU correctness-only** candidate: N=32,
16 slots, composite degree 1, 23 Q primes, 24 P primes, input/output scale
1,073,741,441 (approximately 2^30), input/output Q subset equal to the first
two primes. The error gate is 2^-10 for both bootstrap and a subsequent square
followed by rescaling. It is not a secure application profile. Larger-ring
TPU runs require a separately validated common manifest.

The initial native CROSS configurations failed the accuracy gate, including a correction-factor-7,
Meta-BTS-precision-4 diagnostic (maximum ramp error about 0.00796). Reports in
`results/` preserve source hashes, exact parameters, and failures. The final
Jaxite patch passed the full recheck after the accumulator-boundary repair;
both targeted regression tests also passed. Raising native Meta-BTS precision
to 8 at correction factor 7 still failed (ramp error about 0.01281).

The subsequent `common-n32-c2.json` candidate uses native correction factor 2
and Meta-BTS precision 4. It passes all four variants' CPU checks without
changing the shared Q/P lists, boundary scales, plaintext range, or error gate.
Native-only tuning fields do not affect SHIP execution; its existing reports
therefore remain applicable to this candidate. The strict correctness comparison
checks all shared fields and the SHIP tuning fields separately.

| Variant | Maximum bootstrap error | Maximum subsequent-square error |
| --- | ---: | ---: |
| Native CROSS with degree-1 patch and correction factor 2 | 0.000479942 | 0.000046806 |
| CROSS SHIP | 0.000200730 | 0.000050142 |
| Correctness-patched Jaxite main + common input adapter | 0.000200641 | 0.000050117 |
| Optimized Jaxite + common input adapter | 0.000200678 | 0.000050133 |

This establishes a small correctness case, not a latency or security claim.
`make_manifest.py` generates larger-ring candidates for independent validation.
`compare.py --correctness-only <four reports>` checks equivalence and errors
without accepting or presenting timing data.

For each variant, select the isolated source checkout with `PYTHONPATH`:

```sh
JAX_PLATFORMS=cpu python benchmarks/ship_equivalence/run_common.py \
  --manifest benchmarks/ship_equivalence/common-n32.json \
  --variant cross-ship --output /tmp/cross-ship-common.json
```

On an idle TPU, use at least `--iterations 3 --profile-dir <fresh-directory>`
using the included `device_profile.py` and `raw_xplane.py` helpers.
`compare.py` requires all four variants to pass, matching
cryptographic boundary parameters and runtime/device metadata, and at least
three wall and device-active timing samples. It refuses CPU timings and
incomplete comparisons. Device-active and wall latency remain separate metrics.

## Source reproduction

For current measurements, apply `cross-cd1-correctness.patch` to CROSS commit
`b6c5e530d2506c28a76879823151aab46c8864d7` in an isolated checkout. The
historical CPU checks used `262054680759f93fdf841a9f6002351ee6e9d948`.
The effective git-diff hash changes when rebasing a patch; reports record
the diff against the actual checked-out revision.
Apply `jaxite-correctness.patch` to the upstream main revision pinned above.
The optimized Jaxite variant is commit
`89797d88374c383a550374c0dfad6e44feced3fb` plus
`jaxite-pr147-optimized.patch` followed by `jaxite-bounded-rescale.patch`. Reports record the SHA-256 of each complete
tracked-source diff; keep these patches separate from the pristine baseline.

## TPU allocation

The comparison uses an authorized seven-day Flex-start v6e-8 VM in
`europe-west4-a`, runtime `v2-alpha-tpuv6e`. Resource provisioning alone does
not establish benchmark correctness or supply latency results.

## PR-review revision

New runs use CROSS `b6c5e530d2506c28a76879823151aab46c8864d7` (rebased on
upstream main) plus the explicit native degree-1 comparison patch. The patch
applies to this revision without changing the production SHIP source. All four
variants use that same CROSS checkout and encrypted input adapter. The
post-bootstrap square now calls public `ctx.he_mul[0].mul`; earlier split-only
checks do not substitute for this gate. Reports include exact source/diff and
harness hashes, libtpu/JAX versions, chip, stack limit, and XLA flags. Persistent
compilation caching is disabled in the matrix runner following the review's
cache-reload failure report. The historical correctness table above predates
these fixes and is not a timing result for this revision.

## Rejected larger candidates

The saved pre-review N1024 native runs failed the `2^-10` gate: maximum ramp
errors 0.0953268181 and 0.0554859896 for the two recorded parameter choices,
although CROSS SHIP passed its N1024 case. N128 native and CROSS SHIP passed
separate correctness runs, but the optimized Jaxite TPU compilation failed.
These partial results do not establish a four-way comparison.

`cross-cd1-scalar-precision.patch` and its opt-in regression preserve a rejected
diagnostic experiment; the matrix runner does not apply it. It is not a
validated remedy for the larger native bootstrap failures.

The shared `dnum=2` is the context/key-switch configuration. Internal schedules
are algorithm-specific: CROSS SHIP's product tree retains up to eight
relinearization partitions per level, while Jaxite uses its configured
partition count. Those choices, and native CROSS's Meta-BTS polynomial
schedule, are not identical algorithms. The equivalence claim concerns the
exact ring/modulus/scale boundary contract and the verified application output,
not bit-identical intermediate arithmetic. All sources use nominal Gaussian
sigma 3.19 (CROSS's default stores its float32 representation); keys/noise are
independently sampled, not identical encrypted inputs.

The recorded TPU matrix uses a 64 MiB compiler stack limit consistently for
all variants. Reproduce from prepared source checkouts on one idle chip:

```sh
ulimit -s 65536
unset JAX_COMPILATION_CACHE_DIR
export JAX_ENABLE_COMPILATION_CACHE=false
python benchmarks/ship_equivalence/run_matrix.py \
  --cross /path/to/cross-reviewed-with-cd1-patch \
  --jaxite-patched /path/to/jaxite-main-with-correctness-patch \
  --jaxite-optimized /path/to/jaxite-pr147-with-recorded-patch \
  --manifest benchmarks/ship_equivalence/common-n32-c2.json \
  --output-dir /path/to/new-results-directory --chip 0 --iterations 3
```

## Optimized Jaxite compiler compatibility

The initial optimized N32 TPU attempt crashed in XLA
`InstructionFusion::CanFuseOnAllPaths`, with over 1,000 repeated stack frames,
even with the common 64 MiB stack limit. Its failed report and log are retained;
no latency is assigned to that attempt.

`jaxite-bounded-rescale.patch` applies **after** `jaxite-pr147-optimized.patch`.
It stages the triangular multi-prime recurrence and scaling-constant product as
`jax.lax.fori_loop` operations instead of unrolling them into one large compiler
graph. Future dropped limbs are updated in a vector; inactive limbs retain
their prior values. The arithmetic and modulus chain are unchanged.

`check_jaxite_rescale.py /path/to/original-patched/rescale.py` checks exact
residues against the original optimized function on CPU for 3, 5 and 24 dropped
primes. Set `JAX_PLATFORMS=cpu` and put the bounded-rescale Jaxite checkout and
CROSS's `jaxite_word` directory on `PYTHONPATH`. The isolated N32/24-drop TPU
rescale compiled and executed in 4.85 seconds; this diagnostic is not bootstrap
latency. The patched variant passed the full encrypted TPU bootstrap and public-square
gates before timing; see [RESULTS.md](RESULTS.md).

Barrier-only attempts and a loop around only the scaling-constant product were
stopped while compiling, before producing correctness or timing results. Their
logs and patches are retained with the raw artifacts. The successful isolated
compiler diagnostic uses loops around both recurrences.
