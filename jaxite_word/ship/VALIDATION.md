# SHIP bootstrap validation and performance

## Comparison with Jaxite SHIP

The native CROSS implementation measured **1.271x faster** than the earlier
locally patched Jaxite SHIP implementation: **21.31% less active TPU time**.
Both runs used a full encrypted slot-to-slot bootstrap at N=16384, all 8192
real slots, identical Q/P primes, weight=31, theta=9, gamma=8 and two output
Q limbs. Setup, compilation, host execution, idle gaps and post-bootstrap
work are excluded from these measurements.

| Implementation | Median active TPU time (ms) |
| --- | ---: |
| Jaxite SHIP (local PR147 implementation) | 1724.291416 |
| First CROSS port | 1576.169903 |
| Native CROSS SHIP | **1356.806534** |

This is a historical comparison across different v6e chips/VMs and
independently randomized keys and ciphertexts, not a controlled
identical-ciphertext A/B. Full-bootstrap timings are not directly comparable to LL13 half-bootstrap
results. The functional profile is not exact HT14 or
security-qualified. The committed [measurement summary](benchmark_results.json)
contains exact prime lists, all timing samples, accuracy checks, trace event
counts and measured-source hashes. The large raw trace/source archives listed
below are retained in the benchmark workspace, not included in this repository.

## PR review checks — 2026-10-29

Fresh CPU validation passed **127 tests and 15 subtests** under Python 3.11,
JAX 0.10.2 and pytest 9.2.0. The two optional large-ring CKKS tests were
excluded. The current SHIP suite has 27 tests, including invalid parameter
and dual-import identity checks. Generic near-modulus multiplication/square
regressions now live in `polynomial_test.py`.

The review tightened configuration validation (including primality and finite
derived scales), made hoisted-kernel cache ownership explicit, removed an
unused control object and clarified formatting. These changes do not alter
the measured evaluator arithmetic. The TPU results below belong to the
2026-10-28 measured revision; the additional validation-only checks were run
on CPU, not reported as a new TPU benchmark.

Reproduce the CPU checks from the repository root:

```sh
PYTHONPATH=jaxite_word python -m pytest \
  jaxite_word/ship_test.py jaxite_word/polynomial_test.py \
  jaxite_word/mapping_compile_test.py jaxite_word/hemul_test.py \
  jaxite_word/finite_field_test.py jaxite_word/bootstrapping_test.py \
  -q -k 'not n4096 and not n8192'
ruff check --isolated --select E4,E7,E9,F,I \
  jaxite_word/ship jaxite_word/ship_test.py jaxite_word/ship_benchmark.py
ruff format --isolated --config indent-width=2 --check \
  jaxite_word/ship jaxite_word/ship_test.py jaxite_word/ship_benchmark.py
```

Ruff 0.17.3 checks and Python compilation passed. Existing unrelated files
were not reformatted wholesale; their patch passes `git diff --check`.

## Native CROSS refactor — 2026-10-28

The second refactor replaces online Jaxite ciphertext/kernel PyTrees with
CROSS rank-five arrays and ordinary CROSS control objects. SHIP is selected
through `ctx.he_bootstrap.configure(scheme="ship", ship_config=...)` and can
be used as an explicit Mapping bootstrap barrier. Evaluation keys remain
runtime operands. Configuration, trusted setup, and online evaluation have
separate lifecycle steps.

CPU validation (JAX 0.10.2): 13 SHIP tests, 86 existing Mapping/arithmetic
tests with 15 subtests, and 12 existing CKKS bootstrap tests passed. The two
optional large-ring CKKS tests were excluded. New coverage includes actual
encrypted Mapping execution, input-contract/readiness checks, idempotent
setup, and calls with key generation/diagonal encoding disabled after setup.
Full-bootstrap tests check both real and unprojected complex errors.
All 13 SHIP tests also passed on TPU v6e chip 7 (JAX 0.9.1, libtpu 0.0.35)
in 344.53 seconds, including setup and compilation. That duration is a test
suite runtime, not an online bootstrap latency.

The arithmetic contract remains uint32 limbs with primes below 2^31 and
64-bit intermediates, Barrett representation, composite_degree=1, batch one.
This refactor does not add composite-scale or Montgomery support.

### Full-size check and active TPU time

N=16384, 8192 real slots, weight=31, theta=9, gamma=8, Q=[31,30,30,30,30,30,30,30]
bits and nine 30-bit P primes. The exact primes match the first integration.
Two output Q limbs remain. Both noisy encrypted fixtures passed the 2^-10
maximum real-slot error gate using CROSS's validating `decrypt_slots`:

| Fixture | Maximum real-slot absolute error |
| --- | ---: |
| Ramp | 0.000486270380 |
| Random, plaintext seed 42 | 0.000421552547 |

One TPU v6e chip (physical chip 6 on `nodev6e8`, `europe-west4-a`), JAX 0.9.1:

| Capture | Active TPU time (ms) |
| --- | ---: |
| 0 | 1356.613103 |
| 1 | 1356.891160 |
| 2 | 1356.806534 |
| **Median** | **1356.806534** |

All three raw XPlane captures contain 66,291 XLA Modules events, one executing
device, no dropped-event markers, and no profiler warnings. Timing is the
union of device module intervals for a warmed full bootstrap. Setup,
compilation, decryption, host work and inter-module idle time are excluded.
No post-bootstrap multiplication is included. Driver log-file permission
warnings were present, as in the first integration, but raw captures completed.

Compared with the first integration's 1576.169903 ms on the same chip and
parameter profile, this run used 13.92% less active device time (1.162x).
These are separate runs with newly randomized keys and ciphertexts, not an
identical-ciphertext A/B. This remains a functional profile, not a
security-qualified HT14 parameter set or a general accuracy guarantee.

### Second-refactor artifacts

Workspace: `bootstrap-comparison/results/cross-ship-second-refactor/`.
`second-result-n14-device.json` contains exact parameters and raw trace
summaries. `cpu-tests.log` and `tpu-tests.log` record the SHIP test suites.
`second-validated-artifacts.tar.gz` includes the measured source, raw captures,
full-run logs and the raw XPlane summarizer.

Archive SHA-256:
`30a38137d75e32dce331bfe7f11accdbdc77eb6b4266c0ac2d380351b2ddfe24`.
Measured source archive SHA-256:
`c120a8381343f19518b6ade19c3dd7bfa8eb73cbf68baf9e98c61f339c7a22df`.
Subsequent local edits affect compatibility imports, the convenience encrypt
helper's real-input guard, documentation and a Mapping variable binding;
the measured evaluator arithmetic is unchanged.

## First integration — 2026-10-27

The port passed 12 new tests on CPU (JAX 0.10.2) and on one TPU v6e chip
(JAX 0.9.1, libtpu 0.0.35). The CPU run of the existing `polynomial_test.py`,
`hemul_test.py`, and `finite_field_test.py` passed 39 tests and four subtests.
The full-bootstrap tests additionally verify ciphertexts encoded/encrypted
through the caller's original CROSS context, and unprojected complex output
error; those additional assertions were checked on CPU at N=32 and N=128.

The complete N=16384 run used `ship_benchmark.py`, weight=31, theta=9,
gamma=8, eight Q primes and nine P primes, exactly matching the previous
functional comparison's prime lists. All 8192 real slots were checked on a
ramp and a randomized vector in [-0.125,0.125]. Each fixture was encrypted
with noise and bootstrapped under the same independently generated key setup.

| Fixture | Maximum real-slot absolute error |
| --- | ---: |
| Ramp | 0.000549852178 |
| Random, plaintext seed 42 | 0.000489316856 |

These errors use CROSS's real-valued `decrypt_slots` with its approximation
validation enabled. The gate was 2^-10 = 0.0009765625. Two Q limbs remain.
This full-size run does not establish an arbitrary-input failure probability
or security qualification.

### Active device time

One TPU v6e chip, physical chip 6 on `nodev6e8`, zone `europe-west4-a`.
Three warmed full slot-to-slot captures:

| Capture | Active TPU time (ms) |
| --- | ---: |
| 0 | 1575.936644 |
| 1 | 1576.215340 |
| 2 | 1576.169903 |
| **Median** | **1576.169903** |

The raw XPlane reader found 66,294 `XLA Modules` events in each capture,
one executing device, no dropped-event markers, and no profiler warnings.
The metric is the union of device module intervals. Setup, compilation,
decryption, host execution, and inter-module idle gaps are excluded.
No post-bootstrap multiplication is included.

The previous SHIP implementation measured 1724.291416 ms at the same
functional parameters: this run used 8.59% less active device time (1.094x).
That is a historical comparison across different v6e chips/VMs and random
keys, not a fresh controlled A/B. The historical CROSS Meta-BTS timing is
not a new measurement of the modified repository.

The TPU driver's attempted `/tmp/tpu_logs` writes produced permission warnings
on this shared VM. They did not prevent execution or complete raw profiler
capture; all tests and accuracy gates completed.

### Artifacts

Local workspace: `bootstrap-comparison/results/cross-ship-port/`:

* `result-n14-device.json`: parameters, errors, samples, and raw trace summaries.
* `tpu-tests.log`: the 12-test TPU run.
* `validated-n14-artifacts.tar.gz`: raw traces, exact measured source archive,
  reports, profiler reader, and execution logs.

Archive SHA-256:
`5875f6571d8346842c1db23ca8302c5c293636230ba8b0198d91d1e2239845c0`.
Measured source archive SHA-256:
`592f39a92b8d62ff7d44ac7e262c4be7965f18199dd5e9bdc6148a8d3f74e07c`.

These artifacts describe the first integration, before the native CROSS
refactor above. Its timings must not be relabeled as second-refactor results.
