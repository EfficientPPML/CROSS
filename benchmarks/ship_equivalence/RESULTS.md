# Matched bootstrap comparison

Completed 2026-11-05 UTC. All four full encrypted TPU runs passed the strict
`compare.py` boundary, runtime, accuracy and timing checks. Each median uses
three samples on the same v6e chip. This is a small-sample functional result;
it is not a statistically established application speedup.

| Variant | Wall median (ms) | Device-active median (ms) | Max bootstrap error | Max post-square error |
| --- | ---: | ---: | ---: | ---: |
| Jaxite main + correctness patch | 128.412 | 14.719 | 0.000200715 | 0.000050140 |
| Optimized Jaxite + correctness / bounded-rescale patches | 123.237 | 3.769 | 0.000200718 | 0.000050137 |
| CROSS SHIP | 107.323 | 1.465 | 0.000200761 | 0.000050149 |
| Native CROSS + degree-1 patch (two-pass Meta-BTS) | 7904.930 | 386.413 | 0.000289082 | 0.000045843 |

Reports and every sample are in `results/matched-n32-*.json`; the accepted
summary is [matched-n32-summary.json](results/matched-n32-summary.json).
Device-active latency is the union of traced TPU module execution intervals;
wall latency includes host dispatch and idle gaps. Do not mix these metrics.

The PR review fixes passed 331 CPU tests and 41 TPU SHIP regressions. A separate
full-size N16384 bootstrap check passed with maximum errors 0.000401648876
(ramp) and 0.000387622470 (random). These full-size checks validate correctness;
they do not supply a matched four-way timing comparison at N16384.

The common profile is a **toy functional benchmark**, not a security-qualified
application or evidence of realistic deployment speedup. Every row must pass
both full encrypted slot-to-slot bootstrap and public post-bootstrap square
checks on ramp and random fixtures before its timings are admitted.

| Shared boundary parameter | Value |
| --- | --- |
| Ring degree / real slots / batch | 32 / 16 / 1 |
| Composite degree | 1 |
| Ordered Q / P prime counts | 23 / 24 |
| log2(Q) / log2(P) | 690.999840 / 719.999380 bits |
| Input and output Q subset | First two Q primes; log2(Qactive) = 60.999990 bits |
| Input and output scale | 1,073,741,441; log2(scale) = 29.999999485 bits |
| Output level / noise scale degree | 1 / 1 |
| Verified remaining multiplication capacity | One terminal public multiply |
| Plaintext range / maximum absolute error gate | [-0.125, 0.125] / 0.0009765625 |
| Context decomposition | dnum=2 |
| SHIP weight / theta / gamma | 5 / 4 / 8 |
| Native transform budget / correction / Meta-BTS precision | [1,1] / 2 / 4 |

Exact ordered primes and fixture seed are in [common-n32-c2.json](common-n32-c2.json).
The same TPU chip, runtime, source adapter, and timing scope are required by
`compare.py`. Wall latency and union-of-device-active time are distinct metrics.
Setup, compilation and the verification square are outside both timing windows;
input conversion and output normalization are inside them.

Jaxite rows use a common CROSS encrypted input adapter. The unoptimized row is
correctness-patched remote main; pristine upstream fails the tested multi-P
configuration. Native CROSS includes the recorded experimental degree-1 fix.
Internal relinearization schedules differ; see the README for the exact scope
of equivalence and source revisions.

## Source and runtime

- CROSS: `b6c5e530d2506c28a76879823151aab46c8864d7` plus
  `cross-cd1-correctness.patch`; effective diff SHA-256
  `fdbc896ffb832fb5d1c28d6fbd835b45dd226834c18b1fe59f2af43521d12fdc`.
  All four rows use this same checkout for their shared adapter/operators.
- Jaxite main: `1c35ddd29179e946ab7d08515053cefd7e76c46e` plus
  `jaxite-correctness.patch`; effective diff SHA-256
  `1d7f5304eaafa5d469e703771395cc4b2b0bcfff1aeb6027df36f8676234b18d`.
- Optimized Jaxite: PR147 `89797d88374c383a550374c0dfad6e44feced3fb` plus
  `jaxite-pr147-optimized.patch` and `jaxite-bounded-rescale.patch`;
  effective diff SHA-256
  `71b588f1d2ed73a001df3ad8bf7be21f03a64ad9d2abdcb9c7f95ee214b4b0d8`.
  The original optimized TPU compilation crashed. The bounded recurrence
  preserves every tested rescale residue (3, 5, 24 drops), and the complete
  patched bootstrap passes both encrypted fixtures and public-square gates.
- Python 3.11.13, JAX/JAXlib 0.9.1, NumPy 2.4.6, libtpu 0.0.35;
  host `t1v-n-43c8c27f-w-0`, chip 0, 64 MiB stack limit, empty XLA_FLAGS,
  persistent compilation caching disabled. No concurrent TPU job ran during
  the accepted timing samples.

## Raw evidence

The local archive `artifacts/pr5-n32-raw.tar.gz` contains all raw TPU traces,
logs, complete effective source patches, the measured harness, validation
reports, and failed/stopped compiler attempts. It is retained locally rather
than committed as a 214 MB binary. Archive SHA-256:
`9bbaf18e0b3b02ebfb9ad87f537086541f0f3553bce14090628231c0ad63386a`.
The per-file [SHA-256 manifest](results/matched-n32-artifact-sha256.json)
is committed. Report trace paths refer to the original TPU `/tmp` directories;
the archive preserves paths relative to `pr5-results-n32/`.

Recheck the four accepted reports locally:

```sh
python benchmarks/ship_equivalence/compare.py \
  benchmarks/ship_equivalence/results/matched-n32-cross-native.json \
  benchmarks/ship_equivalence/results/matched-n32-cross-ship.json \
  benchmarks/ship_equivalence/results/matched-n32-jaxite-patched.json \
  benchmarks/ship_equivalence/results/matched-n32-jaxite-optimized-loop2.json
```
