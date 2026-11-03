# Problem 3: non-AI track

Read [the handout](doc/problem.html) or [the Markdown handout](doc/problem.md).
The short competition schedule is **compact-v1**, with two explicitly reduced
CNNs. The original six LeNet/AlexNetTiny fixtures and baseline remain available
as the legacy full-model track; their results and SOTA rubric are separate.

The compact models are defined in `src/compact_models.py`:

| Case | Input | Encrypted circuit |
|---|---|---|
| `student1.in` | 1 × 28 × 28 grayscale | Conv(1→2, 5×5, stride 4, padding 2), square, Linear(98→10) |
| `student2.in` | 3 × 16 × 16 RGB | Conv(3→4, 3×3, stride 4, padding 1), square, Linear(64→10) |

Both cases use batch one. They are synthetic arithmetic checks, not trained
image-classification accuracy. Their three multiplication levels permit a
secure N=8192 reference with a 42-bit composite scale. The runners check the
actual full Q/P chains against the 128-bit rule, require ciphertext inputs and
outputs, and apply the same logit and argmax gates as the full-model runner.

From `src/`, use:

```sh
make compact-reference-test
make competition-baseline
# Implement cnn_opt.py; begin with baseline_student.build_mapping.
make competition-evaluate
make competition-score
```

The entire competition evaluation has a 300-second wall budget: process
startup, both model setups, one warmup and three measured requests per model,
validation, and cleanup. The parent kills its worker process group on timeout
and returns failure. Setup and requests run serially. Each run writes its
report and logs below `src/suite/run/compact-baseline/` or
`src/suite/run/compact-candidate/`, including source/fixture hashes, actual rings,
software versions, peak host RSS, and available TPU memory statistics.

The compact rubric has a maximum of 20 points: both models must pass and the
whole evaluation must fit the budget; each model earns five correctness
points and up to five speed points, reaching full speed points at 2× the
measured reference server speed. A correct baseline earns 10/20. The scorer
requires a successful measured baseline with matching fixtures, models,
software, and TPU type. Organizers must freeze their calibration and perform
protected candidate reruns; contestant-authored reports are not trusted
judging evidence. There is no unmeasured SOTA constant in this rubric.

The [recorded compact reference](../../validation/p3_compact_baseline.json)
completed the whole evaluation in **145.00 seconds** on one TPU v6 lite
(148.25 seconds measured by the controller), including cleanup. Both cases
passed, with maximum logit errors below 2.2e-8. This is a measured result for
the recorded software versions and source hashes, not a runtime guarantee on
other machines. The baseline self-score is 10/20.

For the legacy full-model track, use `make reference-test`,
`make baseline-plan`, and `make baseline-all`. The latter checks all six
original public cases with one cold request each; `make baseline-test` checks
only test1. `make student-baseline-test` evaluates those same full models with
an optional lower-precision secure N=16384 profile. These commands can take
many minutes and are not the compact competition schedule. The legacy
20-point SOTA bonus still requires a separately named and measured SOTA
reference and trained assets before it can be awarded.

The runner's `setup_s` measures Mapping construction. `warmup_s` and
`warmup_server_s` retain excluded warmups; `cold_start_included` marks
`--warmups 0`. Its `total_s` includes fixture loading, setup, requests and
release for a locally constructed mapping. A prepared mapping remains owned by
the competition evaluator, which releases it once and measures that cleanup
in the complete worker time. The parent records failed evaluations on
cancellation or launch failure and stops remaining owned worker processes.
Existing request timing and numerical
gates are unchanged.

Set `CROSS_ROOT` when using an extracted package. `make submit` creates a local
archive after `REPORT.md` and `cross.patch` exist. `make clean` removes runtime
scratch. CPU tests do not establish TPU latency; use the recorded full
competition evaluation when planning the event.
