# P5 implementation and experiment status

## Portable PR checkpoint — 2026-11-05

This PR packages P5 and its required shared runtime against current `main`,
preserving the policy that all five problems allow coding agents. Use
[TPU_TESTING.md](TPU_TESTING.md) to collect measurements on another host.
The development experiment history below refers to the original local checkout
and local evidence directory; those large archives and machine-specific queue
scripts are not bundled in this code PR. Historical measurements must not be
relabeled as measurements of this new checkout. No P5 TPU run or speedup is
verified yet. Fresh validation of the PR checkout:

- 74 P5/package tests passed in77.20s, including the opt-in real encrypted CPU
  HerPN, fusion and rotation-cache checks.
- 556 shared-runtime/core tests and64 subtests passed in368.64s;11 tests were
  skipped. This is CPU regression coverage, not TPU validation.
- `make -C cadathlon all reference-test` passed all30 public plaintext cases.
- The214-file contestant archive extracted successfully and matched its source
  files. Its P5 test1 encrypted CPU collector passed one warmup and three
  requests, median server5.30338s, max error7.72539e-8 and exact top1 agreement.
  All126 frozen files were verified; the collector and child exited0/reaped.
- All7 local P5 handout links resolve. The PR diff passes `git diff --check`.

The extracted encrypted check preceded this documentation update and HTML
whitespace normalization. All101 Python runtime sources relevant to that check
remain byte-identical; the document renderer is outside encrypted execution.
No raw experiment directory, machine-specific queue, key or ciphertext is
part of this PR. The full local logs remain under the development evidence
directory in `pr-preparation-20261105/`.

## Historical development evidence

Updated 2026-11-04, 10:44 UTC. The WISE scheduling / TPU acceleration goal is
**in progress**. No P5 TPU execution or speedup is claimed. Raw evidence lives
under `../../validation/p5_wise_20261104/`; each collector report is authoritative
for its current process state. P1–P4 measured artifacts are unchanged.

## Implemented

- Pinned WISE planner at `485655782dc331e6ebb48b159acac8ba924a9933`, with unchanged
  upstream packers, BSGS plans, sparse transforms, biases and output folds.
- Keyless CROSS execution, compiled rotations/arithmetic, bounded public
  plaintext caching, actual Q×P security and ciphertext placement checks.
- Six fixed MLP/CNN workloads, independent PyTorch references, direct/BSGS
  controls, raw fresh-request timing and immutable runtime collection.
- Native Winograd routing and its signed 4→16→4 transforms, integrated into
  whole CNNs, including padded tile planes and multiple ciphertext blocks.
- Pool-friendly Toeplitz tile layouts and a pretrained hybrid matching WISE's
  4×4 Toeplitz first layer / 2×2 Winograd second layer.
- The complete supplied MNIST checkpoint: two convolutions, two pools, two
  dense layers and all three learned HerPN activations. No truncation,
  retraining, activation replacement or normalization change.
- Original native WISE/OpenFHE CPU evaluation and tuning, with 128-bit security
  enforced by OpenFHE, preserved failures, thread/ring/layout selection,
  runtime source snapshots, and hashes of loaded native binaries.
- Five-problem CADathlon build, handout and release integration.

## Correctness and layout evidence

`pretrained-oracle-v1/result.json` validates the original checkpoint on all
**10,000 plaintext MNIST test images: 9,914 correct (99.14%)**. Both optimized
layouts also match the independent original HerPN/PyTorch model on two distinct
images and perturbed versions. This is plaintext accuracy, not encrypted
accuracy on the full dataset.

At batch one / 16,384 slots, the pretrained Toeplitz plan has 16,752 public
multiplications. The tiled Toeplitz plan has **1,514**, and the hybrid has
**2,514**, with all nine layers intact. Single-image public-plan errors are
6.22e-15 and 7.99e-15 respectively (`pretrained-layouts-v2/`). These are operation
counts and ideal-circuit checks, not accelerator speed measurements.

The original 19 schedule/Winograd tests passed in 7.77s, including distinct input seeds,
small slots, block crossings and the complete tiled CNNs. A broader checkpoint
passed 38 schedule/scoring/profile tests in 13.75s. The later secure arithmetic
check passed **five tests in 66.10s**, including real encrypted learned
quadratic composition and nine successive levels retaining values up to 32
at the final modulus (`final-headroom-tests.log`). These overlapping counts
must not be added. The package checkpoint passed 41 tests with two opt-in HE skips in 13.93s;
those encrypted checks passed separately above. The extended native-comparison
scorer passed 23 tests. All five build/reference targets (30 public fixtures)
and five HTML/PDF handouts rebuilt successfully. These overlapping checks are
recorded separately in the raw logs. The final interleaved MLP/CNN scheduling
and extended scoring checkpoint passed 42 tests in 7.50s.

The complete encrypted Winograd test4 CNN passed on CROSS CPU with one warmup
and three requests: median server **75.45017s**, external process **323.40666s**
(`cpu-winograd-network-v1/`). This is slower than the original Toeplitz
28.95733s on that CPU case. The earlier isolated convolution proof remains
`winograd-cpu-v1.json`: max error2.33901e-8, cold server7.08578s. Its source
observation was collected during execution; it is not a prelaunch archive.

## CPU measurements

The original CROSS Toeplitz six-case schedule completed successfully in
3147.86s externally, with all archived sources unchanged. It used eight
allocated CPU cores, one warmup and three measured requests (`cpu-bsgs-v2/`):

| Case | Median server | External case process | Status at this checkpoint |
| --- | ---: | ---: | --- |
| test1 | 4.31003s | 29.55728s | Completed, passed |
| test2 | 9.15892s | 52.50984s | Completed, passed |
| test3 | 6.41163s | 39.35517s | Completed, passed |
| test4 | 28.95733s | 135.99133s | Completed, passed |
| test5 | 97.67222s | 425.01927s | Completed, passed |
| test6 | 596.63961s | 2465.37452s | Completed, passed |

A second whole six-case CROSS collection using the earlier tiled layouts
(`cpu-tiled-v3/`) completed in **1731.61074s**, with all sources unchanged.
Its six medians were4.31191,9.13236,6.37765,28.96278,92.47955 and251.76666s.
Later interleaved layouts supersede its batched cases; it remains an intact
historical control. The initial two CPU proofs remain unchanged as
`cpu1-v1.json` and `cpu2-v1.json`, with their archived earlier runtime.

Native WISE/OpenFHE uses the unchanged backend, OpenFHE v1.4.2.6 commit
`94289854bb024bcef7e6b5c9a59eda5e9e204770`, and actual `HEStd_128_classic`
validation. All six workloads now have at least one passing repeated CPU
measurement across `native-cpu-tuning-six-v1/` and `native-cpu-tiled-v2/`.
The original-layout tuning grid is still running. Later single-NUMA and
interleaved-layout runs improve these baselines (`native-cpu-mlp-numa-v3/`,
`native-cpu-mlp-interleaved-v4/`, `native-cpu-batch-inner-v4/`). Best observed medians:

| Case | Median server | Threads | Layout |
| --- | ---: | ---: | --- |
| test1 | 0.15220s | 2 | Original |
| test2 | 0.37519s | 2 | Interleaved features/images |
| test3 | 0.31983s | 2 | Original |
| test4 | 0.33977s | 4 | Tiled Toeplitz |
| test5 | 0.59496s | 4 | Tiled Toeplitz, images within channels |
| test6 | 1.52025s | 4 | Tiled Toeplitz, images within channels |

These use degree16,384, FLEXIBLEAUTO, 42-bit scales, one warmup and three
requests. Other measured thread counts/scaling modes and all rejected N8192
attempts remain in the tuning ledgers. Native and CROSS have different secure
profiles; these values do not constitute a matched hardware speedup. Concurrent
CPU work and allocation are recorded, so final contest calibration must be
repeated under its specified resource conditions. The older cores88–95 allocation
crossed a NUMA boundary; its results remain unchanged. Later native tuning pins
threads within one NUMA node and records binding/placement settings.

The native **complete pretrained network** passes one warmup and three fresh
requests for the first test image, with exact top-1 agreement and errors well
below the unchanged 1e-3 / nRMSE1e-4 gates. Single-NUMA median server latency is
35.46873s (one thread), 22.29355s (two), 15.64924s (four), and **15.46036s (eight)**.
The N32768/FLEXIBLEAUTO profile uses 42-bit scales and a 60-bit first modulus
(`native-pretrained-numa-v3/`). The older cross-NUMA 35.78163s/31.20384s runs
remain in `native-pretrained-calibration-v2/`. These are one-image trained-model
correctness and repeated timing, not full-test-set encrypted accuracy.

The complete nine-layer pretrained network also **passes CROSS encrypted CPU
execution** (`cpu-pretrained-tiled-v3/`): max error2.72541e-7, nRMSE2.18784e-8,
exact top-1 agreement, one correct example. This was one cold request: setup
73.01046s, server1239.51360s, external process1326.05575s. It is not a warm
performance calibration. All source snapshots stayed unchanged and the child
was reaped with exit0. A repeated CROSS calibration is running in
`cpu-pretrained-calibration-v4/` and is **not completed**. The native eight-image
extension completed successfully (`native-pretrained-batch8-v4/`): all eight
classifications agree and are correct, max error2.13041e-7, nRMSE1.05623e-8,
cold server73.80305s. Its repeated warm calibration is **not completed**.

The latest batched CNN layout groups distinct images within channel planes:
test5 convolution products fall from522 to74, and test6 from1,988 to128.
Their pooling and final dense layers remain intact. Nineteen independent
multi-input/ciphertext schedule checks passed; the native encrypted runs above
pass too. The corresponding CROSS test5 and test6 both passed, at60.40109s and103.29951s
medians (`cpu-batch-inner-v4/`). Their four-core allocation differs from
the earlier eight-core experiment, so this is not an isolated speedup claim.
The new interleaved MLP test2 passed CROSS at a7.41503s median and native CPU
at0.37519s (`cpu-mlp-interleaved-v4/`, `native-cpu-mlp-interleaved-v4/`).
Batch-two pretrained plans also pass against original and perturbed inputs
with max error7.11e-15 (`pretrained-batch-inner-v2/`).

## Preserved failures and diagnostics

- `native-cpu-tuning-v1/` and `native-small-ring-v1/`: OpenFHE rejects all
  attempted N8192 profiles at the requested 128-bit security; they are not
  passing speed measurements.
- `cpu-pretrained-tiled-v1/`: failed before key generation because the earlier
  small-prime chain exhausted its prime neighborhood. The longer CROSS profile
  now uses 46-bit scales and distinct primes, still under the actual Q×P bound.
- `native-pretrained-v1/`: both original 43-bit-first-modulus layouts fail the
  numerical gate. Client layer diagnostics locate the error in the final dense
  output, after eight accurate layers. A 60-bit first modulus retains the
  logits' integer range; the successful repeated run above includes that fix.
- `native-pretrained-headroom-v1/`: the client diagnostic passes all nine layers
  and final logits (max error1.29512e-7). Diagnostic times include intermediate
  decryption and are excluded from performance selection. Its older collector
  exits1 because there is no eligible performance entry; the successful child
  result is preserved. The collector now reports diagnostic completion
  separately from performance eligibility.
- `cpu-pretrained-tiled-v2/`: the earlier 47-bit-first-modulus run completed
  with a failed numerical gate (max error16.62536, top-1 disagreement). Its
  failed request took1200.56192s on the server. The corrected60-bit-first
  run above passes; the failed result remains unchanged.

The extracted-release collector now honors `CROSS_ROOT` for core code while
preserving the released planner/common sources. Four packaging tests pass,
including a marked released-source snapshot without JAX initialization.
The273-file release-v3 archive matches its source bytes; all35 local links
resolve. Its extracted collector also passed real encrypted test1 with external
`CROSS_ROOT`: one cold request, server14.00710s, external19.08301s, sources
unchanged and child reaped with exit0. This is a release smoke test, not a warm
performance calibration (`release-v3/verification.json`, `encrypted-smoke/`).
`native-fastest-v1.json` records the fastest eligible native result for each
fixed case, selected by the scorer from raw repeated measurements. It reports
no TPU comparison.

## Further execution tuning

The pretrained first dense layout now preserves the pooled image stride for
both tiled Toeplitz and hybrid plans. Both complete plans match the independent
model on eight original and perturbed images (`pretrained-fc-layout-v3/`,
70.61698s). The64-image tiled Toeplitz plan also passes original and perturbed
inputs; its hybrid counterpart also passes. The complete64-image public-plan
check finished in522.14932s with unchanged sources and exit0. This establishes
full-network plaintext equivalence on64 original and64 perturbed inputs,
not encrypted accuracy (`pretrained-batch64-v1/`).
The repeated eight-image native4/8-thread grid completed in802.52674s, with all
sources unchanged and every attempt passing (`native-pretrained-batch8-fc-v5/`).
Tiled/hybrid medians are44.38152s/36.98631s at four threads and
42.96628s/37.69441s at eight. The16/32-thread extension completed successfully
in742.89225s. Tiled/hybrid medians were37.57230s/33.11580s at16 threads and
40.85030s/36.65891s at32 (`native-pretrained-batch8-wide-v5/`). All eight labels
are correct. The best observed eight-image median is now33.11580s.
`native-pretrained-fastest-v1.json` selects only eligible warm repeated results:
The earlier selection records15.46036s for one image and36.98631s for eight;
later selection snapshots retain the improved result above. They include rejected
cold, diagnostic, failed and incomplete entries and claims no TPU comparison.

An opt-in `--linear-chunk` path fuses small groups of WISE public products and
modular additions using the original CROSS kernels. It preserves the original
composite rescale and exact schedule. A real encrypted test passed exact
ciphertext-byte, modulus, scale and decoded-value comparisons for chunk sizes
1,2,8, mixed input levels, nonuniform coefficients/bias and two fresh inputs
(`fused-linear-tests-v1.log`,8.82s). All six full-network CPU cases passed in
1023.78201s (`cpu-fused-eight-v5/`), with unchanged sources and exit0. Server
medians were6.00138,7.23264,8.85720,39.26090,60.65178 and103.80853s. Some are
slower than historical controls, whose core allocations differ. Fusion remains
optional; no speedup is claimed.

The extended scorer checks pretrained sample counts and the distinct1e-3 /
nRMSE1e-4 accuracy gates; `--pretrained` reports separate ratios without
synthetic-suite points. The package/schedule/scoring checkpoint passed52 tests
in12.79s (`fusion-package-scoring-tests-v5.log`). These are overlapping scopes,
not additions to earlier test counts. New runner reports separately record
planning and plaintext verification time, executed/available devices, thread
settings and available allocator memory statistics. Native reports additionally
record pre-initialization and individual worker affinities because OpenMP can
pin the main thread while its workers use the rest of the allocation.

The preserved274-file release-v5 archive has SHA256
`87550171f0c6545e5c2284f5dbb16503509497a1d30a38967937c636c17f9e50`.
All archived source bytes and35 local links pass verification. Its extracted
fused test1 also passes real encrypted evaluation with an external CROSS core:
server12.71354s, collector17.84438s, one cold request. New timing/device metadata
is present; CPU allocator statistics correctly report unavailable. The outer
process exited0 and was reaped. The release reflects its recorded build
checkpoint; this status file may include later evidence.

A complete hybrid eight-image CROSS cold check passed in
`cpu-pretrained-hybrid8-fused-v5/`, with fusion8 and a1024MiB public cache.
All eight classifications match, with max error2.80774e-7 and nRMSE1.18784e-8.
The single cold server request took5043.01720s; collector time was5144.04665s.
`pretrained8-cold-v9.json` verifies149 frozen files, unchanged planner bytes
against the queued runtime and complete child receipts. Session44727 exited0
and was reaped. This establishes full-network encrypted correctness on those
eight images; it is ineligible for warm speed comparisons and does not establish
10000-image encrypted accuracy.

## Bounded rotation caching and extended CPU controls

The optional public rotation cache retains level-specific compiled controls and
key representations under an explicit byte budget. Ciphertexts are never cache
entries; it uses the original CROSS rotation kernel and completion barrier.
An encrypted test passed exact ciphertext, level, scale and decoded-value
comparisons across three keys, alternating levels, fresh inputs, explicit
clearing and a one-byte uncached fallback (`rotation-cache-tests-v6b.log`,19.01s).
The first test attempt failed on the test's assumption that fresh ciphertexts
always carry an explicit noise-degree attribute; both paths correctly preserve
its absence/default1. That failed log is retained as`rotation-cache-tests-v6.log`.

Two sequential, same-source and same-allocation comparisons completed:
`cpu-rotation-comparison-v6/` uses cases1,5,6 with cache0/256MiB on cores104–107;
`cpu-winograd-cache-v6/` uses the complete Winograd CNN test4 with cache0/512MiB
on cores8–15. Both use one warmup and three requests. All requests passed;
outer collectors exited0 and were reaped. The independent readback in
`cache-comparison-v7.json` checks raw numerical/timing gates, all requested
cases, matching schedules/profiles and actual frozen source bytes.

| CPU workload | Cache0 median | Cached median | Observed ratio |
| --- | ---: | ---: | ---: |
| test1, tiled schedule | 5.86313s | 5.38739s | 1.0883x |
| test5, tiled CNN | 60.87653s | 60.93955s | 0.9990x |
| test6, tiled CNN | 104.46779s | 102.67630s | 1.0174x |
| test4, Winograd CNN | 77.10412s | 75.45244s | 1.0219x |

These are observed medians from one ordered pair of trials, not a statistical
guarantee or TPU result. Case5 did not improve. Cache byte peaks/hit counts and
all raw samples are retained. Cases2,3,4 with the same256MiB cache, source and
four-core allocation also passed in `cpu-cached-remaining-v7/` (269.61877s;
outer exit0, reaped). `cpu-cached-six-v7.json` verifies all six fixed cases,
raw repeated gates, child receipts and identical150-file frozen source bytes
across the two collections. Case1–6 medians are5.38739,6.82025,8.59070,39.51988,
60.93955 and102.67630s. This supplies six matched CPU reports for the queued
TPU comparison; it is not a single continuous whole-suite wall-time measurement.

All ten native N16384 pretrained attempts at30/32/34/36/38-bit scales and a
60-bit first modulus were rejected by OpenFHE's128-bit security gate
(`native-pretrained-precision-v6/`,347.10768s). A separate actual-context preflight
also rejects the six tested28–32-bit configurations with39/40-bit first moduli
(`native-headroom-preflight-v6/`). No model execution or performance is claimed
for that preflight, and no security gate was relaxed. The native collector now
accepts explicit`--first-bits` settings for preserved headroom experiments.

The native64-image tiled encrypted attempt completed in
`native-pretrained-batch64-v6/`, with one warmup and three measured requests,
median264.69148s. All64 classifications match on every request. Its first warmup
had max error3.07421e-7, nRMSE1.15653e-8 and server264.93871s.
The hybrid comparison also completed: median197.27058s, all64 classifications
matching in the warmup and every repeat, worst max error2.80637e-7 and
nRMSE1.13628e-8. The complete two-layout collector took2377.80640s, preserved
all source bytes and child reaping, and its outer process exited0 and was reaped.
`native-pretrained-fastest-v3.json` independently checks archived source bytes,
runtime binaries, pinned WISE sources/assets and complete repeated results. It
selects15.46036s/33.11580s/197.27058s for batches1/8/64 from14 eligible reports,
while explicitly retaining19 ineligible attempts. This is the observed CPU
baseline; no TPU ratio is available.

The new275-file release-v6 includes the rotation cache and passes all source-byte
and35 local-link checks. Its preserved archive SHA256 is
`118d21ff57e055a57d880b4d1e7d3fc17df46525f82297e4c31168d98f8c7c25`.
Four packaging tests passed in5.70s. Extracted encrypted test1 passed one warmup
and three requests with cache256MiB: median server5.51783s, peak retained cache
77,635,680bytes,96 binding hits and75 key-state hits. Sources stayed unchanged,
and the outer process exited0 and was reaped. This is a release smoke test,
not the controlled cache comparison (`release-v6/`).

## Queued TPU experiments

The one-shot local collector `tpu-queued-v7/queue.py` is running and waiting for
the existing TPU sweep to finish. Its150-file source snapshot and launch plan
are frozen. The plan includes all six tiled cases, three Winograd CNN cases,
and the complete pretrained one/eight-image networks, each with one warmup and
three measured requests. Pretrained jobs also require their corresponding CPU
checks to finish successfully. It does not retry executed failures.

Seventeen queue tests passed in0.78s, including live-owner/device-holder guards,
failed/incomplete CPU prerequisites, source drift, single execution and rejection
of incomplete receipts. Real test subprocesses verify failure preservation,
reaping, restart rejection and cleanup of an owned orphan. No test initializes
TPU. Actual preflight observed parent944985 live and child2992744 holding all
eight VFIO devices. There are1639 unreadable process descriptor tables/entries;
the scan only covers visible holders, and libtpu's exclusive acquisition remains
the final authority. The queue changes no device ownership or libtpu locks.
No P5 TPU experiment has started at this checkpoint.

## Not completed

- TPU six-case and pretrained measurements, schedule/cache/batch tuning, matched
  speed ratios, and proof that the best TPU configuration beats tuned native CPU.
- Final completion audit and contest calibration after the remaining evidence.

The frozen unrelated Qwen job (parent944985, child2992744 at this checkpoint)
remains live and owns the TPU. It must finish and release the allocation before
P5 TPU experiments start. Do not cancel or restart it to obtain the device.

## Completed pretrained CPU calibration and direct native comparison

The full original nine-layer WISE MNIST CROSS run in
`cpu-pretrained-calibration-v4/` completed one warmup and three measured requests,
with median server1182.46971s and client-to-client1182.79240s. All requests pass
the unchanged encrypted gates; worst max error2.34516e-7, nRMSE1.96007e-8 and
top1agreement1. Collector time was4826.94615s. Independent readback verifies
all148 archived sources and child receipts (`pretrained-cpu-warm-v8.json`);
outer session71254 exited0 and was reaped. This is one selected image, not the
complete10000-image encrypted test set. Its earlier frozen runtime differs
from the queued TPU runtime; eventual comparison must check exact case,
profile and schedule identity and disclose the execution controls.

`score.py --native-only --native-cpu ... --tpu ...` now compares complete warm
native CPU/TPU measurements directly. It awards no contest points and does not
supply the six matched CROSS CPU/TPU pairs required by the existing rubric.
Both synthetic and pretrained modes retain their complete-workload gates;
pretrained reports additionally require the exact pinned checkpoint, HerPN,
image and label asset hashes. Forty-six scoring tests pass in0.18s, including
the existing contest formula, native-only synthetic/pretrained comparisons,
slower-TPU reporting, missing/changed assets and rejection of failed/incomplete
reports (`native-only-scoring-v8b.log`). These test fixtures are not TPU
measurements. The running queue retains its original immutable source snapshot.

The historical original-layout native tuning grid also finished all48 numerical
attempts (six cases,1/2/4/8 threads and two scaling modes), with every child
exiting0/reaped and unchanged archived sources. Its old runner lacks the per-run
source and loaded-library before/after hashes required by the current scorer;
all48 remain ineligible for scored speed comparisons. The raw evidence is
unchanged (`native-original-grid-v8.json`). None of its reported best medians
beats the eligible selections already preserved in `native-fastest-v1.json`.
Outer session52016 exited0 and was reaped. No repeat of this slower historical
grid is needed for the queued comparisons.

The updated275-file release-v8 preserves the direct-native scorer and regenerated
P5 HTML/PDF handout. Its archive SHA256 is
`21b3db175471be8cc613c2b3d7083f4a4d49741a32fbea787b3fd8484a8e7506`.
All archive bytes matched the build checkout and extracted files, and all35
local links resolved. Four packaging tests passed in4.68s. The extracted scorer
loads its native-only CLI and rejects a real CROSS CPU report supplied as TPU,
with exit1 and no result JSON (`release-v8/verification.json`). No TPU was
initialized by that negative check, and no new encrypted arithmetic is claimed
for this scorer/documentation update. The prior release-v6 remains preserved.

## Rotation-cache capacity and strongest matching CPU controls

CNN case5 had zero rotation-key state hits at a256MiB cache budget. The same
frozen workload with512MiB passed one warmup and three measured requests:
median59.79810s versus60.93955s at256MiB. It retained375,140,020 bytes (357.76MiB)
and recorded150 state hits instead of zero. Session88862 exited0 and was reaped.
The initial readback caught a thread-environment difference: this run used
OMP1/MKL1, while the256MiB run used OMP4/MKL unset. The failed equality check
is preserved in `cache512-readback-initial-v9.json`. The observed ratio is1.0191x;
it cannot be attributed solely to cache capacity. A fresh512MiB attempt with
the original thread settings also passed in `cpu-cache512-case5-matched-v9/`:
median59.83494s, the same375,140,020-byte retained-cache peak and150 state hits.
Session11677 exited0 and was reaped. `cache512-comparison-v9.json` independently
verifies all three attempts and the restored thread settings. The matched
recorded-control ratio is1.0185x; different observation times and other active
jobs still prevent treating it as an isolated statistical speedup. Both512MiB
attempts and the failed initial equality check remain preserved.
Retained cache size is not total process/device memory, and no TPU cache benefit
is established yet.

`cpu-fastest-matched-v9.json` selects the fastest recorded eligible CROSS CPU
control matching each queued tiled schedule and security profile. It verifies
raw warm/measured gates, complete collector/reaping receipts and archived source
bytes, retaining23 eligible reports and12 rejected reports. Case1–6 medians are
4.31191,6.82025,6.37765,28.96278,59.79810 and102.67630s. Cases1/3/4 use the earlier
eight-core allocation; cases2/5/6 use four cores. Different cache controls and
runtime revisions are recorded explicitly; unavailable historical thread
metadata is recorded as null rather than inferred. This selection avoids overstating
TPU speedup by using slower available CPU controls; it is not one continuous
suite or a claim of globally optimal CPU performance. The separate tuned native
CPU baseline remains required for broader CPU-system claims.

A public check of the queued one-image pretrained plan confirms that its exact
WISE schedule and actual Q/P profile match the completed CPU calibration,
despite different frozen runtime revisions. It matches the independent original
checkpoint oracle to6.21725e-15 (`queued-pretrained-plan-v9.json`,10.59632s;
session36441 exited0/reaped). It generated no encryption contexts and ran no
TPU work. The eventual comparison must still pass encrypted TPU gates and
disclose its execution controls.
