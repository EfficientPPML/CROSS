# Author validation

## PR #6 follow-up review — updated 2026-11-03 (in progress)

Reviewed merged PR `EfficientPPML/CROSS#6`, merge commit
`98ca6833397069e4228c265d75fef64884218a28`, against the five issues in the
supplied `issue.md`. **The PR is a partial resolution, not a competition
readiness sign-off.** Its own description accurately lists the unresolved
transformer references and five-minute evaluation target.

| Reported issue | Evidence and current disposition |
| --- | --- |
| 1. P1 standalone key mismatch | Fixed in PR #6: the lazy client context and evaluator receive the same service-owned pair. Key-lifecycle regressions, all six reduced-ring local fixtures, and the real upstream student-profile suite pass. Results and their source snapshots are recorded below. |
| 2. P2/P4 encrypted goldens and calibration | **Unresolved.** Both goldens remain plaintext oracles. The current frozen twelve-case TPU sweep has passed optimized Qwen test1 (cold runner 1431.57s, external process 1438.63s), BERT test1 (cold runner 2322.02s, external process 2329.45s) BERT test2 (cold runner 2528.87s, external process 2536.06s), and Qwen test2 with three genuine refreshes (cold runner 6870.74s, external process 6877.97s). A separate historical snapshot passed Qwen test2 with three genuine refreshes (cold runner 12125.28s). These validate individual public toy fixtures; they do not establish a controlled speedup. Complete six-case sweeps, full-checkpoint/file-stage integrations and scoring calibration remain unfinished. The historical native audits found approximation-domain failures in every tested BERT input and Qwen context. A training-derived BERT candidate now passes all 3,668 training trajectories and all nine held-out ranked accuracy checks, including 345/408 correct, with fixed-domain coverage. This is a plaintext arithmetic result; encrypted composition remains unfinished. Qwen's unchanged approximate candidate passes all 416 calibration contexts but fails public held-out validation: 22 trajectories stop outside fixed SiLU domains and three subsequent free-running entries remain unmeasured. Full encrypted model integrations and calibration are still unresolved. Runners distinguish plaintext oracles, accept explicit implementations, enforce a requested TPU, validate repeats and report cold/total timings through cleanup. See the scoped transformer evidence below. |
| 3. Actual-ring 128-bit enforcement | Implemented in all four local runners and P1 stage 3 in PR #6. Current regression tests cover live full Q/P chains and ciphertext/context binding. This is an execution contract, not isolation from arbitrary contestant Python. |
| 4. P3 TPU aborts at batch 2/4 | **All six original-profile cases have passed across separate historical snapshots.** Cases 1–4 passed in a four-case TPU sweep, each in a fresh process with unchanged source hashes; LeNet batch-2/4 errors were 3.77e-13 / 8.84e-13. That sweep took 2225.23s (`validation/p3_legacy_20261102/`). Test5/test6 passed earlier, with runner totals 661.50s / 501.43s (`validation/p3_review.json`). A complete six-case sweep against the final promoted sources is still pending; the shorter compact calibration does not replace it. |
| 5. About five minutes per full evaluation | **Both shorter schedules revalidated after runtime promotion.** P1 student-v2 ranks C3/C4 and completed its full upstream harness schedule in 257.56s, including three repeats per case, archiving, cleanup and calibration publication. Both 100-image cases passed on one/eight chips; all 103 measured files stayed unchanged. P3 compact-v1 completed both smaller CNNs in 145.00s, including setup, warmups, three repeats per model and cleanup; all 119 measured files stayed unchanged. Matching new calibrations are archived below. The six-case legacy schedule and pretrained transformer workloads are outside these five-minute results. |

Additional correctness problems found during this review:

- P1 no longer silently substitutes fewer devices when a case requests more
  than the backend has. Missing hardware now produces an explicit failure.
- Submission module loading now follows Python import semantics, supporting
  dataclasses and postponed annotations while isolating different file paths.
- P1 scoring rejects invalid timings/quality counts, incomplete own-model
  comparisons, mismatched archive identity and failed baseline calibration.
  Exact labeled archives cannot fall back to another run. The student-v2
  scorer additionally requires the complete successful evaluation and immutable
  calibration proof; legacy/pre-v2 archives cannot satisfy that requirement.
  The fresh baseline self-score gives 30 execution points and zero speed points.
- Daemon cleanup checks process identity instead of trusting a stale PID file,
  leaves libtpu's global lock alone, bounds shutdown, and waits for exit before
  another service starts. Readiness requires a live process and listening
  socket; incomplete client requests have a bounded read timeout.
- Transformer diagnostics, timing and shared helper modules are preserved in
  submission/release packages. Plaintext outputs still fail ciphertext gates.
- P3 prepared mappings are borrowed by the per-case runner and released once
  by their suite owner. Local runs release their own mapping. Cleanup failures
  preserve the primary evaluation error and do not skip other owned mappings;
  interrupted preparation also releases already-created mappings. The parent
  records failed evaluations on termination or launch failure and stops owned
  descendant processes, including helpers left after a worker exits. A focused
  58-test CPU timing/lifecycle/scoring run passed after these changes. Historical
  compact TPU timings below retain their original source snapshots.

The earlier P1 run repeated the first 266.37-second evaluation after evaluator
hardening and finished in 262.14 seconds externally. Its original report and
source hashes remain unchanged in `validation/p1_student_pre_v2/`. The later
student-v2 run completed in **260.3756 seconds externally** (260.3016 seconds
inside the evaluator), with both cases passing, cleanup complete and 103 exact
source snapshots unchanged. Its matching model/input/device/software identities,
case archives and published immutable calibration passed the strengthened
scorer. This historical calibration is retained unchanged; the later post-promotion calibration below is now current.

Current validation: `make -C cadathlon all reference-test` passed, including
all 24 plaintext fixture checks (not encrypted transformer tests). A focused
CPU regression run passed **507 tests and 88 subtests**, with **6 skips**, covering CADathlon helpers, key generation, polynomials, Mapping,
BSGS, codec preparation, packing, bounded prime generation and demo contracts.
The skips include accelerator-only and explicitly opt-in encrypted integration
checks; separate real TPU results are identified in the artifacts above. Follow-up changes and accelerator measurements will
be recorded separately; these counts must not be presented as full TPU or
full-contest coverage.

A later CPU checkpoint passed **538 tests and 64 subtests**, with **8 skips**,
in 210.52 seconds. It covers the CADathlon tests plus the changed codec,
packing, Mapping, BSGS, prime-generation and bootstrap suites, including the
small N256 functional bootstrap checks. This is not a secure-ring TPU refresh
result. A subsequent focused run passed **43 tests** after synchronization
was changed to wait on ciphertext arrays without retaining NTT contexts in
the Polynomial registry. The transformer runner also releases each completed
request's ciphertexts before allocating the next request. These changes have
memory-lifetime regressions; the earlier P1/P3 timing artifacts retain their
original source snapshots and are not silently relabeled as later-source runs.

A subsequent broad CPU checkpoint passed **606 tests and 64 subtests**, with
**10 skips**, in **230.58 seconds** (233.04 seconds including process overhead).
It covered all `cadathlon/tests` and the changed CKKS context, codec, Mapping,
BSGS, packing, composite-prime and bootstrap suites on CPU cores 148–151 with
`JAX_PLATFORMS=cpu`. All **203 recorded Python source files** were unchanged
between its before/after snapshots. Six opt-in secure polynomial/refresh
checks, two opt-in large bootstrap checks and two inapplicable zero-diagonal
subset cases were skipped; these are not encrypted transformer or TPU passes.
The [checkpoint report](validation/cpu_checkpoint_20261102/report.json) records
the exact command, environment, package versions and individual skip reasons,
alongside raw pytest/JUnit output, resource data and source hashes.

After integrating the guarded uniform-scalar encoder, rotation binding reuse,
tensor-only bootstrap primitive and experimental bounded transformer adapters,
a fresh working-checkout CPU run passed **929 tests and 64 subtests**, with
**10 skips**, in **274.14 seconds** (276.64 seconds including process overhead).
All **302 recorded source and fixture files** were unchanged. The
[integration checkpoint](validation/cpu_integration_20261102/report.json)
includes the exact source archive, raw output, JUnit results and environment.
It also covers the later packaging, cancellation and source-provenance fixes;
the skips retain their explicit accelerator/large-integration scope.

The separate [combined secure CPU checks](validation/combined_runtime_cpu_validation.json)
retain an initial isolated snapshot with **375 passing tests, nine missing-resource
failures and two skips**. The affected subset passed **155 tests** after only
the missing test resources were restored; these overlapping counts are not
added. Real secure polynomial, codec, rotation, tensor and bounded-metadata
checks passed in the original snapshot. Two complete N256 bootstrap comparisons
were bit-exact in payload, scale and noise-scale degree; that intentionally
insecure arithmetic fixture is not a secure-ring or TPU refresh result.

A later [isolated three-change candidate](validation/combined_runtime_candidate_cpu_20261103/README.md)
passed **326 tests and 68 subtests**, with **7 skips**, in **167.18 seconds**;
all 171 captured files stayed unchanged. It combines giant-step queue scheduling,
native key-generation allocation changes and opt-in scoped NTT reuse, with
packaging/ownership regressions and all twelve ideal graph contracts. Separate
real secure linear and two-lane/multi-level rotation checks matched baseline
ciphertext payloads and metadata exactly. The optional second-device guard
passed in a fresh two-device CPU process. Its patch and exact sources are
archived for review. It was unapplied at that checkpoint and was later included
in the reviewed runtime promotion below. The component checkpoint adds no
full-model, accelerator or HBM claim.

A fresh [integration of all five candidate patches](validation/all_candidates_cpu_20261103/README.md)
passed **1,131 tests and 93 subtests**, with **11 skips**, in **357.04 seconds**
(360.00 seconds externally). All **320 recorded source and fixture files**
stayed unchanged, and each exact source-archive member was independently
verified. The combined 28-file patch includes the queue/key-allocation changes,
optional HERot/HEMul NTT reuse, guarded dispatch-cache policy, model opt-in
wiring and owned-key-directory collector fixes. It reconstructs every changed
source from archived base bytes. This was an isolated, unapplied candidate
at that checkpoint. After the Qwen proof exited, it was included in the later
runtime promotion below; Qwen retains its original frozen source evidence. The explicit skips
and earlier component proofs are recorded separately; this adds no complete
encrypted transformer, TPU performance or scoring-calibration claim.

The [runtime promotion](validation/runtime_promotion_20261103/promotion.json)
applied the reviewed 32-file union after the Qwen process exited and released
its source freeze. Every changed file matched the exact tested bytes. The
[final integration](validation/runtime_followup_integration_cpu_20261103/summary.json)
passed **362 tests and 9 subtests**, with **4 explicit skips**, plus two
separately passing device guards. It includes facade and nested rescale NTT
reuse and serialization-safe deferred BConv construction. The separate
[collector follow-up](validation/transformer_collector_outer_failure_cpu_20261103/summary.json)
passed **90 tests**, preserving primary cancellations and truthful failed
metadata when initial inspection or final publication fails. The broader
lazy-BConv checkpoint passed **1,146 tests and 93 subtests**, with 12 skips.
These are CPU/component results; ring profiles and bootstrap arithmetic were
not changed by this promotion.

Fresh complete TPU evaluations then passed against those promoted sources:
P1 student-v2 took **257.561762s externally**
(**257.467304s internally**), with all 103 source files unchanged,
96/100 accuracy and complete own-cleartext agreement for both C3/C4 cases.
Its evaluation ID is `9250e83535a84d98a3918c629d64380c` and its baseline self-score is
**30 execution points, zero speed points**. The [P1 archive](validation/p1_student_post_promotion_20261103/README.md)
retains exact sources, aggregate proofs, process status and raw logs.
P3 compact-v1 took **145.001872s externally**
(**144.906203s internally**), with all 119 source/fixture files
unchanged. Both models passed, with maximum errors **1.39457e-08** and
**1.22242e-08**, exact top-1 agreement and server medians
**0.02108278s** / **0.01956712s**. Its self-score is **10/20**,
with zero speed bonus. The [P3 archive](validation/p3_compact_post_promotion_20261103/process.json)
preserves the separate current calibration. All owned evaluation processes
exited zero and were reaped, and cleanup completed.

These runs held the TPU exclusively and used CPU affinity 0–127 while an
independent encrypted CPU routing proof continued on cores 144–147. Their
recorded conditions differ from the older calibration's CPU allocation;
the new elapsed times establish the updated schedules' budgets, rather than
an isolated optimization speedup. Historical reports and the prior release
bytes remain unchanged. Full BERT/Qwen six-case encrypted sweeps,
pretrained/file-stage integration and full-workload calibration remain open.

Environment for this review: Python 3.13.18, JAX/jaxlib 0.12.0.1,
NumPy 2.5.4, PyTorch 2.15.0+cpu. This differs from the originally reported
author environment. The source is `/home/jianming_gatech/CROSS` on branch
`fix/competition-readiness`; the older `CROSS_dev` checkout was not modified.

A later local release checkpoint contains 207 files, including the packed
experimental adapters and their shared dependencies. All 24 relative HTML
links resolve inside the extracted archive, and all four extracted problem
sets pass their build and plaintext-reference targets with `CROSS_ROOT`
pointing to the frozen core checkout. See the [archive checksum and raw checks](validation/release_checkpoint_20261102.json).
This is a packaging checkpoint, not encrypted transformer validation; nothing
was published or uploaded.

The pre-promotion release checkpoint after the earlier recalibrations and
P2 staging repair contains **222 files**. It includes the bounded adapters, P1's matching
student-v2 manifest and nine immutable proof files, and P3's new compact
calibration. All **24 local HTML links** resolve in the extracted archive,
every archived file matches the checkout, and all four extracted build and
plaintext-reference targets pass. See the
[release checksum and raw checks](validation/release_integration_20261103.json).
The P2 staging repair separately passed **39 CPU tests**, including rejection
of altered helper snapshots and mixed checkout/staged imports; its
[exact sources and results](validation/p2_staging_cpu/report.json) do not claim
encrypted file-stage or full-model execution. A separate
[check from the extracted release](validation/p2_released_staging_20261103.json)
successfully staged the pinned unchanged harness and verified every copied
helper. All three experimental adapters loaded through their matching staged
runner without initializing JAX and correctly reported the five missing
full-model file stages.

The first post-promotion local release contained **234 files**, including the three
new optimized-runtime modules, the latest matching P1/P3 calibrations and both
historical/current immutable P1 proof directories. The current P1 proof still
contains exactly nine aggregate files. Every archive member matches the
checkout, all **24 local HTML links** resolve, and all four extracted build
and plaintext-reference targets pass. The
[release check](validation/release_post_promotion_20261103.json) records archive
SHA256 `c3cc40a2ad3a7203572471e5011a1e6191db40aa2379d64a23301cf605f2d1b7`
and matching calibration identities. A separate
[released P2 staging check](validation/p2_released_staging_post_promotion_20261103.json)
verified the pinned harness and helper bytes, and all four experimental
adapters—including the optimized one—loaded without JAX initialization and
correctly reported the five missing full-model file stages. Packaging success
adds no encrypted transformer or pretrained-model claim. Nothing was uploaded
or externally published.

The preceding local release contains **235 files**, adding the tested reusable
streaming-linear helper. Its [exact release verification](validation/release_streaming_linear_20261103.json)
passed all file, calibration identity, local-link and extracted build/reference
checks. The separate [released P2 staging check](validation/p2_released_staging_streaming_linear_20261103.json)
also passed. The measured P1/P3 sources are unchanged. This remains a local
package checkpoint; the five full-model file stages are unfinished.

The latest local release contains **237 files**, adding the reviewed integer-affine
and static scaled-Newton helpers. Its SHA256 is
`a24f256b5c7e42ba4257028a26d51971a5dc5b37a5ca5a2a157a79b4102d3e71`.
The [release check](validation/release_integer_newton_20261103.json) verifies
all exported bytes,24 local links, both new imports without JAX initialization,
and all four extracted build/plaintext-reference targets. The
[released P2 staging check](validation/p2_released_staging_integer_newton_20261103.json)
also passes. The [promotion record](validation/integer_newton_promotion_cpu_20261103/PROMOTION.md)
binds88 focused tests and the current secure two-update regression; it does not
relabel the earlier thirteen-update experiment. All103 P1 and119 P3 measured
source files are unchanged, preserving their existing calibration identities.
Full transformer file-stage implementation and calibration remain unfinished;
the archive has not been externally published.

P1 follow-up artifacts: `validation/p1_student_profile.json` contains all six
TPU fixture results and trained 100-image checks on one/eight chips;
`validation/p1_student_suite.json` contains the full 257.56-second post-promotion
student-v2 C3/C4 harness evaluation and self-score. Exact source bytes, raw logs,
complete reports and external timing are preserved in
`validation/p1_student_post_promotion_20261103/`. The earlier 260.38-second
calibration remains in `validation/p1_student_v2_baseline_20261102/`. Its separate
`src/suite/baseline_manifest_student.json` gives 30 execution points and zero
speed points when scoring that baseline against itself, with the matching
proof under `src/suite/calibration/student-baseline/`. These are real
encrypted results, not plaintext-fixture checks.

P3 follow-up evidence is in `validation/p3_review.json`. The encoder's
production N32768 output remained bit-exact while its median component time
fell from 0.17458s to 0.05924s. An opt-in Mapping path passes immutable model/key
arrays as resident operands instead of embedding huge HLO literals. P3
student test6 passed at N16384 (total 263.54s, max error 1.95e-8), but a
concurrent two-full-model evaluation exceeded the 300s budget. The explicitly smaller
compact-v1 competition models subsequently passed their entire evaluation in
153.01 seconds (peak host RSS 10.75GiB, peak reported TPU allocation 1.42GiB).
The unchanged original models are not advertised as meeting five minutes. Invalid
composite-prime requests that previously looped forever now fail explicitly.

After the mapping-ownership, cancellation and cleanup fixes, a fresh complete
compact-v1 baseline passed in **148.35 seconds externally** (**148.25 seconds**
inside the controller). Both models passed their real encrypted gates, with
maximum logit errors **6.26e-9** and **2.17e-8**, exact top-1 agreement, and server
medians **0.02112s** and **0.01966s**. All **119 recorded source/fixture files**
were unchanged and are preserved in the
[follow-up archive](validation/p3_compact_followup_20261102/process.json).
That historical baseline self-score is **10/20**, with zero speed bonus.
The current `validation/p3_compact_baseline.json` now contains the later
post-promotion report described below; the prior 153.01-second calibration
is preserved byte-for-byte as
`validation/p3_compact_baseline_pre_lifecycle_followup.json`.

The later original-profile P3 sweep used `--baseline --warmups 0 --repeats 1`
in four sequential fresh processes. Every case passed its numerical, argmax
and actual-ring security gates at N32768. Core, demo, runner and shared-security
source hashes were unchanged throughout this sweep; the parent exited zero
after 2225.23 seconds. These original runs retain the default captured-constant
compiler path, rather than opting into the compact schedule's setup profile.

| Original case | Model / batch | Whole child process | Maximum logit error |
| --- | --- | ---: | ---: |
| test1 | LeNet / 1 | 548.31s | 1.49e-13 |
| test2 | LeNet / 2 | 603.69s | 3.77e-13 |
| test3 | LeNet / 4 | 619.34s | 8.84e-13 |
| test4 | AlexNet-tiny / 1 | 453.82s | 4.03e-13 |

The [four-case report](validation/p3_legacy_20261102/report.json) retains
per-case results, raw stdout/stderr, GNU-time resource records, source hashes
and an artifact checksum manifest. Earlier original test5/test6 outcomes in
`validation/p3_review.json` retain their original measurements and provenance;
they are not relabeled as runs against this later frozen source snapshot.
The review therefore has direct encrypted results for all six original cases
across the recorded runs, not a single final-source six-case timing sweep.
The five-minute claim applies to the separately measured compact-v1 schedule.

The separate [transformer sweep collector](validation/check_transformer_suite.py)
now runs the six public BERT or Qwen fixtures in sequential fresh TPU processes.
It retains raw stdout/stderr, per-process elapsed time, request samples, package
versions and before/after source hashes. It rejects stale output directories,
source changes, inconsistent timing records and CPU results. Partial sweeps
remain incomplete; even six passing synthetic fixtures do not certify the
pretrained models or establish scoring calibration. The default measurement
schedule uses one warmup and three repeats per case. No timeout is imposed
unless an explicit per-case deadline is supplied.

Its 39 focused CPU tests include real dummy subprocesses for timeout,
SIGTERM cleanup, and a child that exits while leaving a helper alive. These
tests validate collection and cleanup only, not encrypted model execution.
For an explicitly selected implementation, collect a new sweep with:

```sh
python cadathlon/validation/check_transformer_suite.py --model bert \
  --implementation /path/to/encrypted_implementation.py \
  --output /new/path/bert-public-tpu-sweep
```

Use `--model qwen` for P4. `--cases 1` is a development measurement and exits
nonzero because the six-case requirement is incomplete. Existing experimental
adapters still require their separately documented validation; the collector
does not bypass a circuit's preflight or original correctness gates.

## Issue repair validation — 2026-10-31 / 2026-11-01

This repair addresses the standalone P1 key mismatch, actual-ring security
gates, and the TPU inverse-NTT fusion failure. It also reduces redundant
key-generation work and releases Mapping resources between runs. It does
**not** establish encrypted BERT/Qwen references or a five-minute complete
C1–C6 evaluation.

### Causes and changes

- P1 constructed its client codec and server Mapping lazily with `keys=None`
  twice. Each call generated a different key pair. `BaselineService` now owns
  one lazily generated pair and passes it to both; externally supplied suite
  keys are preserved.
- P2 and P4 `.golden` files are plaintext numerical oracles, not incomplete
  adapters around an existing encrypted implementation. Their six numerical
  reference cases each pass exactly. Encrypted attention, nonlinear
  approximations, normalization, and depth/refresh planning are still missing.
  Removing the ciphertext check would conceal this gap. P2's measured CROSS
  baseline is absent; P4's 60-second scoring target is a constant, not a
  measurement. No encrypted transformer baseline is claimed by this repair.
- All four local runners now call `he_params.qp_is_secure` on live contexts'
  complete Q and P chains and bind every ciphertext to a checked context.
  P1 stage 3 checks both its client and lazily built evaluator, before marking
  the service ready, and records those actual rings in its manifest. Declared
  profiles and reduced output moduli cannot substitute for a full-ring check.
  These checks do not sandbox hostile contestant Python; independent judging
  and evaluator audit remain necessary.
- TPU fusion combined the batch/component reshape feeding the inverse
  NTT with its bitcast/convolution, producing an invalid scaled-dimension
  window stride. An `optimization_barrier` keeps that boundary explicit.
- `Mapping.release()` now drops executables, bindings, owned context caches,
  and failed-construction state. JAX's process-wide tracing caches are cleared
  as documented in the API; unrelated JIT functions may retrace.
- P1 stage 3 now accepts an optional `warmup_input()` hook. The baseline
  encrypts one zero image and repeats it across one full physical lane group;
  this warms its exact fixed Mapping shape once instead of encrypting and
  executing 1000 dummy images. Other services retain full-batch warmup unless
  they implement the hook. Input/output security gates still run and ranked
  requests still use every image with fresh encryption.
- Encoder-owned BSGS codec caches are evicted on release and preprocessing
  failure. Immutable public FFT root and rotation tables are cached with
  bounded capacity instead of being rebuilt for each diagonal; their numeric
  construction is unchanged.
- Native key switching unnecessarily inverted and retransformed every Q
  secret limb, although its input was already in evaluation form. It now
  reuses Q directly and only inverts q0 to center-lift into P. The scalar
  arbitrary-precision path remains as an independent exact oracle. Randomness,
  distributions, modulus chains and key-generation version are unchanged.
- Submission archives now preserve the problem/common directory hierarchy
  and include shared validation and the P1 suite configuration, without
  including runtime keys or daemon artifacts.

### Measurements and limits

Initial validation uses Python 3.13.17 and JAX/jaxlib 0.12.0 on the available
eight-device TPU host. Additional isolated checks use the report's compiler
packages (JAX/jaxlib 0.11.2, libtpu 0.0.48), but Python, NumPy and PyTorch
versions still differ from the full author environment. See the exact metadata
and results in `validation/issue_repair.json`.

- P1 `runner.py --case ../test/test1.in --baseline --warmups 0 --repeats 1`:
  passed, max absolute error `1.2703726959273354e-13`, top-1 agreement 1.0.
  Cold end-to-end time was 116.294 s; first server call was 114.670 s and
  included lazy Mapping construction. Constructor-only `setup_s` was 0.527 s
  and must not be interpreted as total setup time.
- Latest-code P1 eight-device warmup: a service configured for batch 1000
  produced eight warmup ciphertexts, passed input and output security checks,
  and decrypted all eight outputs correctly (max absolute error `1.2671e-13`,
  top-1 agreement 1.0). Construction/encryption/execution/decryption took
  120.423 s; whole process including teardown took 130.01 s. This validates
  the warmup hook; it does not measure a ranked 1000-image request.
- P3 `runner.py --case ../test/test2.in --baseline --warmups 0 --repeats 1`
  passed end to end at batch 2: max absolute error `2.7606e-13`, top-1
  agreement 1.0. Setup was 906.866 s, cold request 2.244 s (server 1.045 s),
  process elapsed 928.40 s, and peak host RSS 134,190,156 KiB. This run started
  before the later FFT-table and encoder-cache refinements. It proves the
  repaired batch-2 path, and also demonstrates the five-minute target is not
  met. It is not a calibrated scoring baseline or a same-environment speedup.
- P3 test3 passed end to end at batch 4 on the reported compiler packages
  using the latest code: max absolute error `6.3052e-13`, top-1 agreement 1.0.
  Setup was 1019.792 s, request 3.907 s (server 1.765 s), runner total including
  cleanup 1034.328 s, process elapsed 1043.25 s, and peak host RSS
  134,091,316 KiB. Full CNN reruns in this repair cover test2/test3; the other
  four P3 public cases were not rerun.
- A sensitive compiler reproducer on JAX/jaxlib 0.11.2 and libtpu 0.0.48:
  original `Polynomial.to_coeffs_form` aborts with exit 134 at N=32768,
  20 towers, batch 2 with exactly `(1 vs. 4) Scaled dimension stride is not
  equal to window bound`; batch 4 fails with `(1 vs. 8)`. The patched version
  returns bit-exact coefficients. The committed five-test class now includes
  the full 20-tower production shapes at batches 2/4 and passes in 32.34 s,
  together with the initial eight-device case. The eight-device case was then
  strengthened to 20 towers too and passed separately in 23.32 s (global
  batch 32, four samples per device). Earlier two-tower checks passed with
  both versions and missed the layout-sensitive bug; the production tests
  were upgraded rather than relying on that inadequate smoke shape.
- A CPU rotation-key microbenchmark at the P1 ring (N=16384, eight Q towers,
  four P towers, dnum=2), with identical fixed samples and one excluded warm-up,
  matched every output bit. Median of three runs: 0.09510 s before and
  0.06895 s after. This measures one component, not full-suite runtime.
- CPU regression checks under the reported compiler packages: 157 passed,
  three TPU-only cases skipped, 41 subtests passed (31.30 s), covering
  baseline lifecycle/warmup/timing, actual-ring gates, packaging, key
  generation, polynomial/Mapping compilation, BSGS and demo API contracts.
  The separate codec suites passed 43 tests and 31 subtests.
- All 24 public plaintext reference checks passed. This is fixture validation,
  not evidence of encrypted BERT/Qwen completion.

The five-minute target remains open: the ranked suite starts new stages and
server processes for C1–C6; model-dependent constants, evaluation keys, and
XLA compilation are repeatedly prepared. The setup optimization and lifecycle
repair here do not justify extrapolating a complete-suite time. Preserve the
ranked workload, security parameters, and timing boundaries when measuring a
future solution; a smaller smoke case is not a replacement benchmark. The
shipped historical P1 manifest totals 1183.23 seconds of setup across C1–C6
before requests. Setup is outside the harness repeat loop, so reducing repeats
alone cannot remove this cost. P1/P3 runner reports now include warmup times,
a cold-start flag, and total time through cleanup while retaining the existing
measurement fields.

## Problem 4 description clarification — 2026-09-21

The title, introduction, milestones, scoring explanation, README, oneliner
and report now state the intended sequence: use or improve CROSS to enable
encrypted BERT inference on TPUs, then optimize benchmark performance and
assignment points. The introduction explicitly identifies CROSS as the
implementation library and `fhe-benchmarking/BERT` as the evaluation benchmark.
The scoring section explains how benchmark measurements feed the CADathlon
rubric. Permission to use AI agents remains explicit.

Rebuilt the HTML and PDF. All 131 Markdown content elements survive HTML
rendering. Both repository links work in both formats; the two repository
roles, TPU objective and AI-agent permission appear on the PDF's first page.
The revised PDF has eight pages with no out-of-page text. Offline browser
checks passed at 390 and 1512 pixels in dark/light themes, without page
overflow or JavaScript errors. See `validation/bert_description_check.json`.
The contestant and BERT submission archives were refreshed. This was a
description update; numerical and encrypted-inference tests were not rerun.

## Problem 4 added — 2026-09-21

`agent_pset/p2_bert` follows the existing template and tutorial visual style, with the
five-section handout, source TODOs, an explicitly plaintext `.golden`, six
synthetic input/output pairs, report template, `oneliner`, `EXCLUDE` and local
build/test/submission targets. **Students' permission to use AI agents is
explicit in both the HTML and PDF**, as well as the README and report.

| Check | Result |
|---|---|
| Build and public fixtures | `make all` and all 24 problem-set reference checks passed; the six new BERT `.out` files match exactly. Fixture hashes are in `validation/bert_fixture_sha256.json`. |
| Independent BERT semantics | `common/check_upstream_bert.py` passed all six cases against Transformers 4.51.3 in float64, using strict loading of every synthetic parameter. Largest discrepancy across embeddings, first block, final encoder and logits was 1.999e-15. Perturbing masked padding left predictions unchanged. Raw records: `validation/bert_upstream.jsonl`. |
| Pinned harness | The eleven source hashes match upstream commit `170bfe567545d74d0fad785a052518351d37bc93`. Staging preserves those bytes and the Apache license, includes seven usable stage CLIs, and refuses an existing destination. |
| Strict benchmark outputs | Synthetic contract checks passed for counts 1/100/200/408, input order and the label-1 tie rule. Missing/duplicate/extra results, nonfinite or wrong-shaped logits, incorrect predictions and nonbinary labels were rejected. The 75% and two-percentage-point accuracy boundaries were checked. See `validation/bert_contract_check.json`. |
| Client preprocessing | The pinned real tokenizer processed a synthetic sentence pair to 128 IDs, masks and token types. Postprocessing and the strict checker accepted deliberately supplied synthetic logits; no model inference is implied by that adapter check. |
| Intended TODO behavior | With `JAX_PLATFORMS=cpu`, the contestant runner exits nonzero with `TODO: implement the CROSS BERT session`. No encrypted BERT implementation or TPU measurement is claimed. |
| HTML/PDF | `make docs` rebuilt all four handouts. BERT has seven PDF pages, with no out-of-page text; the AI permission and benchmark hyperlink are present. Offline Chromium checks passed at 390 and 1512 pixels in dark/light themes, with working mobile navigation, no page overflow or JavaScript errors. All 120 Markdown content elements survive rendering, and internal navigation/PDF links resolve. See `validation/bert_html_check.json` and `validation/bert_pdf_check.json`. |
| Packaging | BERT submission creation and the 135-file contestant release passed archive-content checks. Goldens/caches are excluded, each archived file matches the workspace, and all six BERT references pass from an extracted release. See `validation/bert_packaging_check.json`. |

The new track is connected to the root Makefile, fixture generator, shared
renderer/navigation and release packager. Full pretrained MRPC inference and
encrypted BERT were not run; model weights and the MRPC dataset were not
downloaded. The tokenizer-only check did download its small pinned assets.
The 100-point rubric, full-width operator tolerances, resource limits and
CROSS performance baselines require organizer preflight as documented in
`ORGANIZER.md` before scored use. The upstream DESILO example is not a CROSS
implementation or a valid substitute for its benchmark baseline.

## Tutorial style update — 2026-09-20

Both HTML and PDF handouts now follow `CROSS_Tutorial/index.html` and its
stylesheet: Inter/Outfit typography, the dark blue-black canvas, purple/pink
accents, sidebar navigation, section index, rounded code blocks and striped
tables. Fonts and their license notices are embedded in the HTML; PDF fonts
are embedded as subsets. The shared renderer rebuilds both formats offline.

Chromium checks passed for both handouts at widths 320, 390, 768, 1024 and
1512 pixels, in dark and light themes. Theme persistence, section anchors,
track/PDF links, desktop navigation toggle, mobile drawer and Escape closing
were checked, with no JavaScript errors or external asset requests. The
rendered content matches every original Markdown heading, paragraph, list,
table cell and code block (133 elements for Qwen, 82 for CNN). The matching
dark PDF palette also applies when printing HTML from light mode.

All PDF pages were checked for out-of-page text, and representative desktop,
mobile and PDF pages were visually reviewed. Qwen remains 7 pages; CNN remains
5 pages. See `validation/style_check.json` and the refreshed
`validation/pdf_check.json`. Problem definitions, starter code and test data
were not changed by this style update. The contestant release archive was
rebuilt with the current HTML/PDF files.

## Problem 3 added — 2026-09-21

`non_agent_pset/p1_benchmark` follows the same template as the other two tracks: `doc/`,
`src/`, `test/`, `README.md`, `oneliner`, `EXCLUDE`, a TODO entrypoint, a
`.golden` reference solution, six `.in`/`.out` pairs and the five template
sections. It is wired into `make all`, `make reference-test`, `make clean`,
`make release`, `common/render_docs.py` and `common/generate_cases.py`.

| Requirement | Evidence and result |
|---|---|
| Scope is further performance improvement | The track optimizes throughput of an already-working CROSS serving path on eight chips, not correctness of a new capability. `make baseline-plan` reports the actual derived plan and prints `slot_occupancy`, the quantity the problem asks teams to improve. |
| Distinct from Problem 2 | P2 is one network per case on one chip, scored on latency. P3 is one network across 1 and 8 chips, scored on throughput `B/T`, with ranked batches 1, 100 and 1000 mirroring the FHE benchmarking suite's single/small/medium instances. Both latency and throughput cases are scored so a batch-only optimization forfeits points. |
| Grounded in CROSS | `make -C non_agent_pset/p1_benchmark/src baseline-plan` produced depth 3, degree 16384, 8192 slots, 4 logical Q moduli, dnum 2, security target 128, operations `matvec, add_plain, square, matvec, add_plain`, and slot occupancy 0.0957 (784 of 8192). |
| Public reference answers | `make -C non_agent_pset/p1_benchmark/src reference-test` passed all six cases with `max_abs_error` 0.0 against the committed `.out` files. Fixtures are labelled `synthetic-arithmetic-not-trained-accuracy`. |
| Real encrypted baseline | `baseline.py` builds an actual `Mapping` and runs encrypt/execute/decrypt through the canonical rank-5 `Polynomial` boundary; the runner rejects any bundle that is not ciphertexts. |
| Intended TODO behavior | `bench_opt.py` raises the documented `NotImplementedError`; `bench_opt.py.golden` is a valid unoptimized solution, excluded from the contestant archive. |
| Packaging | `make all` and `make release` passed. The archive now holds 80 files including 24 p3 files, and contains no `.golden` member. |
| Opaque bundle boundary | `build_service` returns encrypt/execute/decrypt/release rather than a `Mapping`, so contestants may change how images map onto ciphertexts. The runner validates every leaf of the bundle is a rank-5 `Polynomial` and allows integer bookkeeping only. |
| Parameter freedom is bounded | The handout permits a different ring only if `he_params.qp_is_secure` accepts it at 128-bit for the ring actually run, and requires reporting `log2(Q)`, `log2(QP)`, the ceiling and the margin. It warns that logit error grows as the chain shortens and that the absolute gates must be rechecked at the chosen parameters. |

Organizer calibration still required before a scored event: freeze trained
weights and hashes, freeze the public/hidden image splits and the repeat
schedule for batches above 256, and measure judge baselines for all six ranked
cases on the exact hardware. The handout's numbers are proposed defaults.

## Problem 3 wired to the benchmark suite — 2026-09-21

The six ranked cases now run through the FHE benchmarking suite's own harness
(`fhe-benchmarking/ml-inference` at `739378698f`), not a runner written here.
Every claim below was produced by running that harness against a clean clone at
`/tmp/p3suite` with `--seed 3 --num_runs 3`.

| Requirement | Evidence and result |
|---|---|
| Harness is untouched, and changes outside it are disclosed | `make -C non_agent_pset/p1_benchmark/src suite-check` reports `harness/ is byte-identical to the pinned commit` and lists the eight installed stage shims. `install_suite.py` refuses to install otherwise, which is the suite's own submission rule enforced mechanically. It does replace `scripts/get_openfhe.sh` and `scripts/build_task.sh` with no-ops, keeping `.orig` backups, because `harness/utils.py::build_submission` invokes both unconditionally and would otherwise download LibTorch and build OpenFHE for a submission that compiles nothing. That is outside `harness/` and so within the rules, and `suite-check` now prints it rather than leaving it to a diff. |
| The harness actually completes | C1 and C2 both ended with `All steps completed for the single inference!` and exit code 0, writing `measurements/single/results-{1,2,3}.json`. |
| Compilation is outside the measured stage | Before the fix, run 1 stage 7 was 125.66 s against 0.13 s for runs 2 and 3, because XLA compiled inside the measured stage. `service_daemon.py::materialize()` now runs a warm-up encrypt/execute on zeros during stage 3. After: stage 3 139.39 s, stage 7 0.1264/0.1258/0.1250 s, `server_execute` 0.0325/0.0324/0.0319 s. The 126 s compile moved into `server_warmup_s`. |
| Cases that share an instance stay distinct | The suite keys output by instance alone, so C1 and C2 both write `measurements/single/`. `suite/archive.py` copies each case to `src/suite/results/<case>_<label>/`; C1 and C2 archives now hold different numbers for the same instance (1 chip 0.0324 s vs 8 chips 0.0372 s median `server_execute`). |
| Correctness is not self-reported, and not taken on trust from the harness either | Two gaps were found in the pinned harness and closed. `run_submission.py` calls `verify_result` with `check=False`, so a wrong single inference leaves no trace in `results-N.json`; and `calculate_quality` divides by the number of predictions written, so a truncated output reports high accuracy over skipped images. `archive.py` recomputes both from the harness's own label and prediction files and records the verdict; C1 and C2 archived `PASS, 1/1 correct` and C3 `PASS, 96/100 correct`. It also copies those files into `results/<case>_<label>/evidence/`, because the next case's key generation wipes `io/<instance>/` — observed directly when C4's stage 2.2 removed C3's predictions. `score.py` awards zero without that verdict — verified by scoring C1's pre-fix archive, which returned `passed: 0` with reason `no frozen harness verdict`. |
| The accuracy gate compares the right two things | The suite's cleartext reference is `harness/mnist/mnist_ffnn_model.pth`, a 784-128-64-10 network, while the ranked model is 784-50-10 with `x**2`. Gating on the harness's `Harness model quality` therefore measures a training difference, not HE fidelity. Measured on the 100 small-instance images: the ranked model scores **96/100 in cleartext and 96/100 encrypted** — the HE path costs nothing — while the harness network scores 97/100. The earlier arrangement would have failed the unmodified baseline twice over: top-1 agreement read 0.98 against a 0.99 gate, and `(0.97 - 0.96) * 100` is `1.0000000000000009`, tripping the 1 pp gate on a binary floating-point artifact. `archive.py` now runs the ranked model in cleartext on the harness's own pixel file, in organizer-owned code, and `score.py` gates against that; both harness-model figures are retained as context only. |
| Gate arithmetic is tested | `make -C non_agent_pset/p1_benchmark/src suite-gates` runs `suite/test_gates.py`: 14/14 gate behaviors correct, covering a healthy case, a short prediction file, a missing verdict, a FAIL verdict, an unavailable verdict, low agreement against the model's own cleartext run, an unavailable cleartext check, a drop of exactly 1 pp (which must pass), a drop beyond it, the measured case where the harness's larger network is 1 pp better (which must not count against the submission), harness-model disagreement being ignored entirely, setup over budget, batch over budget, and the single instance that legitimately has no `Quality` block. It needs no accelerator or suite checkout and runs as part of `make all`. |
| Harness timing semantics documented rather than assumed | Stages 1, 2.2 and 3 execute once outside the run loop, so every `results-N.json` repeats one setup measurement and every `Total` carries it; only stages 4–10 are per-run. The scorer names these fields `one_time_setup_s` and `total_s_including_one_time_setup` rather than implying per-run figures, and ranks on stage 7. |
| The three runs are not three samples | `run_submission.py` re-creates `np.random.default_rng(seed)` **inside** the run loop, so all runs derive the same `generate_input` seed and evaluate the same images. Reproduced directly: base seed 3 yields `1742692731` on every iteration, contradicting the harness's own adjacent comment. The handout, `suite_config.json` and `ORGANIZER.md` now state that run-to-run spread measures machine noise only. |
| Upstream reference numbers preserved and labelled | `measurements/` is tracked in git at the pinned commit and holds the reference implementation's February 2026 results, which a fresh checkout makes indistinguishable from a submission's. They are copied to `src/suite/results/_upstream_reference/` with provenance: single 7.6145 s, small 647.0703 s (accuracy 0.9600), medium 6505.6093 s (accuracy 0.9740). `score.py` prefers per-case archives and labels any fallback in `read_from`. The README states these came from undescribed hardware and are context, not a measured speedup. |
| Process split respected | Client stages are pinned to `JAX_PLATFORMS=cpu` by the installer, so only the resident server touches the accelerator; one TPU admits one process. Client and server share the harness-generated key pair, and ciphertext bundles cross the process boundary through `bundle_io.py`, which preserves each ciphertext's tracked CKKS scale. |

All six ranked cases were then measured end to end through the harness, median
of three runs, on one host with eight TPU v6e chips:

| case | images | chips | stage 7 | img/s | server execute | encrypted | cleartext | agreement |
|---|---|---|---|---|---|---|---|---|
| C1 | 1 | 1 | 0.1271 s | 7.87 | 0.0325 s | 1/1 | 1/1 | 1.0 |
| C2 | 1 | 8 | 0.1329 s | 7.52 | 0.0378 s | 1/1 | 1/1 | 1.0 |
| C3 | 100 | 1 | 3.4374 s | 29.09 | 3.0463 s | 96/100 | 96/100 | 1.0 |
| C4 | 100 | 8 | 1.0276 s | 97.31 | 0.6006 s | 96/100 | 96/100 | 1.0 |
| C5 | 1000 | 1 | 32.9975 s | 30.31 | 30.4529 s | 976/1000 | 976/1000 | 1.0 |
| C6 | 1000 | 8 | 8.9178 s | 112.14 | 5.5829 s | 976/1000 | 976/1000 | 1.0 |

Every case returned `PASS` with `own_cleartext_agreement` 1.0: across C3–C6
the encrypted output matched the same model in cleartext on all 1100 images,
with no disagreement at any batch size, and the encrypted and cleartext label
counts are equal in every case. Scaling is the headroom the track is about —
30.5 ms per image on one chip whether the batch is 100 or 1000, 5.45× rather
than 8× from eight chips at batch 1000, and eight chips *slower* than one at
batch 1 (37.8 ms against 32.5 ms) because one image cannot fill eight lanes.

Scorer verified end to end: `score.py --label baseline` against
`baseline_manifest.json` returns ratio 1.0 for all six cases, 30.0 execution
points and 0.0 speed points, which is what the handout specifies for a correct
baseline. The speed curve was checked separately — 2x yields 5 points, 4x
yields 10, 8x clamps at 10 and a regression yields 0, so six cases at 4x plus
the 10 evidence points reach exactly 100.

Packaging: `make all` and the 24 public reference cases pass with
`max_abs_error` 0.0 in all four problems. `make clean` now removes
`src/suite/run` (sockets, pidfiles and daemon logs), and `make release`
produces a 170-file archive, 79 of them P3, with no `.golden`, no
`__pycache__`, no runtime scratch and no per-image evidence. The ranked
cases' ground-truth labels and the baseline's own predictions are organizer
artifacts and are excluded: shipping them would let a submission reproduce the
expected answers by lookup rather than by evaluating the circuit. Contestants
receive `baseline_manifest.json`, which carries only the aggregate numbers a
ratio is computed from. `common/package.py` now honors `#` comments in
`EXCLUDE`, which it previously read as literal patterns.

Not established here: baselines on contest hardware rather than the author's
host, and any accuracy figure for the trained checkpoint under the frozen
hidden image splits rather than the suite's seed-derived selection.
`baseline_manifest.json` in this package records the runs above and is
indicative; `ORGANIZER.md` requires the judge to measure its own.

## Reorganized into two problem sets — 2026-09-21

The four problems were renumbered and grouped by whether coding agents are
allowed. `p3_benchmark` became `non_agent_pset/p1_benchmark`, `p4_bert` became
`agent_pset/p2_bert`, `p2_cnn` became `non_agent_pset/p3_cnn`, and `p1_qwen`
became `agent_pset/p4_qwen`.

| Requirement | Evidence and result |
|---|---|
| Every reference rewritten | 24 files carried the old directory names; all were rewritten to the pset-qualified paths, and `grep` for the four old names across the tree now returns nothing. Handout titles, subtitles and the one cross-reference between tracks were renumbered to match. |
| The extra directory level does not break paths | The problems sit one level deeper, so 12 files needed adjusting: four `src/Makefile` includes, `common/Makefile.problem` (`CROSS_ROOT`, `compileall`, `package.py`), six `parents[2]` to `parents[3]` in the runners and references, and `stage_lib.py`'s reach for `common/`, which was the problem's sibling and is now the pset's. `make all` and 24/24 reference cases pass at the new depth, and `stage_lib.load_model()` resolves `common/` and loads the checkpoint. |
| Rendering survives the nesting | `render_docs.py` derived the problem number from the path's first token, which is now the pset name and raised `ValueError: invalid literal for int()`. It now takes the leaf. The cross-track nav needed a third `..`. All four handouts render, every relative link resolves, and the eyebrows read PROBLEM 01–04 in the new order. |
| The suite wiring survives the move | The installed shims embed absolute paths and pointed at the old directory. `suite-install` regenerates them; a full ranked case (C1, `--num_runs 3`) then ran end to end through the unmodified harness and archived `PASS, 1/1 correct` to the new location. `suite-check` still reports `harness/` byte-identical to the pinned commit, and the gate self-test is 14/14. |
| Packaging | `make release` produces 170 files — 105 under `non_agent_pset`, 62 under `agent_pset`, 3 shared — with no `.golden`, `__pycache__`, runtime scratch, per-image evidence, or stale directory names. |
| Scoring scheme is consistent across documents | P1 is the essential problem at 100 points under its rule-based rubric; P3 is the bonus at 20 points, all or nothing, requiring both end-to-end HE-on-TPU correctness and `T <= 1.5 * T_sota` against a named open-source implementation. README.md, P3's section 4, both subtitles and ORGANIZER.md state the same rule, and the rendered PDFs carry it. |

Not established here: the state-of-the-art implementation P3 is measured
against is not yet named or measured. `ORGANIZER.md` requires organizers to
pin it and measure `T_sota` on contest hardware before release, since the
bonus gate is meaningless without it.

## Original problem-set validation

Validated in the supplied workspace on 2026-09-19 against CROSS commit
`e652a56f897d8ed814978359e5a2b0c78f46641d`.

| Requirement | Evidence and result |
|---|---|
| Follow the supplied template | Both problems have `doc/`, `src/`, `test/`, `README.md`, `oneliner`, `EXCLUDE`, TODO source, reference `.golden`, six `.in`/`.out` pairs, and the five template sections. |
| AI-agent assignment targets full Qwen3-0.6B | P1 specifies the pinned checkpoint, all 28 layers, Q/K norm, GQA/RoPE, SiLU MLP, final norm, all 151936 logits and four-token generation. Small fixtures are explicitly partial-credit development cases. |
| Non-AI assignment covers both networks | P2 fixes the actual QuadLeNet/QuadAlexNetTinyInfer definitions and gives each three ranked workloads and 45 execution/performance points. |
| Ground the assignment in CROSS | `make -C non_agent_pset/p3_cnn/src baseline-plan` produced depth 7, degree 32768, 16384 slots, 8 logical Q moduli, dnum 4, security target 128 for both actual model plans. |
| Public reference answers | `make reference-test` passed all six Qwen and six CNN cases. Exact fixture/input hashes are recorded in `validation/public_fixture_sha256.json`. |
| Independent Qwen semantics check | `common/check_upstream_qwen.py` passed all six fixtures against Transformers 4.51.3. Maximum absolute discrepancy was 1.8314e-7; see `validation/qwen_upstream.jsonl`. It also checks decoder-block outputs and composition of all five partial-credit operator interfaces. |
| Build and intended TODO behavior | `make all` passed; both `.golden` files compile; direct invocation of each main contestant TODO raises the documented `NotImplementedError`. These intentionally unsolved starters do not pass HE tests. |
| HTML/PDF delivery | `make docs` passed. Qwen has 7 pages; CNN has 5. Required sections survived rendering. Text bounds passed on every page; selected pages were visually reviewed. See `validation/pdf_check.json`. |
| Submission and release packaging | Both `make submit` targets and `make clean` passed. The final contestant archive has 56 files and excludes goldens, hidden paths, caches and organizer material. Reference tests passed for both tracks from an extracted archive with `CROSS_ROOT` set. |
| Reference provenance | The bundled Qwen config SHA-256 matches `test/model_source.json`; checkpoint revision is pinned. No model weights or private datasets were downloaded or fabricated for this package. |
| Scoring and limits | Both rubrics sum to 100. Timing boundaries, numeric gates, accuracy gates, security invariants, runtime limits, partial credit and tie breaks are specified. Values are proposed rules awaiting organizer calibration. |

## Real encrypted LeNet smoke check

The following command completed successfully on one visible TPU device:

```bash
python cadathlon/non_agent_pset/p3_cnn/src/runner.py \
  --case cadathlon/non_agent_pset/p3_cnn/test/test1.in \
  --baseline --warmups 0 --repeats 1
```

Setup took 1002.88 seconds (16.7 minutes). The single encrypted request had
maximum absolute logit error 9.364e-14 and matching argmax. Its measured server
and client-to-client times were 1.081 and 2.014 seconds respectively. Raw
output is in `validation/lenet_he_smoke.json`; the author environment is in
`validation/author_environment.json`. This was a synthetic functional check
with **no warm-up and only one sample**, not a trained accuracy result,
calibrated warm latency, official baseline or throughput claim. JAX reported
12.36 GB of captured compilation constants, corroborating the setup/memory
concerns discussed in the handout. This number is not measured peak HBM.
The raw run's nRMSE diagnostic predates the near-zero denominator floor;
the CNN acceptance checks use the unchanged elementwise error and argmax gates.

## Limits of the original validation

The original validation covered the authored problem set, public numerical
references, CNN planning, build and packaging. Later follow-up results are
scoped separately above and below, including P1's trained 100-image checks
and the measured short schedules. They do not establish full-checkpoint Qwen
inference, official hidden-case judging, or a completed transformer solution.
The `.golden` oracles remain explicitly plaintext. The source checkout does
not bundle the two trained P3 CNN checkpoints or official hidden prompt/image
suites.

Event organizers must complete the asset and hardware preflight in
`ORGANIZER.md` before running a scored contest. Those event operations are
distinct from delivery of this problem proposal and its starter materials.

### Transformer follow-up work remains experimental

The secure encrypted polynomial integration checks passed on CPU, including
inverse square root, RMSNorm, GELU and LayerNorm. Fixed-domain numerical
planning checks cover all twelve public fixtures, but those planning checks
are not encrypted full-model runs. After an initial TPU attempt exhausted HBM,
the bounded-memory implementation passed **one real encrypted Qwen public
fixture, test1**, on TPU: maximum absolute error 0.03049173, NRMSE 0.01361694,
and top-1 agreement 1.0. The single cold run took 3830.42 seconds (about 64
minutes), with no warmup or repeated timing samples. Its actual N=131072,
depth-41, dnum=8 ring has log2(QP)=2669.51 below the 128-bit classical ceiling
of 3523; this fixture required no refresh. See the [raw result](validation/qwen_encrypted_test1_tpu_raw.json),
[stage log](validation/qwen_encrypted_test1_tpu.log), and
[evidence including unchanged start/end source hashes and approximation contracts](validation/qwen_encrypted_test1_tpu_evidence.json).
The result explicitly reports `full_qwen_evaluated=false`. This historical
Qwen run used the per-feature graph and its own source snapshot; it does not
validate later packed graphs or the remaining Qwen public cases.

A separate later **real encrypted packed BERT public test1** passed on one TPU:
maximum absolute error **0.0043024723861**, NRMSE **0.011767178719**, and top-1
agreement **1.0**. The single cold run used zero warmups and one measured
request. Its runner total was **4475.26686s**, external process time
**4482.29332s** (about 74.7 minutes), setup **657.81609s**, and server execution
**3793.55823s**. It used N=131072, depth 50 and dnum=5 with actual-ring security
enforcement; the log reports QP rounded to 3425 bits, below the 128-bit
classical ceiling of 3523. This fixture required no homomorphic refresh.
Both `full_bert_evaluated` and `mrpc_accuracy_evaluated` are false, and the
implementation status remains `experimental-unvalidated`.

The BERT child exited zero and was reaped. All **86 captured runtime/fixture
files plus the launch wrapper** were unchanged and archived before subsequent
integration. Public rotation keys used `/dev/shm`; its recorded free space
was 760,920,805,376 bytes before setup and again after cleanup. This setting
has no controlled disk-versus-tmpfs timing comparison. See the
[scoped result](validation/bert_packed_test1_tpu.json),
[raw runner output](validation/bert_packed_test1_tpu_raw.json),
[stage log](validation/bert_packed_test1_tpu.log),
[process and source-hash evidence](validation/bert_packed_test1_tpu_evidence.json),
and [exact frozen source archive](validation/bert_packed_test1_tpu_sources.tar.gz).
A later **real encrypted bounded packed Qwen public test2** passed on one
**TPU v6 lite**, including exactly **three completed homomorphic refreshes**.
Maximum absolute error was **0.0266620106002**, NRMSE **0.0185225562206**, and
top-1 agreement **1.0**, under the unchanged runner gates. This single cold
request used zero warmups and one repeat: runner total **12125.28009s**,
external wrapper total **12132.90537s** (about 3 hours 22 minutes), setup
**1114.55122s**, and server execution **10983.44591s**. The secure N=131072,
depth-49, dnum=4 profile used Q100/P24 and the packed period-128 layout; the
stage log rounds actual QP to 3491 bits, below the 128-bit classical ceiling
of 3523. The server remained keyless and refresh required no client callback.

The Qwen2 child and wrapper exited zero and were reaped. All **146 captured
runtime, fixture, collector and wrapper files** were unchanged; their exact
startup bytes were archived and verified before releasing the runtime freeze.
The unique public-key directory was removed, restoring `/dev/shm` to its
recorded 760,920,805,376 free bytes. The archived stack observations contain no
frame locals; their overhead was not calibrated, so this is a correctness
proof with one observed cold timing, not a scoring baseline or speed claim.
See the [scoped result](validation/qwen_bounded_test2_tpu.json),
[raw runner output](validation/qwen_bounded_test2_tpu_raw.json),
[stage log](validation/qwen_bounded_test2_tpu_log),
[process and provenance evidence](validation/qwen_bounded_test2_tpu_evidence.json),
and [exact source archive](validation/qwen_bounded_test2_tpu_sources.tar.gz).
It explicitly retains `full_qwen_evaluated=false` and
`implementation_status=experimental-unvalidated`; the earlier Qwen1 and
BERT1 results keep their distinct source snapshots and graph implementations.

These separate one-fixture successes do not complete either six-case public
sweep, the full pretrained models, file-stage integrations, or transformer
scoring calibration. No completed P2/P4 reference or calibrated transformer
baseline is claimed.

A genuine sparse homomorphic refresh subsequently passed on one TPU for the
periodic signed pattern `[-0.75, -0.25, 0.25, 0.75]`, checked across all 32768
slots. Its actual uniform-ternary ring was N=65536 with 46 Q and 6 P towers,
dnum=8, and log2(QP)=1472.015 below the 128-bit classical ceiling of 1747.
The keyless evaluator restored two usable levels; maximum errors were
0.000564558 after refresh and 0.000822600 after a subsequent encrypted square.
The process exited successfully. Component timing through client validation
was 2387.14 seconds and excludes final cleanup and process startup. See the
[scoped evidence](validation/secure_refresh_validation.json),
[unmodified result](validation/secure_refresh_tpu_smoke_raw.json), and
[stage log](validation/secure_refresh_tpu_smoke.log). The recorded source hashes
are startup snapshots: a lazily imported experimental dependency may have
changed during setup, so they do not certify every final loaded source.

This small-range N=65536 check does not validate the proposed N=131072 refresh
profile, magnitudes near 14, accumulated errors in deep transformer graphs,
or repeated-refresh performance. Separate CPU checks establish exact encrypted
parity and control-array release for experimental compiled rotations and
tensor/relinearization operations; these are not TPU memory or latency
measurements. Complete encrypted BERT/Qwen references remain unvalidated.

[CPU follow-up evidence](validation/secure_refresh_cpu_followup.json) separately
records secure rescale/plaintext-multiply/relinearization parity and secure
public averaging/scale-normalization checks. The full bootstrap optimization
and two-pass projection experiments in that artifact use an explicitly insecure
N=256 arithmetic fixture. They preserve the advertised output levels and guide
the next large-ring validation; they do not establish secure refresh at large
magnitudes or deep-transformer accuracy.

An isolated secure N=65536 CPU probe of the older single-pass path, with
public bound 16 and signed inputs through magnitude 14, failed the unchanged
0.001 error gate. Maximum absolute errors were 0.00918 after refresh and
0.25692 after an encrypted square; maximum relative errors were 0.00189 and
0.00377. It returned level 3, ran for 6175.49 component seconds and exited 1
with unchanged snapshot hashes. This path omitted the newer two-pass
correction and periodic projection. See its [unmodified result](validation/secure_refresh_cpu_large_single_raw.json),
[stage log](validation/secure_refresh_cpu_large_single.log), and
[verified source snapshot](validation/secure_refresh_cpu_large_single_sources.tar.gz).

A separate N=131072 TPU check passed for 128 linearly spaced signed values
from -14.2 to 14.2, repeated across all 65536 slots. It used public bound 16,
two-pass correction at precision 10 and homomorphic periodic projection. The
actual ring had 100 Q and 24 P towers, dnum=4 and log2(QP)=3490.162 below the
128-bit classical ceiling of 3523. The keyless evaluator returned level 25;
maximum absolute errors were 0.0000196293 after refresh and 0.000557472 after
an encrypted square, both below the unchanged 0.001 gate. Maximum relative
errors were 0.0000771542 and 0.000154304. Component time was 5314.35 seconds;
external time including startup and cleanup was 5337.16 seconds. The process
exited 0, all 77 captured source hashes stayed unchanged, and its temporary
public-key store was removed. See the [scoped result](validation/secure_refresh_tpu_deep_validation.json),
[raw result](validation/secure_refresh_tpu_deep_raw.json),
[process evidence](validation/secure_refresh_tpu_deep_process.json), and
[measured source archive](validation/secure_refresh_tpu_deep_sources.tar.gz).
Other packed periods, full-model error accumulation and a short transformer
schedule remain unvalidated.

A separate [public profile comparison](validation/qwen_refresh_profile_planning_20261103/README.md)
preserves actual generated secure Q/P rings and all refresh events for twelve
fixtures. The prospective depth-50/dnum-5 profile reduces Qwen2's ideal
schedule from three refreshes to two, with public bounds [0.5, 4], while
increasing its raw rotation-key storage from 27.61 to 33.96 GiB. **It is not a
blanket replacement:** BERT cases 4–6 would request bound 32, exceeding the
earlier bound-16 component's amplitude scope; deeper Qwen cases also move
some refreshes to bound 16. More levels are not monotonic: the archived Qwen2
schedule uses two refreshes at depth 52 but three at depth 53. All 21 ideal
graph evaluations pass their fixture gates and declared domains, excluding
CKKS and refresh noise. Each prospective profile still needs complete refresh
validation at its actual period and amplitude, followed by end-to-end
encrypted gates. BERT test1's depth-50/dnum-5 success used no refresh. The
current depth-49/dnum-4 Qwen2 run and its required three actual refreshes
remain unchanged; no new encrypted success or performance claim follows
from this planning work.

After that run was archived, bootstrap plaintext preparation gained a private
array interface while preserving the existing list interface. The candidate
matched 41 CPU residue/payload cases, including the larger ring dimensions;
conversion of identical precomputed residues took 4.1184 seconds through
nested lists and 0.00985 seconds through the array. The integrated change
passed 108 focused codec and scalar-add tests in 1.06 seconds. These CPU
measurements are recorded separately from the earlier TPU run in the
[candidate evidence](validation/secure_refresh_bulk_array_cpu.json) and its
[exact source archive](validation/secure_refresh_bulk_array_cpu_sources.tar.gz).

The pinned full BERT checkpoint was also checked offline against Transformers
4.51.3 on three fixed public sentence pairs. The complete twelve-layer NumPy
reference matched its logits to 8.89e-16, with no missing or mismatched weight
tensors. These examples exceeded the small-fixture approximation intervals
for attention exponentials and reciprocals, LayerNorm variance, GELU and the
pooler tanh. This is an empirical domain audit, not MRPC accuracy evaluation,
a proof of interval bounds, or encrypted inference. See the
[audit and source hashes](validation/pretrained_bert_domain_audit.json) and
[verified checkpoint manifest](validation/pretrained_bert_assets.json).
The separate [pinned upstream harness contract check](validation/bert_harness_contract_followup.json)
passed all seven stage CLIs and output-validation checks for all four sample
counts; it does not implement the five encrypted file-stage handlers.

A separate [CPU serialization component](validation/ckks_file_roundtrip_cpu_20261103/README.md)
completed a real client/server/client round trip in three fresh processes.
Its N=8192, Q8/P2, dnum=4 ring has log2(QP)=212.4557 below the 128-bit ceiling
of 218. The public-only server evaluated an affine map, encrypted square,
positive/negative rotations and weighted sum; client decryption of all 8192
values across two lanes had maximum error **1.1266e-7**, below the fixed 1e-6
gate. All 100 source hashes stayed unchanged, all processes exited zero, and
owned temporary key/input/ciphertext files were removed. Internal collector
time was 12.54 seconds. The archive includes the initial metadata rejection
and a separate 67-test transport/manifest checkpoint. This supplies evidence
for file transport and key separation only; it implements no pretrained BERT
layer, refresh path, MRPC evaluation or production file-stage hook.

A separate [streaming row-layout CPU proof](validation/bert_streaming_layer0_cpu_20261103/README.md)
matched the pinned full-width BERT layer-zero block on seven cases, including
all 128 query rows, nontrivial key masks and a partial tile. Maximum absolute
error against the NumPy/Transformers oracles was **1.95e-14**; the run exited
zero in 12.12 seconds with all four measured sources unchanged. This uses exact
plaintext nonlinear functions, not encrypted or polynomial BERT. Retaining
padded queries exposes centered-score domain failures, and raw exponentials
reach 4.50 million. Its conditional depth and payload/key inventory retain the
unimplemented nonlinear/refresh requirements and distinguish the core BSGS
staged-key path from the generic sequential-key path. Full-slot refresh,
full-model accuracy and accelerator memory remain unvalidated.

A later [score-only attention prototype](validation/bert_score_only_cpu_20261103/README.md)
passed the same seven plaintext cases with fixed polynomial/Newton softmax
arithmetic: maximum context error was **5.26e-10**, with zero failures of the
declared conditional domains and six source files unchanged. The narrower
historical domain still fails on 22 tile records. Public rotation composition
avoids extra stored keys, but the best tested map schedule still needs over
**one million rotations per 128-token example**, before other model work.
This and the abstract 226-level softmax schedule are practical limitations.
Its reciprocal intermediates can reach 128 (the declared domain permits 256),
so bounded final probabilities do not justify the bound-16 refresh path.
This is not a feasible encrypted full-model design or a refresh/accuracy proof.

To reproduce the domain audit with the manifest's local checkpoint snapshot
and an environment containing `transformers==4.51.3` and `safetensors`:

```bash
JAX_PLATFORMS=cpu python cadathlon/validation/check_pretrained_bert_domains.py \
  --snapshot /path/to/pinned-bert-snapshot --output /tmp/bert-domain-audit.json
```

The corresponding pinned full Qwen checkpoint audit exercises all 28 layers
and all 151936 final logits on three short public prompts. The NumPy planning
graph matched Transformers to a maximum absolute difference of 8.73e-6, with
matching top-1 tokens, and matched the independent NumPy oracle to 6.22e-14.
Observed intermediates exceeded the toy intervals for residual and Q/K
normalization variance, attention exponentials and reciprocals, and SiLU.
This likewise establishes neither encrypted inference nor generation or
accuracy on a benchmark dataset. See the [audit](validation/pretrained_qwen_domain_audit.json)
and [verified checkpoint manifest](validation/pretrained_qwen_assets.json).
Reproduce it with `check_pretrained_qwen_domains.py` and the same `--snapshot`
and `--output` arguments, using the pinned Qwen snapshot from that manifest.

The isolated public-transformer collector follow-up passed **78 CPU lifecycle tests in 5.53 seconds**. It pins each owned key directory's device/inode identity, retains moved or replaced directories, and stops the sweep when process, directory or handle cleanup is incomplete. The outer Qwen wrapper also retains its parent when inner cleanup is unknown or failed. Cancellation and primary errors survive secondary cleanup or evidence-writing failures. The [reviewed candidate and exact sources](validation/transformer_owned_key_collector_cpu_v2_20261103/README.md) remain unapplied during the frozen Qwen2 run. The earlier 65-test snapshot is retained as superseded evidence. These are lifecycle checks, not encrypted results or an OS sandbox.

A separate [full-width BERT linear component proof](validation/bert_fullwidth_linear_cpu_20261103/README.md) passed on CPU using the pinned layer-zero query (768×768), intermediate (3072×768) and output (768×3072) weights. All 8192 decoded slots, including padding, had maximum error below **3.93e-11** against a fixed 1e-6 gate. The secure N16384, Q4/P3, dnum2 run completed in 594.91 seconds with 100 unchanged exact source/provenance files and successful key-directory cleanup. Each map used one public vector repeated twice. The proof does not establish a chained BERT block, TPU performance, full-model inference or MRPC accuracy.

An [isolated dispatch-cache retention probe](validation/dispatch_cache_retention_cpu_20261103/README.md) repeated the secure full-width BERT projection checks and passed below 4.61e-11 maximum error, with 100 unchanged exact sources and successful cleanup. Removing only the evaluator's global JAX cache clear reduced this observed CPU run from 594.91 to 379.40 seconds; host peak RSS increased from 922904 to 1002216 KiB. Fresh keys and uncontrolled host load prevent a controlled speed claim. Cached-trace/registry lifetime, deep-chain, refresh and accelerator memory behavior remain unvalidated. Review also reproduced a Polynomial output-registry failure outside the explicit-array path, so the exploratory patch is **not ready for promotion**.

An isolated optional extension of the scoped NTT provider to HEMul control construction is preserved in [hemul_scoped_ntt_cpu_20261103](validation/hemul_scoped_ntt_cpu_20261103/README.md). Its incremental five-file patch applies after the unchanged eleven-file combined candidate and remains unapplied. The final CPU checkpoint passed 123 tests plus 6 subtests (one optional-device skip, separately passed), with 137 captured sources unchanged; a separate 24-test fixture checkpoint captured all 187 source/fixture files. Secure N8192/Q6/P2/dnum3 batch-two multiplication, tensor/relinearization and actual Bootstrap multiplication dispatch matched ordinary payloads and metadata exactly, with maximum new-component decrypted error 5.0420e-8. All six public NTT fields, fallback backends, owner/control retirement and cache rebuild were checked. These are CPU component and conditional planning results; no complete refresh, encrypted twelve-case pass or TPU speedup is claimed.

A [factored BERT score-routing plan](validation/bert_factored_scores_cpu_20261103/README.md) passed all 128 public forward/inverse map checks and seven pinned full-width layer-zero plaintext cases. Public bit routing reduces map-only work to **18,176 direct rotations** per 128-token example, or **27,392 signed-power rotations**, with nine added levels in each direction. Pinned block error stayed below 2.50e-10. Independent review adds 784 segmented-reduction rotations per example and corrects the unfused softmax depth estimate to at least 227. Full-period refresh, large reciprocal intermediates, actual ciphertext scheduling and end-to-end feasibility remain open; this is not an encrypted model result.

The subsequent [secure full-slot routing component](validation/bert_factored_routing_secure_cpu_20261103/README.md) passed one actual forward/inverse map, tile7/offset15, on all 65,536 slots. Errors were **1.63e-9** forward at level9 and **2.22e-9** roundtrip at level0 against the fixed 1e-6 gate, including hostile inactive padding. The N131072/Q38/P10/dnum4 run used 34 actual plaintext multiplications/rescales, 18 path levels, 18 rotations and 16 additions; its actual QP passes the 128-bit security gate. It exited and was reaped with code0 after 9981.80 internal seconds, with 97 exact sources unchanged and owned key storage removed. This is the archived isolated candidate from before the later runtime promotion, not a new-source full attention, refresh, model or TPU result.

The [isolated optimized-adapter wiring](validation/optimized_transformer_wiring_cpu_20261103/README.md) passed **216 CPU tests, with 3 skips**, in **66.80 seconds**, with all **344 captured files unchanged**. Its eleven-file follow-up patch applies after the archived combined candidate, HEMul scoped-NTT extension and guarded dispatch policy. Separate experimental BERT/Qwen adapters explicitly enable scoped NTT reuse and guarded dispatch caching; existing defaults, ring selection and Bootstrap stage clears remain unchanged. Extracted-package preflight, staged P2 helper closure, flag propagation and cleanup were checked. The patch remains unapplied and supplies no complete encrypted-model, TPU-memory, new refresh-profile or timing result.

Two [secure CPU scaled-Newton component runs](validation/scaled_newton_step_cpu_20261103/README.md) evaluated the same single update over 105 public test combinations. The N8192/Q8/P1, dnum8, scale40 profile **failed** the unchanged 1e-6 gate: maximum update error was **1.15975e-4**. The N16384/Q8/P3, dnum4, scale56 profile **passed**, with product, correction, and update errors of **1.6847e-11**, **5.2588e-10**, and **4.1656e-9**. Both actual QP rings satisfy the archived 128-bit security ceiling, and both runs passed metadata, source, and cleanup checks. All 98 source files in each archive were verified; both processes are absent. The lower run's exit 1 was observed; the higher run has a passing report but no recoverable numeric exit code. This proves one update only and establishes neither a full reciprocal nor refreshed softmax or encrypted BERT.

The separate [rescale-facade NTT candidate](validation/rescale_scoped_ntt_cpu_20261103/README.md) passed **97 CPU tests and 3 subtests, with 3 skips**, including secure batch-two payload/metadata parity, multi-level drops, Bootstrap dispatch and owner retirement. A public-only N131072/Q100/P24 probe matched all 24 NTT fields for singleton and odd/even Q prefixes. Observed cold control setup was 3.641s ordinary versus 0.545s scoped on CPU; host load and order were uncontrolled, so no TPU or model speedup is claimed. The patch remains unapplied, preserves rescale arithmetic and defaults, and excludes HEMul-internal rescale constructors. Exact sources and both validation-fixture failures are retained separately from the final passing run.

A [lazy public BConv candidate](validation/lazy_bconv_cpu_20261103/README.md)
passed **1,146 CPU tests and 93 subtests**, with **12 skips**, in **370.22 seconds**;
all **323 recorded files** stayed unchanged. It avoids building unused cached
BConv controls during ordinary NTT/field lookups, while preserving eager mode,
direct field access, construction device, serialization and owner lifetime.
Secure batch-two affine/square payloads and metadata matched the eager path
exactly, with maximum client error **3.97e-8**, while cached BConv construction
was forbidden. A separate two-device check passed. The initial green candidate
had a reproduced pickle regression; that failure and the fixed rerun are
preserved separately. This two-file follow-up remains unapplied and adds no
complete model, accelerator or timing-calibration claim.

The [combined follow-up integration](validation/runtime_followup_integration_cpu_20261103/README.md)
then passed **362 CPU tests and 9 subtests**, with **4 skips**, in **199.51 seconds**.
All **325 recorded files** were unchanged. The 32-file candidate combines the
earlier five reviewed patches with facade/nested rescale reuse and the fixed
lazy BConv change; every changed file reconstructs from its archived base.
Secure component errors stayed below **8.12e-8** at their original gates, and
both skipped device guards passed separately on two logical CPU devices. The
two opt-in large refresh tests remain outside this checkpoint. The patch is
still unapplied and leaves ring profiles and Bootstrap arithmetic/stage clears
unchanged; it establishes no complete model or calibrated TPU baseline.

The separate [nested HEMul rescale follow-up](validation/hemul_nested_rescale_cpu_20261103/README.md) remains unapplied: 110 CPU tests plus 3 subtests passed (3 skips), with 175 exact source files unchanged across its regression and precision diagnostic. Four constructor keywords reuse the existing scoped public-table provider; core arithmetic, Bootstrap and the disabled split-CD2 branch are unchanged. Secure CD1/CD2 and compiled-CD2 component ciphertexts match their baselines. The archive retains a real CD1 48-bit input precision failure at the unchanged 2e-6 gate, alongside the genuine 56-bit pass; it makes no new full-refresh, model, TPU or calibration claim.

A [complete first-round scaled inverse](validation/scaled_newton_first_round_cpu_20261103/README.md) passed eight genuine encrypted Newton updates on CPU, with maximum final error **2.3084e-8** against `4/(128*x)` and worst intermediate-update error **2.3183e-8**, below the unchanged **1e-6** gate. The secure N65536/Q50/P7, dnum8 profile checked 258 public interval points across all 32768 slots, finished at level2, and used no refresh. All 95 exact sources stayed unchanged, cleanup succeeded, and exit0 was observed. This isolated combined+HEMul+guarded-dispatch snapshot took 2359.25 seconds and peaked at 11899048 KiB host RSS. It establishes this first-round inverse only; the later independent inverse proof below does not compose normalization rounds or validate refreshed softmax, a complete model or accelerator behavior.

The [joined full-checkpoint QKV file-boundary component](validation/bert_qkv_file_component_cpu_20261103/README.md) passed on CPU: two distinct actual BERT embedding rows crossed a strict ciphertext boundary, a separate keyless server applied the full layer-zero 768×768 Q/K/V projections, and only after server exit the client checked all 24,576 slots. Worst error was 5.4563e-11 at the fixed 1e-6 gate. The secure N16384/Q4/P3/dnum2 run used no refresh, preserved all 128 source hashes, and removed owned key/input/ciphertext files. Native preprocessing was 2.6757s internally, the three cryptographic subprocesses 215.5866s, and total collector 219.2424s; these are uncalibrated CPU observations. The archive separately retains a reviewed 34-test collector cleanup follow-up, with no numerical-rerun claim. It does not implement a full layer, model, MRPC run or any production file-stage handler; the remaining implementation breakdown is explicit.

The isolated reusable streaming-linear follow-up now has a genuine three-process pinned BERT QKV file-boundary PASS on two distinct embedding rows: all 24,576 physical slots meet the unchanged `1e-6` gate, worst error `3.0744e-11`, secure N16384/Q4/P3/dnum2, no refresh. Its 132-source snapshot is unchanged and all processes/temporary keys were cleaned. Independent ideal/metadata checks cover F4096 rectangular 768/3072 widths, both slot capacities, partial tiles and hostile padding; the actual encrypted proof remains full 768×768 Q/K/V only. The [reviewable module and package/staging closure evidence](validation/streaming_linear_qkv_cpu_20261103/README.md) includes an unapplied six-file patch, preserved setup failures, and unchanged/disjoint 103-file P1 and 119-file P3 calibration inventories. This is necessary incremental full-checkpoint work, not a complete BERT/Qwen file-stage or model result.

An isolated [integer-affine and two-update Newton feasibility proof](validation/integer_affine_newton_cpu_20261103/README.md) passed on secure N16384/Q8/P3/dnum4 CPU: six integer affine variants matched an independent residue oracle exactly, and both Newton updates met the fixed `1e-6` gate over all 8192 slots, with worst update error **8.9368e-10**. Existing add/subtract/plaintext-add primitives avoid the correction's rescale; a public backward scale planner sets the real initial affine encoding scale, and every actual binary64 metadata value matches without relabeling. All 96 sources, input payloads and input metadata stayed unchanged; cleanup succeeded and exit0 was observed after 10.58 seconds. The initial assertion failure is retained separately. Only two updates were evaluated, so true-inverse error remains 3.78723; this candidate is unpromoted and supplies no full reciprocal, softmax, refresh or model claim.

The subsequent [thirteen-update integer-correction inverse proof](validation/integer_newton_thirteen_cpu_20261103/README.md) passed every 32,768-slot checkpoint at the fixed 1e-6 gate, with worst circuit error **1.8294e-8**. Four independently packed conditional domains passed against `4/(128*x)` at the predeclared updates 9/12/13/13, with maximum errors 1.8258e-8, 1.6400e-8, 1.5592e-8 and 1.6664e-8. Actual secure N65536/Q54/P7/dnum8 arithmetic consumed levels25 through1 with exactly planned scales, no metadata relabeling and no refresh. All 99 source files and input ciphertexts remained unchanged, cleanup passed, and terminal exit0 was observed and reaped. The uncontrolled CPU run took 3189.52s and peaked at 13,202,848 KiB host RSS. This validates the four independent inverse components; the helper remains unpromoted, and composed softmax/model execution and scoring calibration remain unfinished.

A [public C32 certificate for all valid key counts](validation/c32_all_valid_key_counts_cpu_20261103/README.md) checks all 640 original K/stage domains for K1–128 without rounding `K/4` into a different inverse. Exact interpretations of the rounded first-affine coefficients retain the declared convergence counts; 1,312,173 binary64 samples provide separate diagnostics. All 640 inactive midpoints are valid, while the constant dummy1 falls outside 318 domains. The certificate documents the required correlated multiplication/restoration order and first-update rounding bounds. It is conditional real arithmetic, not a CKKS noise, refresh or composed-softmax guarantee, and does not extend the measured K128 encrypted run to arbitrary K.

The reusable streaming-linear component was subsequently [promoted locally](validation/streaming_linear_promotion_20261103/promotion.json). All six promoted files match the reviewed 75-test package/staging checkpoint, and the module matches its genuine QKV proof. None of the 103 measured P1 or 119 measured P3 files changed, so their matching calibrations remain applicable. The rebuilt [235-file local release](validation/release_streaming_linear_20261103.json) matches the checkout byte-for-byte, resolves all 24 local HTML links, and passes all four extracted build/reference targets. Its SHA256 is `6c7a99c45402d277c08aa81485da998e765a2c32ce0725cf5595d3e921589cee`. The [released P2 staging check](validation/p2_released_staging_streaming_linear_20261103.json) also passed with the new helper in the exact inventory. Full-model/file-stage completion is still unproved; no external publication occurred.

The separate frozen 199-file transformer sweep has passed [optimized Qwen test1](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/qwen_test1/process.json): maxabs 0.030491700345, nRMSE 0.013616929560, top-1 agreement 1, cold runner 1431.5731s and external process 1438.6299s. Both actual-ring checks and TPU payload records are retained, the process exited zero, owned keys/handles were removed, and source bytes stayed unchanged. This is one cold public fixture, with instrumentation overhead included and no steady-timing/full-model calibration claim. The sweep continues from its own immutable snapshot; later helper additions to the live checkout do not rewrite that evidence.

The same sweep subsequently passed [optimized BERT test1](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/bert_test1/summary.json): maxabs 0.004302472197, nRMSE 0.011767177989, top-1 agreement 1, cold runner 2322.0219s and external process 2329.4523s. Both actual N131072/Q102/P20 rings pass the security gate, and input/output payloads are on TPU. The child exited zero and was reaped; owned keys and handles were removed, and all 199 exact sources still match. The checkpoint retains sampled resource data and its measurement limitations; it supplies no complete six-case, full-model, scoring-calibration or controlled-speedup claim.

[Optimized BERT test2](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/bert_test2/summary.json) also passed: maxabs 0.0008575034272, nRMSE 0.003188205659, top-1 agreement 1, cold runner 2528.8669s and external process 2536.0557s. Actual security/device checks, all 199 source hashes, terminal exit0, reaping and owned-key cleanup passed. Sampled aggregate process-tree RSS peaked at 46,827,896,832 bytes, with the recorded sampling limitations. Three of twelve cases are complete; Qwen test2 is running, and neither model has a complete six-case sweep or full-workload calibration.

A [packed full-width LayerNorm plan](validation/bert_row_layernorm_plan_cpu_20261103/README.md) passes eight public ideal cases against independent NumPy arithmetic, checking all 16,384 slots with hostile padding and partial tiles. Maximum difference is 7.11e-15. The two fixed native layer-zero input rows fit the unchanged attention/MLP inverse-square-root domains. The eleven-level schedule has a secure N32768/Q26/P4 profile and 22 planned rotation keys. Exact public and native sources are archived separately. This is planning and native preprocessing only; no encrypted LayerNorm, cross-ring composition, full-model accuracy or performance result is claimed.

The subsequent [LayerNorm candidate prelaunch checks](validation/bert_row_layernorm_prelaunch_cpu_20261103/README.md) passed 155 arithmetic, fixture, transport and lifecycle tests against 111 unchanged sources. Review found and fixed acceptance of self-consistently rehashed token IDs and substituted expected values; both original attacks are now rejected. The archive retains the earlier reproducer and native checks alongside the repaired versions. These checks do not establish encrypted LayerNorm accuracy or implement the full P2 file-stage contract.

The first [encrypted LayerNorm attempt](validation/bert_row_layernorm_cpu_20261103/README.md) **failed** the unchanged 1e-6 gate: attention output error was 3.63752e-6, while the other seven checkpoints passed across 131,072 slot comparisons. The secure N32768/Q26/P4/dnum8 run completed server computation, then the client reported the numerical failure. All 111 sources stayed unchanged, every child was reaped, owned files were removed, and terminal exit1 was observed after 1152.50s. The failed evidence is retained unchanged. A separate secure N32768/Q26/P3/dnum13 precision60 run **passed** with the same inputs, polynomials and1e-6 gate: worst error3.22296e-8 across all131,072 slots, with MLP output2.02293e-9. Root reaped exit0 after1642.08s; all111 measured sources and archive bytes matched, all children were reaped and owned files/handles removed. This validates the fixed-input component only; full-model domain coverage, common-ring composition, refresh and MRPC accuracy remain separate requirements.

The [full-slot score reduction audit](validation/bert_row_score_reduction_cpu_20261103/README.md) passed 59 public NumPy checks, including hostile unused slots, partial masks and nonrepeating rows, with maximum difference 9.10e-13. It supplies a rotation/mask schedule, not encrypted softmax. A separate [two-level score-routing candidate](validation/shallow_score_routing_prelaunch_cpu_20261103/README.md) passed 160 prelaunch checks and an actual secure N131072/Q12/P3 preflight. Its [genuine encrypted proof](validation/shallow_score_routing_secure_cpu_20261103/README.md) now passes eight checkpoints and all524,288 slot comparisons, worst absolute error1.13035e-9 against the fixed1e-6 gate. Both gains use distinct anchors and hostile padding at one tile/offset. All99 frozen sources match; session16776 exited0 and was reaped, and owned keys were removed after9259.62s internally. Six codec entrypoints were blocked during arithmetic; the historical collector did not patch the separate `decode` entrypoint, and source review confirms all decoding followed complete server work. A separate [callback and metadata repair](validation/shallow_score_routing_hardening_cpu_20261103/README.md) reproduces and fixes boolean-NSD and callback-mutation gaps and passes252 source-bound ideal tests; that version has no encrypted rerun and remains outside the release. Reduced depth is a circuit property, not a measured speedup or full-model result.

A [full-slot refresh candidate](validation/fullslot_refresh_preflight_cpu_20261103/README.md) passed 33 prelaunch tests against 99 unchanged sources and an actual public resource/security audit. Its CPU experiment uses the secure N131072/Q100/P24 profile with period65,536, bound16 and meta precision10. All input slots are distinct, including fixed boundary/interior probes; the earlier period128 alias proof cannot validate this branch. The planned checks retain the strict1e-3 gates for refresh and an encrypted square at levels25/24. The genuine run is in progress, with no passing full-slot refresh, full-model or calibration result yet.


The [full408 native MRPC audit](validation/mrpc408_fixed_domain_audit_20261103/README.md) completed against the pinned full12-layer float32 checkpoint and exact upstream seq128 tokenization. Native accuracy is345/408 (84.5588%). **Fixed-domain coverage failed on all408 examples**:41 of61 valid-query domain checks and18 of60 padded-query checks exceed their unchanged contracts somewhere. Layer2 centered scores reach[-37.64,43.57] against[-28,27], the finalMLP variance reaches[.931,7.724] against[1.075,5.182], and pooler inputs reach[-27.24,27.33] against[-24,24]. The collector's exit0 denotes audit completion; the archive explicitly records domain_coverage_passed=false. All26 captured sources match their archive and before/after hashes. A separate exact derivation covers base seeds42/43/44: native accuracies for100 examples are82%,79%,84%, and for200 examples81%,86%,83%; full408 is unchanged. These are native accuracy observations, not encrypted results. The earlier three-pair approximate evidence remains historical and does not establish these domain contracts for MRPC.

The [conditional common-ring plan](validation/bert_normalized_squaring_integration_plan_20261103/README.md) checks actual secure primes/P coverage, levels, binary64 scale recurrences and public key payloads for four profiles and all128 valid-key counts. Its five public tests and108-source archive passed independent verification without a CKKS context, keys or backend execution. The current depth49 refresh output at level25 cannot sustain later static reciprocals plus their product/reserve for K>=38. Candidate depth52/53 profiles yield levels28/29; their224-key dense-graph stores alone require161.44–358.31GiB. The plan also identifies that a1e-3 refresh error on a denominator near1/128 is a12.8% relative perturbation, so the pending refresh component gate cannot establish normalized mass or model accuracy. Bound64, common-ring precision, approximation-domain coverage and correlated error control remain unvalidated. These conditional schedules are not a full-model implementation or performance result.


The [full-size native Qwen workload audit](validation/qwen_native_workload_audit_20261103/README.md) completed all28 checkpoint layers for48 distinct public prefill inputs (16 each at8/16/32 tokens),32 teacher-forced decisions and32 free-running contexts from8 four-token sequences. All151936 final logits per context matched the unchanged independent exact NumPy oracle gates: worst absolute error3.26455e-5 and nRMSE2.09303e-6, with32/32 teacher choices and8/8 whole sequences agreeing. **The fixed approximation contracts failed in all112 contexts**, including156 of197 contracted sites;784(layer,K) softmax entries are missing forK5–32. Native range-reduced exponential inputs reach[-.83423,.87581] against[-.5,.5], and learned Q/K-normalized outputs reach magnitude486.79544, so BERT/toy refresh bounds do not transfer. The run exited0 after100.23s, all21 measured sources match exact archived/before/after bytes, all processes were reaped, and no JAX/HE/TPU was used. These are exact semantic checks and failed native-domain coverage, not approximate/HE accuracy, hidden-suite scores or scoring calibration. The earlier three-prompt plan is retained unchanged.


The [chained-rotation linear candidate](validation/chained_streaming_linear_secure_cpu_20261103/README.md) passed genuine secure CPU evaluation of three independent pinned BERT weight matrices:768×768,3072×768 and768×3072. Every8192-slot output passed1e-6; worst error is3.64730e-11 over24576 comparisons. Distinct rows, a partial tile, hostile unused slots and a signed source mask were checked. At N16384/Q4/P3/d2, all three maps use a union of only four rotation keys while keeping86/122/122 rotations and one level per map. The measured137-source snapshot, post-computation-only decoding, input immutability, terminal exit0 and cleanup were independently verified. Default direct mode remains unchanged in the isolated candidate. This is a key-storage optimization with a measured small-ring component check;332.97s is not a controlled speedup, full65536-slot encrypted proof or full-model result.

A [separate probability-only refresh plan](validation/bert_probability_refresh_plan_20261103/README.md) checks five accepted secure profiles, one intentionally rejected insecure profile and640 public K/profile schedules. Candidate refresh outputs31/32 can leave two levels after thirteen static Newton updates and the retained numerator product, avoiding denominator refresh. Prospective chained linear keys reduce the complete planned key union from224 to116; raw stores are371.11GiB for depth55/p60/d28 or181.14GiB for depth56/p56/d14. These are conditional public counts, not measured common-ring execution. Probability refresh still perturbs mass, signs and ratios, and the fixed first-block schedule does not cover the new training-derived degrees.

[Public learned-normalization bounds](validation/learned_normalization_bounds_cpu_20261103/README.md) use pinned weights and exact rational outward checks to bound all25 BERT and113 Qwen normalization outputs in exact real arithmetic. Coordinate energy gives sqrt(D)*abs(gamma) for RMSNorm; the zero-sum centered constraint sharpens LayerNorm to sqrt(D−1)*abs(gamma)+abs(beta). The maximum bounds are76.24955 for BERT and6144 for Qwen, covered by powers of two128 and8192 respectively. These bounds are independent of sampled activations but do not certify inverse-square-root input domains, polynomial error, finite-precision arithmetic or noisy ciphertexts. Refresh placement and numerical validation must account for those additional errors; the global power-of-two covers are not automatically recommended refresh profiles.


The [training-derived BERT candidate](validation/mrpc_training_domain_candidate_20261103/explicit_v2_complete_training/README.md) completed all 3,668 training examples without loading labels or held-out validation. All nonlinear inputs and normalized-squaring denominators stayed inside the frozen contracts. The unchanged candidate then [passed the held-out 408-example audit](validation/mrpc_training_domain_candidate_20261103/explicit_v2_heldout_validation/README.md): 345/408 correct (84.5588%), matching native accuracy, with complete domain coverage. The 100-example counts for seeds 42/43/44 were 82/79/84; the 200-example counts were 162/172/166. All nine actual ranked gates (at least 75% accuracy and at most two percentage points of loss) pass. Separate stronger logit diagnostics also pass, with maximum absolute error 0.00335088. An [independent readback](validation/mrpc_training_domain_candidate_20261103/independent_final_verification.json) verifies the exact 17 training and 20 held-out sources, fixed candidate hash, dataset tokenization identity, and all 130 rounded-affine reciprocal certificates. Training and held-out elapsed times were 2041.66s and 239.88s. These finite plaintext results do not supply encrypted composition, a full file-stage implementation, runtime calibration or arbitrary-input guarantees.

Qwen's [separate calibration protocol](validation/qwen_calibration_protocol_20261103/README.md) was frozen before measurements and uses 288 prefill inputs plus 128 teacher-forced/free-running contexts. Its tokenized inputs are disjoint from the archived public validation fixtures. The [416-context native calibration](validation/qwen_native_calibration_cpu_20261103/README.md) passed all exact semantic checks over 151,936 logits per context: maximum absolute error 7.74980e-5 and nRMSE 2.90427e-6, with 64/64 teacher choices and 16/16 free sequences matching the independent oracle. All 169 calibration input sites were observed, all 22 sources and seven model assets stayed unchanged, and terminal exit0 and cleanup were verified. The 370.20s native run supplies calibration ranges only; approximation selection, approximate trajectory coverage and encrypted model validation remain separate work.

The [standalone chained linear helper](validation/standalone_chained_linear_secure_cpu_20261103/README.md) subsequently passed the same three separate full-width encrypted matrix maps using its final isolated API: worst error 1.89684e-11 over 24,576 slots, secure N16384/Q4/P3, one level per map and four supplied rotation keys. All 366 exact sources and input ciphertexts stayed unchanged; the process exited0 after 335.09s internally, was reaped, and removed its owned keys. The [four-file local promotion](validation/standalone_chained_linear_promotion_20261103/promotion.json) preserves both existing linear helpers and all 199 active TPU-sweep sources. A further 189 focused main-checkout tests pass. The [rebuilt 238-file release](validation/release_standalone_chained_20261103.json), SHA256 `467a612e720c22a8ab42e334de786d4c194cde4d1e93fba80c6a4a05550edc04`, matches the checkout, resolves all 24 local HTML links, and passes all four extracted build/reference checks. [Released P2 staging](validation/p2_released_staging_standalone_chained_20261103.json) and an [independent release inspection](validation/standalone_chained_linear_release_review_20261103/report.json) also pass. The unchanged 103-file P1 and 119-file P3 measured inventories preserve their existing calibration applicability. This optional helper is a component result, not a full-ring encrypted model, measured speedup or external publication.


The frozen TPU sweep has now passed [optimized Qwen test2](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/qwen_test2/summary.json), including three genuine homomorphic refreshes. Maximum absolute error was 0.02666965, nRMSE 0.01852704 and top-1 agreement 1. The cold runner took 6870.74s; the external case process took 6877.97s. Actual Q/P security, TPU payload boundaries, all 199 frozen source bytes, terminal exit0, reaping and owned-key cleanup were verified. Four of twelve cases are complete and BERT test3 is running. These timings include cold setup and instrumentation; the checkpoint supplies neither a controlled speedup nor completed scoring calibration.

The [full-slot score-reduction proof](validation/bert_score_reduction_secure_cpu_20261103/README.md) passed all four encrypted CPU outputs over 262,144 physical-slot comparisons. Signed means and squared probability denominators cover K128/full rows and K15/partial rows with noncontiguous masks, distinct row/head values and hostile padding. Worst error was 2.16312e-9 against the unchanged 1e-6 gate. The actual N131072/Q10/P3/d4 profile passes security/P coverage; means finish at level2 and denominators at level1. All 104 measured sources stayed unchanged, client decoding followed complete server computation, input ciphertexts remained unchanged, and exit0/reaping/cleanup were independently verified after 4267.64s internally. This component remains isolated pending callback-continuation hardening and composition; it is not encrypted softmax or full attention.

A [fixed-v2 first-block plan](validation/bert_first_block_v2_public_plan_20261103/README.md) now binds the actual training-derived coefficients and all 128 valid-key counts. Its independent verifier reconstructs 1,024 public schedules from six archived sources. Every K128 normalization round uses thirteen reciprocal updates; the older stage-specific counts do not apply. Common-ring composition, repeated refresh error, learned normalization bounds with polynomial/noise margins, and all-key score accumulation still need measured evidence. The [tracked-residual reciprocal component](validation/tracked_residual_reciprocal_secure_cpu_20261103/README.md) now passes genuine encrypted checks on secure N65536/Q32/P9/p60: all thirteen estimate/residual checkpoints over32,768 physical slots and all130 original stopping rules pass fixed1e-6 gates. Worst true inverse error is3.22786e-8; worst branch-consistency defect is1.29132e-10. Thirteen updates consume fourteen levels while preserving the same exact-real rounded-affine Newton polynomial. Both branches and the separately executed final residual square stayed encrypted until all outputs were complete. All110 sources match, session97650 exited0 and was reaped, and cleanup completed after2169.09s internally/2169.90s externally. The helper remains isolated; common N131072 composition and joined normalization error still require evidence.


The frozen [Qwen approximate candidate](validation/qwen_calibration_candidate1_cpu_20261103/README.md) then [passed all 416 calibration contexts](validation/qwen_approximate_calibration_cpu_20261103/README.md). Every context checked 421 native operator outputs, 28 native block outputs and all 151,936 final logits, with complete declared-domain coverage. Worst operator, block and logit nRMSE values were 0.00245019, 5.29253e-5 and 1.11430e-4 respectively, below the unchanged 0.05/0.05/0.10 gates. Prefill choices matched 288/288, teacher-forced decisions 64/64 and complete free-running sequences 16/16. All 31 runtime sources and seven model assets remained unchanged; the process exited0, was reaped and completed in 993.65s internally (995.12s externally). Candidate coefficients were selected only from the separate calibration inputs; no public validation extrema or outputs selected them. Long Newton fallbacks and rational SiLU still require substantial encrypted scheduling and precision work. This finite plaintext calibration is not a hidden-suite score, language-task accuracy result, encrypted model or runtime calibration.

The [tracked-residual reciprocal prelaunch](validation/tracked_residual_reciprocal_prelaunch_cpu_20261103/README.md) passed 77 final tests and an actual secure N65536/Q32/P9/p60/d4 public preflight, with all 110 sources verified. It retains the exact frozen BERT affine coefficients and all 130 original reciprocal stopping rules. The 13-update production schedule consumes 14 levels, compared with 25 for the existing recurrence, while separate estimate/residual consistency gates account for CKKS drift. Its genuine encrypted run has begun with all fixed 1e-6 gates; no numerical pass or full-ring composition is claimed yet. Earlier preflight evidence and the subsequent JSON-safe numerical-failure hardening remain separately preserved.


The same unchanged Qwen candidate subsequently **[failed public held-out validation](validation/qwen_heldout_candidate1_cpu_20261103/README.md)**. All 112 required entries are accounted for: 109 native attempts, 87 complete approximate outputs, 22 declared-domain stops, and three explicitly unmeasured free-running suffix entries. All 16 S16 prefills stop at layer1 SiLU, one S32 prefill at layer3 SiLU, and the Q6_05 sequence has four teacher and one free context stop at layer9 SiLU. Computed outputs pass their numerical/semantic gates, but incomplete trajectories remain failures: prefill is31/48 against44 required, teacher28/32 against29 required, and7/8 complete free sequences against6 required. The overall result is false. All 38 sources and seven assets stayed unchanged; terminal exit1 was observed and reaped after293.76s internally/294.73s externally. The source archive and original calibration prerequisite remain pinned, and no coefficient/domain changes followed this validation. The separate calibration pass must not be presented as general held-out coverage. A revised design needs independent evidence and fresh validation. A separate [public-weight bound extraction](validation/qwen_public_silu_bounds_cpu_20261103/README.md) now verifies86,016 gate-row bounds without model execution or fitting validation extrema. Native exact-real layer bounds reach3200.2377 before polynomial/noise margins, and56 fresh tokenized inputs are frozen with zero prior-input equality overlap. Uniform normalization-polynomial certificates and revised SiLU selection remain separate work; the original held-out failure is unchanged.
