# Author validation

## PR #6 follow-up review — updated 2026-11-06 (in progress)

Reviewed merged PR `EfficientPPML/CROSS#6`, merge commit
`98ca6833397069e4228c265d75fef64884218a28`, against the five issues in the
supplied `issue.md`. **The PR is a partial resolution, not a competition
readiness sign-off.** Its own description accurately lists the unresolved
transformer references and five-minute evaluation target.

The [November 6 readiness review](validation/readiness_review_20261106/README.md)
rechecked the issue evidence and recovered public status after the original tool
sessions became unavailable. Three original BERT workers remain present with
their original process identities. The BERT4 diagnostic setup reports success
after3151.16s; root verified its exact260-file sources, public report chain,
recorded child exit0, process-group absence and retained workspace identities.
The original outer-tool terminal receipt is unavailable, so strict terminal
admission and the server/validator stages remain pending. The Qwen calibration
public report also says complete/pass, but independent terminal acceptance is
pending and its private outcomes remain unopened. These observations do not
resolve the encrypted-reference or full-model requirements.

| Reported issue | Evidence and current disposition |
| --- | --- |
| 1. P1 standalone key mismatch | Fixed in PR #6: the lazy client context and evaluator receive the same service-owned pair. Key-lifecycle regressions, all six reduced-ring local fixtures, and the real upstream student-profile suite pass. Results and their source snapshots are recorded below. |
| 2. P2/P4 encrypted goldens and calibration | **Unresolved.** Both goldens remain plaintext oracles. The frozen twelve-case TPU sweep is complete: seven passes (BERT tests1–3 and Qwen tests1–4) and five failures. BERT test4 failed decoding after five refreshes, BERT test5 reported an OverflowError after eight refreshes, BERT test6 failed strict approximation validation after eight refreshes, and Qwen tests5 and6 failed approximation validation after 19 refreshes each. None of the five failed cases produced decoded-logit metrics. The original parent exited1; all source and cleanup checks passed. The [TPU sweep archive](validation/optimized_transformer_tpu_sweep_20261103/README.md) preserves these separate results and their numerical and timing limits. Qwen test4 took 21146.86s cold runner time (21154.16s external), with nRMSE0.01601423, exceeding the separate full-Qwen HE0.01 gate and the five-minute target. A separate historical snapshot passed Qwen test2 with three genuine refreshes (cold runner 12125.28s). These validate individual public toy fixtures; they do not establish a controlled speedup. Successful six-case sweeps for both models, full-checkpoint/file-stage integrations and scoring calibration remain unfinished. The historical native audits found approximation-domain failures in every tested BERT input and Qwen context. A training-derived BERT candidate now passes all 3,668 training trajectories and all nine held-out ranked accuracy checks, including 345/408 correct, with fixed-domain coverage. This is a plaintext arithmetic result; encrypted composition remains unfinished. Qwen candidate1 passes all 416 calibration contexts but fails public held-out validation: 22 trajectories stop outside fixed SiLU domains and three subsequent free-running entries remain unmeasured. A separate public-bound-derived candidate2 passes the same complete 416-context calibration but fails fresh validation: nine inputs stop outside unchanged normalization domains. Its original published112-context regression passes, but its fresh failure remains preserved. A separate candidate3, selected by a uniform public-domain policy, passes all 416 calibration contexts and both the frozen fresh112 and original published112 plaintext evaluations without changing coefficients between them. Its separately fixed S1 row0 trajectory fails the final normalization domain; the exact-function prefix exceeds the same bound, and the original calibration contains no S1 case. Full encrypted model integrations and scoring calibration remain unresolved. Runners distinguish plaintext oracles, accept explicit implementations, enforce a requested TPU, validate repeats and report cold/total timings through cleanup. See the scoped transformer evidence below. |
| 3. Actual-ring 128-bit enforcement | Implemented in all four local runners and P1 stage 3 in PR #6. Current regression tests cover live full Q/P chains and ciphertext/context binding. This is an execution contract, not isolation from arbitrary contestant Python. |
| 4. P3 TPU aborts at batch 2/4 | **Resolved on the current frozen snapshot: all six original-profile cases passed in one TPU sweep.** LeNet and AlexNetTiny batches1/2/4 all passed; maximum absolute error was9.141108009425025e-13 and every top-1 agreement was1.0. The original parent exited0 after3369.48s (56.2 minutes). All128 measured files matched start/each case/end/current/archive, and root verified process-group and owned-key cleanup. The [complete six-case archive](validation/p3_original_six_tpu_20261105/final/README.md) preserves the source-bound numerical and timing evidence. This original cold schedule remains separate from the compact145-second result. |
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

The [remaining historical scaled-Newton inverse components](validation/scaled_newton_remaining_rounds_cpu_20261103/README.md) passed all thirteen genuine encrypted CPU checkpoints over65,536 slots each, with worst circuit error5.32140e-8 at the unchanged1e-6 gate. Four independent K128/s4 segments selected at updates9/12/13/13 have worst true-inverse error5.31867e-8. The secure N131072/Q84/P27/dnum3 ring passed actual-ring/P-coverage checks; all97 measured sources matched, cleanup succeeded, and root reaped exit0 after26194.746s internally. This preserves the older three-level weighted update and does not validate the newer static-integer/tracked path, compose normalization rounds, or establish softmax, refresh or model behavior.

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

The [tracked-residual reciprocal prelaunch](validation/tracked_residual_reciprocal_prelaunch_cpu_20261103/README.md) passed 77 final tests and an actual secure N65536/Q32/P9/p60/d4 public preflight, with all 110 sources verified. It retains the exact frozen BERT affine coefficients and all 130 original reciprocal stopping rules. The 13-update production schedule consumes 14 levels, compared with 25 for the existing recurrence, while separate estimate/residual consistency gates account for CKKS drift. That prelaunch was followed by the genuine encrypted thirteen-update pass documented above; common N131072 composition remains unvalidated. Earlier preflight evidence and the subsequent JSON-safe numerical-failure hardening remain separately preserved.


The same unchanged Qwen candidate subsequently **[failed public held-out validation](validation/qwen_heldout_candidate1_cpu_20261103/README.md)**. All 112 required entries are accounted for: 109 native attempts, 87 complete approximate outputs, 22 declared-domain stops, and three explicitly unmeasured free-running suffix entries. All 16 S16 prefills stop at layer1 SiLU, one S32 prefill at layer3 SiLU, and the Q6_05 sequence has four teacher and one free context stop at layer9 SiLU. Computed outputs pass their numerical/semantic gates, but incomplete trajectories remain failures: prefill is31/48 against44 required, teacher28/32 against29 required, and7/8 complete free sequences against6 required. The overall result is false. All 38 sources and seven assets stayed unchanged; terminal exit1 was observed and reaped after293.76s internally/294.73s externally. The source archive and original calibration prerequisite remain pinned, and no coefficient/domain changes followed this validation. The separate calibration pass must not be presented as general held-out coverage. A revised design needs independent evidence and fresh validation. A separate [public-weight bound extraction](validation/qwen_public_silu_bounds_cpu_20261103/README.md) now verifies86,016 gate-row bounds without model execution or fitting validation extrema. Native exact-real layer bounds reach3200.2377 before polynomial/noise margins, and56 fresh tokenized inputs are frozen with zero prior-input equality overlap. The later uniform certificates and revised SiLU selection are documented below; the original held-out failure is unchanged.

The [uniform Qwen normalization certificates](validation/qwen_rms_uniform_certificate_cpu_20261103/README.md)
bound all eight selected post-attention inverse-square-root polynomials on
their unchanged domains, using exact rational arithmetic and the actual rounded
coefficients and affine constants. The largest certified exact-real relative
error is 5.02903e-7. All four inputs stayed unchanged; the run exited0 and was
reaped after 298.12s, with 14 regression checks. An
[independent review](validation/qwen_uniform_norm_independent_review_20261103/README.md)
checked the certificate algebra and retained report provenance. These bounds
exclude later floating-point/CKKS error and require actual domain membership.

The separate [Qwen candidate2 selection](validation/qwen_calibration_candidate2_cpu_20261103/README.md)
changes only the 28 SiLU rules. Public weight bounds, uniform normalization
certificates and a fixed 9/8 engineering margin give integer domain radii
48–3601. Six sites use balanced polynomials and 22 use rational-tanh fallbacks;
all 113 normalization rules and every softmax rule remain unchanged. Fixed
full-domain and central grids pass the unchanged 1e-4 absolute selection target;
these sampled SiLU checks are not uniform approximation proofs. The selected
manifest is `0af45a8fc6d13b9d429ee1af2311b2cb6ac0aa4cde02eca56478472762482489`.
Its abstract full-model depth reaches 12,675; encrypted scheduling, noise and
practical runtime remain unresolved.

That frozen candidate2 [passed the complete 416-context calibration](validation/qwen_candidate2_approximate_calibration_cpu_20261103/README.md):
288/288 prefill choices, 64/64 teacher-forced choices and 16/16 complete free
sequences match native execution. Every declared domain, all 421 operator
outputs, 28 block outputs and 151,936 logits per context pass the fixed gates.
Worst operator/block/logit nRMSE values are 0.00207554, 7.49190e-5 and
1.84967e-4. All 53 sources and seven model assets remained unchanged; exit0,
reaping and cleanup were verified after 1031.27s internally. Independent
readback verifies the source archive, all 416 token trajectories and aggregate
maxima. The separate fresh validation subsequently failed, as documented below; the
original published regression subsequently passed against the same coefficients.
Calibration success does not replace either obligation or establish encrypted inference.

A [factored two-level score route](validation/factored_shallow_score_routing_cpu_20261103/README.md)
reduces the full 128-map routing key union from 276 to 66 without adding levels
or rotations. At the actual secure N131072/Q100/P14/p60/dnum8 profile, raw
routing-key payload falls from 245.8125 to 58.78125 GiB. All 421 source-bound
ideal checks pass with ten unchanged inputs, including noncanonical scale
alignment through real weighted operations. Its separate
[genuine common-profile proof](validation/factored_score_routing_p60_cpu_20261103/README.md)
now passes all twelve 65,536-slot checkpoints, with maximum error
1.63619e-10 at the fixed 1e-6 gate. Both canonical and genuinely encrypted
noncanonical inputs pass forward/inverse routing; all outputs precede client
decode. Root verified all 111 source files, the exact archive, terminal exit0
and owned-key cleanup. The CPU run took 11,826.06 seconds and peaked at
63,453,668 KiB RSS. It establishes this routing component, with no refresh,
complete attention, model, file-boundary or TPU result.

The [complete attention-tile assembly](validation/bert_attention_tile_assembly_cpu_20261103/README.md)
now implements all 128 score and probability-times-value terms for 16 queries,
using already projected full-width Q/K/V inputs. All 198 ideal-arithmetic and
lifecycle checks pass against independent dense references over 65,536 physical
slots, with nine source/input files unchanged. Each direction requires 4,328
rotations and 128 ciphertext products; the shared union has 89 rotation keys.
This is an isolated implementation with actual binary64 metadata equations,
not measured CKKS error, nonlinear softmax, a complete block or file-stage
inference. No main module or release was changed.

A separate [p60 full-slot refresh prelaunch](validation/bert_fullslot_p60_refresh_prelaunch_cpu_20261103/README.md)
passed 47 source-bound checks with 103 unchanged files. Its genuine experiment
is now running at the same N131072/Q100/P14 profile with independently encrypted
bound16 and bound128 inputs, each using all 65,536 nonperiodic slots. Both
restored refresh outputs and subsequent encrypted squares retain strict 1e-3
absolute gates, with all server work before client decoding. The older p56
refresh experiment remains unchanged and active. Neither experiment has a
terminal numerical result yet; neither supplies repeated-softmax or full-model
evidence.


Candidate2 **[failed the frozen fresh validation](validation/qwen_candidate2_fresh_validation_cpu_20261103/README.md)**:
all 112 native contexts were attempted, 103 approximate contexts completed,
and nine prefills stopped outside unchanged normalization domains. The stops
occur at layer1 or layer2 input inverse square root for the same three inputs
in each S8/S16/S32 bucket. There are no unmeasured free-running suffixes.
Completed outputs pass their numerical checks, but prefill39/48 misses the
required44; teacher32/32 and complete free sequences8/8 pass. All63 sources and
seven assets remained unchanged, exit1 was observed and reaped, and the run
took266.34s internally. The original published112 regression subsequently passed
under the same frozen manifest and arithmetic, with no fitting between runs.
The original candidate1 failure also remains unchanged.

The [softmax metadata and construction repair](validation/bert_softmax_metadata_guard_preflight_cpu_20261103/README.md)
corrects a separate implementation error: a real weighted operation can return
canonical scale plus or minus an ULP, whereas the earlier ideal double always
returned exactly canonical scale. The corrected planners replay the actual
binary64 operation order and preserve the original encoded coefficients,
nominal affine targets and numerical gates. The initial owner also binds and
revalidates all typed contract fields and borrowed objects at construction.
The preserved failing reproductions and successive2241/2326-test source-bound
passes remain distinct. Corrected initial/normalization collectors passed292
and510 preflight tests before their genuine common-profile experiments with
the original1e-6 gates. The corrected initial component passes its encrypted
checks; the original normalized-round component fails its inverse-error gate.
Its separate gain4 successor passes the fixed-fixture validation recorded below.
Both historical failures remain preserved.


The [published112 regression for candidate2](validation/qwen_candidate2_published_validation_cpu_20261103/README.md)
passes every domain/numerical gate:48/48 prefills,32/32 teacher choices and8/8
complete free sequences agree. Worst operator/block/logit nRMSE values are
0.00260099,5.55492e-5 and1.45595e-4. All63 sources and seven model assets stayed
unchanged, exit0 was observed and reaped, and the run took275.70s internally.
The fresh and published suites used identical candidate and evaluator bytes,
and their distinct frozen input manifests remain separately bound. This
published-suite success does not supersede the fresh failure or establish
full encrypted inference.

The isolated [complete layer0 softmax owner](validation/bert_softmax_tile_assembly_cpu_20261103/README.md)
composes the initial polynomial and all five tracked normalized-squaring
rounds, with explicit full-period B2/meta10 refresh calls selected from public
levels only. After any final continuation refresh, it applies a genuine public
mask to suppress padding. Its source-bound476-test ideal check passed with109
unchanged inputs in37.64s; exit0 and reaping were verified. Independent review
also reproduced and fixed an actual-engine slot-layout guard gap and a retained
exponential callback gap in this new copy. Neither active initial experiment
was changed. At K128, the public path is45→initial23→round5, followed by four
refresh25→round7 steps; a final continuation refresh25→mask24 leaves room for
PV21→dense20→LayerNorm9. Identity refresh doubles in these tests establish
composition only. Full-period B2 error, repeated domain stability and the joined
ciphertext result remain unmeasured.

A [first-attention integration inventory](validation/bert_first_attention_integration_cpu_20261103/README.md)
passes135 public checks with33 exact sources. It verifies four pinned768×768
matrices and computes the full graph/refresh union as129 rotation keys,
114.890625GiB of raw key payload. For eight live query tiles, the public graph
uses73,120 rotations and3,336 ciphertext products plus40 B2 refreshes; bootstrap
internal work is additional. The compact contract preserves original learned
positions and evaluates only valid query rows, while the existing attention
API requires eight K/V tiles. These counts do not establish practical latency,
peak memory or encrypted block accuracy. Serial block composition is being
checked separately; complete five-stage file integration and full-model
measurements remain unfinished.

The [pinned-v2 row normalization owner](validation/bert_v2_row_norm_owner_cpu_20261103/README.md)
now covers layer0 attention and MLP normalization with their exact learned
affine tensors and frozen inverse polynomials. The current candidate passed369
source-bound tests and228 public scale plans with114 unchanged files. Both
sites consume11 levels: attention20→9 and MLP24→13. Independent review found
three callback/API gaps in the earlier candidate; the separate correction
guards captured outputs, removes callback-added semantic bounds, and rejects
bound-aware arithmetic before work. The original349-test result and negative
reproductions remain preserved. Root independently verified both archives.
These are ideal arithmetic and ownership checks, not an encrypted result for
the v2 coefficients or a complete block.

The [public Qwen RMS envelope audit](validation/qwen_rms_outer_domains_public_cpu_20261103/README.md)
computed all113 exact-real native denominator envelopes from the complete
embedding table and310 pinned tensors, without a model forward or coefficient
selection. Twelve preflight checks passed; the65.18s extraction exited0 and
was reaped with seven sources and seven assets unchanged. The common lower
bound is epsilon1e-6; upper bounds range from0.0205677 to3.08865e13. These
conservative bounds discard correlations and do not transfer automatically
to the approximate or encrypted graph. The illustrative ideal Newton costs
of61–233 levels select no coefficients and establish no practical schedule.
Root independently verified the archive and113 envelope records.

A [new56-prompt validation freeze](validation/qwen_fresh56_outer_domain_freeze_cpu_20261103/README.md)
is disjoint by text and full/truncated token equality from400 prior records.
The tokenizer-only preparation reproduced those prior records, passed ten
checks, and exited0/reaped in2.39s with11 unchanged sources. Its token manifest
is `b20f28fc66260d8bd0ec704d6138da9a60ba513a53111d789cfd10f431dd80af`.
No model output or coefficient selection was performed. Root independently
verified the frozen inputs; candidate2's earlier fresh-validation failure
remains unresolved.

The [full-width layer0 FFN tail](validation/bert_v2_ffn_tail_ideal_cpu_20261103/README.md)
passed150 source-bound checks in28.46s with125 unchanged source/public-plan
files and unchanged external checkpoint/fixture hashes. The actual chained
768→3072→768 maps, GELU64, residual and v2 MLP normalization ran over all nine
tiles of the fixed K15/K128 fixtures. Every physical output slot passed the
independent dense polynomial comparison; worst final error was1.20e-14.
The fixed polynomial differs from the independent exact-function reference by
at most2.75e-4, a diagnostic without a new native-framework or ranked-model
claim. Per tile, actual calls match286 rotations,80 ciphertext products and
7857 plaintext-product terms. Two identity refreshes test the public schedule
25→24→10→25→24→24→13 and one-ULP metadata propagation. Independent review
reproduced and corrected malformed refresh-plan/key/helper acceptance and
callback/layout gaps; the original draft and reproducers are retained.
Root independently verified the archive and all125 current source bytes.
These inputs are native attention outputs. Joined approximate attention/FFN,
genuine refresh/error measurements and complete file-stage inference remain
separate requirements; no main runtime or release was changed.

The [serial first-attention owner](validation/bert_first_attention_owner_cpu_20261103/README.md)
passed108 source-bound checks in225.08s with130 unchanged files. It composes
the four full-width projections, all128-key QK/PV terms, complete five-round
softmax, residual and v2 attention normalization on the fixed K15/K128 inputs.
All65,536/524,288 output slots pass the1e-8 independent polynomial gate, with
worst differences3.55e-14/2.84e-14. Propagated valid-query domains and exact
native-reference parity were checked; approximate/native differences remain
diagnostics. Root verified every current and archived source and the unchanged
private fixture hash. Five/40 identity refreshes exercise level/scale
composition only. The schema and owner do not implement ciphertext transport
or the five complete BERT handlers; joined layer0 and genuine encrypted
accuracy remain outstanding.

[Exact-real envelopes for all twelve GELUs](validation/bert_all_gelu_refresh_bounds_cpu_20261103/README.md)
now give fixed refresh bounds `[16,64,32,32,32,32,16,64,64,64,64,16]` after
an explicit9/8 headroom factor. The certificate handles rounded domain maps
that extend slightly outside[-1,1], using exact Chebyshev growth bounds and,
for layers0/2/6,4097 rational grid points plus a global derivative envelope.
The final28-test source-bound run exited0/reaped with three unchanged files;
two independent reviews and an independent exact Clenshaw readback passed.
The preserved original run has identical twelve results; a generic subunit
helper edge was corrected separately. No inputs or coefficient fitting were
used. These certificates exclude floating/CKKS error and do not establish
full-slot B32/B64 refresh accuracy or propagated model-domain validity.

The genuine [historical full-width FFN file component](validation/bert_ffn_file_component_cpu_20261103/README.md)
has completed: all three32768-slot checkpoints pass their unchanged1e-6
gates, with worst errors4.30e-9 for the first dense map,2.71e-9 for GELU48
and2.11e-9 for the final dense map. Fresh client/server/client processes
enforced keyless server work and completed every encrypted output before
decoding. The135 frozen sources match the archived/start/end/current bytes;
exit0, reaping and temporary-directory cleanup are verified. Total collector
time was18,600.42s on four CPU cores. Root independently verified the archive
and98,304 checked slots. This uses the older N65536/depth17/GELU48 circuit;
it supplies no v2 GELU64, residual/normalization, full-block or model claim.

The [joined layer0 ideal composition](validation/bert_layer0_composition_cpu_20261103/README.md)
passed103 checks in245.89s with140 unchanged sources. Actual first-attention
output objects feed the FFN owner directly. All nine K15/K128 tiles pass the
independent complete-block polynomial gate, with worst errors7.55e-15 and
8.77e-15; exact dense/native full BertLayer parity is within1.07e-14. Root
verified every current/archive source. The K128 graph actually executes75,408
rotations,3,976 ciphertext products and56 identity refreshes; its public key
union is130, totaling115.78125GiB in raw rotation pairs. These counts exclude
bootstrap internals and are not runtime/peak-memory measurements. Neither
identity refresh nor this single joined layer establishes encrypted or
complete12-layer/file-stage readiness.

The [all-layer GELU owner](validation/bert_all_gelu_owner_ideal_cpu_20261103/README.md)
preserves the exact twelve selected balanced polynomials and passes180
source-bound checks with119 unchanged files. Seventy-two full-slot cases
check4,718,592 outputs against independent Chebyshev evaluation, worst
error5.68e-14; all388 supported layer/level schedules retain two levels.
Layer0 has identical ideal output bytes, metadata plans and primitive calls
to the earlier specialized owner. This remains ideal-only evidence; its
common-ring encrypted layer0 experiment is being prepared separately.

The [full12-layer public continuation plan](validation/bert_twelve_layer_continuation_plan_cpu_20261103/README.md)
passes1,536 layer/key-count paths plus pooler continuation, with117 unchanged
sources. It resolves the level-budget design by refreshing later inputs to25,
splitting initial softmax after normalized t, and preserving each site's
selected power or balanced method. Minimum MLP output is3. The proposed K128
schedule counts1,064 encoder refreshes and1,067 including pooler; this is
neither an optimized lower bound nor evidence of practical latency. The
prefix/tail, generalized norms, pooler and complete file handlers still need
implementation and numerical validation. Root independently verified the
public recurrence and archive.

The [Qwen RMS repair feasibility audit](validation/qwen_rms_repair_feasibility_public_20261103/README.md)
passes40 source-bound public checks and independently verifies113 exact-real
residual certificates,904 level schedules and33 rounded-affine reciprocal
certificates. It identifies a numerical obstacle: a prospective tracked
inverse-square-root recurrence fails66/113 binary64 lower-endpoint checks,
including49 initializations rounded to an unstable fixed point. Exact-real
18–60-update certificates do not repair that cancellation or transfer native
domains to the approximate graph. No replacement candidate, model run or
coefficient selection was performed; candidate2's fresh failure remains
decisive. Root independently verified this archive. Public embedding-row
bounds and an approximate-graph perturbation analysis are the next evidence
steps before selecting another candidate.

Subsequent independent review found that this feasibility archive's v2
initial-softmax mass formula incorrectly retained a rounded1/K factor that
cancels between denominator and output. The v2 artifacts are preserved; a
[separate corrected v3](validation/qwen_rms_repair_feasibility_correction_public_20261103/README.md) now passes49 source-bound tests and an operation-level K3 regression.
Its independent verifier checks the shared-alpha cancellation for all K1–32,
as well as113 inverse certificates,904 schedules and33 reciprocal bounds.
The corrected initial mass is1−r; the rounded K*(1/K) factor belongs only
in the denominator-domain enclosure. Root verified this corrected archive.
The tracked-normalization cancellation findings remain distinct. The v2
archive verifier alone did not establish that mass formula's correctness.


The [public Qwen embedding and Q/K audit](validation/qwen_embedding_row_domains_public_cpu_20261103/README.md)
passed22 focused tests with eight unchanged sources and seven pinned assets.
The exact stored-value embedding denominator interval is
[9.028572102258705e-5,0.0018623166357577261] across151,936 rows, improving
the first normalization condition ratio to20.6269. RoPE pairwise gamma bounds
tighten26/28 exact-native Q/K bounds. Root verified archive and rational
aggregate consistency; the offline verifier does not reread the full table.
Native FP32 reduction, approximate-graph domain propagation and CKKS error
remain uncertified. No replacement candidate or fresh56 model run occurred.

The [conditional approximate-norm output certificate](validation/bert_approximate_norm_output_bounds_public_20261103/README.md)
now covers all24 BERT-v2 selected polynomial methods using actual rounded
public coefficients. The largest ideal affine bound is76.2971, or85.8342
with9/8 headroom, so all sites fit B128 conditional on their variance domains
and shared ideal moments. Thirty-one source-bound tests passed. Root
independently checked all24 polynomial identities and envelopes and reread
all48 pinned affine tensors. Floating/CKKS and independently perturbed
feature/variance errors remain outside this certificate.

The [split initial-softmax implementation](validation/bert_split_initial_softmax_cpu_20261103/README.md)
passed510 checks with125 unchanged sources. Its real primitive composition
implements scores21→mapped t18→explicit B2 refresh25→initial probabilities6
for all12 layer contracts, preserving their16/32/64 divisors. The suite
covers384 full-slot cases, all1536 layer/key-count schedules and callback
guards. Root verified every current and archived source. Refreshes here are
identity doubles; the complete later squaring rounds and full model remain
separate validation obligations.

The [all24-site normalization owner](validation/bert_all_row_norm_owner_ideal_cpu_20261103/README.md)
passed338 frozen checks with121 unchanged sources. All288 numeric cases
(18,874,368 slots) meet1e-8 against independent Chebyshev evaluation, with
worst error2.39184e-11. All816 site/level schedules retain two levels, and
layer0 preserves specialized-owner outputs and primitive calls. A development
reserve-counting error was fixed with its source and short reproducer retained.
Root verified current/snapshot/archive bytes; the collector reaped its child.
The outer tool handle was lost at context compaction, so no independent outer
exit/reaping claim is made. This is ideal primitive evidence, without model
inference, real refresh or full file stages.

The [common-ring GELU64 prelaunch checks](validation/bert_gelu64_commonring_prelaunch_cpu_20261103/README.md)
passed209 checks and independent source/security review. The subsequent
[genuine GELU64 component](validation/bert_gelu64_commonring_component_cpu_20261103/README.md)
passes all nine fixed1e-6 gates across589,824 checkpoint-slot comparisons.
Worst error is4.93048e-9 at T64; final polynomial error is1.96483e-11.
Root verifies all128 current/start/end/archive files, actual QP security and
physical P coverage, noncanonical prestage scale1.506919487555368e18, output
level10 and authoritative exit0/reaping. The endpoint-domain diagnostic remains
false because encryption noise can cross a nominal endpoint; the exact-erf
diagnostic9.15556e-5 is separate from the fixed polynomial gates. There is no
refresh, complete block/model evaluation or calibrated timing claim.

The [encrypted Q/K score-file component](validation/bert_qk_score_file_component_cpu_20261104/README.md)
passes both65,536-slot output checks at the unchanged1e-6 limit. Its192 raw
native score anchors are nonzero; worst full-slot error is2.92361e-9. The
separate fixed key mask excludes the selected tile, so its all-zero check is
explicitly vacuous. Two full768-by768 learned projections and the nine-level
score-routing graph run in a keyless server; no refresh runs. Root verifies
all307 current/start/end/archive sources, actual log2(QP)1368.73113, L7/L6
outputs, five exit0/reaped phases and removal of owned private/key directories.
Collector time80,586.66s is an uncalibrated component observation. This does
not establish full attention, full BERT or ranked accuracy.

The fifth completed optimized-reference TPU case was
[BERT test3](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/bert_test3/summary.json),
with maximum absolute error0.0007734462, normalized RMSE0.004451666 and
matching top1. The runner took11,106.24s; its five bounded refreshes and
199 frozen sources were checked, and the child exited0 with cleanup.
The [later sweep status](validation/optimized_transformer_tpu_sweep_20261103/README.md)
records eight completed outcomes, including the subsequent BERT4 decoder failure.
These public reference cases do not establish full-size pretrained-model readiness.


The [generalized full-width FFN owner](validation/bert_all_ffn_tile_owner_cpu_20261103/README.md)
passed99 source-bound checks across all12 layers,24 component cases and108
tiles, with133 unchanged sources. Worst actual-loop error against the
independent dense/polynomial oracle is4.79144e-10 at the unchanged1e-8 gate.
All216 refreshes are identity doubles; independent native attention-LN inputs
initialize each component. Root verified current and archived sources and
all reported cases. These inputs do not constitute a joined12-layer model.

The [pooler rational-tanh B2 certificate](validation/bert_pooler_tanh_bound_public_20261103/README.md)
passed21 source-bound tests and independent exact verification. The actual
rounded base polynomial is bounded by0.587481; the six shared-square rational
iterations are each bounded by1. A separate additive refresh error at most
1/32 would preserve the next reciprocal domain, but that error has not been
established for CKKS. This is a conditional magnitude certificate only.

The [complete pooler/tanh/classifier implementation](validation/bert_pooler_head_ideal_cpu_20261103/README.md)
passed114 source-bound checks with125 unchanged sources. Actual chained dense
loops apply the full768×768 and2×768 pinned matrices, selecting originalCLS0
through genuine source masks. Twelve full-slot tanh cases and eight synthetic
CLS head cases have worst independent-oracle errors4.44e-16 and8.33e-17.
The planned level12 input produces level2 logits after three explicit B2
identity refreshes, and a regression confirms refreshed padding is masked.
Root verified the source archive; independent review found no concrete
algebra, scale, masking or callback blocker. This is neither encrypted
execution nor a full-encoder/model accuracy result.


The [generalized complete softmax owner](validation/bert_layer_softmax_owner_cpu_20261103/README.md)
passed521 frozen checks with127 unchanged sources, covering every publicK
with all three selected round counts, all12 endpoints and1536 schedules.
Root verified current and archived bytes. Identity refresh still supplies
no encrypted accuracy evidence.

The [generalized attention owner](validation/bert_layer_attention_owner_cpu_20261103/README.md)
subsequently passed 216 frozen tests and all 24 layer/K15/K128 cases.
Across 7,077,888 physical output slots, maximum ideal error is 2.48655e-11
at the fixed 1e-8 gate; the independent native oracle stays within
2.84218e-14 at its 1e-11 gate. All declared domains pass. Root verified
all 138 current/snapshot/archive source files, case aggregates and terminal
cleanup. Each case starts with its independent native prior-layer input;
this is component evidence using identity refresh, not a joined trajectory
or encrypted model result.

The [client embedding preparer](validation/bert_client_embedding_inputs_cpu_20261103/README.md)
passed45 checks with three unchanged sources and six pinned assets. It reads
only the five actual embedding tensors, applies learned positions before
compaction, and packs all eight full-slot tiles. K15/K128 and a noncontiguous
K11 case agree with independent NumPy embedding arithmetic within8.89e-16.
Public metadata omits token IDs and private target IDs; native encoder/head
constructors are forbidden. This helper has not yet encrypted its outputs.

The [public server-model cache](validation/bert_server_public_model_cpu_20261103/README.md)
passed28 final frozen checks with seven unchanged sources, extracting and
reading back all196 pinned tensors/74 matrices as immutable exact lifts.
Independent review reproduced cleanup masking, replacement-directory deletion
and malformed-header allocation defects; each is fixed with retained
reproducers. Header validation now precedes bounded array construction.
Root verified the final archive and unchanged public assets. This is real
public preprocessing, with no keys, client inputs or encrypted computation.

The [full-model transport prototype](validation/bert_full_model_transport_cpu_20261103/README.md)
passed71 checks with nine unchanged sources and independently reviewed
boundaries. It binds eight ordered level49 request tiles and one level2
response to public identities, masks/positions, an opaque request nonce and
the ordered request digest. Full100MiB/6MiB payload shapes are tested using
zero arrays and fake polynomial wrappers. This does not establish real
ciphertext transport or complete five-handler execution.

The [full native comparison fixture](validation/bert_full_native_reference_cpu_20261103/README.md)
passed15 checks for the two fixed inputs, and the separate
[independent full12 clear reference](validation/bert_independent_full_reference_cpu_20261103/README.md)
passed35 checks and all312 propagated nonlinear-domain observations. Exact
functions match actual native blocks/attention within5.69e-14; approximate
logits pass the original P2 tolerance on these two inputs. All later native
arrays are comparison references only; joined execution must feed its own
preceding output objects through every layer. These are neither encrypted
results nor the required ranked100/200/408 evaluations. Root verified both
archives. Full12 primitive composition, the five complete handlers and final
benchmark measurements remain in progress.

The [client key-generation helper preflight](validation/bert_client_keys_preflight_cpu_20261103/README.md) passed42 frozen tests with40 unchanged source/public-input files. Actual QP security and P coverage are checked before production key-generation calls; tiny mocked producers exercise the complete130-rotation inventory, serialization and owned cleanup. Root verified all current/archive bytes. The complete public store requires125,380,329,472 raw bytes plus52,428,800 private-key bytes; the117.82GiB storage reservation excludes working memory. No actual keys, CKKS context, or encrypted file-stage execution were generated by these tests.

The [independent final-logit scale enclosure](validation/bert_response_scale_public_20261103/README.md) passed13 tests with11 unchanged sources. Exact rational bounds enclose the final level3-to2 weighted-sum metadata for every accepted positive finite source scale in three binary64 values. Client response validation therefore checks a fixed public enclosure instead of trusting the server to declare its own expected scale. Root verified current/archive bytes and replayed the certificate. This binds metadata only; it proves neither payload correctness nor encrypted numerical accuracy.

The [five-handler integration draft](validation/bert_full_file_stages_mocked_cpu_20261103/README.md) passed112 frozen mocked checks with24 unchanged source/public-input files. Tests cover all five handlers, private ID/order/request bindings, independent response-scale checking, cancellation cleanup, and two consecutive upstream-style runs with reused paths. Independent reproductions fixed nested-file replacement deletion and primary-error masking in evidence collection; both failures remain archived. The final snapshot includes the reviewed server-failure mapping quarantine policy. Root verified current/snapshot/tar bytes and reaped exit0. Keys, codecs and graph execution are mocked here; no actual full-model file-stage or ranked evaluation pass is claimed.

The [authenticated key-context loader checks](validation/bert_key_contexts_preflight_cpu_20261103/README.md) passed104 frozen tests with49 unchanged sources. Public rotation arrays remain read-only memory maps; the server has no PKE or secret key in its context and both key-generation aliases plus seven codec entrypoints are guarded before setup. Review reproduced unsafe mapping cleanup after partial setup and failed lazy rotations. The corrected failure path retains unresolved public mappings until process exit, rejects loader reuse and preserves the failure; normal successful cleanup closes the maps. The earlier100-test proof and negative reproducer remain separate. Root verified current/archive bytes. These are mocked-context/public-security checks, with no actual key generation or HE inference.

The [joined full12 baseline](validation/bert_full_model_owner_cpu_20261103/README.md)
completed all 46 tests and both K15/K128 trajectories: 24 stages plus pooler/head,
14,417,920 physical-slot comparisons and 2,648 domain/bound observations.
Maximum stage error was 7.79560e-10 at the unchanged 1e-7 gate. Both cases
passed the original P2 logit and prediction checks; the larger native-logit
error was 0.000293576. Root verified all 137 current/archived sources, both
current assets and terminal evidence. The process exited zero and was reaped.
The 136/1,067 refreshes are identity doubles; this establishes joined ideal
arithmetic, with no HE or ranked accuracy result.

The [actual first-handler key-generation preflight](validation/bert_full_key_generation_preflight_cpu_20261103/README.md) passed29 frozen checks plus actual security/capacity verification over58 sources. The subsequent [actual key-generation run](validation/bert_full_key_generation_cpu_20261103/README.md) passed and was reaped with exit0. It generated all130 rotations plus the relinearization and public keys:263 public arrays occupy125,380,363,136 NPY bytes. The actual N131072/Q100/P14/p60/dnum8 parameters pass the classical128 security bound. Worker time was3272.128s and collector time3273.428s onCPU136–139. Root verified all58 original/execution/archive source hashes, report bindings and process cleanup. The owned key workspace remains available for later handlers; secret material is excluded from the archive. This completes key generation only, with no encrypted inference or ranked result.

The isolated [immutable-mask guard candidate](validation/bert_immutable_mask_guard_cpu_20261103/README.md)
passed 118 NumPy checks with five unchanged sources. Standard masks now have
immutable bytes storage; complete storage/shape/type bindings permit reuse
of their content digest, while custom mutable-backed masks retain full
hashing. Root verified exact archived and current bytes. Local guard-only
median time fell from 1.234 ms to 12.29 microseconds, excluding construction
and arithmetic. This is neither a full-model nor an HE speedup claim.
The separate [joined-model integration](validation/bert_full_model_immutable_masks_cpu_20261103/README.md)
now passes all 46 tests and both complete K15/K128 trajectories at the original
numerical gates. Across 14,417,920 physical-slot comparisons, worst stage
error is 7.79560e-10 at the fixed 1e-7 gate. All 2,648 actual domain/bound
observations pass; native-logit errors are 4.78943e-5 and 2.93576e-4 with
matching predictions. Root verified all 137 current/snapshot/archive files,
the unchanged assets, exact operation counts, case aggregates and cleanup.
The only runtime change from the baseline is the mask guard; two provenance
files also change. The 136/1,067 refreshes are identity doubles. This is
joined ideal-arithmetic evidence, with no HE, ranked accuracy or model-speedup
claim. The original baseline subsequently completed unchanged, as recorded above.

The [upstream size0 input fixture](validation/bert_size0_input_fixture_cpu_20261103/README.md)
passed 20 source-bound checks and actual tokenizer-only preparation in
3.318 seconds. The exact upstream seed42 derivation, sorted sampling and
original target order select one 128-token record with 32 valid positions.
Pinned local dataset/tokenizer resolution is explicit; model construction
and forward calls are blocked. Independent reconstruction matches the
selection and tokenization bytes. Root verified all 22 source files,
six asset hashes and six retained private files, plus the archive and
terminal records. This prepares the future size0 input; it adds no
embedding, encryption, full-model or ranked accuracy result.

The [second-handler collector preflight](validation/bert_public_model_stage2_preflight_cpu_20261103/README.md)
now passes 72 tests with 65 unchanged sources. Review found that the actual
6,461,050-byte plan exceeded the earlier 2 MiB source-member limit. The
successor permits 8 MiB per member within the unchanged 16 MiB total bound,
pins the exact first-handler source inventory, and tests the actual archive
plus both size-limit rejections. The original proof remains separate.
Its execution path requires an
externally authenticated, successful first-handler report and retained key
workspace before public model preprocessing. It calls the actual handler,
authenticates all 196 returned tensors, and binds the retained cache inode
and canonical manifest to completed worker evidence. Root verified the
current sources and archive. These preflight checks use mocked lifecycle
operations plus real public import guards.

The [actual public model preprocessing run](validation/bert_public_model_preprocessing_cpu_20261103/README.md)
subsequently passed and was reaped with exit0. It extracted and authenticated
all196 public tensors, including74 matrices, and read them back as immutable
exact FP64 lifts. Worker time was101.604s and collector time102.914s. Root
verified65 original/execution/archive source files, unchanged checkpoint
assets, the retained model-cache inode and canonical manifest, and the
third-handler completion gate against the actual report. The generic process
helper's unused `passed=false` placeholder is preserved and explained in the
archive; the worker result, terminal status and strict collector gates pass.
No model forward pass, CKKS context, input encryption or decryption occurred.

The [third-handler collector preflight](validation/bert_input_encryption_stage3_preflight_cpu_20261103/README.md)
latest successor passes 135 mocked checks and denies/restores all 21 forbidden entrypoints
in the actual installed libraries. Root verified all 81 proof files
(80 candidate files plus the external preflight driver), execution snapshots,
archive bytes and recorded imported-module hashes. The actual client path
requires completed key, model and size0-input evidence, preserves private
ordering, and validates every byte container in all eight upload tiles.
Review fixed output containment before writes and cancellation precedence;
the successor also accepts the authenticated real plan within explicit source
size bounds. The original proofs and reproductions remain archived. Every actual attempt retains
its remaining I/O for later use or diagnosis. This checkpoint executed no
embeddings, encryption, decryption or model inference. The latest successor
adds reviewed key-context and outer cleanup repairs, publishes an explicit
completion flag only after its cleanup scope exits, and republishes failure
if cleanup raises after an earlier report. Moving the final process exit
outside the termination guard prevents exit0 from concealing a restoration
failure. Preserved reproductions cover both reporting failures.

The [actual size0 input-encryption run](validation/bert_input_encryption_cpu_20261103/README.md)
passed and was reaped with exit0. The real third handler produced eight
ordered ciphertext tiles containing 838,860,800 raw payload bytes. Worker
time was 192.471s and collector time 194.649s. Root verified all 80 current,
execution and archived source files, completed prerequisites, process cleanup,
and the explicit terminal-scope flags. Independent readback authenticated all
eight payload containers, headers, residues and digests, plus the retained
private input/order/request binding and permissions. Keys and ciphertext
payloads are retained outside the public archive. This completes encryption
and transport validation only; server inference and decrypted accuracy are
separate obligations.

The isolated [stage4 engine cleanup follow-up](validation/bert_stage4_engine_cleanup_v2_cpu_20261103/README.md)
passed 25 lifecycle and real-signal checks with 13 unchanged sources.
Independent reproductions found cancellation precedence, interrupted cleanup
and partially installed signal handlers; the original failures and first
candidate remain preserved. The final candidate blocks signals during handler
installation/restoration and attempts every compiled-owner cleanup before
closing the shared NTT provider. Root verified current/archive bytes and
the unchanged original engine. This candidate is reserved for a separate
stage4 assembly; the measured key-generation sources and frozen stage3 runtime
are unchanged.

The [stage4 source assembly](validation/bert_stage4_source_assembly_cpu_20261103/README.md)
passes ten static checks over198 proof files. It combines the59-file handler
runtime with137 joined-model files and the reviewed engine-cleanup overlay.
All30 shared core files match; the two same-name module resolutions also match
exactly. Static resolution covers75 local dependencies without loading their
modules. Root verified current and archived bytes. This establishes an import
closure only; that checkpoint executes no server computation.

The subsequent [server-stage preflight](validation/bert_server_compute_stage4_preflight_cpu_20261103/README.md)
passes 113 mocked checks over 219 proof files and authenticates the completed
first three stages using public evidence. A separate 218-file probe denies
and restores all 25 forbidden entrypoints in the installed libraries and
binds 69 loaded local modules. Root verified both exact source archives and
current files. The 217-file candidate includes the reviewed cleanup
successors and is now frozen for the actual size0 full BERT12 server run.
That run remains active; decryption, numerical accuracy and ranked evaluation
are still pending.

The [key-context cleanup successor](validation/bert_key_contexts_cleanup_v2_cpu_20261103/README.md)
passes 127 checks over 77 unchanged proof sources: the existing 104 checks,
10 deterministic signal checks, nine sampled subprocess cancellations, and
four mask-mutation regressions. The [outer cleanup helper](validation/bert_stage3_cleanup_successor_cpu_20261103/README.md)
passes 30 checks, and the [engine mask-capture successor](validation/bert_engine_mask_capture_successor_cpu_20261103/README.md)
passes 37 checks. Each saves the prior handler or mask before a mutating
operation and preserves the applicable primary-error precedence. Root
verified current and archived bytes. Main-thread signal masks do not make
process-wide signals atomic when native worker threads exist: the observed
counterexample and failed stress run remain preserved, and these proofs
explicitly exclude that stronger guarantee. These changes establish scoped
cleanup behavior, not cryptographic or numerical correctness.

The [size0 client reference](validation/bert_size0_client_reference_cpu_20261103/README.md)
passes20 checks with23 unchanged sources and all156 approximate domain
observations on the retained K32 sample. Independent exact arithmetic differs
from FP64/eager native logits by at most3.553e-14. Approximate logits differ
from the official default FP32/SDPA result by at most6.193e-5 and pass the
original P2 tolerance with matching prediction. Root verified the public
archive and private input/target/reference binding. This is client-only
plaintext evidence; it supplies no encrypted or ranked accuracy result.

The [Qwen candidate3 public feasibility proof](validation/qwen_candidate3_uniform_norm_feasibility_public_20261103/README.md)
and [fixed polynomial selection](validation/qwen_candidate3_uniform_selection_public_20261103/README.md)
expand all113 original normalization domains by the same factor-four policy.
The fixed degree ladder selects36 certified polynomials and retains77 Newton
fallbacks; all610 unsuccessful polynomial attempts remain recorded. An
[independent exact-rational review](validation/qwen_candidate3_independent_certificate_review_cpu_20261103/README.md)
recomputed all36 certificates and checked the unchanged28 downstream radius
implications. Root verified both public archives. These certificates concern
ideal arithmetic within declared domains; they do not establish model-domain
coverage or CKKS accuracy. Ninety-three sites still lack a two-level reserve
from capacity25.

The frozen candidate subsequently passed the [complete original calibration](validation/qwen_candidate3_approximate_calibration_cpu_20261103/README.md):
all 416 contexts satisfy the unchanged numerical, domain and exact-semantic
checks. Top-1 agreement is 288/288 for prefill, 64/64 for teacher forcing,
and 64/64 for free-running steps, with 16/16 complete sequences matching.
Worst operator, block and full-logit nRMSE are 0.00206666, 0.0000748209
and 0.000185609. Root verified all 55 current/archive source files, seven
current model assets, workload accounting and numerical aggregates; the
process exited zero and was reaped after 1363.922 seconds. This is full
plaintext Qwen28 calibration, with no HE or held-out result in that run. The
separately frozen fresh prompts and original published regression were
evaluated afterward; neither was used for this candidate's selection.

The [fresh56 validation preflight](validation/qwen_candidate3_fresh56_preflight_cpu_20261103/README.md)
passes 116 tests plus 19 subtests over 74 unchanged sources while preserving
all 55 calibrated runtime files exactly. It binds the successful calibration
before opening the frozen fresh input bodies and retains the original
44/48 prefill, 29/32 teacher and 6/8 complete-sequence gates. Every required
context must also pass its numerical and domain checks. A repaired read-state
record distinguishes unstarted, attempted and confirmed reads on failure;
the earlier proof is preserved. Root verified the final source/archive bytes
and that preparation opened no fresh inputs or checkpoint payloads.

The subsequent [fresh112 evaluation](validation/qwen_candidate3_fresh56_validation_cpu_20261103/README.md)
passed all domain, numerical and exact-semantic gates. Top-1 agreement was
48/48 for prefill, 32/32 for teacher forcing and 32/32 for free-running
steps, with all eight complete sequences matching. Worst operator, block
and logit nRMSE were 0.00159283, 0.000155439 and 0.000169702. The process
exited zero and was reaped after 375.599 seconds. Root verified all 76
current/archived source and input files, seven current model assets,
per-context metric gates and workload accounting. The candidate manifest
and numerical gates stayed unchanged. This is plaintext Qwen28 evidence
on the separately frozen fresh prompts. Encrypted Qwen remains unverified.

The [original published regression](validation/qwen_candidate3_published_regression_cpu_20261103/README.md)
also passed all 112 contexts with the same frozen candidate: 48/48 prefill,
32/32 teacher-forced and 32/32 free-running top-1 matches, with all eight
complete sequences matching. All domain, numerical and exact-semantic gates
passed. Worst operator, block and logit nRMSE were 0.00258340, 0.0000555667
and 0.000145261. The process exited zero and was reaped after 395.908 seconds.
Root verified the 71 current/archived sources, seven current assets and every
context's accounting and gates. Its separate [preflight](validation/qwen_candidate3_published_preflight_cpu_20261103/README.md)
passed 110 tests plus 19 subtests over 70 unchanged sources before opening
the published input bodies. Candidate2's failed fresh evaluation remains
preserved. These candidate3 results establish the declared plaintext checks;
full encrypted Qwen execution and hidden-suite coverage remain outstanding.

The [client-decryption scaffold](validation/bert_client_decryption_stage5_preflight_cpu_20261103/README.md)
passed 169 mocked checks over 88 unchanged sources, including all 59 copied
handler runtime files. Root verified current and archived bytes and reviewed
the private result/receipt path. The gate authenticates completed server
evidence before private reference reads; size0 acceptance uses the original
P2 tolerance against official FP32 logits and the same predicted class.
This preparation performed no actual decryption or private payload reads.
The full encrypted server run and subsequent client numerical checks remain
pending; no ranked encrypted accuracy is established.

Its [reviewed dependency successor](validation/bert_client_decryption_stage5_ready_cpu_20261103/README.md)
preserves all implementation bytes, passes the same 169 checks over 89
sources, and denies/restores 21 installed entrypoints with 50 verified local
module bindings. Root verified the archive and current sources. A separate
[actual pending-server check](validation/bert_stage5_pending_server_gate_cpu_20261103/README.md)
authenticated the completed first three stages and rejected the actual
in-progress server report before either later client gate or any private
payload read. Its 90-source archive is verified. Actual decryption remains
dependent on successful completion of the encrypted server run.

The isolated [prepared rotation-key proposal](validation/prepared_rotation_cache_mock_cpu_20261103/README.md)
passes 38 mocked checks over five unchanged sources. It retains at most two
completed key states within a 2 GiB logical-array budget, with explicit
authenticated backing ownership and failure retention. Real-loader wiring
and encrypted parity remain pending. A separate [decomposition review](validation/bert_rotation_decomposition_public_20261103/DESIGN.md)
identifies same-input work shared by the routing rotations while preserving
each rotation's own ModDown, first-component addition and coefficient map.
Root verified all 13 reviewed source files. Neither proposal changes the
active runs or establishes an encrypted speedup.

The [full Qwen public implementation plan](validation/qwen_candidate3_full_he_plan_public_20261103/README.md)
passes 159 checks over 16 unchanged sources. Root verified its archive,
current files and recomputed public schedules. All 113 normalization rules
retain their frozen polynomial coefficients or Newton iteration counts;
segmented inverse-only refresh gives a level-feasible continuation for the
93 sites that otherwise lack capacity. The 260 internal inverse refreshes
are counted before tile multiplicity and outer-boundary refreshes.
Required conditional raw-input bounds reach 32,768 and learned-weight
normalization-output bounds reach 8,192. Those ranges and propagated error
remain unproved in HE. The plan preserves D128 heads, 16 Q/8 KV heads,
RoPE, causal K-specific softmax and all 151,936 vocabulary outputs across
38 streamed blocks. It supplies implementation requirements and raw storage
counts, with no completed encrypted owner, measured peak or runtime claim.

The subsequent [ordinary rotation decomposition proof](validation/bert_ordinary_rotation_digits_public_cpu_20261103/README.md)
passes 372 exact residue comparisons and 180 rejection checks across small
N16/N256 rings at six Q-prefix partition boundaries, using actual JAX on
public synthetic operands. All original rotation methods remain unchanged;
the added method preserves the original completion body after substituting
prepared digits. Nine collector regressions and 40 exact source hashes pass,
with terminal exit zero and reaping. Root verified current/archive bytes,
AST identity and all recorded comparisons. The earlier collector failure,
which included its own output copies in source discovery, is preserved.
This establishes scoped arithmetic parity; group ownership, genuine encrypted
parity, common-ring execution and performance remain separate obligations.

The [prepared-key loader successor](validation/prepared_rotation_loader_mock_cpu_20261103/README.md)
passes 116 mocked checks over 14 exact sources. It authenticates existing
public key files before exposing an optional cache owner and retains original
cleanup objects through retirement. Two reproduced failures, including a
replacement that could hide pending buffers, remain preserved with the
earlier 108-check proof. Root verified the successor's current/archive bytes,
test counts and cleanup changes. The 2 GiB cache cap is per arithmetic owner;
real encrypted parity and measured memory remain pending.

The [Qwen head-routing owner](validation/qwen_head_routing_owner_ideal_cpu_20261103/README.md)
passes 72 ideal/guard checks over 12 unchanged sources. Its actual operation
loops implement D128 half-split RoPE, eight-to-sixteen KV-head expansion and
causal QK score packing for S1/8/16/19/32. Independent dense/scalar oracles
check every physical slot, including padding and noncontiguous positions.
The owner checks each addition's level and left-owned scale independently;
the earlier metadata failure is preserved. Root verified the archive,
current sources and terminal flags. Cryptographic primitives/security are
mocked in these tests; encrypted attention and full Qwen remain outstanding.

The [ranked input-fixture preflight](validation/bert_ranked_input_fixture_preflight_cpu_20261103/README.md)
passes 32 mocked checks over 27 exact sources for all nine size100/200/408
and seed42/43/44 combinations. It preserves the original upstream selection,
ordering and tokenizer bodies, private permissions and interrupted cleanup.
Root verified the source/archive hashes, absent producer processes and zero
attempted real asset reads. This is preparation only, with no ranked HE
accuracy or benchmark result.

The subsequent [actual ranked fixtures](validation/bert_ranked_input_fixtures_cpu_20261103/README.md)
contain all nine cases and 2,124 selected records from those exact 27 sources.
Root independently verified every private file, selection, tokenizer byte and
permission; all three full-408 inputs are identical. Public evidence contains
only counts and hashes. The collector's terminal flags pass; its historical
outer exit/reap observation is unavailable and is explicitly left unknown.
The separate private readback exited zero and was reaped. No model forward,
encrypted ranked inference or ranked accuracy has been measured here.

The [Qwen attention-value routing successor](validation/qwen_pv_routing_owner_ideal_cpu_20261103/README.md)
passes 88 ideal/guard checks over 17 unchanged sources. Root verified current
and archived bytes and the test count. It composes the head-routing layout
with causal probability-by-value aggregation across every physical slot.
Probabilities are explicit public stand-ins; these tests do not implement or
validate encrypted softmax. The combined S32 head/PV key set has 117 indices,
excluding other full-model and refresh requirements. Encrypted parity remains
pending.

The first [ordinary rotation-group proof](validation/bert_ordinary_rotation_group_public_cpu_20261103/README.md)
passes 51 mock/collector checks and 372 ordered public-array comparisons over
45 exact sources. Root verified the archive and current bytes. Review then
found an uncovered all-zero callback path that could replace a pending source
payload before cleanup inspected it. This historical proof is preserved;
an explicit-original-array successor is being prepared before loader or
encrypted use. The arithmetic comparisons remain scoped public evidence,
and do not establish lifecycle completeness or an encrypted speedup.

The separate [rotation-group cleanup successor](validation/bert_ordinary_rotation_group_successor_public_cpu_20261103/README.md)
retains original source, result and control arrays and rejects truncated
binding inventories. Four reproductions fail against the original source;
all 58 successor checks and the same 372 public-array comparisons pass over
46 unchanged sources. Root reviewed the narrow change, verified current and
archived bytes, and checked producer absence. The corrected owner still needs
real loader integration and genuine encrypted parity before adoption.

The [prepared-key encrypted-parity preflight](validation/prepared_rotation_parity_preflight_cpu_20261103/README.md)
passes 92 mocked checks, 28 installed denial/restoration checks and the actual
public prerequisite gates. Root verified all 233 proof files, including the
231 frozen candidate sources, and the absent/reaped child processes. Its
genuine CPU run now compares 24 ordinary/prepared rotations on the same
authenticated input at L49 and after an actual level reduction to L25.
It reuses the completed key and input stages without client decoding or model
execution. Exact ciphertext parity, cleanup and cache-hit results are pending;
logical cache caps do not establish memory usage or a performance improvement.

The separate [BSGS routing proposal](validation/bert_bsgs_routing_plan_public_cpu_20261103/README.md)
passes 392 public NumPy checks over ten unchanged sources, covering all 128
tile/offset maps in both directions and both forward gains. All 50,331,648
physical-slot comparisons exactly match independent direct scatter. Grouping
the existing output masks preserves two-level depth and 28 plaintext terms,
reduces internal rotations from 26 to 11 per map, and increases rescale calls
from two to seven. Its 36 routing keys are a subset of the current 66. This
is public algebra and operation-count evidence only; changed key-switch and
rescale order requires a new encrypted numerical and performance proof.

The [Qwen RMSNorm owner](validation/qwen_rmsnorm_owner_ideal_cpu_20261103/README.md)
passes 295 owner/public/guard checks and six separate public-parameter checks
over 36 unchanged sources. Its actual loops cover all 113 frozen sites using
authenticated learned gamma vectors, all physical slots, exact selected
polynomials or Newton counts, and the existing 260 inverse-only refresh
events. Root verified current/archive bytes, both test counts and process
retirement, and reviewed the accessor-replacement repair. These tests use
ideal arithmetic, mocked security and identity refresh. Raw-input/output
refresh bounds, encrypted errors and full-model composition remain unresolved.

The [ordinary-group parity preparation](validation/bert_ordinary_group_parity_preflight_cpu_20261103/README.md)
passes 214 mocked checks, 28 installed denial/restoration checks and the actual
public prerequisite gates over 258 unchanged sources. Its separately frozen
[execution-ready successor](validation/bert_ordinary_group_parity_ready_cpu_20261103/README.md)
passes the same checks over 259 sources while preserving every runtime byte.
Root verified both archives, current sources, exact readiness differences and
process retirement. Neither preparation executed encrypted arithmetic.
The subsequent [genuine ordinary/grouped rotation comparison](validation/bert_ordinary_group_rotation_parity_cpu_20261104/README.md)
completed all **66 ordered pairs** at L49 and legitimately reduced L25:
58 nonzero pairs, eight zero aliases, 116 rotation calls and six shared
decompositions. All **1,314,914,304 ciphertext words** and typed metadata
matched exactly. Root verified all259 original/execution/start/end/archive
source entries, the selected collector gates, worker cleanup, process retirement
and retained borrowed-directory identities. The raw nested process record's
unused `passed: false` initializer is preserved; typed exit0, `result.passed`,
the authenticated worker proof and all selected gates passed. Both rotation
paths share implementation dependencies. This component result establishes
representation parity; decoded accuracy, full-model correctness and calibrated
performance remain outside its scope.

The [Qwen continuation bounds](validation/qwen_softmax_silu_bounds_public_20261103/README.md)
pass 93 checks over 12 unchanged sources. Root verified the archive/current
files, recomputed certificates and inspected the frozen arithmetic expressions.
Exact rational analysis certifies the proposed B2 events, the completed
six-update initial inverse at B4, and the existing larger K-specific inverse
bounds. A scheduled K8 inverse exceeds four, preserving that distinction.
The 22 rational-SiLU sites retain all 145 doublings. Four softmax layers have
zero exponential-domain endpoint margin; these exact-real results exclude
floating-point, encoding, CKKS and refresh errors and establish no uniform
perturbation tolerance or full-model encrypted result.

The [optimized TPU toy sweep](validation/optimized_transformer_tpu_sweep_20261103/README.md)
completed all twelve cases on November5: seven passes (BERT1–3 and Qwen1–4)
and five failures (BERT4–6 and Qwen5–6). The [final report](validation/optimized_transformer_tpu_sweep_20261103/final/report.json)
and [root terminal checks](validation/optimized_transformer_tpu_sweep_20261103/final/root-terminal-admission.json)
preserve the original parent exit1 and the complete source/cleanup evidence.
Root verified all199 current/archive source files, completed-child retirement,
owned-key cleanup and input/output TPU/security records. Qwen4 took21146.86s
cold runner total (21154.16s external), with maxabs0.02584769,
nRMSE0.01601423 and top1 agreement1.0. It passes the existing toy gate but
exceeds the separate full-Qwen HE nRMSE0.01 gate and contest timing target.
All twelve child processes and the original parent are absent, their process
groups and owned key directories are clear, and all199 source files still match
the archive. All six Qwen public cases completed, with two failures.
The [BERT4 failure](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/bert_test4/DIAGNOSIS.md)
occurred after both toy blocks and five logged refresh completions, when the
decoder's approximation-error estimator overflowed to infinity and rejected
the output. No decoded-logit error, nRMSE or prediction metric was produced.
Both recorded boundaries used TPU ciphertexts with secure actual QP, but the
security report's overall failure remains preserved. Output level11 and finite
scale do not establish numerical correctness. Root verifies all199 sources,
six exact case files, child retirement and owned-key cleanup; the original
source snapshot stayed unchanged. This synthetic two-layer fixture took13866.62s externally.
Intermediate ciphertext diagnostics are needed to locate the first failure.

The [BERT5 failure](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/bert_test5/README.md)
is a separate three-layer, batch-two, length-12 toy fixture. It completed eight
logged refreshes and reached the output TPU/security checkpoint, then reported
`OverflowError: int too large to convert to float`. No decoded error, nRMSE or
prediction metric was produced. The recorded output has 40 Q limbs and finite
scale; actual QP remains within the unchanged 128-bit ceiling. These metadata
checks do not establish numerical correctness or identify the precise throw
site. Root verifies all 199 source files, six exact case files, child retirement
and owned-key cleanup. External process duration was 19,834.48s; this failed run
is not a calibrated baseline. The final parent report preserves this failure.

The [Qwen5 failure](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/qwen_test5/README.md)
reached all six layer attention/MLP checkpoints and completed 19 refreshes,
then failed strict client decoding: estimated standard deviation4.099e48
exceeded the unchanged0.03125 limit. The three-layer public toy returned
a TPU ciphertext at L5 with12Q limbs, NSD1 and finite scale; those metadata
checks do not establish numerical correctness. No decoded accuracy metric
was produced. Root verified all199 source files, six exact public case files,
child exit1/reaping, group absence and owned-key cleanup. External duration
was30,241.93s. The first corrupting operation remains unknown; the saved
progress messages cannot localize it. This rejection differs from BERT5's
recorded conversion exception and is not a calibrated baseline.

The [Qwen6 failure](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/qwen_test6/README.md)
is a separate three-layer public toy fixture with batch 1, sequence length 16
and seed 9106. After six layer milestones and 19 logged refresh completions,
strict decoding rejected an estimated standard deviation of 6.314e81 against
the unchanged 0.03125 limit. No decoded-logit metrics were produced. The output
has 12 Q limbs, NSD 1 and finite scale; its Q product is within binary64's
finite range. These facts do not locate the first numerical divergence.
Root verified all 199 frozen sources, six exact case files, collector-recorded
exit 1 and reaping, current PID/group absence and owned-key cleanup. External
duration was 31,490.24s. Sampled aggregate process-tree RSS peaked at
141,938,708,480 bytes, subject to shared-page double counting and missed
transients. The failed run is not a passing benchmark or full-model result.

The [BERT6 failure](validation/optimized_transformer_tpu_sweep_20261103/checkpoints/bert_test6/README.md)
completed eight logged refreshes and all six layer milestones, then strict
client decoding rejected an estimated standard deviation of1.036e26 against
the unchanged0.03125 limit. No decoded-logit metrics were produced. The case
took41,889.49s externally; progress logs do not locate the first encrypted
numerical divergence. The complete twelve-case sweep took200,478.90s and
ended with seven passes and five failures. The original six-case P3 baseline
sweep subsequently completed with **all six cases passing** in3369.48s. Its
original parent exited0; all128 measured sources/fixtures and the auxiliary
sentinel matched their snapshots, and all completed groups and owned child key
paths were absent. Root checked the empty key parent's original identity before
removing it. The [final six-case archive](validation/p3_original_six_tpu_20261105/final/README.md)
preserves every case, exact cold timing, source archive and terminal receipt.
These synthetic results do not replace trained accuracy or the separate compact
five-minute profile.

After P3 cleanup, the separately reviewed [BERT4 diagnostic setup](validation/bert_toy4_actual_setup_20261105/README.md)
started on CPUs160–167 with a required TPU, fresh workspace parent and exact260-file
activation. Root observed its original collector and worker live at13:38:35 UTC.
On November6, the [public setup readback](validation/readiness_review_20261106/report.json)
verified consistent successful collector/stage/worker reports, recorded child
exit0 and reaping, all260 sources across two trees and three archives, current
process/group absence, and both retained workspace identities. Collector-reported
elapsed time was3151.16s. Original session85433 is unavailable and its outer-tool
exit/reap receipt was not recovered; strict terminal admission remains pending.
Server and validator have not started, and numerical diagnosis remains pending.
Original arithmetic and precision gates are unchanged, and the disabled
decoder-comparison design is not selected.

The [Qwen5 public CPU replay](validation/qwen_toy5_public_replay_cpu_20261104/README.md)
passed 140 packed/dense semantic comparisons with maximum absolute difference
2.66454e-14 across 64 exact replicas. All 22 original polynomial input-domain
checks and 19 simulated refresh input-bound checks passed. The replay independently
derived 19 refreshes and 47 checkpoints, ending at L5 with 12Q limbs. Its clear
output passes the original toy gate: maxabs 0.0393266, nRMSE 0.0182984 and top1
agreement 1.0. Root verified all 27 source files, terminal exit0/reaping and the
saved report; the later process observation was not a live observation. Refresh
is plaintext identity with simulated nominal metadata. This evidence does not
locate the encrypted failure, establish encrypted accuracy or satisfy the separate
full-Qwen HE nRMSE0.01 gate. The original TPU failure and queue remain unchanged.

The [BERT5 public conversion diagnosis](validation/bert_toy5_overflow_diagnosis_cpu_20261104/README.md)
passes 19 synthetic checks over nine source/proof files. Using the recorded Q40
geometry, it reproduces overflow when converting a centered integer before
division by scale, even when the scaled value is representable. An isolated
two-expression candidate preserves the normal conversion path and original
precision gate. NumPy performs CPU guard arithmetic; CRT and FFT are explicit
test substitutions. The earlier 18-test run's incorrect symmetry expectation
is preserved with its correction. Root verifies both proof histories and the
source bindings. The actual throw site remains unproved, the candidate is not
installed, and no encrypted precision improvement follows from these checks.

The [executable scaled-CRT decoder candidate](validation/ckks_scaled_crt_decoder_cpu_20261104/README.md)
passes 35 clear-arithmetic CPU checks with real CRT and FFT, including 65,536
output slots at degree 131,072 using the recorded Q40 moduli. Its only runtime
changes are one scaling helper and the two coefficient-conversion calls; the
original 1/32 approximation gate and normal conversion results remain unchanged.
Root verifies all 207 proof sources, the test results and retirement of the
worker-recorded process/group. The first development run's TPU-library metadata
import assertion and a later reviewer path error remain preserved with their
separate corrections. The final harness explicitly blocks the TPU import and
uses CPU devices. The original 199-source sweep remains unchanged, and this
isolated candidate establishes neither BERT5 recovery nor an encrypted result.

A [source-only Toy4 comparison design](validation/bert_toy4_decoder_comparison_design_20261104/README.md)
preserves the queued diagnostic's original decoder and numerical gates. A later
client could compare both decoders on one retained RNS matrix only after separate
producer/consumer admission and fresh authentication of the retained workspace.
Root verifies the 20 selected source/public-metadata files. No comparison role
is implemented or authorized, and current public BERT5 evidence cannot support
retrospective decoding. This design adds no numerical result or queue change.

The optional [BERT client decoder proposals](validation/bert_scaled_crt_decoder_client_proposals_cpu_20261104/README.md)
preserve the original 122/125-source clients as launch defaults. Root verifies
the separate copies, four-file deltas, decoder AST equivalence and unchanged
producer gates. Their fixed final six-Q response has at most 180-bit centered
coefficients, so the original integer-to-float conversion cannot overflow there.
These proposals have no new preflight or launch grant and imply no BERT repair.

A bounded [clear CPU decoder comparison](validation/ckks_decode_clear_cpu_microbenchmark_20261104/README.md)
produces bit-identical outputs in all 20 full-degree calls on ordinary synthetic
coefficients. Original/candidate median times are 0.40627/0.40522s at Q6 and
1.27272/1.27604s at Q40. Four alternating pairs per case on a shared host support
only these observations, with no speedup, regression or full-HE timing claim.
Root verifies all 210 source files, retained synthetic dataset hashes, timing
summaries and process retirement. No runtime optimization or promotion follows.

The [ranked client-reference preflight](validation/bert_ranked_client_reference_preflight_cpu_20261103/README.md)
passes69 mocked checks,12 installed denials across9 entrypoints, public producer
authentication and public-asset rechecks over42 exact proof files (41 candidate
plus the external driver). Root independently verified the archive, current
sources, all six asset hashes, test count and process retirement. The frozen
candidate is cleared to produce the nine private reference sets from408 unique
records using official FP32/SDPA and separate diagnostic FP64/eager execution.
Official publication, exact-reference parity, approximation and ranked accuracy
remain separately gated. No ranked encrypted result follows from this preflight.

The subsequent [actual ranked references](validation/bert_ranked_client_references_cpu_20261103/README.md)
passed on408 unique examples using the frozen41-source candidate. All nine
100/200/408 cases pass the additional clear-reference numeric/prediction checks
and the75%-accuracy/2pp-drop checks. Official ranked HE acceptance remains
accuracy-based; prediction agreement is diagnostic. Official FP32/SDPA and
approximate predictions agree throughout; all408 cases
score345/408 (84.5588%). Root independently verified11 private reference files,
all54 original inputs, ordered case reuse and every63,648 domain observation.
The maximum exact/native-FP64 diagnostic error is1.27898e-13; maximum
approximate/official-FP32 logit error is0.00335416. Native execution and the
independent readback exited zero and were reaped. This supplies private clear
references; ranked encrypted accuracy and repeated competition timing remain
unmeasured.

The [Qwen causal-softmax public design](validation/qwen_causal_softmax_public_design_20261103/DESIGN.md)
passes52 checks over18 unchanged files and derives896 literal layer/K schedules.
Root independently verified the archive, current sources, XML count, recomputed
plans and child absence. It preserves each original reciprocal/squaring count,
all K-specific inverse bounds and an explicit actual-V bypass for K1. The
serial S32 design requires11,656 full-slot refresh calls across28 layers; that
cost and the exact-real/perturbation limits remain explicit. A new owner and
separate PV bypass successor are under ideal validation; this plan establishes
no encrypted correctness or practical timing result.

The [implemented BSGS routing owner](validation/bert_bsgs_routing_owner_ideal_cpu_20261103/README.md)
passes606 ideal/production-metadata checks over28 unchanged sources. Root
verified current/snapshot/archive bytes, XML counts and exit0/reaped producer
absence. Its actual loops match864 independent full-slot scatters
(56,623,104 slots), including all128 mappings and noncanonical scales. The
independent metadata oracle executes21 unchanged production method bodies.
Three pre-fix failures preserve the missing key/map binding and stale-output
annotation findings; both are repaired, with nested NTT/field guards added.
Required36-key identities remain subject to authenticated-loader content
stability. No finite-field execution, encrypted1e-6 result, physical-memory
bound or speedup is established; a separate actual proof remains required.

The [Qwen causal-softmax/PV owner proof](validation/qwen_causal_softmax_owner_ideal_cpu_20261103/RESULT.md)
passes194 ideal/public/guard checks over35 unchanged sources. Root read both
owners, verified current/archive bytes, the194-test XML and producer absence,
and recomputed the896 public schedules. The independent oracle preserves the
frozen sum-before-alpha order, all K2–32 reciprocal/squaring counts, full-slot
padding and the actual-V K1 bypass. Two source-bound failures preserve an
incomplete final-callback metadata fingerprint; the repaired structure guards
pass. These are ideal primitives and identity refresh. Synchronous arithmetic
is not enforced in this frozen version, and general asynchronous failure cleanup
is unproved; a separate enforcement successor is being prepared before HE use.
No full Qwen encrypted, physical-memory or latency result is established.

The [BSGS encrypted-proof public plan](validation/bert_bsgs_routing_he_public_plan_cpu_20261103/README.md)
passes25 checks over8 unchanged sources. Root verified current/archive bytes,
the XML count and an independently recomputed plan:18 ordered checkpoints,
1,179,648 compared slots,72 rotations and42 routing rescales, plus one separate
input product/rescale. The two canonical/noncanonical cases retain the dense
hostile-input inverse oracle and exact binary64 scale recurrence. Raw stored
input/output payload totals348MiB, excluding keys, temporary files and memory.
The tests completed with observed exit0; separate PID/reaping history was not
captured. No encryption, numerical CKKS gate or speedup follows from this plan.

The [ranked file-stage plan](validation/bert_ranked_file_stage_plan_cpu_20261103/README.md)
passes38 public checks over40 unchanged sources. Root independently verified
current/archive bytes and all38 XML cases. It specifies nine fresh official
harness runs:100/200/408 records with seeds42/43/44, accuracy at least75% and
native-to-HE accuracy drop at most2pp. Per-run time sums official stages5–9;
the median across three seeds determines each size's throughput. Prediction
agreement is diagnostic. A408-record run needs321.14GiB raw encrypted IO before
keys; the current disk cannot hold even a100-record batch. Generalized handler
and readback successors are being prepared. Ranked HE execution remains pending.

The [Qwen synchronous-execution successor](validation/qwen_causal_softmax_synchronous_cpu_20261103/RESULT.md)
passes232 ideal/public/guard checks over43 unchanged sources. It requires exact
low-memory flags for arithmetic and refresh and binds the original completion
method through callbacks. Unchanged production-method tests with a mock barrier
verify completion before callbacks and rejection on barrier failure/cancellation.
Root reviewed both owner changes, verified current/archive bytes, all232 XML
cases and completed-child absence, and recomputed the896 public schedules.
The194-check predecessor and four expected low-memory rejection failures remain
preserved. No real backend or HE ran; failed-readiness resource quarantine is an
outer dedicated-process obligation, and full Qwen integration is still pending.

The [ranked stage preparation successor](validation/bert_ranked_file_stage_preflight_cpu_20261103/README.md)
passes141 public/mock checks over241 unchanged sources. Root verified current
and archived bytes, XML counts and the completed producer's absence. It binds
all nine reference cases, all8N inputs/N outputs, private result order and
accuracy-only acceptance; synthetic tests execute the unchanged file-handler
loops at100/200/408 records using explicit crypto/model/transport doubles.
Cleanup-only cancellation regressions and their repairs are preserved. This
version has no actual ranked worker launcher or encrypted batch result. Fresh
official setup/worker wrappers are being prepared separately; existing size0
keys, IO and receipt state remain untouched.

The [fresh ranked setup and worker preflight](validation/bert_ranked_official_worker_preflight_cpu_20261103/README.md)
passes 294 synthetic tests and 106 installed guard checks over 272 unchanged
sources, including all 241 predecessor files. Root verified current/archive
bytes, XML counts, installed module bindings and the completed producer's
absence. Fresh key/model setup reports bind the official directory layout to
ordered all-N stage handoffs; cleanup and outer termination must finish before
success is accepted. Seven literal entrypoint bodies and original timing
aggregation are present. The proof workers combine preprocessing/encryption
and decryption/postprocessing, so their durations remain diagnostics. The
original seven-process harness integration, size0 encrypted acceptance and
sufficient storage remain prerequisites. No ranked keys or encrypted batch
were produced by this preflight.

The [Qwen conditional join plan](validation/qwen_candidate3_joined_continuation_public_20261103/README.md)
passes113 public checks over27 unchanged sources. Root verified current/archive
bytes, XML counts, child absence and an independent full plan recomputation.
Refreshing before RoPE leaves scores atL20, below the existing softmax'sL21
entry; the new plan places conditional refreshes after RoPE and adds nine
reserve events for L2 norm outputs. It preserves all113 norm rules and original
nonlinear counts. The S32 schedule totals12,931 full-slot refreshes, with
conditional post-RoPE boundsB64–2048. None of868 generic centered-score envelopes
proves the frozen exponential domains. These are explicit exact-real and scale
assumptions; real refresh errors, domain coverage, joined encrypted execution
and practical memory/timing remain unproved.

The [public Qwen Q/K/V export](validation/qwen_qkv_public_parameters_cpu_20261103/README.md)
contains metadata for all 84 full matrices. Root independently checked all
117,440,512 saved FP64 values against exact BF16 promotions from the pinned
checkpoint, including file digests and checkpoint hashes before and after.
The successful export retains its original four-source snapshot. A later
descriptor-handoff leak and its separate three-source, 14-test repair remain
recorded without attributing the export to repaired code. Tensor bodies stay
outside the archive. This is public parameter preparation; no model forward
or encrypted inference ran.

The [remaining public Qwen decoder parameters](validation/qwen_decoder_public_parameters_cpu_20261103/README.md)
passed export and independent readback from ten unchanged sources. All 28 O
projections, 84 gate/up/down matrices, and both complete embedding/vocabulary
names are present. Actual BF16 bytes establish that the latter two share one
physical matrix: 114 named descriptors refer to 113 files. Root checked all
478,543,872 unique FP64 values against the checkpoint, including all file and
payload digests. The public export took 26.215 seconds; session 81225 exited
zero, was reaped, and its child/group cleanup completed. Final preparation
passed 53 synthetic and nine independent checks. Earlier cancellation and
malformed-return failures remain preserved. Matrix bodies stay outside the
archive; no model forward, backend or encrypted inference ran.

The [disabled BSGS proof preflight](validation/bert_bsgs_routing_he_preflight_cpu_20261103/README.md)
passes 361 synthetic checks and nine installed probes over 142 unchanged sources.
Subsequent review preserves two failures in that exact closure: extreme finite
decode values can prevent failure-report serialization, and an exited leader
can leave descendants while the old collector reports complete reaping. Its
readiness check also accepted an integer alias for true. This predecessor is
retained as preparation evidence and is not cleared for encrypted execution.

The separate [BSGS ready preparation](validation/bert_bsgs_routing_he_ready_cpu_20261103/README.md)
passes 402 synthetic checks and nine installed probes over 147 unchanged sources.
Root verified current/snapshot/archive bytes, all XML cases, public key gates,
direct-child retirement and owned-group absence. Ten explicit overlays bind the
disabled predecessor to exact-boolean readiness, serializable failure metrics
and process-group cleanup. Original body failures and cleanup-only cancellation
retain their intended priorities. Installed probes initialize the CPU backend
but execute no encryption or model work. This is preparation, not a numeric pass.

The subsequent [genuine BSGS fixtures](validation/bert_bsgs_routing_encrypted_fixtures_cpu_20261103/README.md)
passed in 151.396 seconds using that frozen 147-source candidate and existing
authenticated keys. Two L49 ciphertexts contain 200 MiB of raw payload for the
fixed public full-slot patterns. Root authenticated both complete ciphertext
readbacks and confirmed client process/group retirement. Session 5310 exited
zero and was reaped. This fixture-only archive records the intermediate state
before the separate server and validation phases described below. No keys or
ciphertext bodies are archived.

The completed [genuine BSGS routing component](validation/bert_bsgs_routing_encrypted_cpu_20261103/README.md)
passes all 18 ordered checkpoints and 1,179,648 physical slots at the unchanged
1e-6 absolute gate; maximum error is 2.0359652806592398e-10. Canonical and actual
noncanonical scales cover forward routing, roundtrip inverses and hostile-input
inverses. Root independently verified all 147 current/execution/archive source
files, all three terminal receipts, complete encrypted payload readbacks before
validation, metric aggregation and process/group absence. The keyless server
took 13,745.564 seconds and the private validator 116.590 seconds; sessions
89442 and 6747 exited zero and were reaped. Routing used 72 rotations and 42
rescales, plus one separate input product/rescale. This component pass does not
establish a controlled speedup, full BERT accuracy or competition latency.

The [prepared-rotation encrypted parity run](validation/prepared_rotation_parity_cpu_20261103/README.md)
passes all 12 pairs at levels 49 and 25: 239,075,328 compared uint32 words have
zero mismatches, with exact metadata equality across 24 real rotations. Root
verified all 231 current/execution/archive source files and the public terminal
evidence. The actual 49-to-25 reduction preserves the ordered modulus prefix,
scale and noise degree. Each level records two publications and four cache hits.
The logical retained-byte ceiling is separate from physical memory; measured
peak RSS was 194,483,196 KiB. Session 41655 exited zero and was reaped. This is
representation correctness evidence; no decryption, model accuracy or measured
speedup follows from it.

The [seven-executable ranked harness proof](validation/bert_ranked_seven_executable_preflight_cpu_20261103/README.md)
passes 45 tests over 300 unchanged sources, including all 272 predecessor files
and 11 original harness sources. Root verified the archive, current files, test
count and absence of all 87 tracked processes. Separate original entrypoints
exercise all nine size/seed cases and explicit repeat handling, with 2,124
synthetic records in the default schedule. Original stage intervals and
sum-before-median aggregation are preserved. Crypto, model, tokenizer and data
are explicit test doubles; genuine execution remains disabled. A subsequently
found preflight CLI exit-status defect is preserved for repair in the separate
genuine bridge. No ranked key generation or encrypted batch ran here.

The [joined Qwen attention ideal run](validation/qwen_attention_join_ideal_cpu_20261103/README.md)
passes 144 checks over 52 unchanged sources. Seven fixtures use authenticated
full Q/K/V and gamma matrices with synthetic rows, covering 589,824 output
slots; maximum error is 2.220446049250313e-15 at the fixed 1e-8 gate. Root verified
source/archive identity, all XML cases, metrics and child/group retirement.
The joined path includes input RMS, Q/K/V, Q/K normalization, RoPE, GQA, causal
softmax and PV, returning level20 attended features (level22 for one token).
It uses ideal primitives and identity refresh; output projection, residual,
MLP, full-model trajectory and encrypted error remain separate work. Encountered
domains pass, but the broad public bounds do not prove all frozen domains.

The [Qwen SiLU ideal owner](validation/qwen_silu_owner_ideal_cpu_20261103/README.md)
passes 181 checks over 34 unchanged sources, including all 28 frozen nonlinear
rules. Sixty fixtures cover 3,932,160 physical slots with maximum error
1.1368683772161603e-13 at the unchanged 1e-8 gate. Root verified current/archive
bytes, test counts, domain/metric accounting and child/group absence. The path
preserves all coefficients and 145 rational doubling events, consuming level24
and returning level6 with actual scale propagation. A separate three-check
verifier repair fixes tuple-versus-JSON-list comparison without changing the
measured sources. These are ideal/identity-refresh results; real refresh errors
and the rational denominator domain remain conditional obligations.

The [actual-matrix Qwen key plan](validation/qwen_actual_matrix_key_plan_cpu_20261103/README.md)
covers all 234 maps from 197 public matrix files: 84 Q/K/V, 112 O/MLP and 38
vocabulary blocks. Root verified its 22-source current/execution/archive closure
and 21-test preflight. Independent public-body readback confirms every actual
diagonal support. The union contains 166 keys, including unreduced conjugation,
and accounts for 147.84375 GiB of raw rotation-key arrays at p60/d8. This excludes
other keys, model/IO arrays, compiler state and duplicate storage; it is neither
a memory-capacity guarantee nor evidence of key generation or encrypted execution.

The [historical initial-softmax run](validation/bert_initial_softmax_historical_failure_cpu_20261103/README.md)
failed an exact canonical-scale assertion after the first fixture reached its
final ciphertext: the actual scale differs by one binary64 ULP. No client
numeric checks ran, and the second fixture did not execute. Session 34588 exited
one and was reaped; root verified all 110 unchanged source files and completed
cleanup. The corrected candidate's completed result is recorded separately.

The [corrected initial-softmax encrypted component](validation/bert_initial_softmax_corrected_component_cpu_20261104/README.md)
passes all14 checkpoints and917,504 slot comparisons at the unchanged1e-6
absolute gate. Worst error is2.8573543531251744e-10; final probability errors
are2.23472802762803e-12 and5.370348610256315e-11. Both fixed public fixtures
use every65,536 slots, including hostile padding and partial rows. The secure
N131072/Q100/P14/dnum8 CPU run completes all server checkpoints before any
client decoding. Root verifies all113 current/start/end/archive sources,
exit0 and process retirement, and removal of owned keys. The86,974.36-second
run is one component measurement: it performs no refresh or subsequent five
normalized-squaring rounds and establishes no full attention, model, ranked
accuracy or calibrated performance result.

The [normalized-round encrypted run](validation/bert_normalized_round_component_failure_cpu_20261103/README.md)
completed all 54 checkpoints and client validation but failed the fixed 1e-6
absolute gate: six K128 inverse estimates reached 1.9912012305667304e-6 error,
and the final true-inverse diagnostic also failed. Both final normalized
probability checks pass (K128 1.2224e-10; K15 9.7461e-11), as do all metadata,
finite and tracked residual/invariant checks. Those successes do not override
the inverse failures. Root verified all 114 unchanged sources and retired
process/group state; session 17163 exited one and was reaped with clean key
cleanup. The failed run remains preserved without revising its arithmetic or gate.

A separate [public sensitivity diagnosis](validation/bert_normalized_round_sensitivity_cpu_20261103/README.md)
finds that the recorded K128 denominator error near 1/128 predicts about
1.991e-6 reciprocal error, consistent with the observed inverse failure. The
combined denominator and residual estimate is 1.9927291e-6; K15 gives
2.7080096e-8. These calculations use rounded aggregate metrics and cannot
establish exact slotwise cause or a rigorous interval certificate. They identify
denominator accuracy as a repair target without changing the failed run's gate.

The [BSGS attention integration](validation/bert_bsgs_attention_join_ideal_cpu_20261103/README.md)
passes 22 checks over 17 unchanged sources. Six complete all-128-key score/PV
fixtures compare 393,216 physical slots with maximum error 4.163336342344337e-17
at the fixed 1e-8 ideal gate. Actual binary64 level/scale recurrences, all 63
required key bindings and callback failure retention are checked. Session 9052
exited zero and was reaped, with the child group absent. Each score/PV path uses
1,520 routing rotations plus 768 head and 120 row rotations, retaining three
critical levels. This uses NumPy/security doubles and establishes no encrypted
attention result or speedup. Integration into the full 12-layer owner is separate;
the current encrypted full-model run remains unchanged.

The [first full BERT BSGS ideal attempt](validation/bert_bsgs_full_ideal_fixture_failure_cpu_20261103/README.md)
failed both full cases because an inherited tracked-reciprocal test helper
returned values without the structural NTT metadata required by the BSGS guard.
The other 78 of 80 checks passed. Root verified all 150 unchanged sources and
fixed asset hashes, observed exit1, and confirmed child/group retirement after
102.95 seconds. No full-case numerical result was produced. A separate successor
corrects those test bindings and adds two focused metadata regressions; its full
82-test run remains pending. The failed archive stays unchanged.

The [genuine seven-executable bridge preparation](validation/bert_ranked_genuine_seven_bridge_preflight_cpu_20261103/README.md)
passes 478 public/synthetic checks over 309 unchanged sources, retaining all 272
parent files and 11 original harness files. Root verified the portable/current
archive, producer absence and core launch/phase/timing code. Preprocessing and
encryption now compose separate authenticated proofs, as do decryption and
postprocessing, preserving the original seven executable boundaries and stage
timers. The original unversioned native-model lookup is bound to the pinned
offline checkpoint cache. Completed size0 stage5, authoritative outer reaping,
and an externally pinned full-count storage grant are required before actual
execution. Envelope, cleanup, exit-status and pytest-discovery failures and
their repairs remain preserved. Session 59964 exited zero and was reaped;
this preparation produced no ranked keys, encrypted batch or timing result.

The [Qwen decoder-tail ideal composition](validation/qwen_decoder_tail_ideal_cpu_20261103/README.md)
passes 193 checks over 54 unchanged sources. Six learned-weight fixtures cover
524,288 physical slots with maximum error 2.8421709430404007e-12 at the fixed
1e-8 gate. Root verified source/archive identity, XML cases, domain and metric
accounting, and child/group absence. Layer0 S1/S19 consumes actual attention
outputs; layer3/15/27 cases use explicit synthetic attended inputs. The original
residual is masked at level25→24 before O projection/residual, post RMS, gate/up,
SiLU, down projection and final residual produce level4 outputs. Actual scales,
padding and all 28 contracts are checked. The preserved history includes an
invalid overflow-prone test perturbation and its stricter finiteness checks.
Session 51690 exited zero and was reaped. Identity refresh and ideal primitives
remain in use; full 28-layer and encrypted accuracy are still pending.

The [prepared-cache engine successor](validation/bert_server_prepared_cache_successor_cpu_20261103/README.md)
passes three separate public/mock suites of 37, 49 and 9 checks over 251
unchanged sources. It constructs two authenticated prepared-rotation owners,
each limited to two entries and 2 GiB of logical storage, while refresh keeps
its ordinary helpers. Root recomputed all 50 level budgets and verified the
archive, process retirement and nine native-thread signal samples. Independent
review found and fixed mutable owner-list and failure-ordering gaps; original
owners remain retained through cleanup. The earlier combined 67-test run still
has one signal-restoration failure, and the finite follow-up samples do not
establish process-wide signal atomicity. The file handler remains ordinary;
this is an unapplied construction seam with no full-engine HE or speed result.

The [Qwen full-model continuation plan](validation/qwen_full_model_continuation_public_20261103/README.md)
passes 114 public, packing and lifecycle checks over 68 unchanged sources.
Root verified all sequence lengths 1–32, the existing 166-key inventory, and
the final last-row-only path: mask level4→3, explicit refresh3→25, frozen
degree64 RMS25→7, then all 38 vocabulary blocks7→6 covering 151,936 logits.
Both audits recorded no parameter/input-body or backend access. This is a
public interface proof, and whole-model implementation is proceeding separately.
The S32 schedule still requires 12,931 refreshes; raw rotation keys total
147.84375 GiB and final ciphertexts total 532 MiB. Those counts establish no
memory or latency pass, and conditional domain/refresh guarantees remain open.

A [public normalized-round precision plan](validation/bert_normalized_scale_precision_plan_cpu_20261103/README.md)
passes 520 checks over 21 unchanged sources and 512 canonical-input schedules.
Higher real mask encoding scales raise numerator/denominator scales by the
square of the gain; a jointly planned inverse lowers its estimate scale by the
same factor while preserving the residual branch and final scale. Root checked
those exact binary64 relations for all 128 public key counts. This changes no
real polynomial, iteration count or gate, but changes encoded operations and
therefore needs a separate encrypted proof. It cannot recover error already
present in the input. The retained failed encrypted run remains failed; the
conditional sensitivity calculations establish no repaired accuracy result.

The [BERT prepared-cache/BSGS compatibility audit](validation/bert_prepared_bsgs_compatibility_public_20261103/README.md)
authenticates all 74 public weight matrices against stage2 and the original
checkpoint, then derives their six exact dense rotation keys. Root independently
rechecked every tensor digest and all signed diagonal supports. The complete
BSGS graph uses 62 keys; including 106 refresh keys gives a union of 118, all
present in the existing 130-key store. No key bodies were opened or changed.
The audit passes with 38 unchanged sources, plus 15 public checks over a
39-source proof. Its earlier inventory failure from generated bytecode remains
preserved. The documented import, handler and cleanup integration is being
implemented separately; this audit provides no combined encrypted result.

The [fixed-Q dnum/P audit](validation/bert_fixed_q_dnum_public_audit_cpu_20261104/README.md)
passes eight checks over 39 candidate sources and 40 proof sources. Actual
public parameter functions check 1,200 Q prefixes across 12 profiles; root
independently verifies partition coverage, integer security ceilings and raw
key sizes. All dnum5 profiles fail the full security ceiling. The dnum6
aux30/31 candidates pass these public checks and require 81.59/80.89 GiB for
118 raw BERT rotation keys, or 114.77/113.80 GiB for the prospective Qwen166
inventory. These figures exclude transient and compiler memory. This audit
establishes neither frozen-maxQ compiled acceptance nor encrypted noise,
latency or peak-memory behavior. Existing profiles and jobs remain unchanged;
using a new profile requires fresh keys and separately bound encrypted proofs.

The [fixed gain4 normalization preparation](validation/bert_normalized_round_gain4_public_cpu_20261104/README.md)
passes 1,084 checks over 131 unchanged sources, preserving all 114 source bytes
from the failed encrypted predecessor. Root verifies all 384 K/valid-row cases,
their unchanged operation counts and fixed 1e-6 gates, source archives and
process retirement. The original two fixtures retain all 54 checkpoint gates.
The maximum ideal error is 7.68e-13. A separate public host-encoder audit
covers 36 nonuniform and 24 scalar records; the first reciprocal weight loses
a factor256 in encoding scale, so these results establish no practical noise
improvement. Genuine component preparation is separate, and the predecessor's
encrypted failure remains unresolved.

The [full28 Qwen focused runtime successor](validation/qwen_full_model_focused_runtime_v3_cpu_20261103/README.md)
passes 150 checks over 87 unchanged sources. Root verifies the archive, XML,
retired child groups and recorded zero parameter/input-body or backend attempts.
The owner binds the fixed p60/d8 ordered Q100/P14 profile, protects partial
activation/setup failures, retains returned backing and feeds actual child
outputs into each subsequent layer and final vocabulary path. Parent tests
use synthetic child arithmetic; tiny independent-reference checks do not
establish a full-model trajectory. The earlier 84-source/144-check version
and its six profile/setup negatives remain preserved. A separate whole-model
ideal harness is being prepared for the predeclared original S1/S8 fixtures.

The [combined prepared-cache/BSGS handler proof](validation/bert_prepared_bsgs_handler_public_cpu_20261104/README.md)
passes 169 selected checks over 265 unchanged sources in 18.42 seconds. It
connects the explicit prepared-loader branch to the same authenticated handle,
engine, graph and linear owners. Synthetic checks verify BSGS helper binding,
ordinary level/checkpoint clearing and original-owner retirement. The new source
guard checks 54 critical module paths and actual exports. Review caught and
fixed a nonexistent loader import and failure-publication gaps; those negatives
remain archived. The initial combined run also remains failed because a
historical parent-only manifest assertion rejected the intended substitutions;
the new complete source guard replaces that single obsolete check. This is a
construction/lifecycle proof with ideal helpers, without a combined genuine
worker, full-model numerical result or performance claim.

A [registered-validator and fixed-partition follow-up](validation/bert_fixed_q_runtime_partition_public_cpu_20261104/README.md)
passes 21 audit checks over 43 candidate/44 proof sources, including 150 actual
abstract key-formatting traces and 450 malformed-shape rejections. All three
registered configurations are rejected: d8/aux30 requires 15 P towers but has
14; d6/aux30 requires 19 but has 18; d6/aux31 requires 18 but has 17. Those
outcomes remain distinct from passing actual-QP security and physical partition
coverage. Root independently checks all 150 partitions and output shapes.
The real dispatch constructor uses an explicitly recorded kernel double;
no kernel controls or encrypted operation run. The earlier raw-axis plan
mistake is preserved. No profile or validator was changed to force acceptance.

The [full12 BERT BSGS ideal successor](validation/bert_bsgs_full_model_ideal_cpu_20261104/README.md)
passes 82 selected checks over 150 unchanged sources in 3,807.88 seconds.
Both original K15/K128 fixtures traverse all 12 blocks and 24 stage comparisons,
then pooler and logits, using actual child outputs. They check 1,703,936 and
12,713,984 physical metric slots, respectively. Maximum stage error is
7.80e-10 against the fixed 1e-7 gate; both original native-logit and prediction
gates pass. All recorded nonlinear domains pass. Root independently verifies
source/archive/asset identities, counts, XML and producer retirement. These
are full-model ideal-array trajectories with identity refresh, without HE,
five-handler, ranked or performance evidence. The earlier missing-NTT-fixture
failure remains archived; its runtime arithmetic was unchanged in this successor.

The [gain4 genuine-component preparation](validation/bert_normalized_round_gain4_component_preflight_cpu_20261104/README.md)
passes 415 mocked and 10 installed-library checks in each of the disabled162
and ready163 source closures. Root verifies both archives and the ready current
execution tree; only five metadata/documentation files and one provenance file
change between them. The authorized next phase encrypts the two original public
fixtures using the existing authenticated key store. Separate server and client
phases require independently pinned completed prior receipts and all 54 original
checkpoint gates. Preparation does not resolve the prior encrypted normalization
failure or establish full-model accuracy.

The [combined prepared-cache/BSGS server preparation](validation/bert_prepared_bsgs_server_preflight_cpu_20261104/README.md)
passes 157 public/synthetic checks and 25 installed boundary denials over 286
unchanged sources. All 265 parent files remain byte-identical. Root verifies
source/archive identities, retired processes and the public dependency gate,
including completed full-model ideal evidence and the 118-key subset of the
existing 130 keys. The worker authenticates the original upload and writes to
a separate owned response root. A separately reviewed launch now runs that
frozen server on CPUs100–103, borrowing the existing authenticated keys, model
and upload. Its result remains pending. A client successor binds that response
and isolates its completion receipt; actual client execution requires the
completed server proof and the separate descriptor-cleanup repair. No numerical
result, performance or physical memory bound follows from preflight or launch.
The first installed-probe import-order failure remains archived.

The [first whole28 Qwen trajectory attempt](validation/qwen_full_model_trajectory_prerequisite_failure_cpu_20261104/README.md)
failed before token/parameter reads because its prerequisite gate expected a
boolean where the frozen calibration report records the integer count 416.
Both fixed fixtures stopped at that metadata check; no numerical result follows.
The [separate count-schema successor](validation/qwen_full_model_trajectory_count_schema_preflight_cpu_20261104/README.md)
passes 86 prelaunch checks over 101 sources, requiring the exact integer count.
Root verifies the original public report metadata, source/execution archives and
retired processes. All 87 runtime files, fixed S1/S8 fixtures and numerical gates
remain unchanged. Its actual trajectory run is separate from this preparation.

The [gain4 encrypted fixture phase](validation/bert_normalized_round_gain4_fixture_cpu_20261104/README.md)
completes both original inputs at level 49 in 147.68 seconds, using the existing
authenticated key store. Root verifies all 163 current/execution/archive sources,
both complete ciphertext headers, residue ranges and digests, terminal receipts
and producer retirement. The 209,715,200 raw ciphertext bytes remain external.
The separate encrypted server and client validation phases have completed,
as recorded below.
Only the final persisted receipt hash is used, because the inherited collector's
stdout hash precedes outer termination-handler restoration.

The [whole28 Qwen ideal trajectory run](validation/qwen_full28_ideal_trajectory_cpu_20261104/README.md)
is an aggregate failure: the original S1 embedding-row0 fixture reaches final
RMS variance 3391.698360993511, above the frozen upper bound 2923.83603515625,
before entering the packed owner. Its failed terminal flags remain unchanged.
The predetermined S8 fixture separately passes all 421 operator, 28 block and
38 vocabulary-block comparisons, covering 31,916,032 physical metric slots
and all 151,936 logits. Maximum block error is 8.10e-11 and logit error 2.93e-13
against the fixed 1e-8 gate. All 1,918 actual domains and both corresponding
clear-reference domain sets pass, with 3,270 identity refreshes. Root verifies
every metric/domain record, all 101 current/execution/archive sources, original
assets and producer retirement. This is ideal-array evidence, without HE,
native-framework, file-handler, ranked or performance readiness.

The [fixed-S1 clear-prefix diagnosis](validation/qwen_s1_final_rms_diagnosis_cpu_20261104/README.md)
reproduces that failure and measures final variance3391.7152988079342 with
NumPy exact functions, also above2923.83603515625. Approximation slightly
reduces the variance by0.0169378144; it does not cause this fixture's domain
excess. All307 preceding approximate domain records pass. The original416
calibration contexts cover S8/S16/S32 prefills and decode prefixes16–19,
with no S1 coverage. Root verifies25 unchanged sources, the original row0
embedding, all28 block/113 variance comparisons and process retirement.
Both prefixes stop before final normalization and the vocabulary head;
no final logits, native-framework or encrypted result follows. Candidate3,
its original failed flags and all gates remain unchanged. The separate expanded
protocol below covers calibration across prefix lengths1–32 and fresh validation.

The [separate-response BERT client preparation](validation/bert_prepared_bsgs_client_preflight_cpu_20261104/README.md)
passes 198 public/synthetic tests and 21 installed boundary checks over 115
unchanged sources. Root verifies source archives, loaded origins and retired
processes. Two explicit receipt-only changes preserve the original default
client path while placing this component's results and completion receipt in
its own private directory. The client requires the completed 286-source server
proof and authenticates response ciphertext bytes before private input, reference
or key access. It retains the original size0 P2 and prediction gates; this is
not ranked acceptance. Review reproduced and fixed standalone worker report
writes into borrowed private roots. That negative and an umask-related test
fixture failure remain archived. No actual client decode ran in preparation.
Post-freeze review also reproduced descriptor-handoff leaks and cleanup paths
that could mask a primary exception or cancellation in inherited readers.
The115-source preparation remains historical evidence and does not establish
current launch readiness. A separate117-source repair preserves the original
formats, cryptographic arithmetic and acceptance gates; its final preparation
and independent readback are recorded separately. The same inherited failure
paths in other frozen server/validator sources require separate successors;
healthy runs retain their exact sources.

The [separate client descriptor successor](validation/bert_prepared_bsgs_client_descriptor_successor_cpu_20261104/README.md)
passes263 tests and21 installed-library denial/restoration checks across117
frozen sources. Root independently verifies every source/archive byte, all40
loaded origins, five explicit overlays against the89-source parent, unchanged
historical115 files and process retirement. Bounded reads retain descriptor
ownership; nested cleanup preserves the original body exception and prioritizes
cleanup-only cancellation. Confirmed leak and exception-masking negatives remain
archived alongside independent69-test receipt and35-test reader retests.
All successful formats, key-generation provenance, cryptographic arithmetic and
size0 acceptance gates remain unchanged. This is preparation evidence; actual
decryption awaits the completed286-source server proof and concrete launch review.
Subsequent call-path review found additional inherited gate and transport reader
cleanup defects. A further separate successor must repair and test those paths
before launch; the117-source evidence remains frozen with this limitation.


The [original stage5 I/O successor](validation/bert_original_stage5_io_successor_cpu_20261104/README.md)
passes279 synthetic tests and21 installed-library denial/restoration checks over122
frozen sources. Root verifies exact current/start/end/archive bytes,40 loaded
origins, seven explicit overlays and82 unchanged files from the89-source parent.
The original217-source producer gate, default file handler and completion/
retirement definitions remain unchanged. Independent review preserves seven
active-reader failures and a pre-import shadow-file execution before repair.
The successor authenticates its external manifest before local helper imports,
rejects unexpected importable files and aliases, and retains descriptor ownership
through failing reads and cleanup. Actual decryption still awaits the completed
original217-source producer and a separately pinned launch.

The [combined client active-reader successor](validation/bert_prepared_bsgs_client_active_readers_cpu_20261104/README.md)
passes317 synthetic tests and21 installed-library checks over125 frozen sources.
Root verifies all bytes,40 loaded origins, retired processes and the exact port
from the reviewed122-source original client. Three repaired gate/transport readers
are byte-identical to that source; the combined286-source producer gate, readback
logic and separate completion-receipt APIs remain byte-identical to117.
There are107 unchanged parent files,10 changes and8 additions. Historical failed
reader/bootstrap proofs remain archived. This preparation does not decode any
ciphertext or authorize a client launch; completed combined-server evidence is
still required.

The [BERT4 checkpoint diagnostic plan](validation/bert_toy4_checkpoint_diagnostic_plan_cpu_20261104/README.md)
passes61 preparation checks over222 exact sources, including the unchanged199
from the failed TPU sweep. Independent dense and packed NumPy references agree
at all96 checkpoints across6,291,456 physical slot comparisons, with maximum
error1.12e-15. All11 ideal polynomial domains and the five fixed refresh bounds
pass for this public toy fixture. Root verifies the original case/profile,
199-key union, source/execution archives and process retirement. The plan
requires all server work and authenticated checkpoint saves before private
decode, preserving the original decoder and output gates. Its3.5GiB raw
checkpoint storage estimate is not a memory peak. Actual encrypted instrumentation,
transport and diagnosis remain unvalidated; this clear-reference
result does not explain or repair the original ciphertext failure.
Implementation review separately corrects the plan's private-secret storage:
the serialized100×131072 uint32 RNS array needs52,428,800 bytes, rather than the
listed1MiB assumption. The archived correction preserves all222 measured sources;
future capacity checks must use the full private payload plus headers/reserve.

The [BERT4 diagnostic successor preflight](validation/bert_toy4_checkpoint_diagnostic_successor_v4_cpu_20261104/README.md)
passes371 checks over260 frozen sources. Root independently verifies the current
tree, both execution copies, archive, test results and child/group retirement.
The original199 numerical files, profile, coefficient definitions, response
admission and scale replay remain unchanged; arithmetic classes and the
instrumented numerical body also match. Repairs cover independent file/spec
origin admission, including symlink and PathLike aliases, and exception
provenance across cleanup layers. A genuine body exception retains priority
even if cleanup rethrows the same object; cleanup-only cancellation still takes
priority over ordinary cleanup errors. Earlier253/276 and258/338 checkpoints
and confirmed negative cases remain preserved. Actual role execution remains
disabled, and the diagnostic stays behind the original TPU12 and P3six queue.

The [expanded-prefix Qwen protocol](validation/qwen_expanded_prefix_protocol_public_20261104/README.md)
passes68 public preparation checks over23 exact sources. It proposes3072
prefills from the original96 prompts at every length1–32, plus128 decode
entries, with the original416-entry workload mapping retained. These are
entry counts, not distinct-input claims. A separate1088-entry validation design
must be tokenized and checked against all expanded calibration prefixes, old
contexts and row0 before any expanded calibration results or selection.
One uniform policy may revise all113 normalization rules; protected SiLU and
softmax rules, numeric gates and semantic acceptance rates stay unchanged.
Row0 remains a post-freeze regression excluded from selection. Root verifies
source/execution archives, unchanged candidate3, declared gates and retired
children. The first count-assertion failure remains preserved. No tokenization,
data-disjointness result, coefficient selection or model execution occurred.


The [expanded-prefix tokenizer preflight](validation/qwen_expanded_prefix_tokenizer_preflight_cpu_20261104/README.md)
passes91 synthetic tests and six installed import denials over33 frozen public
source/metadata files. Independent reviews reproduce and repair pre-import
symlink aliases, descriptor cleanup failures, nested cleanup-origin handling
and final-publication ordering. The earlier32-source checkpoint and all measured
negative cases remain archived. An external manifest authenticates helper bytes
before imports; a separate tokenizer-only grant leaves the preparation source's
execution flag false. No model evaluation is part of this stage.

The [actual tokenizer-only freeze](validation/qwen_expanded_prefix_tokenizer_actual_cpu_20261104/README.md)
passes with3072 calibration prefill entries and1024 new validation entries,
plus16 and8 authenticated decode seeds. Their unique prefill counts are2792
and1021; duplicates remain in every original denominator. Root independently
reauthenticates all five old input files and four tokenizer assets, verifies
every ordered prefix and seed mapping, checks private0700/0600 output identities,
and recomputes zero known-context and append-only-family prefix intersections.
New prompts have60–63 natural tokens. The3200/1088 protocol totals include future
decode work: no future suffix tokens have been generated or frozen. Token bodies
remain external; public archives contain source and aggregate evidence only.
The outer process completed with exit0 and retired children. Equality separation
does not establish semantic independence, model accuracy or generalization;
expanded native/exact observation and normalization selection are still pending.

The [Qwen observer workload plan](validation/qwen_expanded_prefix_observer_plan_public_cpu_20261104/README.md)
passes62 synthetic checks over38 exact sources. It retains all3200 ordered
entries without deduplication:3072 prefills,64 native-teacher contexts and64
free contexts generated by the frozen candidate3. This initial generator uses
none of the three new-candidate budget. Dependent invalid suffixes remain
unmeasured, and incomplete observation cannot admit selection. Initial ranges
use native/exact inputs; baseline-generator ranges remain separate diagnostics.
The public producer proof, source archive and process receipts verify. No
actual token/model body, forward pass, selection or HE is part of this plan.

The [initial Qwen observer preflight](validation/qwen_expanded_prefix_observer_preflight_cpu_20261104/README.md)
passes226 synthetic checks and eight backend-import denials over52 frozen
sources. Root verifies the current/execution/archive bytes and retired processes.
Its separately granted actual run stops during a Transformers optional
TensorFlow probe, before checkpoint construction or forward; an additional
network probe is denied. The failed run exits1 with source equality, input
reauthentication, closed private handles and an empty private journal. The
worker's historical `model_forward:true` field was set before runtime entry
and is stage intent, not evidence of a forward. These outcomes remain preserved.
A separate startup successor makes backend selection explicit and adds a real
import-only check with all actual input/model bodies still forbidden. No
completed3200 observation or selection admission follows from the52-source run.

The [Qwen startup successor](validation/qwen_expanded_prefix_observer_import_successor_cpu_20261104/README.md)
passes274 synthetic checks, eight import denials and a separate actual Qwen
class-import smoke over the same61 frozen sources. The smoke constructs no
pretrained model, reads no token/checkpoint bodies and records zero audit
denials. It verifies14 backend sources and three import-probe sources, permits
stateless activation registration, and restores the temporary discovery/socket
settings. Forbidden optional backends remain unavailable to discovery and
rejected by actual imports. Root independently verifies both source archives,
terminal receipts and retired process groups. Failed57/59/61 smoke checkpoints
remain archived. The separately granted3200-context observation below uses this
exact successor. Native-forward attempts and completions have separate counters.

The [independent initial-range reconstruction preflight](validation/qwen_initial3200_reconstruction_preflight_cpu_20261104/README.md)
passes17 synthetic helper checks and seven independent regression checks. Three
earlier cleanup failures remain preserved. Its reviewed script requires a
successful, independently reaped observer before opening private journals,
then checks all9666 files, ordered3200 prefixes and choices, per-context input
counts and reconstructed aggregate ranges. Publication uses a fresh held output
directory outside protected inputs. Selection additionally requires a separate
successful reconstruction outer receipt; this preflight executes no actual
reconstruction, reads no private observations and admits no candidate.

The [actual initial Qwen observer](validation/qwen_expanded_prefix_initial_observer_cpu_20261104/README.md)
completes all3072 prefills,64 teacher contexts and64 frozen-baseline free
contexts. All3200 native/exact semantic checks pass over151936 final logits;
all3232 native calls, including32 bare invariance calls, complete. Both processes
exit0 with unchanged61 sources, zero audit denials, restored hooks and closed
private handles. Root separately reconstructs all9666 private journal files,
ordered prefixes and choices, per-context113 normalization/28 SiLU/causal-score
coverage and aggregate ranges. The reconstructed native/exact union matches
the producer; baseline-generator ranges remain excluded. Private journals and
the reconstructed snapshot remain private. This establishes initial input
observations, not a new candidate's calibration, held-out validation or HE.

The [initial normalization-selector preparation](validation/qwen_expanded_prefix_selector_preparation_cpu_20261104/README.md)
passes66 synthetic tests with18 component files and19 proof members. It applies
the same113-site expansion, certificate ladder and newly certified Newton
fallback to synthetic bounds, preserves protected rules and retains failed
assessments. Its proposed admission checks require complete3200 initial
evidence and a native/exact union excluding baseline-generator ranges. Actual
admission and selection remain disabled; completed producer/root evidence,
private-journal reconstruction and authenticated learned gate bounds are still
required before a real candidate can be selected.

The [granted-selector path preflight](validation/qwen_expanded_prefix_selector_admission_cpu_20261104/README.md)
passes150 synthetic checks over36 component sources and one external proof
driver, preserving the original18 sources. It requires the completed61-source
initial proof, independent journal reconstruction and a separate root-review
exit/reap receipt before reading the single authenticated snapshot. Review
repairs cleanup skipped when the signal shield cannot initialize, and makes
any audit denial veto success. The failed outer-audit checkpoint and independent
cleanup negatives remain archived. Source/process checks pass; this preflight
itself uses synthetic evidence and performs no actual observation or selection.

The [actual first expanded-prefix selection](validation/qwen_expanded_prefix_initial_selection_cpu_20261104/README.md)
completes from the authenticated initial3200 native/exact union. All113
normalization sites are supported:36 balanced polynomials and77 freshly
certified Newton rules with15–24 iterations. Root replays all648 polynomial
attempt certificates and grid checks, the77 Newton fallbacks, every uniform
domain, all28 protected-rule checks and all32 abstract depth calculations.
The complete37,283,359-byte assessment history is retained. SiLU, softmax,
exp8 and reciprocal rules remain unchanged. The first new manifest is frozen
as `344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a`;
actual selection and independent replay both exit0 with retired processes.
Its abstract depths are6992 atS1 and14398 atS32; earlier refresh plans do not
transfer automatically. Candidate3200 calibration, held-out1088 validation,
row-zero regression and HE remain unmeasured for this manifest.

The [candidate-checker component preparation](validation/qwen_expanded_candidate_checker_component_cpu_20261104/README.md)
passes79 synthetic component checks and seven harness checks over29 sources.
Its dispatcher preserves each candidate's own free choices, stopped suffixes
and full denominators. The single native capture owner keeps421 output metrics
separate from113 variance inputs. Review reproduced fabricated-ledger revision
eligibility and repaired typed evidence and causal-prefix admission; the
negative cases remain archived. Source archives and process receipts verify.
This component does not include the numerical adapter or actual worker added
by the separate runtime preflight below, and establishes no candidate-model or
encrypted result.

The [strict candidate-checker runtime preflight](validation/qwen_expanded_candidate_checker_runtime_preflight_cpu_20261104/README.md)
passes696 synthetic checks and ten import denials over115 frozen sources. It
checks all421 operator outputs,28 blocks and151936 logits separately from113
normalization input variances and every propagated nonlinear domain. Complete
scientific failure, native/exact comparison failure and interrupted execution
have distinct admission rules; only completed candidate-specific failure can
support a separately authorized revision. The failed113-source fixture proof
and reproduced cleanup-cancellation defect remain preserved with their repairs.
Root verifies current/execution/archive equality and retired processes. Candidate
binding remains pending in this measured preparation; no actual candidate model,
validation, row-zero regression or HE is executed by this preflight.

The [ready candidate-checker preflight](validation/qwen_expanded_candidate_checker_ready_cpu_20261104/README.md)
passes 738 synthetic checks and ten import denials over 139 frozen sources. It
binds all 17 public proof files for the completed initial observer and first
selection. Only the exact selector-assessment member receives a 64 MiB bound;
ordinary source files retain 8 MiB and the combined source/proof bound is 96 MiB.
The six reviewed numerical files remain unchanged. An independent construction
failure probe exposed skipped cleanup when allocating the signal shield; its
three failures and passing control remain archived, and the separate successor
passes the identical four cases. The earlier 137-source, 727-test proof remains
preserved. Root independently verifies the final source/archive/proof chain,
738-test result and process retirement before issuing the single calibration
grant. The actual first candidate is now running on CPUs 172–175 using that
exact source snapshot. Its 3,200-context aggregate result remains pending;
held-out validation on 1,088 contexts, row-zero regression and HE have no execution authority from this
grant. Public proof byte limits are not resident-memory guarantees.

The separate [first-pass freeze assembly preflight](validation/qwen_candidate1_freeze_issuer_preflight_cpu_20261104/README.md)
passes 88 synthetic checks over six exact sources. It binds the completed
producer reports, an independent reconstruction review, and separate process
exit receipts. The reconstruction's full source manifest, archive and grant
are pinned alongside its entry script. The original missing-helper-proof
counterexample and its corrected regression remain preserved; the same corrected
test fails on the old issuer and passes on the successor. Root verifies the source
archive, test result and retired proof process. This is public proof assembly
preparation only: no actual candidate freeze, private reconstruction, validation
or model execution occurs in this proof. The original 79-test development run
has only a post-terminal source snapshot and reported direct exit, not a source
binding captured at startup.

The [heldout and row-zero runtime preflight](validation/qwen_expanded_heldout_runtime_preflight_cpu_20261104/README.md)
passes 478 synthetic checks and ten import denials over 167 frozen sources,
including all 139 unchanged candidate-checker files. It preserves the heldout
gates of 922/1024 prefills, 29/32 teacher steps and 6/8 whole free sequences;
free-step agreement remains diagnostic. Row-zero evaluation uses the same
first calibration-PASS freeze and its own execution grant. A separate root
receipt can admit row zero after no heldout attempt or a retired failed attempt,
without admitting a heldout result or allowing feedback and tuning. Review
reproduced three cases that incorrectly reported completed validation reads as
unread; the identical four-case successor passes and preserves uncertain reads
explicitly. Root verifies exact source/archive equality, all test results and
the observed collector/worker process retirement. This is synthetic preparation
only: actual heldout validation and row-zero evaluation remain unmeasured and
have no execution grant. Subsequent review found the same late-publication
exit-code collision described below in this runtime; a separate successor is
required before actual execution.

The [candidate terminal-reconstruction preflight](validation/qwen_candidate_terminal_reconstruction_preflight_cpu_20261104/README.md)
passes 139 synthetic checks over 44 exact sources, including eleven backend
import denials. Root independently verifies the source archive, test results
and recorded process retirement. The verifier checks recorded statistics and
aggregates; it does not recompute unavailable activation or logit arrays. The
earlier 40-source startup failure and independent reader/causal-prefix repairs
remain preserved. A subsequent producer review reproduced an exit-code
collision: a late report-close error can return the same status as a completed
scientific failure. Failed results from the current 139-source producer cannot
authorize revision under this terminal contract; changing a consumer does not
repair that producer evidence. The running calibration and frozen proof sources
remain unchanged. No
actual reconstruction grant or first-PASS freeze has been issued.

The [candidate terminal-exit successor](validation/qwen_candidate_terminal_exit_successor_cpu_20261104/README.md)
passes 781 synthetic checks and ten import denials over 141 frozen sources.
It preserves normal scientific statuses 0/1 and assigns status 70 to exceptions
escaping authenticated work, malformed returns and diagnostic-output failures.
The identical 43-case CLI probe has 36 expected failures and seven passing
controls against the predecessor, then passes all 43 cases against the
successor. The six numerical files, protocol and selected manifest remain
unchanged. Root independently verifies source/archive equality, the full test
result and observed process retirement. This proof neither reclassifies the
running original producer's result nor issues a new execution grant.

The [heldout terminal-entry successor](validation/qwen_heldout_terminal_entry_preflight_cpu_20261104/README.md)
passes 563 synthetic checks and ten import denials over 171 frozen sources.
It applies the same status-70 guard to heldout and row-zero worker/collector
entrypoints, with 85 focused checks included in the full suite. All 139
inherited files and all scientific rules remain unchanged; only the heldout
entrypoint, test-count metadata and a no-grant diagnostic assertion change
among the earlier 167 files. Root verifies exact source/archive equality and
process retirement. Earlier failures, development snapshots and documentation
corrections remain preserved. This is preparation only, with no actual heldout
or row-zero execution grant.

The [consumer source-readiness audit](validation/consumer_readiness_source_audit_cpu_20261104/README.md)
checks 20 selected source files and five complete source maps. Existing
reconstruction, freeze and heldout consumers still bind the original 139-source
producer; the 141-source successor needs explicit consumer updates and new
grants. An independently admitted original-producer PASS remains compatible
under the reviewed terminal scope. Its failed-result branch remains barred
because the exit-status ambiguity is unresolved in that frozen producer.
This static audit reads no actual result or private input and grants no run.

The [fixed-Q public control probe](validation/bert_fixed_q_public_controls_cpu_20261104/README.md)
constructs real HERot/BConv controls for all50 prefixes of each of three
profiles, using metadata descriptors in place of polynomial allocations.
All37 forbidden-entry probes deny, with zero forbidden calls during control
construction. Root verifies49 source/archive files and independently recomputes
all11,092,910 numeric constant entries and4,008 table hashes without importing
the runtime. The16 tests and process retirement also pass. Actual QP security
and physical partition coverage pass, but the unchanged registered policy still
rejects all three profiles. This establishes no key binding, compiled rotation,
encrypted precision, speedup or physical memory bound. The initial audit-traversal
typo failure remains preserved; no profile or validator was changed.

The [gain4 validator bootstrap successor](validation/bert_normalized_round_gain4_validator_bootstrap_preflight_cpu_20261104/README.md)
passes543 synthetic tests and10 installed-library checks over185 frozen sources.
Its new consumer source map remains separate from the unchanged163-source
fixture/server producer. Two explicit loader I/O overlays preserve the original
54 checkpoints and numerical gates. Review of the earlier181-source preparation
reproduced helper execution before source admission; that evidence remains
archived. The successor binds an external manifest digest before helper imports.
Root independently verifies all185 sources,160 static import resolutions,
process retirement and two fresh canary rejections at the worker/collector
entrypoints. Both execution flags remain false and completed-server pins remain
null. Actual validation requires a separately reviewed ready successor bound
to the completed original encrypted server.

The [gain4 encrypted server phase](validation/bert_normalized_round_gain4_server_cpu_20261104/README.md)
completed all54 checkpoints for the unchanged K128/16-row and K15/9-row public
fixtures in36,769.57 seconds. It executed two level49-to25 reductions, six
plaintext products,46 production ciphertext products, two diagnostic squares,
54 rescales and28 rotations, without refresh, key generation or server codecs.
Root independently verified all163 current/execution/archive sources, both
fixture/server public terminal gates, the final persisted parent/worker report
binding, cleanup and process retirement. Original keys, inputs and outputs
remain external and retained. The separate numerical validation is recorded
below. This server completion alone does not establish accuracy, full-model
execution, ranked performance or competition readiness.

The [separate ready gain4 validator](validation/bert_normalized_round_gain4_validator_ready_cpu_20261104/README.md)
passes543 synthetic and10 installed boundary checks over187 exact sources.
Its runtime and numerical gates are unchanged from185; the new metadata binds
the completed163-source server and independent root review while keeping the
legacy fixture/server entrypoints disabled. Root verifies source archives,
test XML, completed producer bindings and process retirement, then authorizes
one genuine validator invocation on CPUs124–127. That run has completed and
retains all borrowed stores; its numerical result follows.

The [gain4 encrypted component validation](validation/bert_normalized_round_gain4_validation_cpu_20261104/README.md)
passes all 54 ordered checkpoints across 3,538,944 physical slots and all 52
derived-error gates at the unchanged absolute tolerance of 1e-6. The largest
error is 1.7275218056056474e-7 at K128/16-row inverse estimate y13, level 8.
All 29 K128 and 23 K15 derived checks pass. The client authenticated both inputs
and all 54 outputs before private key access; final readback matches the
completed server. Root verifies all 187 current/execution/archive sources,
typed planned levels and scales, all 106 numerical records, loader and signal
cleanup, and process retirement. Client validation took 269.82 seconds; the
separate server phase took 36,769.57 seconds. No retry, gate change, borrowed-store
retirement or promotion occurred. This establishes the fixed gain4 component
on two public fixtures; refresh, full attention, full-model and ranked-case
accuracy remain unproved by this experiment.

The [Qwen causal-softmax cohort prototype](validation/qwen_softmax_cohort_synthetic_cpu_20261104/README.md)
passes136 synthetic checks over197 exact sources, with all191 inherited files
unchanged. It groups query rows by their original Newton iteration count and
refresh bound while retaining each query's coefficients, positions and the
166-key plan. Full-slot comparisons cover sequence lengths1–32, both squaring
schedules, the literal predecessor, inactive lanes, returned scales, PV layout
and callback failure retention. At S32 across28 layers, softmax refresh calls
fall from11,656 to3,760. Root independently verifies these operation counts,
the source archive, test results and process retirement. The full-model join
still selects the literal implementation; encrypted accuracy, integrated
execution and measured acceleration remain untested.

The [expanded-candidate full28 Qwen preparation](validation/qwen_expanded_full28_runtime_preparation_cpu_20261104/README.md)
passes all180 public and synthetic checks over199 exact sources, with zero
audit denials and verified process retirement. It derives fresh certificates
for113 normalization rules while preserving the protected arithmetic rules and
parameter provenance. The final degree256 schedule reaches all151,936 vocabulary
outputs at level2; its final raw-input refresh bound becomes8192 under the
unchanged bound rule. The owner also rejects replacement of its refresh/check
callbacks before invoking them. Failed preparation and callback-reproducer
results remain archived. These checks establish source admission, metadata and
synthetic owner behavior; actual calibration, encrypted refresh accuracy at8192,
final-logit precision and full-model file-stage execution remain unproved.

The [original-profile P3 dynamic-constants candidate](validation/p3_original_dynamic_constants_preparation_cpu_20261104/README.md)
passes10 CPU checks over100 exact sources. Its sole baseline change enables
`dynamic_constants=True` in the mapping constructor. Six frontend and packing
comparisons retain the original models, weights, batches and ring, three
existing clear compilation checks pass, and an independent AST comparison
confirms the one-keyword delta. The mapping builder in the six comparisons is
a recording stub. No encrypted or TPU run, setup-time improvement, production
promotion or change to the pending original six-case sweep is claimed.

The [integrated Qwen cohort preparation](validation/qwen_cohort_full_owner_preparation_cpu_20261104/README.md)
passes 330 checks over 208 exact sources, with zero audit denials and verified
process retirement. The full-owner attention constructor now selects the
reviewed cohort softmax. Synthetic integration covers original positions,
K1 bypass, V23 to PV20 (or PV22 for S1), callback failures and retained owners;
parameter authentication, normalization, head and linear stages use explicit
synthetic substitutes. All 32 public sequence plans preserve the 166-key union,
protected arithmetic and final RMS3 to all 38 vocabulary outputs at level2.
At S32, the integrated plan requires 3,760 softmax refresh calls and 5,035 total,
compared with the literal plan's 11,656 and 12,931. Root independently verifies
these counts, the six explicit parent overlays, source archives and terminal
outcome. Historical development failures retain their original evidence scope.
This establishes integrated synthetic behavior and operation counts; learned
whole-model accuracy, encrypted composition and measured acceleration remain
unproved, and actual execution remains disabled.

The [Qwen file-adapter preparation](validation/qwen_file_adapter_preparation_cpu_20261104/README.md)
passes 191 focused public and synthetic checks over 238 exact sources. It
combines bounded parameter-file readers, retained file identities, one/two
input-container admission, complete 38-output admission before the private
callback, original embedding placement and all 151,936-logit readout. Tests
include full physical RNS container shapes, synthetic parameter files and a
decoder that reuses its scratch array; no cryptographic operations occur.
The NPY reader now checks declared header length before reading that header.
Its preserved negative test records the old oversized-read request, while the
earlier 71-test run retains its failed wrapper result from four pytest cache
files. The final run disables that cache writer, preserves all 208 inherited
sources and retires both processes cleanly. Root independently verifies the
source archive, XML and authoritative terminal receipt. Actual role admission,
key/context loading, genuine L49-to-L25 reduction and backend session execution
remain separate work; the contestant starter is unchanged.

The [Qwen server file-engine preparation](validation/qwen_file_engine_integration_preparation_cpu_20261104/README.md)
passes **12 focused recording tests over 366 exact sources**. It joins the
reviewed key reader, ordinary engine composition and file publisher with an
explicit parser namespace and shared cleanup policy. Tests use 335 recording
key mappings and synthetic contexts/arithmetic, then write all 38 full-shape
synthetic response containers. File consumers retire before the key-loader
scope exits; late failures retain the original owner graph. Three preserved
negative cases showed that replaced close callbacks and late handle/factory
mutations escaped detection. The successor captures original callbacks and
checks ownership after publication and cleanup; the identical 12 cases pass.
A separate preserved five-case probe verifies cleanup-only cancellation
precedence across the joined helpers. Root verified both negative histories,
source deltas, archives, XML and terminal process evidence. The independent
reader120, composition47 and publisher27 checks retain their separate scopes.
Actual factories remain disabled; real key/context loading, full encrypted
execution and numerical acceptance remain pending.

The [Qwen client file-stage preparation](validation/qwen_client_stage_preparation_cpu_20261104/README.md)
passes **19 focused recording tests over 373 exact sources**, preserving all
366 server sources. It places original embeddings and noncontiguous positions
into one or two fresh L49 inputs, authenticates all 38 response files before
private-loader entry, and requests each ciphertext's actual scale, all 65,536
slots and approximation validation. Tests verify immutable snapshots before
decoder-buffer reuse and original ciphertext retirement before key-scope exit.
Corruption, nonfinite values, callback replacement, late ownership changes and
cleanup failures reject as expected. Root independently verified the exact
test selection, source archive, empty backend-import audit and authoritative
exit-0/reap receipt against its live process observation. These tests use owned
synthetic files and recording keys, contexts and codecs; they do not execute
the selected real decoder or establish HE correctness, model accuracy or speed.
The 12 server cases remain a separate result; actual client factories are disabled.

The [Qwen key-producer preparation](validation/qwen_key_producer_preparation_cpu_20261104/README.md)
passes **30 focused synthetic tests over 71 exact sources**: 15 writer cases
and 15 recording producer cases. A bounded synchronous writer replaces buffered
NPY output and rejects retained-sink access after its original descriptor owner
detaches. Small owned files match NumPy's control bytes exactly; partial writes,
invalid progress, extent limits, descriptor reuse and cleanup failures are checked.
Recording fixtures verify the literal 166-rotation order, 335 public descriptors,
original secret-object identity, prerequisites before generation, and manifest
and cleanup ordering. All 57 selected predecessor files and the generation
composition remain unchanged. Root verified exact collection/XML, source and
archive equality, NumPy source identities, and terminal exit/reap evidence
against its independent live observation. The buffered predecessor's concern
was source-reviewed; no old runtime corruption was reproduced. These checks
establish neither real generated-key compatibility nor backend security,
capacity, model/HE accuracy or performance. Actual producer entry remains disabled.

The [Qwen model-stage preparation](validation/qwen_model_stage_preparation_cpu_20261104/README.md)
passes **48 focused tests over 38 exact sources**, preserving all 29 runtime
and provenance files: 28 admission cases using synthetic evidence, 10 identity
checks on small owned files, and 10 recording traversal and cleanup cases.
Admission binds externally expected model, source, report and terminal hashes;
repinned malformed reports still reject on their contents. Held root and file
identities are checked before parameter hashing, header reads or mappings.
Recording traversal covers 198 logical matrices, 197 physical descriptors and
113 gamma vectors, drops temporary views and retains original cleanup callbacks.
Root independently verified exact collection/XML, current and archived sources,
NumPy identities, an empty backend audit and exit-0/reap evidence against its
live process observation. These tests establish neither the integrity of the
retained full parameter bodies nor scientific calibration, HE accuracy or speed.
Those component entry points remain disabled. The separate readonly successor
below supplies current-store validation; model execution still needs its own grant.

The [Qwen keygen authority preparation](validation/qwen_keygen_authority_preparation_cpu_20261104/README.md)
passes **48 pure synthetic checks over 39 exact sources**, preserving all 28
parent files: 17 launch/model/source cases, 25 generated-readiness cases, and
6 retained-state and entry-denial cases. The checks bind model evidence to its
selected consumer, distinguish generation and request/response sources, and
require externally expected generated identities and normal terminal evidence.
Malformed proofs reject after surrounding hashes are recomputed. Root verified
the source archive, exact test order, empty backend/NumPy audit and authoritative
exit-0/reap receipt. Root observed both processes after exit; the owner's late
observation race remains preserved. These fixtures establish no actual model
admission, generated keys, resource/security execution, role readiness or HE
result. Actual key-generation entry remains disabled.

The [Qwen model-store readonly validation](validation/qwen_model_store_readonly_validation_cpu_20261104/README.md)
completed successfully for **197 physical matrices, 198 logical matrix visits
and 113 gamma vectors**. The separately granted 42-source process checked
headers, extents, hashes, finite values and original held identities, completed
readback and cleanup, and exited normally with direct reap. Root verified the
source archive and current file identities, then admitted the actual report
through the unchanged model-admission gate. The public capsule also preserves
the 62 focused tests on 54 sources, metadata-only expectation acquisition and
source-review repairs; no model bodies are included. This establishes retained
model-file integrity only. Model forward, calibration, generated keys, HE
accuracy and downstream execution remain unproven here. Consumer pins name
the reviewed preparation boundary; future actual roles need separately reviewed
source maps, current held-store checks and their own grants.

The [Qwen held-parent key-store preparation](validation/qwen_keygen_held_parent_preparation_cpu_20261104/README.md)
passes all34 focused checks over83 source files containing the exact74-file
runtime successor. The writer requires independently supplied canonical parent
metadata and checks the original held descriptor, named directory and UID/sticky
policy before creation and during later binding checks. Its serialization,
core34 and recording producer remain unchanged. Tests cover real owned-directory
replacement, malformed metadata, tiny genuine NumPy files and original descriptor
retirement after failure; UID changes and signal shields use disclosed test seams.
Root verified ordered XML, imports, all source/archive copies, normal zero exit,
reaping and both process-group absences. No independent live observation was
available for this subsecond run. The capsule also preserves conditional resource
estimates and a separately reviewed one-hash authority prerequisite update;
that update has static review only. It does not rerun the earlier48 authority
tests or authorize actual key generation. The capsule README records its original
pre-copy staging state; this paragraph records its repository publication.

The [Qwen client-authority preparation](validation/qwen_client_role_authority_preparation_cpu_20261104/README.md)
passes **95 pure synthetic checks over 53 exact sources**, preserving all39
runtime/public-metadata parent files: 29 conditional-role, 33 phase and33
server-envelope checks. They verify independently pinned model/key/source
linkages, a required later phase grant, distinct server versus request provenance,
all38 response descriptors, disjoint paths and retained-state reauthentication.
Root verified the exact XML/node selection, unchanged source/archive bytes,
empty NumPy/backend audit, live original processes and normal exit0/reaping.
The 138.98-second duration measures this test suite only. The capsule preserves
the unrun52 fixture corrections and unrun harness metadata corrections separately.
Its93 prerequisite identities are synthetic map entries; those runtime files
were not executed. These checks establish no actual file/private-read ordering,
generated keys, resource readiness, model execution or encrypted accuracy. The
later response-root implementation is outside this measured source snapshot.


The [Qwen key-generation runtime preparation](validation/qwen_keygen_actual_runtime_preparation_cpu_20261104/README.md)
passes **61 focused synthetic checks over 97 exact sources**, containing the
85-file runtime successor: 15 output-owner, 19 policy/capacity-seam, six recorded
generation/lifetime and 21 entry/terminal cases. Review found two metadata files
with incompatible JSON encoding; the successor changes only their encoding and
the suite tests both preserved old controls. Root verified ordered XML,
source/archive equality, an empty genuine-backend audit, normal exit0 and direct
reaping. The run completed within the launch call; there was no independent live
observation. Its 0.503-second duration measures the small test suite only.
Capacity observations and generation primitives are explicit doubles; three
cases temporarily install module-name doubles and remove them before the final
guard. No actual keys, RNG, context, original model/key/ciphertext bodies or
resource readiness were exercised. Actual entry remains disabled, and the
capsule preserves the source-only85 history and separate focused-run grant.
Its README records the original pre-copy state; this paragraph records repository
publication. Full-model encrypted correctness and timing remain outstanding.

The [Qwen own-process audit preparation](validation/qwen_proc_self_audit_preparation_cpu_20261104/README.md)
passes **56 focused checks: 24 client and 32 key-generation cases** over exact
472- and 92-file source packages. The correction permits only three literal,
read-only `/proc/self` paths; write flags, aliases and unselected paths remain
denied. Three unchanged client controls reproduce the earlier alias rejection.
Each case installs the real audit hook and `os.open` wrapper with an explicit
synthetic admission. Positive cases read only their own status, cgroup or mount
metadata; retained receipts contain no proc contents. Root verified ordered XML,
unchanged source/archive bytes, empty backend audits, normal exit0 and direct
reaping, with 96 client artifacts and 32 keygen child observations preserved.
The independent process observation arrived after exit. Test cleanup and negative
probe corrections were reviewed before execution; no earlier runtime failures
are claimed for those drafts. This evidence establishes neither capacity nor
actual role readiness, generated keys, model execution or encrypted accuracy.
The earlier61 and95 checks remain separate evidence; actual entries remain disabled.

The [Qwen toy5 checkpoint preparation](validation/qwen_toy5_checkpoint_preparation_cpu_20261104/README.md)
passes **48 focused checks over 261 sources**, including all 47 checkpoint mappings,
key-before-query refresh order, 19 synthetic refresh-return observations, binary64
scale replay, recording Ledger state checks and entry-denial cases. The first run
had 42 passes and 6 failures: its fixture included 38 refresh bookkeeping phases in
the intended 47 checkpoints. The successor selects the six named layer boundaries;
all 250 runtime files, expected trace indices and test assertions remain unchanged.
Both runs and their exact sources, logs, owner receipts and reviews are preserved.
Root verified ordered XML, source/archive equality, genuine NumPy origins, empty
backend audits, observed exit codes and direct reaping. Independent process
observations arrived after exit. The corrected suite took 6.31 seconds; this
measures only these tests.
Public toy arithmetic uses ideal refresh slots and synthetic return metadata.
No ciphertext export, actual decoder/loader, HE context, capacity check or learned
model was exercised. These checks establish neither encrypted accuracy nor the
first encrypted divergence; actual diagnostic roles remain disabled.

The [Qwen server snapshot preparation](validation/qwen_server_admission_snapshot_preparation_cpu_20261104/README.md)
passes **18 focused recording-state checks over 38 sources**, including 28 unchanged
runtime files. Cases cover retained object identities, mutation rejection, caught
and uncaught reentry, sticky failures, callback binding and clean retirement.
The tests use synthetic non-proof bytes, explicitly bypass PublicContract's
constructor validation and replace the full packet validator with a recording
function. They do not test scientific proof admission or server integration.
The earlier 37-source run failed all 18 cases during fixture construction before
exercising the guard. Its exact sources, failures and missing child-owner receipts
remain preserved; no child-process absence claim is made for that run.
The corrected fixture records child ownership before success assertions. Root
verified exact sources and ordered XML, empty backend audits, 72 regular child
artifacts, 17 recorded aliases, normal exit0 and direct reaping. Independent
readback confirmed the evidence after exit. The 1.70-second duration measures
only this small suite. No actual backend, capacity, model, key, private proof,
calibration or HE execution occurred. The server's capacity denial remains in
place; performance and full-model encrypted correctness remain outstanding.

The [Qwen device-capacity preparation](validation/qwen_device_capacity_preparation_cpu_20261104/README.md)
passes **30 focused synthetic checks over 50 sources**: eight document-join cases
and 22 state/ownership cases. The real pure device and resource validators run
against fabricated metadata, with two phase-validation recordings. Seventeen
children use an explicitly temporary JAX-shaped fixture; no genuine backend or
device initialization occurs. Failed permits reject their next public operation
after the original identities and benign fixture behavior are restored.
The capsule preserves the source-only device identity correction and the unrun
49-file test predecessor. Its successor strengthens the existing assertions;
all 34 runtime/fixture files and 30 selected cases remain unchanged.
Root and independent review verified exact sources/archives and ordered XML,
restored fixture identities, empty genuine-backend audits, 120 regular child
files and 21 recorded aliases. Root observed normal exit0 and direct reaping;
the later process observations found all recorded PIDs/groups absent. Local
module origins were asserted by the source-bound children, without an emitted
origin table. The 4.18-second duration measures this suite only.
These results establish neither real capacity, reservation, lease exclusivity,
native compatibility, scientific admission nor actual server execution. The
server's capacity denial remains selected. The README records the package's
pre-copy state; this paragraph records its repository publication.

The [Qwen server snapshot integration preparation](validation/qwen_server_snapshot_integration_preparation_cpu_20261104/README.md)
passes **36 focused synthetic cases over 405 sources**, preserving 391 runtime
files. The source successor binds transitive cleanup callbacks and retains
original dispatchers so replaced exports are rejected before invocation. Cases
exercise the real snapshot, runtime/control, backend-constructor and Finalizer
paths with explicitly named admission, resource, core, audit, loader, stage and
evidence recordings. Routine checks add no full-validator or ID/digest-derivation
calls; full boundaries still reread the tiny owned public fixture. This is not
full scientific validation. The definition-only entry fixture excludes genuine
CLI bootstrap, and temporary NumPy/JAX names provide no native backend evidence.
The earlier 404-source run had 25 fixture-setup failures and 11 passes. Its direct
fixture supplied bound methods where plain functions were required; two local
delegates correct that fixture without changing runtime391 or the 36 assertions.
Both runs, including the failed run's absent aggregate report, are preserved.
Root and independent review verified exact sources/archives and ordered XML,
144 child artifacts, six tiny synthetic fixtures and 15 recorded aliases for the
successful run. Root observed both harness processes live, then normal exit0 and
direct reaping; post-exit checks found all 38 recorded PIDs/groups absent.
The 57.33-second duration measures these tests only. No actual capacity,
reservation, scientific admission, borrowed model/key/ciphertext body access or
HE execution occurred. The server's capacity denial remains selected; full-model
encrypted correctness and performance remain outstanding.

The [Qwen toy5 role preparation](validation/qwen_toy5_role_preparation_cpu_20261104/README.md)
passes **55 focused tests in 56 owned children over 427 sources**, including the
unchanged 389-file runtime. Cases cover synthetic authority joins, recorded
key/constructor/cache/NTT ownership, partial returns and original barriers,
tiny owned publication/assets I/O, all 47 recorded public checks before private
entry, full 65,536-slot decode snapshots and disabled-entry terminal behavior.
Ten children use genuine NumPy with four source hashes and three module origins
pinned; this does not authenticate its complete native dependency tree. The
decoder ABI is exercised with a recording codec, without actual decryption.
Root observed the collector and worker live, then normal exit0 and direct reaping.
Root and independent review verified unchanged source/archive bytes, ordered
collection/XML, 392 child receipt/log files and all 58 recorded PIDs/groups absent.
The single run took 76.03 seconds; this measures only these preparation tests.
The public capsule includes one exact final source archive and sparse overrides
that reconstruct the earlier 18/382/385/387/389 file maps. Historical archive
digests are retained as provenance; their original tar bytes are omitted.
Fixture bodies and the deliberately retained FIFO are not copied or opened;
pytest aliases are recorded without following them. Source corrections and
authoring failures retain their original scope and history.
No genuine key generation, backend/context/device/capacity observation, borrowed
model/key/ciphertext body, actual role, HE inference, encrypted first-divergence
or speed improvement is established. Actual diagnostic entry remains disabled.
Full-model encrypted correctness and performance remain outstanding.

The [Qwen server device integration preparation](validation/qwen_server_device_integration_preparation_cpu_20261105/README.md)
passes **59 focused recording cases over 460 sources**, preserving all 403 runtime
files. Cases cover device-packet and native-open-flag joins, resource arithmetic,
snapshot and owner transitions, loader/stage/engine controls, audit policy,
publication limits and cleanup. Scientific/model/key admission, installed backend
and core operations, arrays, native ownership/synchronization and host/device
observations use explicitly named recordings. The definition-only entry omits
the original final dispatch; temporary JAX and NumPy names provide no genuine
backend or CLI-bootstrap evidence.

Root and independent review verified ordered collection/XML, exact current,
execution and archived sources, 236 bounded regular child records, 14 fixture
files inspected only through metadata, and 15 unfollowed aliases. Root observed
the collector and worker live, then their normal exit0 and direct reaping;
all 59 child waits were joined and all 61 original PID/process-group pairs were
absent after completion. The 1746.10-second duration measures these tests only.

Both failed predecessors remain separate evidence: the 458-source run failed
all 59 cases during fixture bootstrap, and the 459-source run had 43 passes and
16 fixture/guard failures. The corrections preserve the runtime and selected
cases; overlapping counts are not combined. One final source archive and sparse
overrides reconstruct all three exact file maps. Historical 458/459 archive
digests remain provenance; their original tar bytes are omitted from the portable
capsule. Its verifier passed after repository copying. Fixture bodies and aliases
were not copied. The selected native authority remains denied; actual capacity,
scientific admission, full-model HE correctness and performance remain unproven.
