# Author validation

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

## Limits of this validation

This validates the authored problem set, public numerical references, actual
CNN planning path, build and packaging. It does not establish a completed
student solution. The full encrypted Qwen implementation is the assignment;
the `.golden` oracle is explicitly plaintext. Full-checkpoint Qwen inference,
trained CNN accuracy, hidden-case judging and leaderboard timings have not
been claimed as validated. The source checkout does not bundle the two
trained CNN checkpoints or official hidden prompt/image suites.

Event organizers must complete the asset and hardware preflight in
`ORGANIZER.md` before running a scored contest. Those event operations are
distinct from delivery of this problem proposal and its starter materials.
