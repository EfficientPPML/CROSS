# Organizer guide

This package is a problem-set proposal and a runnable development scaffold.
It does not claim that full Qwen HE inference is already implemented or that
the proposed time/error limits have been demonstrated on contest hardware.
Students are asked to implement that capability. The CNN reference uses the
existing real CROSS pipeline.

The four problems are split into two sets. `non_agent_pset` is solved without
coding agents and holds **P1 (essential, 100 points)** and **P3 (bonus, 20
points)**, so a team there can reach 120; its total is essential + bonus.
`agent_pset` holds **P2** and **P4**, both 100 points, with agents allowed.
Keep the two sets on separate leaderboards, and the two `agent_pset` problems
on separate leaderboards from each other.

## Format and difficulty

Use an eight-hour event: roughly 45 minutes for onboarding, 75 minutes for
baseline/operator checks, four hours for implementation, and two hours for
verification and reporting. For `agent_pset`, a full encrypted transformer is
an ambitious final milestone. The staged rubric is intentional. If requiring
most teams to finish full Qwen in one day, provide an organizer-tested full
baseline before release; building that baseline is additional research work,
not something the current library or this package silently supplies.

Do not silently replace the full model with the public tiny fixtures. If
preflight shows that full-model scoring is infeasible, extend the event or
publish an explicitly revised challenge before registration. The objective in
the authored handout remains the complete Qwen3-0.6B model.

Judging P3 differs from every other problem here, and needs preparation the
others do not. It is **open-ended and all or nothing**: 20 points when a team
demonstrates both end-to-end HE-on-TPU correctness and performance no more than
50% slower than the state of the art (`T <= 1.5 * T_sota`), nothing otherwise.
Two things must be settled before release. **Name the state-of-the-art
implementation**, pin its commit, and measure `T_sota` yourself on the contest
hardware with the same models, images and batch sizes; a number quoted from a
paper is not a baseline, and a cross-hardware comparison is not a speedup.
**Decide what counts as a correctness demonstration** and publish it, since
there is no per-case rubric to fall back on: require per-image agreement
against the team's own cleartext run, not against a different network, and
require an evaluator process that never receives the secret key or a plaintext
activation. Expect to reject fast submissions whose correctness is unproven;
that is the intended behavior, not an edge case. Because the award is binary,
write the verdict down with the evidence that produced it, and apply the same
standard to every team.

P3, the `non_agent_pset` bonus problem, optimizes execution of two fixed
networks. Unlike the 2019
pruning template, it does not reward removing parameters or retraining. This
keeps its score attributable to CROSS/accelerator optimization and makes an
eight-hour hands-on track practical with pre-staged assets.

The benchmark problem (P1) optimizes throughput of one fixed network across one
and eight chips, and it is scored through the FHE benchmarking suite's own
harness rather than a runner written here. That is deliberate: correctness is
decided against the suite's ground-truth labels and performance comes from its
`Timing` block, so a team cannot self-report either.
`src/suite/install_suite.py` refuses to run if `harness/` differs from the
pinned suite commit, which is the suite's own rule. It does replace
`scripts/get_openfhe.sh` and `scripts/build_task.sh` with no-ops, keeping
`.orig` backups, because `harness/utils.py::build_submission` always invokes
both and a pure-Python submission has nothing to compile; `suite-check` reports
this. Confirm it before judging, and confirm no other path outside
`submissions/mnist/` differs from the pinned commit. P1 is the essential
problem of `non_agent_pset` and is solved without coding agents.

Two properties of the pinned harness mean `results-N.json` is not by itself a
correctness verdict, and a judge must not treat it as one. The single instance
has no `Quality` block: `run_submission.py` calls `verify_result` with
`check=False`, so a wrong single inference neither fails the process nor leaves
any trace in the measurement file. And `calculate_quality` divides by the
number of predictions the submission wrote rather than the number of images, so
a truncated prediction file reports a high accuracy over the images it skipped.
`src/suite/archive.py` recomputes both from the harness's own label and
prediction files, stores the verdict in `src/suite/results/<case>_<label>/
case.json`, and `score.py` awards a case zero without it. That verdict covers
the **final** run only, because the harness overwrites `io/<instance>/` on each
run; for a ranked judgement also read the per-run `[harness] PASS`/`FAIL` lines
from the harness's own stdout, and preserve every run's `io/` directory.
Archiving is also what keeps C1 and C2 apart: the suite keys its output by
instance alone, so the two cases that share the single instance and differ only
in chip count would otherwise overwrite each other.
`make -C non_agent_pset/p1_benchmark/src suite-gates` self-tests the gate arithmetic and needs
no accelerator or suite checkout.

Two further properties of the pinned harness affect how its numbers may be
read. `--num_runs 3` does **not** sample three image sets: `run_submission.py`
re-creates its RNG from the base seed inside the run loop, so every run derives
the same `generate_input` seed (`--seed 3` yields `1742692731`) and evaluates
the same images. The spread across runs is machine noise; to sample input
variation, run separate invocations with different published base seeds. And
stages 1, 2.2 and 3 execute once outside that loop, so every `results-N.json`
repeats one measurement of key generation and model preprocessing, and every
`Total` includes that one-time setup. `Total` is not per-run latency; stage 7,
`Encrypted computation`, is the only ranked quantity measured per run.
The upstream checkout also ships the reference implementation's own
`measurements/<instance>/results-N.json`. `score.py` prefers the per-case
archive and labels any fallback read, but confirm `read_from` in its output
points at `src/suite/results/` before accepting a score.

Most importantly, the suite's `Quality` block does **not** contain the
cleartext reference this problem's accuracy gate is written against.
`harness/cleartext_impl.py` runs `harness/mnist/mnist_ffnn_model.pth`, a
784-128-64-10 network; the ranked P1 model is 784-50-10 with an `x**2`
activation. `Harness model quality` is that other network's accuracy, so the
agreement between it and a submission's encrypted output measures a difference
in training, not HE fidelity, and must never decide a case. `archive.py` runs
the ranked model in cleartext on the harness's own pixel file and records
`verify.own_cleartext`; `score.py` gates the argmax agreement and the accuracy
drop against that, and keeps the harness-model figures as context. Verified on
the measured baseline: 96/100 cleartext and 96/100 encrypted on the 100
small-instance images, against 97/100 for the harness network. Scoring the
earlier way would have failed the unmodified baseline on both the 0.99
agreement gate (it read 0.98) and, through a binary floating-point artifact in
`(0.97 - 0.96) * 100`, the 1 pp accuracy gate. Re-validate both gates against
your own weights before release, and keep `archive.py` organizer-owned: a
contestant who could edit it could write their own correctness verdict.

## Assets to freeze before distribution

1. **Environment:** pin the CROSS commit, Python/JAX/jaxlib/libtpu/PyTorch/
   NumPy versions, host CPU/RAM, TPU type, one-chip allocation, thread counts
   and relevant environment variables. Save `pip freeze`, `jax.devices()`,
   and a clean `git diff`/commit manifest. Verify isolated chip allocation;
   slicing `jax.devices()[:1]` in a harness is not a resource scheduler.
2. **Qwen:** acquire the exact model revision in `agent_pset/p4_qwen/test/model_source.json`
   plus tokenizer/config files. Hash every artifact. Provision them before the
   event; do not make download speed part of the competition. Check license
   notices when redistributing. `src/export_weights.py SNAPSHOT OUTPUT` exports
   local safetensors to `.npy` files with a manifest (requires `safetensors`
   and PyTorch). Load via `np.load(path, mmap_mode='r', allow_pickle=False)`.
   The config hash only checks the configuration: separately verify shard
   hashes and all 28 layers' expected tensor names/shapes against the pinned
   snapshot. Export success alone is not proof of checkpoint identity.
3. **CNNs:** acquire MNIST/CIFAR-10 and train once outside contest time:

   ```bash
   cd /home/jianming_gatech/CROSS
   python demos/prepare_demo_assets.py --models lenet alexnet_tiny --epochs 30
   ```

   Confirm supported model names with `--help` for the pinned checkout.
   Freeze the exported binaries and their hashes. Convert weights into
   dictionaries with exact PyTorch state-dict names. In `CanonicalDemo`,
   `load_weights()` permits a partial dictionary, so explicitly assert that
   supplied keys equal `model.state_dict().keys()` and validate all shapes.
   Use `QuadLeNet` and `QuadAlexNetTinyInfer(client_downsample=False)`. Exported
   AlexNet weights already contain BatchNorm folding. Neither the demo's
   random defaults nor its random-input fallback can be used for label scores.
4. **Data:** freeze 64 public calibration and 256 private test images per CNN
   from disjoint IDs. Freeze the Q1–Q6 calibration/hidden embedding or prompt
   suites in the handout, including exact tokenizer settings, IDs, position
   IDs and masks. Keep the reference-answer files out of contestant archives.
   Include near-boundary nonlinear values, cross-lane-distinct images,
   attention to non-final tokens and inputs different from warm-up inputs.
5. **Original-model oracles:** store full CNN logits/labels and original Qwen
   layer outputs, last-position logits and decode sequences. Use the original
   pinned Qwen implementation in an explicitly frozen dtype/backend, and
   compare the NumPy reference on matching small cases as an independent
   implementation check. Tiny random cases are never full-model goldens.
6. **Benchmark suite (P1):** clone `fhe-benchmarking/ml-inference` at the
   commit pinned in `non_agent_pset/p1_benchmark/src/suite/suite_config.json`, verify
   `harness/` is untouched with `make -C non_agent_pset/p1_benchmark/src suite-check`, and
   provision the MNIST data before the event so download speed is not part of
   the competition. Freeze `non_agent_pset/p1_benchmark/weights/mlp_784_50_10.pth` and its
   hash manifest, or replace it with your own trained weights for the same
   `Linear(784,50) -> x^2 -> Linear(50,10)` architecture and re-hash. Publish
   the image selection: the suite derives it from `--seed`, so record the seed
   and the repeat schedule for ranked batches above 256 images. Measure the
   judge's own baseline with `make -C non_agent_pset/p1_benchmark/src suite-baseline` on the
   exact hardware and commit the resulting `baseline_manifest.json`; the
   authored one in the package is indicative only.
7. **Actual encrypted preflight:** run baseline CNN inference on every official
   batch; confirm numerical/accuracy gates, key/setup time, per-batch budgets
   and memory on the exact judge. Validate 128-bit parameters at actual Q/P
   sizes. For Qwen, establish operator and full-width block feasibility and
   calibrate approximation intervals/thresholds on public data. Establish a
   full encrypted run before advertising full-model completion as an expected
   eight-hour outcome. No full HE Qwen golden is bundled here.

Publish a versioned `contest_manifest.json` containing artifact SHA-256 values,
environment, hardware, limits, accuracy gates and baseline measurements; sign
off each item above. The handouts' numbers are proposed defaults until this
manifest is frozen. Do not change thresholds after seeing team submissions.
Training and calibration are event operations, not required to read or run
the supplied numerical fixtures.

## Judge procedure

The bundled runners are smoke tools, not a security boundary. Never trust a
team's `passed` field, self-reported times, claimed parameter security or
memory values. Use organizer-owned driver/checker code with fixed inputs and
fresh randomness, hold reference outputs outside the evaluator, and rerun on
exclusive hardware. Disable network access for final evaluator runs.

Audit the candidate's `execute`/`evaluate` path and CROSS patch: no decryption,
reference loading, hidden plaintext inference, result lookup, weak randomness,
reduced parameter set or prompt-dependent public preprocessing. For Qwen,
inspect every transformer operator and demonstrate a separate evaluator
process receiving no secret key or plaintext activations. A wrapper named
`Polynomial` is necessary at the API boundary but is insufficient proof that
the implementation actually performed HE. Client-side embedding lookup and
final decryption are permitted; intermediate client refresh is not.

For approximate Qwen, also execute the contestant's declared approximate
circuit in plaintext to separate numerical HE error from model approximation.
Compare against original Qwen independently. Save all numeric outputs before
argmax; a token string alone cannot establish logit fidelity. Check long
chains for exhausted levels and validate every bootstrap transition. The
whole-model claim requires all layers and all vocabulary logits, not a
constant-size slice extrapolated to the full model.

Instrument prepare, encryption, synchronized server execution, transfer and
decryption separately. Record physical memory with the fixed profiler;
`Mapping.estimate_live_memory()` is a logical estimate. Apply runtime/RSS
limits externally to the complete process tree. The example CLI does not
enforce cgroup limits. Measure baseline and candidate with identical timing
boundaries; warm-up is allowed, excluding prompt-specific cached answers.
Apply the formulas in the handouts only after independent case verdicts.

Keep reference and candidate processes separate, and rerun borderline cases
or ties consistently. A failed case earns zero for that case; one failed CNN
must not erase credit for a valid case on the other model. In the Qwen track,
awarding operator/block credit does not establish full-model completion.

## Packaging and review

`make release` produces `contestant-release.tar.gz` with the three template-style
problem directories and shared support. `EXCLUDE` removes `.golden` files and
hidden/full-reference paths. It is still necessary to inspect archive members
before distribution. Author-only files (`ORGANIZER.md`, generation and audit
tools) are not in the contestant archive. Public numeric `.out` files stay in.
The full author package, including source docs and reference goldens, remains
in this directory. No external publication is performed.

`make submit` packages source, `cross.patch`, `REPORT.md`, `results*.json` and
shared support; it does not package the datasets/checkpoints or test answers.
Apply patches to a fresh pinned CROSS checkout, import from that checkout,
and run the organizer-owned cases. Public starter TODOs are intentional.

The supplied template's `.golden` convention is followed for reference code:
the CNN golden is a valid unoptimized encrypted solution; the Qwen golden is
only a complete plaintext numerical oracle for the decoder semantics. This
distinction is stated in the student handout. A full encrypted Qwen solution,
hidden checkpoint-specific answers and measured judge baselines must not be
implied by the presence of a file with the `.golden` suffix.

## BERT benchmark preflight (Problem 2)

AI agents are explicitly allowed for students in this track, including coding,
debugging, testing, profiling and optimization. Keep that permission in both
rendered handouts and require disclosure and verification in the report.

Use the pinned `fhe-benchmarking/BERT` commit and harness hashes in
`agent_pset/p2_bert/src/benchmark.py`. `stage_benchmark.py` copies its untouched harness
and the CROSS stage adapters into a fresh directory, without copying DESILO
code. Freeze the model/tokenizer revision, full checkpoint hashes, GLUE MRPC
dataset revision, environment and all selected record identities. Upstream
model loading uses an unpinned `main`: provision an isolated cache whose main
reference is the approved snapshot and enforce offline loading during judging.
Do not run the public benchmark against whichever checkpoint is currently
online. Retain the upstream Apache license with staged harness files.

Verify paired tokenization, padding/truncation to 128, evaluation mode and all
12 encoder layers through the trained pooler/classifier. Client embedding
LayerNorm and public attention masks are allowed as described in the handout.
Enforce server filesystem isolation externally: it must not see client text,
tokens, dataset target IDs, labels, secrets or decrypted results. Separate
subprocesses alone do not enforce that boundary. Inspect serialized upload
and key artifacts and all evaluator source paths, including CROSS patches.

Require complete sample counts (1, 100, 200, 408) and ordered finite logits.
Run an organizer-owned copy of `check_result.py` after the upstream harness.
The pinned harness prints accuracy failures without necessarily failing its
process and compares size-0 labels with `check=False`. Its batch metric can
also accept incomplete output counts. A successful process exit is not a
correctness verdict. For size 0, run the explicit harness cleartext command
before the supplementary checker. Ranked sizes require at least 75% accuracy
and at most a 2-percentage-point drop against the frozen plaintext model.

Use three separate `--num_runs 1` invocations with published seeds and fresh
encryption, preserving every run's files before they are overwritten. A fixed
seed under `--num_runs > 1` repeats the selection at this revision. Size 3
always uses all 408 validation examples. Distinguish setup, dataset/reference
work and online stages 5–9; the upstream `Total` is not online latency.
Control retained `io/server_data` and page-cache behavior for every candidate.

Supply public calibration inputs and full-width hidden operator/block cases;
validate the proposed error gates, runtime/RSS/storage limits and three
full-checkpoint ranked workloads. Establish a correct CROSS implementation
and measured baseline for all ranked sizes before publishing speed scores.
The bundled BERT `.golden` is a plaintext oracle only. No full HE BERT run,
MRPC accuracy result, calibrated CROSS BERT baseline or feasible eight-hour
completion is implied. Large evaluation runs may follow the development
window. Never substitute DESILO performance numbers for a CROSS baseline.
