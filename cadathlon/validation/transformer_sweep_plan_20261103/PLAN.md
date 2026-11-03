The executable scope is a six-public-case encrypted TPU regression and timing
sweep for each optimized toy adapter. There is no executable full-workload
calibration publisher yet. Even a successful 1-warmup/3-repeat sweep must retain
`scoring_calibrated=false` and `full_model_evaluated=false`.

The reviewed arithmetic comes from
`/tmp/runtime-followup-integration-20261103/checkout`, the composed 32-file
candidate. The prepared commands select its separate corrected-collector copy,
`/tmp/transformer-collector-outer-failure-20261103/checkout`. The two-file
collector follow-up must accompany the composed candidate at promotion. It
does not change encrypted arithmetic. No TPU command below has been executed
by this review; the existing Qwen lease and runtime freeze remain in force.

After the current process is terminal/reaped and the parent explicitly assigns
the next TPU reservation, use new evidence paths and an existing public-key
parent with enough free space. These are independent commands, run sequentially:

```sh
bash /tmp/transformer-sweep-plan-20261103/run_public_sweep.sh \
  bert /tmp/bert-optimized-six-warm-NEW /dev/shm
bash /tmp/transformer-sweep-plan-20261103/run_public_sweep.sh \
  qwen /tmp/qwen-optimized-six-warm-NEW /dev/shm
```

The script uses Python from the supplied `jaxite` environment, cores160–167,
explicit TPU backend, one OMP/BLAS thread, level-cache capacity1 and NTT-cache
capacity2 (the existing bounded-Qwen proof settings). It binds CROSS/PYTHONPATH
to the selected checkout, removes inherited XLA overrides and the incompatible
rotation-key-skipping option, and disables bytecode writes. Set `SWEEP_ROOT`
to the reviewed promoted checkout when appropriate. Preserve the exact source
archive and environment for that path before launching; verify its bytes
against the approved patch set, and retain collector source hashes afterward.

All six cases are explicit. Each case runs in a fresh sequential subprocess,
creates one session and fresh keys, performs one validated warmup and three
validated measured encrypt/evaluate/decrypt requests, then closes the session.
That is 24 requests per model, 48 across both models. Fresh encryption uses
the same fixture plaintext for every request. The warmup is excluded from
the reported medians; setup and cleanup are outside each request but inside
the runner's total. `server_s` includes synchronized evaluation;
`end_to_end_s` includes encryption, input/output gates and synchronization,
evaluation, and decryption, without a separate network transport. Fixture
oracle comparison and numerical checks are outside each request timer.

The script supplies no timeout for healthy work. `--cleanup-grace-s 30` is a
cancellation grace period, not a work deadline. A separate diagnostic can use
`--warmups 0 --repeats 1` with all six cases and a fresh output path; it retains
`cold_start_included=true` and is not a steady measurement. Partial case
selections cannot set `all_six_passed`. Do not apply the full-workload handout's
per-prompt timeout to this toy diagnostic without explicitly changing its
declared scope.

The public CPU-only plan in `preflight_plan.json` uses the exact constructor
profiles and all-ones setup masks, and prohibits key generation. All twelve
actual generated Q/P profiles pass the repository's classical128-bit and P
coverage checks. The following public-key requirements include the store's
header allowance and 64MiB reserve, but exclude other memory/storage:

| Case | Circuit / chain depth | Period | dnum | Rotation keys | Store GiB |
| --- | --- | --- | --- | --- | --- |
| BERT1 | 50 / 50 | 512 | 5 | 27 | 16.147 |
| BERT2 | 50 / 50 | 2048 | 5 | 37 | 22.104 |
| BERT3 | 93 / 49 | 2048 | 4 | 117 | 56.735 |
| BERT4 | 93 / 49 | 8192 | 4 | 199 | 96.455 |
| BERT5 | 136 / 49 | 32768 | 4 | 226 | 109.533 |
| BERT6 | 136 / 49 | 65536 | 4 | 138 | 66.907 |
| Qwen1 | 38 / 38 | 32 | 2 | 17 | 3.881 |
| Qwen2 | 67 / 49 | 128 | 4 | 57 | 27.672 |
| Qwen3 | 123 / 49 | 512 | 4 | 81 | 39.297 |
| Qwen4 | 123 / 49 | 2048 | 4 | 116 | 56.251 |
| Qwen5 | 179 / 49 | 1024 | 4 | 100 | 48.501 |
| Qwen6 | 179 / 49 | 8192 | 4 | 212 | 102.752 |

Reserve more than the maximum109.533GiB before starting, recheck actual free
space and unrelated jobs at launch, and treat tmpfs pages as host memory.
One case's successful cleanup permits sequential space reuse. This inventory
does not establish host/device fit, peak HBM, per-request time or refresh
accuracy at these periods. In particular, the full65536-slot BERT6 layout has
no repeated aliases; a period128 refresh proof cannot establish its behavior.

Every request requires Polynomial ciphertext wrappers, actual live CKKS Q/P
security at both the encrypted input and output, valid sigma and secret
distribution, ciphertext-prefix binding, finite shape-correct decoded output,
and numerical gates. BERT requires elementwise `abs(error) <= .02 + .01*abs(ref)`
and matching binary predictions; Qwen's toy gate requires maxabs<=.20 and
nRMSE<=.10. Qwen toy top1 is reported but is not a pass criterion. These local
checks do not audit secret separation or arbitrary evaluator internals.

Evidence is written under `report.json` and each `testN/{process.json,
stdout.json,stderr.log}`. Require overall `passed=true`, all-six completion,
unchanged source hashes, all request counts, and successful process/key/handle
cleanup. A numerical case failure is retained and the collector continues
through later cases. Source drift or incomplete cleanup stops the sweep;
external cancellation stops and reaps the owned process group. A child that
leaves descendants is failed and stopped. Cleanup removes only the owned
temporary key subtree after device/inode verification; unrelated parents are
retained. Missing, corrupt or incomplete evidence is a failure, never a
baseline. Fresh output creation rejects stale attempts.

The corrected outer collector persists an initially false/incomplete record
before fallible metadata reads. It preserves primary cancellation/failure if
final source inspection or evidence publication also fails. If final report
publication fails after the last child succeeds, the prior incomplete sweep
record remains and the command raises; case records cannot substitute for
successful final publication.

Remaining evidence gaps are concrete: the runner discards the manifest
returned by its actual-ring gate, and it reports requested device kinds/counts
rather than an execution-device identity audit. The collector hashes sources
but does not archive their bytes. It does not collect transfer volumes,
operation/refresh counts, peak aggregate process-tree RSS, device peak memory
or hardware/driver identity. Preserve exact bytes plus external resource and
device records for any subsequent performance claim; `/usr/bin/time` peak RSS
alone is not a simultaneous aggregate-process-tree measurement. The source
manifest is not an arbitrary third-party import-closure audit. Package versions
and source paths must remain fixed for a comparable run.

P2 full calibration remains blocked by the five missing encrypted file-stage
functions. The optimized adapter's `--preflight` exits1 and names those missing
interfaces, despite its available toy `create_session`; this was reproduced
on CPU. The full contract is all12 pretrained BERT layers, pooler/classifier,
sequence128 and ranked100/200/408-pair MRPC instances. It needs three separate
one-run harness invocations per size with published seeds42,43,44, fresh
encryption, strict `check_result.py`, unchanged ordered sample identities,
accuracy>=75% and at most2 percentage-point loss against the pinned cleartext
checkpoint. Immediately archive each overwritten harness measurement/output.
Online latency is the median sum of the five outer stages5–9; toy server or
request medians cannot supply it. Once the file stages actually exist, the
documented commands are:

```sh
# Provision pinned upstream/checkpoint/tokenizer/MRPC assets first.
python stage_benchmark.py --upstream /path/to/pinned-BERT \
  --destination /path/to/new-bert-run
cd /path/to/new-bert-run
python harness/run_submission.py 1 --num_runs 1 --seed 42
python submissions/check_result.py --benchmark-root . --size 1
# Archive raw inputs/results/predictions/checker output before any next run.
# Repeat independent runs for seeds43/44 and sizes2/3 with their own archives.
```

The three official `T_base_i` values must come from a correct CROSS BERT
implementation on identical hardware, data, security, timing and cache policy,
then be frozen before a speed leaderboard. No baseline manifest/scorer or
publication command for P2 is currently supplied; the toy collector cannot
create one.

P4 full calibration requires frozen Q1–Q6 workloads: full-width operators,
layer0/S8, all-layer16-prompt suites at S8/S16/S32, and eight four-token decode
sequences. Use an unrelated fresh warmup per shape, score each distinct prompt
once, and report suite median/p95. Compare HE to the declared approximate
circuit separately from approximate-to-original Qwen, with the published
full-suite top1 and decode gates. Current toy repeats do neither comparison
and do not implement those datasets or the full operator/decode interfaces.
There is no executable Q1–Q6 judge/calibration publisher yet. The handout's
60-second efficiency target is a scoring constant, not a measured baseline.
No toy command can fill this gap.
