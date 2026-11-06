# Problem 2: Enable and optimize BERT on TPUs with CROSS

Read the [HTML assignment](doc/problem.html) or [PDF](doc/problem.pdf).

**Use or improve CROSS to enable encrypted BERT inference on TPUs, then
optimize it for better benchmark results and a higher assignment score.**

1. **Use the [CROSS library](https://github.com/EfficientPPML/CROSS) and improve it as needed**
   to implement encrypted BERT and execute its server computation on TPUs.
   Submit changes to the library in `cross.patch`.
2. **Evaluate your results with [fhe-benchmarking/BERT](https://github.com/fhe-benchmarking/BERT).**
   Run your CROSS implementation through its pinned evaluation harness.
   First pass the correctness and accuracy gates; then reduce latency and
   increase throughput. The handout's scoring rubric converts these validated
   benchmark measurements into assignment points.

**Students are explicitly allowed to use AI agents** for this workload,
including implementation, debugging, testing, profiling and optimization.
Disclose the tools used and verify their work in `REPORT.md`.

The target is `google-bert/bert-base-cased-finetuned-mrpc`, all 12 encoder layers,
the CLS pooler and two-class MRPC head, with sequence length 128.

Implement `src/bert_he.py` using CROSS and extend the library where needed.
The package includes seven upstream-compatible
stage entrypoints, working tokenization and output adapters, a strict result
checker, six synthetic reference fixtures, and a staging tool that preserves
the pinned upstream harness. The upstream DESILO submission is background
material; it is not this track's implementation or performance baseline.

```bash
cd src
make all
make reference-test           # six small plaintext checks, no downloads
make test                     # intentionally fails until bert_he.py is implemented
make submit                   # creates a local submission.tar.gz
```

For actual benchmark staging and commands, follow section 3 of the handout.
`make benchmark-check BERT_ROOT=/path/to/BERT` verifies the pinned checkout;
`make benchmark-stage BERT_ROOT=/path/to/BERT BERT_RUN=/new/run/directory`
copies its unchanged harness, your current source, and the packaged shared
runtime helpers into a new run directory. The helper copies live in
`submissions/_cross_common/`; `cross_staging_manifest.json` records their
SHA-256 hashes alongside the stage sources. Stage and adapter imports reject
missing or changed staged helpers. Restage after changing your source or
helpers. Set `CROSS_ROOT` to your patched core checkout. Staging dependencies
does not implement the five encrypted file-stage TODOs or validate full BERT.
Use the staged `submissions/runner.py` when selecting a staged experimental
adapter; mixing already-imported helper copies from different runs is rejected.

The `.golden` is explicitly a **plaintext numerical oracle**. This starter
does not include a completed encrypted BERT solution or claim MRPC accuracy
or TPU performance. Organizers must freeze assets and calibrate the proposed
100-point, eight-hour rubric before an event.

Runner diagnostics and timing:

```bash
python runner.py --preflight  # exits 1 for the shipped unimplemented starter
python runner.py --case ../test/test1.in --implementation /path/to/encrypted_impl.py \
  --require-tpu --warmups 1 --repeats 3
```

Run these commands from `src/`. Remove `IMPLEMENTATION_STATUS = 'unimplemented'`
when implementing the starter. Preflight checks callable interfaces only; it
cannot establish encrypted correctness. `--implementation` selects a separate
module without overwriting your work. Passing a `.golden` reports an explicit
plaintext-oracle error before device initialization.

Every warmup and measured request uses fresh encryption and must pass the
ciphertext, live-ring security and numerical checks. `server_s` and
`end_to_end_s` are medians; `samples` contains each measured request and
`warmup_samples` contains excluded warmups. `cold_start_included` identifies
runs with no warmup, and `total_s` includes setup and cleanup. `--require-tpu`
prevents a CPU fallback from being mistaken for a TPU measurement. These
synthetic fixture times are **not full-model performance baselines**. A
completed encrypted reference and measured full-workload scoring calibration
remain organizer prerequisites.

`src/bert_reference.py` is an **experimental, unvalidated encrypted toy circuit**.
It uses separate client/evaluator contexts, public polynomial approximations
and genuine homomorphic refresh for deeper graphs. It has not passed all six
encrypted fixtures and does not implement the full-model file-based harness
stages. Select it explicitly with `--implementation`; the default student file
is unchanged. Its large public evaluation keys may require tens of GiB of disk
space; `CROSS_PUBLIC_KEY_DIRECTORY` selects their temporary directory. Keys are
deleted when the session closes. Operator and plaintext planning tests are not
evidence that this complete reference or the MRPC benchmark is ready.
