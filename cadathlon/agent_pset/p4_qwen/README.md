# Problem 4: AI-agent track

Read [the handout](doc/problem.html) or [the PDF](doc/problem.pdf).
Implement `src/qwen_he.py`, starting from the exact semantics in
`src/reference.py`. `src/export_weights.py` prepares local official weights.
The full target is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B);
the six public fixtures are small decoder tests.

From `src/`, run `make all`, `make reference-test`, then `make test` after
implementing the TODO. `CROSS_ROOT` must point to the supplied CROSS checkout
if this package is extracted elsewhere. `make submit` creates a local archive.
The plaintext oracle earns no encrypted-computation credit. See the handout
for full-model workloads and the report template for required evidence.

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

`src/qwen_reference.py` is an **experimental, unvalidated encrypted toy circuit**.
It uses separate client/evaluator contexts, public polynomial approximations
and genuine homomorphic refresh for deeper graphs. It has not passed all six
encrypted fixtures or the full Qwen workload. Select it explicitly with
`--implementation`; the default student file is unchanged. Its large public
evaluation keys may require tens of GiB of disk space;
`CROSS_PUBLIC_KEY_DIRECTORY` selects their temporary directory. Keys are
deleted when the session closes. Operator and plaintext planning tests are not
evidence that this complete reference or the full benchmark is ready.
