# Problem 1: encrypted MNIST throughput — agents allowed

Read [the handout](doc/problem.html) or [the PDF](doc/problem.pdf).

**AI agents are allowed.** You may use coding agents and AI assistants to
design, implement, debug, test and profile your solution. Disclose the tools
and assisted work in `REPORT.md`, and independently verify generated code and
results. All correctness, privacy, security and scoring requirements apply.

Implement `src/bench_opt.py`; `src/baseline.py` is a correct unoptimized CROSS
starting point that places one image per ciphertext. Both the latency cases
(batch 1) and the throughput cases (batch 100 and 1000) are required.

From `src/`, run `make reference-test`, `make baseline-plan`,
`make baseline-test`, then `make test` after implementing the TODO.
`make baseline-plan` prints the actual derived plan and the slot occupancy the
problem asks you to improve. The HE commands may take minutes and gigabytes.
Set `CROSS_ROOT` if this package is extracted outside the repository.
`make submit` makes a local archive. Public synthetic cases verify arithmetic;
official accuracy runs use organizer-supplied trained weights and images, as
specified in the handout.

The local runner's `setup_s` times only the service constructor. The baseline
initializes keys and compiles lazily on its first request. `warmup_s` and
`warmup_server_s` retain the excluded warmup samples; with `--warmups 0`,
`cold_start_included` is true and the first measured request includes any
lazy setup. `total_s` includes fixture loading, setup, all requests and service
release, so use it when planning an evaluation slot. These smoke timings do
not replace the ranked suite's timing boundaries.

Suite stage 3 accepts an optional `service.warmup_input()` hook returning a
dummy ciphertext bundle that covers every physical execution shape used by
that service. The baseline encrypts one zero image and repeats its ciphertext
across one complete device lane group: every later group runs the same
Mapping. This avoids encrypting and evaluating the entire 1000-image batch
just to warm that Mapping. Both input and output security checks still run.
Services without the hook retain full-batch warmup, and ranked input counts
and timing boundaries are unchanged. This reduces warmup work; it does not
establish that the full six-case evaluation meets a five-minute budget.
