# Problem 3: benchmark track

Read [the handout](doc/problem.html) or [the PDF](doc/problem.pdf).
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
