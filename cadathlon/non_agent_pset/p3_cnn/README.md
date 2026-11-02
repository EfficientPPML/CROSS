# Problem 2: non-AI track

Read [the handout](doc/problem.html) or [the PDF](doc/problem.pdf).
Implement `src/cnn_opt.py`; `src/baseline.py` is a correct unoptimized CROSS
starting point. Both LeNet and AlexNetTiny are required.

From `src/`, run `make reference-test`, `make baseline-plan`,
`make baseline-test`, then `make test` after implementing the TODO.
The HE commands may take minutes and gigabytes. Set `CROSS_ROOT` if this
package is extracted outside the repository. `make submit` makes a local
archive. Public synthetic cases verify arithmetic; official accuracy runs
use organizer-supplied trained weights and images, as specified in the handout.

The local runner's `setup_s` times only the mapping constructor. First-request
compilation may add further cost. `warmup_s` and `warmup_server_s` retain the
excluded warmup samples; with `--warmups 0`, `cold_start_included` is true and
the first measured request includes any lazy setup. `total_s` includes fixture
loading, setup, all requests and mapping release, so use it when planning an
evaluation slot. Existing request timing and correctness gates are unchanged.
