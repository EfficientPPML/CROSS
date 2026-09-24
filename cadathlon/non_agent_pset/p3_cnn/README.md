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
