# Problem 1: AI-agent track

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
