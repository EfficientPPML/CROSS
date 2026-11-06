# Exploratory evaluator cache-retention probe

The isolated probe passed the same three genuine encrypted full-width BERT projection gates as the [original component proof](../bert_fullwidth_linear_cpu_20261103/README.md). Maximum absolute error was4.61e-11 across all8192 slots. All100 captured sources stayed unchanged, exact archived bytes were independently verified, and the process exited0 with successful key-directory cleanup.

The only runtime change removed the evaluator's global `jax.clear_caches()` call. Explicit control/binding, Polynomial registry and assembled NTT cache evictions remained. The process took379.40seconds, compared with594.91seconds in the preceding original run; peak host RSS was1002216 versus922904KiB. These separate CPU processes used fresh random keys and uncontrolled host load. The observations justify further investigation but are not a controlled speed comparison.

**Do not promote this patch yet.** This shallow N16384 Q4/P3 experiment does not establish cached-trace lifetime safety, Polynomial-registry compatibility, deep-chain/refresh behavior, or TPU/HBM safety. The [exploratory.patch](exploratory.patch) preserves the measured change; it is not a production-ready opt-in. [summary.json](summary.json) records exact artifact hashes, source differences and limitations.
