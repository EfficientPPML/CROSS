Client-only size0 reference evidence
===================================

The exact upstream size0/base-seed42 input (one example,32 valid tokens) passed
the unchanged independent full12 approximate reference on2026-11-03. All156
nonlinear domain observations passed. Approximate logits differed from the
official/default-FP32 native model by at most6.192054842985018e-5, satisfied
`abs(error)<=0.02+0.01*abs(native)`, and produced the same prediction under the
upstream class1-on-ties rule. This one sample is not ranked accuracy evidence.

The official comparison used the pinned model's default FP32/SDPA path. A
separate FP64/eager native run checked the independent exact reference at every
attention output, block output, pooler, and both logits; worst error was
3.552713678800501e-14 against the fixed1e-12 parity gate. The approximation
received only original-position client FP64 embeddings and computed every
later state itself. FP64 results are not relabeled as official FP32 results.

Preflight20 tests and the actual4.8194s run each preserved23 source files,
including exact reviewed helpers and selected installed native Python sources.
Six pinned model/tokenizer/dataset assets and all six original fixture files
remained byte-identical. Package versions are recorded; this is not a complete
binary-environment or performance comparison. CPU affinity168–171, four BLAS/
OMP threads, offline assets. Preflight session76137 and actual session94161
both exited0 and were authoritatively reaped; the actual PID is absent.

No keys, ciphertexts, encrypted runtime, server handler, labels, new tokens,
coefficient selection, or validation-domain changes were involved. The original
fixture inventory and permissions are unchanged. Four two-logit reference
vectors (official FP32, native FP64, independent exact FP64, approximate FP64)
and the private target/input binding remain only in a separate0700 workspace,
in one0600 `references.json`. This archive contains hashes and aggregate checks,
not text, token IDs, labels, logits, or activation arrays. Native intermediates
were transient comparison values and were not persisted.

`python verify_archive.py` independently verifies the source archives, counts,
gates, prerequisite report/source binding, and terminal records. Optional
`--private` additionally checks the external private reference/input hashes,
private target binding, permissions, and recomputes the two-logit gates without
printing values. Neither mode runs a model. The final-source copies in the tar
are historical measured bytes; do not edit them to repair later findings.
