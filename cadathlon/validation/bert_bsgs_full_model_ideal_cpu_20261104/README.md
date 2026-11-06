# Full12 BERT BSGS ideal evidence

Both fixed K15/K128 trajectories completed. All82 selected tests pass in
3807.88 seconds over150 unchanged sources. Session2171 exited0 and was
authoritatively reaped; root verified the producer and owned child group were
absent, rehashed the original checkpoint/client fixture and checked every
current, execution-snapshot and archived source byte.

| Case | Physical metric slots | Maximum stage error | Pooler error | Logit error | Native logit max error |
|---|---:|---:|---:|---:|---:|
| K15 |1,703,936|4.0161e-10|4.3255e-12|2.1210e-12|4.7895e-5|
| K128 |12,713,984|7.7956e-10|2.0629e-11|4.7003e-12|2.9358e-4|

Every case checks all24 attention/block stages and all pooler/head slots at
the original1e-7 ideal gate, with actual preceding owner output objects fed
into later stages. The frozen original tests also enforce the P2 native logit
predicate and prediction agreement. All302/2346 actual nonlinear-domain
observations and156 independent observations per case pass. Only original
embeddings enter the graph; no reference intermediate replaces an owner output.

K15 uses81,845 rotations,7,008 CT products,517,267 PT products,37,398 rescales
and136 identity refreshes. K128 uses538,189 rotations,59,236 CT products,
2,058,877 PT products,269,701 rescales and1067 identity refreshes. Per query
tile, BSGS reduces both-direction attention rotations8656→4816 and increases
rescales768→2048; CT/PT counts and critical routing depth remain unchanged.
These operation counts and ideal CPU timings establish no HE speedup.

The graph uses explicit NumPy slot/metadata primitives and identity refresh.
This is full12 ideal trajectory evidence for two fixed cases, with no actual
encryption, five-handler completion, ranked benchmark, calibrated accuracy,
peak-memory or latency pass. Genuine routing and prepared-rotation proofs
remain separate component evidence.

The [first full attempt](../bert_bsgs_full_ideal_fixture_failure_cpu_20261103/README.md)
remains failed:78 of80 tests passed, while both full cases stopped on missing
NTT metadata in inherited ideal tracked-reciprocal stubs. This successor fixes
those two test primitive bindings and adds their two regressions. Runtime,
nonlinear coefficients, inputs, gates and operation counts are unchanged.
The final selector retains33 original cases plus2 metadata and47 collector
checks; exactly13 historical collector tests are replaced. The49-test narrow
collector proof and the earlier partial K15 receipt are preserved separately.

Run `python verify.py .` for portable public source/XML/metric verification.
Add `--current /tmp/bert-full-model-bsgs-integration-v2-20261103` to compare
retained sources. No input arrays, model weights or key/ciphertext bodies are
included in this artifact.
