# BERT QK routing component: prelaunch checks

The isolated five-file [routing-helper patch](routing_helper.patch) passed
**252 CPU tests in 10.05 seconds** (10.405 seconds for the wrapper), with all
**307 captured files unchanged**. [report.json](report.json),
[sources.tar.gz](sources.tar.gz), [junit.xml](junit.xml) and [tests.log](tests.log)
retain that checkpoint. These tests and the native preflight perform **no
encrypted QK computation**. The separately launched encrypted component must
be judged from its eventual terminal result.

`FactoredScoreLayout` and `FactoredScoreRouting` extract the public bit-routing
network from the [earlier actual routing proof](../bert_factored_routing_secure_cpu_20261103/README.md).
The new helper borrows a keyless arithmetic owner, validates the fixed full-slot
layout, and masks/rotates/adds only. It neither creates keys, invokes client
codecs, refreshes nor closes the borrowed context. Forward and inverse routes
each consume nine levels. A public gain can join the first forward mask;
gain1/8 is the proposed fused QK score normalization. The previous standalone
encrypted result does not by itself validate this new wrapper or fused branch.

Tests cover all 128 maps against independent indexing, hostile padding,
gain1/8, exact level/count behavior, keyless contracts, package extraction,
backend-free P2 staging, helper provenance/tampering, strict paired-input
transport, failed/stale preprocessing, pinned model tensor bytes, nonvacuous
raw references, metadata scale recurrence and cleanup/publication failures.
The proposed paths do not overlap either P1/P3 measured inventory; the
[identity check](calibration_source_check.json) found all 103 and 119 existing
measured files unchanged. No new calibration is claimed.

The [native preflight](native-preflight.json) preserves the unchanged fixed
128-token public pair and full checkpoint Q/K matrices. Query tile0/key tile7,
offset15 produces **192 nonzero raw scores**, range **[-7.7559954, 5.5273658]**.
Independent projection and head-score oracles differ by at most 1.43e-14 and
6.67e-15. All selected keys are publicly masked padding; their optional masked
output is therefore zero. A genuine component must check the raw scores first,
over every physical slot, to avoid a vacuous result.

The [public preflight](public-preflight.json) reuses secure N131072/Q38/P10/dnum4,
actual log2(QP)=1368.7311. It plans 93 direct rotation keys (17.4375 GiB raw),
a 192 MiB relinearization pair, and reserves 8.18 GiB for concurrent FFN work
plus a 4 GiB margin. The intended chain ends at raw level7 and separately
masked level6, with fixed1e-6 client gates and no bootstrap. Native references
and embeddings remain client-only; two uploaded ciphertext digests are bound
jointly to each response. Decryption follows the keyless server's exit.

[Public planning](planning/plan.json) records the larger 128-token schedule;
its 16,728 direct rotations and 110-key inventory are not an executed full
attention result. The [fusion check](planning/fusion.json) establishes only
ideal NumPy equivalence. No softmax, V aggregation, complete attention/model,
MRPC accuracy, TPU performance or practical full-model runtime is established.

Historical evidence remains distinct: the earlier 251-test checkpoint predates
cleanup-only cancellation preservation. The 34-test transport checkpoint
predates pinned tensor digests. A wrong pin lookup, an initially wrong product
scale oracle, and two test-preparation errors are retained in named subfolders;
none launched encryption. No key arrays, ciphertext payloads, private native
inputs or plaintext reference arrays are included.

The source archive preserves bytes under `repository/` and `prototype/`.
For reproduction restore `repository/` as `checkout/`; restore prototype files
beside it, moving `source_origin.json`, `prior_transport_sources.json` and
`run_prelaunch_checks.py` to the lab root as recorded by the absolute source
inventory. The latter script reruns the six selected test files; change only
its output directory if retaining the archived run. No production files were
modified by this candidate.
