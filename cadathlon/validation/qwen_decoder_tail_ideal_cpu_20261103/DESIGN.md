# Candidate3 decoder tail: isolated composition successor

This owner receives the actual attended2048 result and original hidden1024 L25 inputs from the completed attention owner. It implements four learned O/gate/up/down maps, post-attention residual and RMSNorm, frozen SiLU, gated MLP, and final residual. No model forward, key ownership, decoding, clear-range inspection, or nonlinear coefficient/count change occurs.

The previous public draft is preserved under `preserved-public-draft-v1`. Its missing original-residual padding mask is repaired explicitly. Before either residual addition, a real weighted sum masks original inputs to valid rows and hidden1024 features: L25→24. This costs one plaintext multiply/rescale per tile, no rotations. Its actual scale supplies the left operand scale in both residual additions. Hostile original padding cannot enter the raw-post refresh or survive the final output. Original wrappers and array owners remain held.

| Stage | Actual level |
|---|---:|
| Original input mask | 25→24 |
| O2048→1024 | 20→19; S1 predecessor22→21 |
| Add masked original as left operand | 19 or21 |
| Explicit raw-post refresh at fixed site bound | →25 |
| Original post RMS rule with inverse-only events | →site-specific2..23 |
| Explicit normalized-post refresh at fixed site bound | →25 |
| Gate and up1024→3072 | 25→24 each |
| Original SiLU rule | 24→6 |
| Multiply retained up | 6→5 |
| Down3072→1024 | 5→4 |
| Add retained unrefreshed raw-post residual as left operand | →4 |

All refreshes are explicit typed construction-bound events. Actual returned scales, including adjacent canonical values, feed existing primitive recurrences. Rational SiLU output is noncanonical; no scale metadata is reassigned. Original norm/SiLU domains and counts remain candidate3 bytes. Layer0 outer bounds are32 and64; all28 fixed bounds come from the continuation report, reaching32768 and8192. These are conditional exact-real plans, not CKKS or refreshed-domain proofs. Unchanged SiLU denominator domains require a refresh-error premise beyond amplitudeB2.

The contract pins the full manifest, public continuation/certificates, layer and sequence count. Public parameters are the authenticated112 O/MLP descriptors from the verified114-name decoder export, plus pinned post RMS gamma. Tensor bodies remain outside source/evidence archives. Every map retains its full dimensions; each layer holds88MiB of FP64 tail weights.

The owner uses actual chained diagonal loops and completed RMS/SiLU owners. Main and separately owned linear facades must share a keyless context and exact synchronous low-memory mode. Helper/accessor/compiler/key/map identities and original/sibling/residual/gate/up/result arrays are guarded around callbacks. Entered failures quarantine borrowed owners; a future Qwen file engine must also quarantine failed-readiness transfers. No general asynchronous reclamation claim follows.

Ideal tests use pinned learned matrices with NumPy primitives and identity refreshes returning adjacent scales. Layer0 tests consume actual preceding attention output objects; later-site tests explicitly use public synthetic attended stand-ins. Independent dense original-Newton/Chebyshev oracles check all physical slots, padding, domains and operation counts. No native/held-out trajectory, model construction or encrypted run is included.
