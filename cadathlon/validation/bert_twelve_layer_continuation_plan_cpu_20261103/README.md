This public-only design binds the unchanged BERT v2 manifest `4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9`. It checks all 1,536 layer/key-count schedules (12 layers × K=1..128), the original selected polynomial evaluation methods, actual reusable rotation-key inventories for all 74 pinned dense matrices, and actual Q/P coverage. It does not execute a model, construct a CKKS context, generate keys, encrypt, or refresh.

The cross-layer bottleneck is explicit: previous MLP outputs have levels 3..13, so they cannot enter the fresh level-49 first-attention API. Refresh each real row tile with a declared B128 bound to level25, giving Q/K/V24 and scores21. Split initial softmax after its score mean and normalization: score21→mean19→normalized t18; explicitly refresh t with B2 to25; evaluate the unchanged exponential/static-six-update reciprocal tail in19 levels to probability6. Five to seven K-specific tracked normalization rounds refresh when their original n+5 depth plus two reserved levels would not fit. A final probability refresh and mandatory real mask give24, then PV21→dense20→attention norm. No source is relabeled as a fresh ciphertext.

The FFN tail refreshes attention-norm outputs with B128 to25, uses dense24, evaluates the selected balanced GELU, refreshes its output with the exact per-layer public bound, and computes dense/residual24 followed by the selected MLP norm. Balanced degree128 norms consume21 levels and leave3, enough for the next explicit refresh. A separate pooler plan keeps the frozen degree12 base and six rational doublings using the original static-five-update reciprocal: three B2 refreshes lead to classifier output2. CLS selection and its mask still need implementation.

| Layer | Attention norm method/degree/depth | MLP norm method/degree/depth | GELU degree/depth | GELU refresh bound | MLP output level |
|---:|---|---|---|---:|---:|
| 0 | power / 12 / 11 | power / 16 / 11 | 64 / 14 | 16 | 13 |
| 1 | power / 16 / 11 | balanced_chebyshev / 48 / 19 | 96 / 16 | 64 | 5 |
| 2 | power / 12 / 11 | balanced_chebyshev / 48 / 19 | 96 / 16 | 32 | 5 |
| 3 | power / 16 / 11 | power / 32 / 12 | 96 / 16 | 32 | 12 |
| 4 | power / 16 / 11 | power / 32 / 12 | 96 / 16 | 32 | 12 |
| 5 | power / 16 / 11 | balanced_chebyshev / 48 / 19 | 96 / 16 | 32 | 5 |
| 6 | power / 24 / 12 | power / 32 / 12 | 64 / 14 | 16 | 12 |
| 7 | power / 24 / 12 | balanced_chebyshev / 48 / 19 | 128 / 16 | 64 | 5 |
| 8 | power / 24 / 12 | balanced_chebyshev / 96 / 21 | 128 / 16 | 64 | 3 |
| 9 | power / 24 / 12 | balanced_chebyshev / 128 / 21 | 192 / 18 | 64 | 3 |
| 10 | power / 24 / 12 | balanced_chebyshev / 128 / 21 | 96 / 16 | 64 | 3 |
| 11 | power / 32 / 12 | power / 24 / 12 | 48 / 14 | 16 | 12 |

The balanced methods remain balanced. Converting degrees48/96/128 inverse-square-root polynomials to monomials would produce huge coefficient cancellation. Even the explicitly selected power32 sites have coefficient L1 up to about1.1e6: passing the library conditioning guard is not an encrypted precision certificate. Actual noncanonical scales must follow the reviewed binary64 operation order; this level-only plan does not replace per-primitive metadata plans.

The common proposed profile is N131072/CD2/depth49/p60/dnum8, Q100/P14, with actual P coverage and log2(QP)=3402.080255733439 below the 128-bit classical ceiling3523. All layers reuse one ring and key set. Actual pinned matrix plans use six linear keys; the graph uses92, full-slot bootstrap106, union130. Raw reusable rotation pairs alone require124,319,170,560 bytes (115.78125GiB), at912MiB each. This excludes relinearization/public keys, ciphertexts, formatted keys, control tables, executables, host overhead and peak device/host memory. Streaming and bounded cache lifetimes remain necessary. The counted design needs1064 encoder refreshes for K128 (1067 including pooler), versus98 encoder refreshes for K1; these are schedule counts, not latency estimates or an optimized lower bound.

The GELU bounds [16,64,32,32,32,32,16,64,64,64,64,16] are the exact-real conditional certificates in [the separate certificate archive](../bert_all_gelu_refresh_bounds_cpu_20261103/README.md), including rounded domain-map endpoint extension. The native learned norm maximum below128 does not by itself bound the approximate circuit or accumulated CKKS noise. B2 t/probability/tanh bounds likewise require declared-domain and approximation-error obligations. Refresh noise can move t outside a fitted interval at its endpoints. There is no clipping, secret-key evaluator, native nonlinear substitution, or client refresh in this design.

No full-slot p60 refresh precision is transferred from sparse period128 or p56 evidence. Actual p60 normalization/restoration and all bounds2/16/32/64/128 need their own encrypted checks and repeated-composition gates. A level-capacity pass is not a numerical readiness claim. The original train3668/held-out408 arithmetic results, existing component passes, and this plan remain separate evidence.

The smallest next implementation is an immutable, layer-indexed initial-softmax prefix/tail owner. It should return t18, accept the caller's genuinely refreshed t25 without assigning metadata, preserve exact layer squaring count/coefficients/static-six inverse, mask refreshed padding, and fingerprint retained exponential/denominator values through callbacks. This solves the concrete inter-layer depth gap. Subsequent owners must generalize all24 normalization sites with their selected strategies; consume T=ceil(K/16) real low-level output tiles and create empty padding tiles by genuine subtraction after refresh; preserve original positions; and compose all12 layers with serial borrowed contexts and bounded live ciphertexts. Generic all-layer GELU now exists as a separately ideal-tested owner, but common-ring HE precision is still pending.

Complete scope also requires actual pooler/rational-tanh/classifier, an embedding-only client boundary, strict model/request/response file handlers, complete encrypted ranked100/200/408 execution and scoring calibration. Neither the first block nor a repeated level schedule completes those requirements.

`report.json` and `sources.tar.gz` preserve117 exact planning sources/public fixtures, archive SHA256 `042d640d80186c31bbd958e5d40afa66c03141eab31415379a38d355d7309b7e`. `preserved-plan-v1` keeps the earlier successful encoder-only draft unchanged. No model tensors or row arrays are included. Run `python verify.py` for an independent offline recurrence and archive check; it needs no model assets or numerical backend.

A later separate [exact approximate-normalization certificate](../bert_approximate_norm_output_bounds_public_20261103/README.md) supplies the conditional ideal-circuit B128 bound for all24 selected norm rules (maximum guarded bound85.8342). The original measured plan/report above is unchanged. Actual encoding/CKKS/refresh error, shared-moment preservation and domain propagation remain unresolved by that certificate.

A later separate [pooler rational-tanh certificate](../bert_pooler_tanh_bound_public_20261103/README.md) establishes the conditional ideal B2 magnitude bound for the unchanged base and six static-five-update doublings. It also separates the narrower denominator-preservation error tolerance from generic B2 headroom. The original public plan and numerical records remain unchanged.
