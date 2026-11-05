# PR #5 review follow-up

Review: https://github.com/EfficientPPML/CROSS/pull/5#pullrequestreview-5762628101

| Review concern | Change / scope |
| --- | --- |
| Public and Mapping multiply/square after SHIP | Shared scale-aware schedule; tensor first at low scale, one post-rescale; reject insufficient modulus capacity. Public, split, whole-Mapping and operations-Mapping regressions. |
| P too small | Planner checks the P/Q product margin, including when Q is extended. |
| Product-tree scale drift | Planner replays every dropped prime and checks both tree and encoder scale. |
| Rescale metadata / refresh cycles | Public rescale updates noise scale degree; public bootstrap/multiply/scale-align/bootstrap regression. |
| Available depth | Plan exposes terminal and between-bootstrap level budgets; documents zero between-bootstrap multiplications for default Q8. |
| Low-scale rotations | Opt-in centered HERot ModDown, automatically selected by low-scale public rotation and Mapping; low-scale BSGS uses it at baby and giant steps. Existing raw OpenFHE convention preserved. N16384 eight-index regression. |
| Sine approximation / decoder check | README and API reference specify gamma=8, amplitude <=0.125 and the sine map; known plaintext comparison is required for accuracy. |
| Setup allocation peak | CM and mux key stacks assembled on host, then transferred once. |
| Missing integration coverage | Post-SHIP public/split arithmetic, Mapping mul/square in both modes, refresh cycle, unsafe parameter rejection, full-ring rotation. |
| Package exports | Exact public export test includes SHIPBootstrap and SHIPConfig. |
| Composite degree 2 / Packing | Unsupported gate retained; default Packing and secure ring configuration incompatibility documented explicitly. |
| Sequential auxiliary-prime drop | HERot approximate drop of P[1:] followed by exact centered drop of P[0]. |
| Selector BAT duplication | Reuses HERot decomposition and key-switch cores; raw uint32 keys; one lift/drop per mux step. Per-level relinearization and selector conjugation keys retained, with rationale documented. No context-key reuse speedup claimed. |
| Eager modular add | JIT conditional subtraction replaces eager uint64 remainder; benchmark separately records synchronized wall time. |
| Profiler incompatibility | Removed unsupported device_tracer_level; validate backend/options before setup; save correctness and wall results before profiling. |
| Security claims | Explicit QP table overrun and unqualified sparse-key distribution; no attack or security claim. |
| Barrett regression / extra reduction | Added reachable troublesome CROSS primes; tensor cross term uses reduce_scaled_sum; square uses canonical conditional subtraction. |
| Repeated/failed setup | Repeated random-source override raises; failed SHIP setup allows reconfiguration. |
| Private secret retention | Clear private transform context, parameter dict and cache secret references after setup. |
| Randomness / noise | Uniform ternary sampler; direct symmetric setup encryption; context sigma propagated; seeded source covers private rotation keys. |
| TPU rounding | Phase encoding uses floor(x+0.5). |
| LOW_MEM / cache lifetime | No unsupported eviction claim; noncollapsed hoisted wrapper cache uses stable level/layout/moduli keys instead of object identity. |
| Import subprocess TPU contention | Import-isolation children force CPU. |
| Memory reporting | Mapping includes SHIP key and transform array storage; allocator/executable/transient limits documented. |
| Missing legacy p during decode | Derive decode working precision from resolved scale when p is absent; regression added. |
| Persistent compilation-cache failure | Cause unconfirmed; comparison runs disable persistent caching and preserve runtime/provenance evidence. |

Historical TPU timings are not measurements of these changes. Validation and
fresh matched measurements must be recorded separately.
