# Joined BERT ideal integration with immutable mask guards

The unchanged complete K15 and K128 fixture gates passed: **46 tests**, all twelve encoder layers, pooler and classifier. Every attention output object feeds the following FFN, and every block output feeds the next attention owner. Only initial embeddings enter from the private fixture; native intermediate arrays remain references. All **137 measured source files** matched before, after, execution snapshot and archive readback. The process exited zero, was reaped, and all recorded process IDs were absent.

The candidate changes one runtime module: `encrypted_score_reduction.py` stores its six standard public masks on immutable bytes-backed arrays and checks construction-bound storage/metadata rather than rehashing their contents on every nested guard. Legacy/custom mutable-backed layouts retain the complete hash fallback. Two additional changed files record honest prerequisite/source provenance. All numerical owners, coefficients, operation schedules, fixtures and gate bytes are unchanged from the pinned parent archive. The parent source archive is retained under `provenance/` so this three-file difference can be independently checked. The original baseline job was never modified or canceled.

| Fixed fixture | Maximum stage error vs independent dense polynomial | Logit error vs dense polynomial | Logit error vs pinned native | Observed domains | Identity refresh calls |
|---|---:|---:|---:|---:|---:|
| K15 | 4.0161e-10 | 2.1210e-12 | 4.7895e-5 | 302 passed | 136 |
| K128 | 7.7956e-10 | 4.7003e-12 | 2.9358e-4 | 2346 passed | 1067 |

All 14,417,920 checked physical stage slots satisfy the unchanged `1e-7` ideal gate. Both fixtures meet the original native logit gate `abs(error) <= 0.02 + 0.01*abs(native)` and agree on prediction. Output level is two. These checks cover the fixed public fixture domains, not arbitrary inputs.

This remains **plaintext ideal-primitives integration with identity refresh**. It does not measure CKKS error, genuine repeated bootstrap, encrypted model execution, five file handlers, MRPC accuracy or ranked readiness. The observed 2046.675 seconds and 7,475,276 KiB peak RSS describe this ideal run under concurrent jobs; they are not an isolated algorithm speedup or HE performance claim.

`measured/` preserves the complete source snapshot, raw test/progress logs and aggregate reports. `smoke/` preserves the earlier 44-test selected smoke: tests passed but the collector intentionally returned incomplete/passed-false/exit-one because both complete fixtures were not selected. No source or historical result is relabeled. `verify_archive.py` checks both scopes and the exact parent difference without importing JAX, PyTorch or model assets. README, summary and verifiers are archive-time additions. No embeddings, keys, ciphertext payloads or model tensors are included.
