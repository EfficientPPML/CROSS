# Isolated causal softmax cohorts

This public/synthetic successor adds `encrypted_qwen_causal_softmax_batched.py` to the exact 191-file expanded-v3 checkpoint (manifest d0846e053f3a369a18768ee69328568be46355f6952a06ef89587902a235f79a). Every inherited file remains unchanged. The full-model join continues to use the original literal-K owner. No model, token, parameter, key or ciphertext body is needed by this preparation; no execution grant or promotion is implied.

The existing 65,536 slots contain sixteen 4,096-slot query rows. Each row's first 512 slots are sixteen heads of 32 key lanes. Queries in the same tile can share an arithmetic schedule only when their original Newton count and original inverse-refresh bound are both equal. The cohort descriptor retains each query's original position, K, binary64 `float(1/K)`, rounded first weight/bias and denominator domain. Positions are never repacked. The five positive-power reduction rotations and five negative-power broadcasts are unchanged; no new rotation or refresh keys are introduced beyond the existing 166-key plan.

At S=32 the exact groups are:

| Tile | K | Newton iterations | Existing inverse bound |
| --- | --- | --- | --- |
| 0 | 2 | 7 | 8 |
| 0 | 3 | 8 | 8 |
| 0 | 4 | 8 | 16 |
| 0 | 5–7 | 9 | 16 |
| 0 | 8–9 | 9 | 32 |
| 0 | 10–14 | 10 | 32 |
| 0 | 15–16 | 10 | 64 |
| 1 | 17–18 | 10 | 64 |
| 1 | 19–28 | 11 | 64 |
| 1 | 29–32 | 11 | 128 |

Slotwise plaintext affine arrays preserve each active query's rounded coefficients. The cohort's first original affine rule fills inactive lanes. Every inactive 32-lane block receives denominator one at its anchor then broadcasts it; one lies inside every original denominator domain. Probability masks keep inactive/future lanes zero at the reduction and publication boundaries. All eight plaintext arrays are immutable bytes-backed float64 arrays, totaling 4 MiB logical data per active cohort, excluding array objects, temporaries, ciphertexts, compiler/device buffers and keys. Refresh error on inactive lanes is masked before the next relevant reduction. Conditional bounds remain caller obligations; this preparation does not certify encrypted trajectories.

K1 keeps the existing direct real-V bypass, with no fabricated probability ciphertext or refresh. Other outputs retain level 24, original query/key positions and the existing `QwenProbabilityTile` interface used by PV. The implementation checks every actual primitive scale and replays returned refresh scales; ideal refresh deliberately alternates one-ULP scale changes. Sharing refresh calls can change encrypted rounding/error, so numerical equivalence in ideal slots is not a bit-for-bit HE claim.

For a cohort with squaring count s and Newton count n, primitive counts are: refreshes 3+2s, rotations 20+10s, ciphertext products 18+2sn, weighted sums 14+s(n+2), ordinary additions 21+10s, plus one ordinary addition per extra cohort. Across 28 layers at S=32 this gives 3,760 softmax refreshes instead of 11,656 literal calls, 20,200 rotations and 32,196 ciphertext products. These are source/synthetic operation counts, not measured acceleration, latency or physical-memory results.

The owner borrows context/key/refresh resources. Original score, live intermediate arrays, completed aggregate, current immutable plaintext arrays and returned payloads remain strongly retained across callback boundaries. An entered failure preserves its primary exception and quarantines the owner and retained originals until dedicated-process exit; it does not claim cleanup or close a borrowed context. Tests simulate process isolation when clearing the synthetic quarantine between cases. Construction binds both refresh-wrapper accessors before invoking the inherited guard; this addresses the practical callback wrapper substitution separately reproduced in the literal predecessor. Binding the new vector factory is additional local hardening, not a general arbitrary-Python tamper guarantee.

Synthetic verification compares every slot against an independent dense fixed-circuit oracle for S1–32 and both squaring families, compares the literal owner on both full tiles, validates all 28 public contracts and exact per-K coefficients, checks all dummy lanes and mixed rules, perturbs inactive refresh lanes, joins unchanged PV including real K1 V, and exercises callback replacement/failure ownership. Preserved development failures include the wrapper/factory admission cases and the wrapper-check-before-invocation ordering cases. No whole learned model or genuine encrypted run is part of this preparation.
