# Frozen candidate3 causal-softmax owner proposal

This is a public implementation design, not an implemented owner or an encrypted result. It uses candidate `7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333` unchanged. No model, gamma, private inputs, calibration records or validation outcomes are read. `input_sources.json` lists the exact ten copied public dependencies; `plan_public.py` authenticates them before deriving 896 layer/K schedules.

## API and physical geometry

Proposed `QwenCausalSoftmax(arithmetic, contract, config_bytes, *, positions, refresher, progress=None)` borrows an ordinary keyless `CompiledArithmetic` and the common full-period bounded refresher. `QwenCausalSoftmaxContract(manifest_bytes, layer)` reconstructs all fields from the pinned manifest. `evaluate_tile(scores, *, query_tile, capture_checkpoints=False)` takes an actual L21 score ciphertext and returns a `QwenProbabilityTile` with `output`, `query_positions`, `key_positions`, `direct_value_queries`, `counts`, and actual metadata. For the sole-query S1 case `output` is `None` and the result explicitly requests the direct encrypted-V bypass.

The frozen head/PV geometry is 65,536 slots, 16 rows of 4,096, 16 query heads, and score lane `4096*row + 32*head + key`. There are at most 32 sorted, unique original positions. Query ordinal `j` admits keys `0..j`, so its public K is `j+1`, including when original positions are noncontiguous. A second tile starts at K17. Future, inactive-row and upper-feature padding is masked by actual public multiplications; no ciphertext value controls a branch.

Each nontrivial query row is evaluated as a separate public branch, with all 16 heads in SIMD. This preserves its exact K-specific reciprocal rule and makes retained denominator ownership explicit. It does not run the largest iteration count on other rows. Branches are processed in increasing query order and added only after their final real masks at L24. Only the current branch and partial packed aggregate remain live by default. Checkpoint capture is an explicit memory-expensive option.

## Literal circuit and refresh locations

| Phase | Actual levels | Frozen operation |
|---|---|---|
| Source branch mask | 21→20 | Mask one query row, 16 heads, first K keys |
| Mean and centering | 20→19 | Five positive power rotations, anchor weight alpha=fl(1/K), five negative broadcasts; subtract from retained masked source |
| Normalize exp argument | 19→18 | Mask valid lanes and multiply by `2**(1-s)`; inactive t=0 |
| Explicit normalized-t refresh | 18→25 | B2, full 65,536 independent slots, 4×4/meta10 |
| Exp8 | 25→21 | The eight stored nonconstant power coefficients and constant; power basis seven CT products; **no second affine input map** |
| Initial mean-exp | 21→19 | Mask/weight with the same alpha, reduce, anchor with dummy1 outside live heads, broadcast |
| Original initial reciprocal | 19→18→15→12→9→6→3 | Rounded first affine and five original three-level Newton updates |
| Explicit completed-inverse refresh | 3→25 | B4 **after all six updates**, never first-affine-only |
| Initial probability | min(21,25)→20→19 | Retained exp times inverse, then the same alpha and branch mask |
| Each of s original rounds | variable→25→24→23→22 | B2 probability refresh, branch mask, square retained numerator, sum/anchor/dummy1/broadcast |
| Original K reciprocal | 22→21→18→15→12→9→6 | Original rounded first affine and updates2..6 |
| Explicit inverse continuation | 6→25, then19/16/13/10/7 | Refresh before update7 at the frozen K bound B8..B128; complete exactly n=7..11 updates |
| Round probability | 18/15/12/9/6 | Retained L23 square times inverse |
| Final packed-branch continuation | variable→25→24 | B2 then mandatory real branch mask; actual post-refresh padding is not silently assumed zero |

Layer squaring counts remain 6 for layers0/1/2/6/11/16 and 5 for the other22. The stored reciprocal counts are K2:n7; K3–4:n8; K5–9:n9; K10–18:n10; K19–32:n11. K1's unused reciprocal rule is preserved in the manifest but is not executed.

The square-reduction stage needs one weighted anchor because its input was already branch-masked before squaring. The exp mean needs two weighted stages because polynomial evaluation produces nonzero inactive values. Dummy1 is inserted at every inactive 32-lane anchor and broadcast across all slots, keeping the ideal inactive reciprocal inputs inside every declared domain. The masks are applied at their actual source coordinates, and each reduction/broadcast uses exactly ±1,±2,±4,±8,±16.

The proposed default result has probabilities at L24. With expanded V at L24, the existing PV geometry reaches L21. This does not solve the preceding RMS/QKV/RoPE capacity and refresh bounds; those remain caller obligations from the full-model public plan.

## Preserve the original K1 direct-V branch

The frozen clear circuit bypasses softmax for query0 and takes actual expanded V0. The new softmax result must not fabricate a probability result for that skipped branch. A separate, explicitly versioned PV successor will accept the result and route actual V0:

* For S>1/tile0, exclude row0 from every probability product and fuse an actual V0 row/feature mask as a second term of the existing final attended-output weighted sum. That term is level-aligned by the real facade and encoded with its actual scale. It adds one plaintext term but no critical level; output remains L21 for P24/V24.
* For S1, take actual V0 through a real output mask; no QK/exp/inverse/refresh is needed by this attention branch. The direct result may retain higher capacity, which the later caller must handle explicitly.
* Later query tiles have no bypass row. No plaintext/native V substitution, synthetic unit-probability ciphertext, or metadata-only lowering is used.

The frozen 88-test PV predecessor remains unchanged. This successor is a required next implementation piece, not an existing capability.

## Metadata, ownership and contracts

The profile stays N131072/CD2/depth49/p60/d8, exact ordered Q100/P14 and delta=1.1467305003699405e18. All weighted stages predict the production order `s * ((delta * integer_drop_pair) / s) / integer_drop_pair`; ordered first-term metadata survives additions, including zero-weight terms. Products use the actual minimum level and corresponding ordered pair. Add/subtract expects minimum input level and the left actual scale, with the ordinary facade's compatible-scale tolerance. No scale is assigned or relabeled.

The public planner replays canonical and alternating delta±1ULP refresh examples. Runtime must instead replan downstream metadata using every actual returned refresh scale, checking all primitive results. The example scale vectors are not a certificate of what a noisy refresh will return.

Typed complete contracts and original arithmetic, reduction, context, layout, key/map and helper/accessor identities are bound at construction, checked before dependent planning/work and after callbacks. Reuse the reviewed RMS full-period refresh binding in a new softmax-specific copy with the exact event inventory and bounds; do not edit the frozen inverse-only owner. It must still independently derive the complete bootstrap plan and bind all five compiled helpers and engine methods.

Retain both wrappers and their original payload references for score input, masked source, mapped t, every exp power, retained exp/denominator, square, inverse, pending refresh source, partial packed output and captured checkpoints. Before/after callback fingerprints cover all retained values, including all-zero and inactive paths; strip inherited public semantic annotations from new outputs. No reentry, context ownership, implicit refresh, encoding/decoding, range inspection or automatic fallback.

## Cost and remaining proof obligations

Per nontrivial query branch with s rounds and n reciprocal updates, excluding bootstrap internals: `18+2*s*n` CT products, `14+s*(n+2)` weighted sums, `20+10*s` rotations, `21+10*s` ordinary adds/subtracts, and `3+2*s` refreshes. Packed aggregation adds one ordinary addition for each branch after the first in a tile. Weighted-sum internal plaintext terms/additions are not included in the ordinary-add count.

For S32, the serial design uses **11,656 full-slot refresh calls across 28 layers**. It prioritizes literal composition and low branch retention; no practical latency claim is made. The head/RoPE/QK/PV + softmax + bootstrap union is162 rotation keys, 144.28125GiB raw pairs at912MiB each. It excludes full-model projection/norm keys, relinearization/public keys and controls. The eight retained exp powers alone occupy382MiB of raw ciphertext payload. These are raw lower-level sums, not peak memory estimates.

The exact-real certificate proves exp bounds, the same-alpha mass cancellation, shared-square denominator induction, and the stated B2/B4/K-inverse envelopes **conditional on the original declared domains and ideal preserved values**. Refresh, floating evaluation, encoding/CKKS and correlated numerator/denominator perturbations remain outside it. Layers3/5/12 hit normalized t=+1 and layer15 hits−1 with zero exp-domain margin; no claim that encrypted inputs remain inside[-1,1] is made. Any future derivative extension analysis must retain original accepted clear domains and coefficients.

Next implementation tests must compare actual source-coordinate primitives against independent frozen-power/original-Newton loops on all physical slots, every K1..32, both squaring counts, S1/8/16/19/32, distinct heads/rows, shifted original positions, hostile future/padding and alternate actual scales. K1 must inspect the actual V path. Mutation tests cover original and retained payloads through first/final and refresh callbacks. All ideal evidence remains separate from an eventual genuine full-slot proof.
