# Full-slot row score reductions (public semantics only)

The59 NumPy checks in `report.json` pass with maximum error9.10e-13. They execute
real full-vector public masks, rotations and additions, with independent dense
oracles. They generate no keys and execute no ciphertext or model operations.
The exact four inspected/script sources are preserved in `sources.tar.gz`.

The layout has16 independent rows at pitch4096. Each row's lower2048 slots store
128 keys ×16 padded heads; only heads0–11 are real. Its upper2048 slots are
inactive. The physical period is65536, with no duplicate aliases.

For a masked score mean, multiply by the public active-slot mask divided by each
row's valid-key count. Add positive-left rotations16,32,64,128,256,512,1024.
Multiply by a public anchor mask selecting row*4096+head for head<12, then add
negative rotations of the same seven amounts. This costs14 rotations/additions,
2 PTCT/rescale levels. Folding1/K into the input mask keeps intermediate values
within the input absolute bound under ideal arithmetic; the unnormalized late
scaling alternative can reachK times that bound.

A squared probability denominator first squares, then uses the same anchor
reduction/broadcast. It costs1 CTCT plus1 PTCT level only when probabilities are
already zero at invalid keys/padding and satisfy the public normalization
contract. The hostile arbitrary-input version first masks and costs3 levels.
The tests cover both paths. Ideal nonnegative normalized probabilities bound
all partial sums of squares by1; this is not a CKKS error/noise bound.

A reciprocal must not see zero at inactive slots. The proposed implementation
can fold a public bias into the existing anchor `weighted_sum` before its
rescale: set inactive head anchors in the lower half and all head anchors in
the upper half to1. Reverse broadcasts then fill every inactive denominator
slot with1, while active row/head denominators are repeated across all128keys.
This counts one plaintext addition within the existing PTCT stage, with no new
multiplicative level. Subsequent inverse domains must explicitly contain1.
Masks of invalid output probabilities remain necessary after nonlinear stages.

The older compact score-only prototype placed16 rows contiguously in32768slots
and duplicated that half. Its original mask differs in2880slots on the15-valid
key fixture, and its duplicate_period assertion rejects these distinct full-slot
rows. Its `reshape(2,16,128,16)` alone still partitions contiguous2048-slot chunks
correctly if given the NEW physical mask; reshaping is not itself the bug. The
NumPy sum/broadcast operations and original replica/mask builders are not a
ready ciphertext reduction implementation and cannot simply be wired in.

The direct reduction key set has14 shifts. Against the current93-key QK
component's public inventory, only+1024 and−1024 are absent. Adding both would
require2 extra key pairs (384MiB raw at that Q38/P10/d4 profile). Composing each
from two same-signed512 rotations instead uses16 execution rotations and no new
keys, with the same multiplicative depth. No runtime benefit is claimed.
