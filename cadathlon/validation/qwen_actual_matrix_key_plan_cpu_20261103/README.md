# Public actual-matrix Qwen rotation plan

The completed public analysis covers all 234 matrix maps: 84 Q/K/V projections,
112 output/MLP projections, and 38 vocabulary blocks (37 × 4096 rows plus 384).
The tied embedding/head alias adds no second server matrix map. Every measured
matrix has all of its geometric diagonals present. Actual matrix keys are
`[1,64,128,60544,61504,62528,63552,64192]`; combining them with the 162 inherited
routing, normalization and bootstrap keys gives **166 keys**, including the
unreduced conjugation index 262143.

At the declared N131072/depth49/p60/d8 Q100/P14 profile, one raw rotation-key
pair occupies 956,301,312 bytes. The 166-key sum is 158,746,017,792 bytes
(147.84375 GiB). This is raw key-array accounting, excluding relinearization,
public/secret keys, model/ciphertext arrays, controls, compiler state, transfers
and duplicate storage. It is not a memory-peak or capacity guarantee. Recorded
chained linear calls total 22,008 per row tile across the 196 layer maps, plus
6,226 for the final-row vocabulary maps; these counts exclude nonlinear and
other graph work and are not benchmark timing.

The exact 22-source closure, 21-test preflight and actual child reports are
preserved unchanged. Parent agent `/root` authoritatively observed session
39684 exit 0 and reaped it at 18:04 UTC; the reviewer did not poll that handle.
The archived child record independently records direct wait 0 and an empty
owned process group, and this archive records a fresh PID/group absence check.
Collector duration was 19.30467984000279 seconds; no calibrated timing is claimed.

Run `python verify.py` for offline checks of all artifacts, both source archives,
all 234 map identities/shapes and chained gap schedules, exact base/combined
key union and byte arithmetic. `--current /tmp/qwen-actual-matrix-key-plan-20261103`
also checks current and both execution copies. The optional `--public-bodies`
reads only the 197 separately exported, pinned public matrix files, rechecks
file/payload/block hashes, and independently enumerates every actual diagonal
support. Public matrix bodies are not included here.

This evidence establishes the key plan for the specified F4096/S1..32 graph.
It does not establish a generated key store, actual-context security preflight,
complete model implementation, numerical HE accuracy, benchmark eligibility or
runtime readiness. Nonlinear-domain and refresh-error obligations remain.
No model forward, key generation, codec, backend or encrypted operation ran.
The source archives contain public code/metadata only, and no weight, key,
ciphertext, token, label or hidden-state body is copied into this archive.
