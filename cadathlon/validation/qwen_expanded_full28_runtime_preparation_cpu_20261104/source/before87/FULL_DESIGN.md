# Frozen candidate3 full28 Qwen join: public implementation contract

This step proves the public schedule and interface only. It does not execute a
model/reference trajectory, read parameter or prompt bodies, generate keys, or
run HE. Existing attention and decoder-tail owners stay byte-identical. The
next runtime will consume original L25 hidden1024 embedding tiles and chain
their actual ciphertext objects through all28 blocks. No native or oracle
intermediate may be substituted into that chain.

Each layer receives L25. Attention returns attended2048 atL20 (S1 direct-V path
atL22). The completed tail masks the original residual25→24, executes the
original four learned matrices and nonlinear rules, and returns L4. Before
layers1..27, each actual prior tail tile is explicitly refreshed4→25 under the
fixed next input-site raw amplitude bound. Those are conditional public bounds;
neither domain preservation nor refresh error is thereby proved. All113 RMS
rules and all28 SiLU/softmax rules retain their selected methods, domains,
coefficients and counts. Nine Q/K reserve events remain unchanged.

Final normalization in the frozen reference applies only to the last token.
The parent must not normalize other valid rows under that site's domain. After
layer27, select the last valid local row and hidden1024 features using one real
mask4→3, then rotate that row to physicalrow0 without a level or scale change.
The public original last-position identity is retained. Explicit B2048 refresh
3→25 precedes the exact degree64 balanced final RMS25→7. The learned final gamma
is pinned. No postnorm refresh is needed. Stream all38 immutable vocabulary
matrix slices at7→6:37 slices4096 wide and the final384, totaling151936 logits.
Chained linear's genuine source masks keep only row0/hidden1024, and its last
slice's inactive output features remain zero. Every returned actual scale is
validated using primitive replay; no metadata is reassigned. Nine public
adjacent-scale examples illustrate this replay, not exhaustive reachability.

All15 possible last-row rotation keys already occur in the independently
verified actual234-matrix166-key inventory, including the unreduced262143
bootstrap conjugation index. No new key is needed for this selection. Raw
rotation-key storage is147.84375GiB, distinct from resident memory. The38 final
L6 ciphertexts total532MiB before transport headers. The plan excludes control,
compiler, temporary and host/device duplicate allocations. It does not establish
the128GiB host limit,600s inference budget,60min key budget or TPU feasibility.
S32 retains12931 literal refresh events. These costs are explicit unresolved
practical requirements, not a performance claim.

Proposed runtime API: a construction-bound FullQwenContract plus one borrowed
synchronous/keyless arithmetic/linear/refresher family and authenticated public
parameter provider. `evaluate(original_tiles)` returns38 ordered logit blocks,
the original last-position identity and exact public block descriptors. Progress
checkpoints expose each actual attention/block result and final norm/head
result without changing arithmetic. Full-layer activation checkpoints are
optional and never used as inputs. The default serial path retains original
input arrays, current source/result arrays and current public matrix owners,
releasing only completed synchronous temporaries. Entered failures require
dedicated-process and outer-loader quarantine. File handlers remain separate.

Reference hookup starts from a separate copy of original embeddings and the
authenticated public dictionary. The byte-identical calibrated
`frozen_arithmetic.py` returns28 block outputs,421 operator labels and all151936
logits. A separate dense NumPy Chebval/original-Newton loop will provide the
independent packed-layout oracle; it must never consume owner or native later
activations. Wholegraph fixtures are predeclared before trajectory execution:
checkpoint embedding row0 atS1, then the first authenticated original calibration
S8 prefix. No fresh/held-out outcomes or token bodies are read for this plan.
Any declared-domain failure must be preserved without clipping, retuning or
choosing a replacement because it passes. Wholegraph execution awaits review.

The current geometry is batch1/S1..32 with public positions and masks. It covers
the stated S8/S16/S32 prefill lengths and repeated-prefill S16→19 generation
shape; full file privacy, measured numeric/argmax gates, all-suite coverage and
actual resources remain obligations. Exact-real score bounds still do not prove
868 centered-domain implications. This schedule does not remove that gap.
