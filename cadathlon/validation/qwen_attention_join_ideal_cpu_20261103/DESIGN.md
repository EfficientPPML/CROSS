# Isolated candidate3 Q/K/V attention join

This successor preserves all inherited owner bytes and candidate7f6478a6.
`QwenAttentionJoin` will take ordered actualL25 row tiles, one layer-bound typed
contract, same-context ordinary synchronous arithmetic/ChainedStreamingLinear,
one borrowed full-slot refresher, actual gamma vectors and authenticated Q/K/V
matrices. It returns attended2048-wide ciphertext tiles, atL20 for nontrivial
causal rows andL22 forS1. Output projection/residual/MLP are later owners.

Per tile: unchanged input RMS, explicit normalized-output refresh25, actual
1024→2048/1024/1024 linears, explicit raw Q/K refresh25, original head128 RMS,
reserve refresh only for the nine norms endingL2, actual half-split RoPE, then
explicit post-RoPE refresh25. K GQA reaches24; V GQA reaches23. Existing head
scores reach21; actual fixed causal softmax reaches24; actual PV retains the
original query0 V0 bypass. All K/V tiles are completed before serial query tiles.
The runtime never inspects ranges or substitutes native/clear intermediate data.

The new outer refresh contract binds stage, layer, entry/output level and exact
public bound from the separately frozen public join plan. It is separate from
the frozen inverse-only binding. Public B64..2048 post-RoPE, up toB32768 raw
normalization ranges, and all868 unproved centered-domain implications remain
conditional. Identity-refresh ideal tests do not establish real refresh errors.

Construction binds the full typed contract, arithmetic/linear/refresh objects,
all original accessors, context/cache/layout/key identity and exactTrue low-memory
flags. Callback checks retain original array references separately from mutable
wrappers, including every input tile, pending K/V tile, RMS basis/variance and
current result. Subphase retention retires only after its successful synchronous
return. Any entered-body exception quarantines borrowed owners and reachable
arrays until dedicated-process exit; it cannot prove readiness of unreturned
primitive outputs. No context/provider/helper is closed by the join owner.

Early guard tests use explicit synthetic matrix descriptors. Root has separately
authorized authenticated public Q/K/V extraction from the pinned c1899 checkpoint;
final ideal fixtures should use these real tensors when feasible. The export
reads only84 named BF16 tensors after full checkpoint/config hashes and records
exact raw/promoted digests. Bodies remain local, outside public evidence archives.
No model construction/forward, key generation or actual encrypted run is granted.
