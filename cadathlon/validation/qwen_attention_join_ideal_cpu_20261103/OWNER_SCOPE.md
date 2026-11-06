# Candidate3 Q/K/V attention composition

The runtime borrows ordinary synchronous arithmetic, a separate or shared
ChainedStreamingLinear facade, and a full-period bounded refresher. It consumes
one or two ordered level-25 ciphertext row tiles. Every row has 4096 physical
slots and 1024 active input features; public original positions are preserved.
It returns 2048 attended features per active row, at level 20 for sequences with
more than one token and level 22 for the original single-token direct-V path.
Output projection, attention residual, post norm, SiLU/MLP, final norm and the
vocabulary projection are outside this owner.

The fixed candidate remains
`7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333`.
All selected RMS polynomials/Newton iterations and softmax circuits are executed
by copied, unchanged predecessor owners. The new code adds typed outer events,
actual chained Q/K/V linears, construction/accessor guards and serial wiring.
Public inputs do not authorize any codec, key generation, clear range inspection
or native-model substitution.

Each tile performs input RMS, a normalized-output refresh, three actual linears,
raw Q/K refreshes, the original Q/K RMS circuits, a reserve refresh only at the
nine prescribed low-level sites, RoPE, then post-RoPE refreshes. K and V expansion
finish at levels 24 and 23. All K/V tiles are retained before query evaluation.
QK scores reach level 21, unchanged causal softmax reaches 24, and actual masked
PV reaches 20. Query 0 takes the existing real V0 route instead of a fabricated
probability value. The caller supplies already refreshed level-25 layer inputs;
the preceding raw residual/input refresh is not hidden inside this component.

All refresh returns are consumed at their actual returned scale. No scale field
is reassigned. Child primitive recurrences and the new linear recurrence validate
real weighted/multiply ordering; the identity-refresh tests alternate adjacent
binary64 scales around the pinned canonical scale.

Static guards cover both arithmetic facades, helper contexts, original accessors,
compiler configuration and actual sparse-matrix key/map identities. Execution
caches may evolve during primitives; their reachable identities are protected
across external callbacks. The parent retains original inputs and completed QKV
and output arrays. RMS basis lifetimes remain bounded by the current norm phase;
softmax/PV use their existing complete bounded retention plus the current parent
callback result. Any entered-body failure retains borrowed owners and reachable
arrays and rejects further joins. A future dedicated Qwen file engine must also
quarantine underlying key mappings on failure; this module is not that engine
and does not certify reclamation after failed readiness.

The fixtures use actual authenticated public Q/K/V and gamma tensors, with fixed
synthetic input rows. No model is constructed or run. Checkpoint-derived tensor
bodies remain outside public evidence archives. The numerical oracle uses dense
matrix products, independent Chebyshev evaluation, and the original static
Newton recurrences. It strictly reports all encountered nonlinear domains.
Cases cover S1/8/16/19/32, original noncontiguous positions, partial tiles, hostile
padding, and separate Q- and K-reserve sites. Primitive NumPy doubles and identity
refresh are explicit; these are not encrypted accuracy or native trajectory
results. Fixed all-slot ideal tolerance is 1e-8, unchanged after failures.

The public envelopes remain conditional exact-real statements. Post-RoPE bounds
range from B64 to B2048; earlier outer raw-normalization bounds reach B32768 in
the full continuation plan. None of the 868 broad score envelopes establishes
the frozen centered-exp domain. Inserted refresh errors, floating arithmetic and
CKKS errors require separate analysis. In particular, the four zero-margin exp
endpoints remain unresolved for perturbations. The S32 full-28-layer plan still
contains 11,656 softmax refreshes and is not an encrypted-readiness claim.
