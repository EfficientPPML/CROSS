# Candidate3 Qwen continuation and join plan

This new public plan preserves candidate `7f6478a6…`, all113 normalization
methods, coefficients and iteration counts, and the frozen causal softmax and
SiLU circuits. It reads19 copied public sources/reports, with no model, raw
gamma tensor, token, activation, label, key or ciphertext reads. It does not
implement or execute the joined model. Earlier component evidence is unchanged.

The prior suggested refresh-before-RoPE placement is insufficient: Q25→RoPE24,
K25→RoPE24→GQA23, then key broadcast22/product21/score mask20. The existing
softmax owner requires scores at21. The corrected conditional schedule refreshes
Q and K **after** RoPE to25; K expansion24/broadcast23/product22/score21 then fit.
Nine existing norm outputs areL2 and cannot enter the current RoPE owner, which
reserves two levels. Those sites need an additional norm-output refresh first:
Q layers5/7/12/15/22/23/24/25 and K layer9. Other Q/K norm outputs can enter RoPE
directly. No normalization update is removed or shortened.

V does not need an extra range-dependent refresh: normalized-input refresh25,
V projection24, GQA23, probability24/PV20, O19, residual19. ForS1 the original
actual-V0 bypass yields attended22/O21. Raw post-attention values are explicitly
refreshed before the fixedL25 post-norm entry. Gate/up input24, unchanged SiLU6,
gated product5/down4/residual4 then require an explicit raw-input refresh before
the next layer'sL25 norm. Final raw refresh25→RMS7→all38 vocabulary blocksL6.
The vocabulary has151936 logits, not a selected top-logit subset.

All28 layers are replayed using the actual ordered-prime binary64 multiplication
and weighted-sum scale expressions. The113 RMS plans call the unchanged public
`rms_scale_plan`. Four supplied return-scale scenarios use delta, its adjacent
binary64 values, or alternating nonsoftmax returns. Final softmax return scale
is separately supplied; internal softmax arithmetic remains its predecessor's
explicit fixed plan. These examples are not an exhaustive reachable-scale
certificate. Every future real refresh return must be replayed; no scale is
relabelled. Residual compatibility is tested with the actual copied production
metadata validator, and additions retain the left operand's scale. The report
stores exact per-norm plan digests and public join metadata, not ciphertexts.

The paired original-coordinate bound is sharper than adding two independent
coordinate bounds. For same-input `v=mean(x²)+epsilon` and uniform inverse error
`|sqrt(v)P(v)|≤1+delta`, pairwise Cauchy gives
`|c gamma_i x_i P(v) ± s gamma_j x_j P(v)| ≤ sqrt(128) gamma_max (1+delta) sqrt(c²+s²)`.
The public gamma maximum and tensor hashes come from the prior exact learned
weight report. This plan computes all40960×64 actual rounded public coefficient
pairs using the frozen expression. It treats those bytes as exact, without
assuming correctly rounded libm. After verifying their absolute values≤1,
the basic binary64 square/sum rounding allowance4u encloses each exact dyadic
`c²+s²`; its resulting upper bound is about1.0000000000000007. Future software
must bind the same table bytes or certify its own tables. No tables are archived.
The resulting conditional post-RoPE refresh bounds areB64..2048, neverB2.

The same coordinate relation gives a generic score/centered-score envelope.
**None of its868 K2..32/layer bounds proves the frozen centered-score domain.**
The predeclared exp domains remain assumptions; this plan does not widen or
clip them. Their known zero endpoint noise margins also remain unresolved.
All these exact-real statements require the original same-input variance
relation. Inverse/pre-RoPE/post-RoPE refresh perturbations, rounded variance,
CKKS error and changed correlations are separate obligations; identity-refresh
planning does not establish them.

This deliberately literal schedule needs638/3270/6278/8043/12931 full-slot
refresh calls forS1/8/16/19/32. ForS32 the breakdown is11656 softmax,520 inverse,
290 rational-tanh and465 outer/reserve refreshes. Raw norm-input bounds still
reachB32768, and norm-output boundsB8192. The known component key union plus
dense-shape linear/vocabulary gaps has166 keys (147.84375GiB raw pairs).
This is a preliminary inventory: actual authenticated matrix zero diagonals
can change chained gaps. Controls, duplicate mappings, relin/public keys,
compiler state and transient payloads are excluded from this storage arithmetic.

The smallest next implementation is a typed outer continuation binding plus a
serial Q/K/V join that keeps real RMS outputs, real projections and original
positions, applies these two explicit post-RoPE refreshes, and feeds actual
score objects into the existing causal-softmax/PV owners. The inverse-only
binding cannot silently authorize these new events. A Qwen SiLU owner, retained
MLP/residual owner, final-vocabulary streaming owner, complete model inventory,
client/server handlers and a dedicated synchronous/quarantine engine remain
missing. No practical HE or full-model correctness claim follows from this plan.
