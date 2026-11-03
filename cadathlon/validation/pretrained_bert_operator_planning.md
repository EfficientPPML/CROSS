# Pretrained BERT operator planning

This is CPU numerical planning on the same three public sentence pairs as the
earlier pinned-checkpoint audit. It is not MRPC accuracy evaluation, a full
approximate-model evaluation, or encrypted inference. The
[script](plan_pretrained_bert_operators.py) and
[report](plan_pretrained_bert_operators.json) preserve checkpoint hashes, source
hashes, versions, per-layer ranges, and operator experiments. Analytic function
approximations use no labels or fitted logits. Their domains are empirical
calibration observations, not guarantees for unseen inputs or CKKS error.

The complete exact NumPy graph matches native Transformers logits within
8.89e-16. Each example has 15, 15, or 16 valid tokens in its 128-token input.
Removing public padding queries and keys, while retaining the original
position-enriched embeddings, preserves **all valid-token hidden states at
every layer** within 1.43e-14 and CLS logits within 8e-16. Masked keys never
contribute to valid queries, and LayerNorm/MLP operations are rowwise. Logical
token rows decrease from 384 to 46 and dense attention pairs from 49152 to 706;
these counts are not measured encrypted speedups or an implemented packing
scheme. The attention mask must already be public.

Padding removal does not eliminate the numerical extremes. Zero-based layer 2
has valid-query centered attention scores from −22.77 to 21.64 and mean-exp
denominators up to 1.5887e8. Layer 10's valid-token MLP LayerNorm variance reaches
266.27, while padding-token variance there stays between 0.861 and 1.249. Valid
non-CLS tokens contain these large variances and can affect later CLS outputs.

The following local-function targets are planning choices: inverse-sqrt
relative error 1e-5, GELU absolute error 1e-3, and tanh absolute error 1e-4 on a
16385-point grid. They are not model-level accuracy requirements. Positive
variance intervals receive 20% slack; other observed intervals receive 10%
span slack and outward integer endpoints. Degrees below are the first
qualifying **tested** degrees, not proofs of minimal degree.

| Layers, zero-based | Operator | Tested qualifying degrees | Current power evaluator limits |
| --- | --- | --- | --- |
| 0–11 | Attention LayerNorm inverse sqrt | 12–16 | Within degree/coefficient limits |
| 0–6, 11 | MLP LayerNorm inverse sqrt | 12–32 | Within degree/coefficient limits |
| 7–10 | MLP LayerNorm inverse sqrt | 48–128 | Exceeds limits |
| 0–11 | GELU | 32–128 | Qualifying tested candidates exceed degree or coefficient limits |
| Pooler | tanh on [−24,24] | 192 | Exceeds limits |

“Within limits” only checks the current power evaluator's degree 32 and
coefficient-L1 budget 1e8. It establishes no encrypted precision or performance.
Chebyshev-basis evaluation could avoid monomial conditioning, but its deeper
schedule, operation count, and noise need separate validation. Fixed Newton
inverse-sqrt iterations are another arithmetic alternative: the three widest
MLP intervals used 11 iterations and produced observed relative errors from
3.2e-12 to4.6e-8. No full circuit level/refresh plan was constructed.

Two softmax alternatives were evaluated on the exact reference's valid score
tensors. Both use fixed public iteration counts and arithmetic; host domain
checks and exact-max comparisons are validation instrumentation.

- A binary polynomial upper-max tournament uses the Chebyshev expansion of
  absolute value and its analytic tail allowance, then range-reduced degree 8
  exponentials and Newton reciprocal. The degree 16 maximum variant has
  observed probability error below 2.34e-10. Its conservative shift allowance
  lowers the reciprocal denominator and requires up to 21 fixed Newton
  iterations. Degree 32 maximum has coefficient L1 about 1.37e9, above the
  current power budget.
- Normalized repeated squaring starts with exp(score/2^r), then squares and
  renormalizes probabilities 5–6 times. It needs no private maximum and has
  observed probability error below 2.32e-10. After each square, the ideal
  denominator lies in [1/L,1]; the experiment widens this to [0.5/L,2] and uses 10
  reciprocal iterations per round. Those intervals remain conditional on
  normalization and do not bound accumulated CKKS errors.

A pooler alternative scales the input by 32, approximates tanh on [−1,1], then
uses five rational double-angle steps, each with five Newton reciprocal
iterations. Base degree 12 yields sampled error 1.59e-6; base degree 16 yields
1.48e-8. This avoids the ill-conditioned direct degree 192 monomial form but
adds substantial encrypted arithmetic if implemented.

All approximation experiments use exact-reference intermediates. Their errors
were **not propagated through the complete model**. The next useful work is
a complete plaintext candidate graph with explicit domain-failure reporting,
followed by genuine encrypted primitive and full-graph validation. Three
examples cannot establish MRPC accuracy or full-checkpoint readiness.

Reproduce with the verified local snapshot and the separate pretrained-model
environment:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
taskset -c 144-147 /tmp/cross-pretrained-validation-env/bin/python \
  cadathlon/validation/plan_pretrained_bert_operators.py \
  --snapshot /home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c
```
