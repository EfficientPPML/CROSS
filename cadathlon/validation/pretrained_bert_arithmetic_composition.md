# Pretrained BERT arithmetic composition

The [complete approximate NumPy forward](compose_pretrained_bert_arithmetic.py)
passes the original P2 numerical and binary tie gates on the same three public
sentence pairs used by the pinned-checkpoint audit. This is validation-only
numerical planning: it performs no encryption or MRPC accuracy evaluation.
The [report](compose_pretrained_bert_arithmetic.json) preserves all 432 observed
domain checks, chosen operators, source and checkpoint hashes, versions, and
per-layer arithmetic costs. It leaves the earlier operator-only report intact.

| Public pair, zero-based | Valid tokens | Maximum absolute logit error | Original gates | Propagated domain failures |
| --- | ---: | ---: | --- | ---: |
| 0 | 15 | 1.59056e-4 | Pass | 0 |
| 1 | 15 | 3.13277e-4 | Pass | 0 |
| 2 | 16 | 8.28338e-5 | Pass | 0 |

The gates require finite logits of the correct shape,
`abs(approx-clear) <= 0.02 + 0.01*abs(clear)`, and the same binary class using
`logit1 >= logit0`, including exact ties. The exact NumPy reference agrees with
native Transformers logits within 8.89e-16. As permitted by P2's input contract,
the client computes the complete embedding module, including embedding
LayerNorm. The approximate graph then covers all 12 encoder layers, pooler,
and classifier. Public padding removal retains the original position-enriched
embeddings; its all-valid-token parity was established in the
[preceding planning report](pretrained_bert_operator_planning.md).

Operator choices were fixed from the preceding analytic-function assessments
before comparing any complete-model logits. Domains came from the same three
public examples with the documented slack; neither coefficients nor iteration
counts were fitted to labels or logits. This candidate uses:

- Degree 8 range-reduced exponential followed by 5–6 probability squarings,
  each with normalization. Reciprocal Newton iteration counts are fixed at
  six initially and ten per square. This path needs no private maximum.
- Degree 12–32 monomial inverse-sqrt where the existing evaluator's degree
  and coefficient budgets allow it. The four wider MLP intervals use 9 or 11
  fixed Newton inverse-sqrt iterations instead.
- Degree 32–128 GELU evaluated in a balanced Chebyshev recurrence. The
  corresponding ill-conditioned monomial coefficients are not evaluated.
- Degree 12 tanh on the input divided by 32, then five rational double-angle
  stages with five fixed reciprocal iterations per stage.

The graph contains no clipping, secret-dependent stopping, or private-value
candidate selection. Domain checks only record validation observations;
failure would not suppress subsequent arithmetic or later failures. No
observed failure here certifies unseen inputs, polynomial uniform error,
quantization, CKKS error, or bootstrap error.

For this chosen abstract schedule, every example uses 3,128 secret tensor
product calls, 4,031 plaintext tensor product calls, 74 public matrix maps,
and 24 key/value layout stages. Secret scalar-element product counts are
51,490,074 for a 15-token example and 55,258,080 for the 16-token example.
These are plaintext tensor-accounting units, **not realized ciphertext counts**.
The model charges one level for each secret product and each public weighted
affine, matrix, masked reduction or layout stage. Its critical path is 3,098
levels. With an assumed initial 49 levels and refresh output 25 with reserve 2,
`ceil((3098-49)/23) = 133` is only an optimistic critical-path refresh lower
bound **for this chosen abstract schedule**. It is not a lower bound for other
circuits: exact negation, integer scaling, constant addition, fusion, and other
arithmetic schedules can reduce the charged affine levels. Conversely, real
packing, parallel branches, precision, and refresh scheduling may require
more work. No concrete full-model packing or rotation inventory is supplied.

Observed intermediate magnitudes reach about 160,166 at an MLP normalization
square, excluding public linear-map partial sums. Thus the toy bound-16
refresh assumptions cannot simply be reused throughout this graph. Repeated
normalization and polynomial evaluation can amplify approximation and encrypted
errors; the float64 result does not establish a feasible precision budget.

The final CPU process exited 0 after 6.42 seconds, with 1,678,272 KiB maximum
RSS, using cores 144–147. This is a plaintext planning run, not an encrypted
performance estimate. All nine recorded sources were unchanged. Eight
[focused regressions](test_pretrained_bert_arithmetic.py) also pass, checking
balanced Chebyshev against independent Clenshaw evaluation through degree 128,
fixed Newton endpoint behavior, unhidden domain violations, and the tie rule.
Raw [forward output](compose_pretrained_bert_arithmetic.log),
[process resources](compose_pretrained_bert_arithmetic.time.txt), and
[test output](test_pretrained_bert_arithmetic.log) are preserved.

Reproduce using the verified local snapshot and separate pretrained environment:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
taskset -c 144-147 /tmp/cross-pretrained-validation-env/bin/python \
  cadathlon/validation/compose_pretrained_bert_arithmetic.py \
  --snapshot /home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c

JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
taskset -c 144-147 /tmp/cross-pretrained-validation-env/bin/python \
  cadathlon/validation/test_pretrained_bert_arithmetic.py -v
```
