# Pretrained Qwen arithmetic planning

The [planning script](plan_pretrained_qwen_arithmetic.py) evaluates all 28 layers
of the pinned Qwen3-0.6B checkpoint, final RMSNorm, and all 151936 final-position
logits using fixed approximate float64 arithmetic. On the same three short
public prompts as the exact audit, every recorded operator/block nRMSE is
below the handout's 0.05 threshold, final-logit nRMSE is below 0.10, and top-1
tokens agree. This is numerical planning, not encrypted execution, hidden-suite
validation, or generation readiness.

| Public prompt | Tokens | Final maximum absolute error | Final nRMSE | Top-1 agrees | Domain failures |
| --- | ---: | ---: | ---: | --- | ---: |
| The sky is | 3 | 5.68341e-4 | 4.13907e-5 | Yes | 0 |
| A small robot | 3 | 5.44104e-4 | 3.06741e-5 | Yes | 0 |
| One plus one equals | 4 | 1.60566e-3 | 1.50900e-4 | Yes | 0 |

The [report](plan_pretrained_qwen_arithmetic.json) records all 84 block
comparisons, 1263 operator comparisons, and 1864 propagated domain observations,
including any failures. The worst recorded operator nRMSE is 1.58153e-4 and
the worst block nRMSE is 2.90324e-5. Metrics use the original denominator floor:
`RMS(approx-original) / max(RMS(original), 0.01)`. The separate requirement for
**HE versus the approximate circuit**, nRMSE 0.01 and maximum error 0.10, is
untested. Three prompts do not establish the required aggregate top-1 rate on
Q3–Q5 or either four-token decode gate.

The exact trace preserves the checkpoint semantics: hidden size 1024, 16 query
heads, 8 key/value heads, head dimension 128, intermediate size 3072, Q/K
RMSNorm before half-split RoPE, grouped-query repetition, causal normalization,
gated SiLU MLP, and tied vocabulary weights. Query projection width is 2048,
so treating it as hidden size would be incorrect. Exact NumPy final logits
match native Transformers within 8.73e-6, with matching top-1; the upstream
implementation uses float32 in some normalization/position internals even
with double weights. Both complete exact logit arrays are preserved in the
[compressed baseline](plan_pretrained_qwen_arithmetic_exact_logits.npz), whose
SHA256 appears in the report. They are not fitted targets for the candidate.

The [operator manifest and assessments](plan_pretrained_qwen_arithmetic_operators.json)
are written before any approximate final logits are computed. They use only
analytic functions and empirical exact-intermediate domains from these public
prompts. Positive RMS variance intervals receive 20% slack; SiLU and centered
score intervals receive 10% span slack and outward integer endpoints. The
fixed local targets are inverse-sqrt relative error 1e-6 and SiLU absolute
error 1e-4 on a 16385-point grid. These local targets are planning choices,
not certified uniform bounds or original model-level gates.

- Of 113 inverse-sqrt sites, 13 use degree 8–32 power evaluation within the
  existing degree/coefficient limits. The other 100 use fixed Newton
  iterations, ranging from 8 to 20, selected from their public intervals.
- Twenty-seven SiLU sites use balanced Chebyshev evaluation with degrees
  24–192. The last layer's empirical domain is [−149,316]; direct degree 256
  still has sampled error 0.05166. That site instead uses
  `SiLU(x) = x*(1+tanh(x/2))/2`, degree 12 base tanh, eight rational double-angle
  stages, and five reciprocal iterations per stage. Its sampled analytic
  error is 1.59e-6. Degree 8 was rejected by the fixed local target.
- Softmax uses degree 8 range-reduced exponential, followed by four or five
  probability-square/normalization rounds. Reciprocal Newton counts are fixed
  from public denominator intervals. Each causal query uses its public number
  of legal keys; the first query has public probability one. No private
  maximum, comparison, clipping, or data-dependent stopping is proposed.

This causal-row representation avoids evaluating nonlinear functions in future
key slots. It is an abstract layout choice, not an implementation of the
existing packed pair layout. A concrete packed circuit must safely initialize
or mask every unused slot before nonlinear evaluation and preserve the same
causal semantics. Recorded domains are observations of the propagated
approximate computation; they do not certify unseen prompts or CKKS error.

Per-layer observations expose substantial conditioning differences. Input RMS
variance starts near 0.00052; the largest post-attention variance reaches about
42842 in layer 25. Q/K variance ranges also differ substantially by layer.
The approximate graph's largest observed intermediate is about 43.12 million,
at layer 25's post-attention normalization square. These magnitudes exclude
internal linear-map partial sums and rule out blindly reusing toy bound-16
refresh assumptions.

The chosen abstract schedule charges one level for each secret product and
each public affine, matrix, reduction, or layout stage. Its critical depth is
9007, with 10139 secret tensor product calls for a three-token prompt and
12559 for four tokens. It has 197 public matrix maps and 56 key/value layout
stages. Tensor calls and scalar-element counts in the report are **not actual
ciphertext counts**. Public query branches run in parallel in this depth model.
Assuming initial depth 49 and refresh output depth 25 with reserve 2 gives the
optimistic critical-path count `ceil((9007-49)/23) = 390` for this chosen
schedule only. This is not a model lower bound: affine fusion or different
arithmetic could reduce levels, while concrete packing, precision, branch
scheduling, and refresh scaling could require more work. No full-checkpoint
packing, key inventory, memory plan, or encrypted error budget is established.

The final CPU process exited 0 in 9.68 seconds with 7759120 KiB maximum RSS on
cores 144–147. This timing is for plaintext planning, with no implied encrypted
speed. Ten source hashes remained unchanged. Six
[focused regressions](test_pretrained_qwen_arithmetic.py) pass, including causal
mask parity and future-key independence, half-split RoPE, grouped-query full
toy graphs where query width differs from hidden size, wide-interval rational
SiLU, and the nRMSE floor. These tests are also plaintext only. Preserved raw
[run output](plan_pretrained_qwen_arithmetic.log),
[resource measurements](plan_pretrained_qwen_arithmetic.time.txt), and
[test output](test_pretrained_qwen_arithmetic.log) accompany the reports.

Reproduce with the verified snapshot and the separate pretrained environment:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
taskset -c 144-147 /tmp/cross-pretrained-validation-env/bin/python \
  cadathlon/validation/plan_pretrained_qwen_arithmetic.py \
  --snapshot /home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca

JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
taskset -c 144-147 /tmp/cross-pretrained-validation-env/bin/python \
  cadathlon/validation/test_pretrained_qwen_arithmetic.py -v
```
