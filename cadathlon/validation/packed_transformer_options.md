# Transformer layout experiments

These are development experiments, not validated encrypted transformer references or competition baselines. Production runtime files were unchanged. Source snapshots are embedded in the associated reports.

## Evidence and first prototype

`check_packed_transformer_layout.py` proves rectangular sparse-diagonal linear maps, masked/GQA attention, segmented reductions, and the complete original Qwen test1 in NumPy. Its packed result matches the current per-feature polynomial circuit within 3.56e-15. Error versus the exact public reference is 0.03050; this is approximation error, without CKKS noise or bootstrap error. The packed Qwen1 circuit uses 41 high-level ciphertext multiplications versus 177 for the current layout. This is an operation count, not a measured speedup.

A minimal encrypted prototype should start with original Qwen test1, its original weights/input/ring/security, and current polynomial contracts. Pack its 16 hidden features, 24 MLP features and 32 vocabulary outputs into one repeated 32-slot feature block. Keep residual tensors and polynomial powers as separate ciphertexts; “one ciphertext per activation tensor” does not mean one live ciphertext for the whole circuit. Single-token Qwen already skips QK/softmax. The first experiment therefore needs only packed linear maps, RMSNorm, SiLU and elementwise products, without a new attention or refresh path.

`check_batched_transformer_features.py` is a separate real HE CPU proof: secure N16384/Q8/P3, dnum3, four distinct fresh ciphertexts, common keys, no rotations. Batch2 and batch4 SiLU results match independent evaluations bit for bit; split/decrypt errors are below 1.2e-8 and scale/nsd match. This proves the nonlinear batch layout only. No bootstrap, full N131072 inference, TPU memory or speed result is implied.

## Sparse linear mapping through existing BSGS

Use `ctx.bsgs_matvec[level, num_slots, n1, n2]` with `preprocess(..., memory_bounded=True, active_diagonal_indices=..., pt_scale=...)`. A matrix-free source exposes `dimension=num_slots` and `materialize_diagonals(indices)`. For W[out,in], a signed diagonal displacement d=in−out has coefficient W[f,f+d] at feature f when both indices are valid, otherwise zero. Repeat that F-wide diagonal pattern across the entire slot vector. Normalize signed displacement modulo num_slots.

The BSGS dimension must remain the full context slot count. Supplying n=F would zero-pad short diagonals instead of repeating them and would break the layout. Generate only the nonzero baby/giant rotations required by the sparse diagonal set, not the dense rotation inventory for num_slots. Keep encoded operator controls bounded to one operator or giant group. To restore the canonical output scale, choose plaintext scale = canonical_scale × dropped-Q-product / input_scale; add biases at the resulting scale.

The four distinct Qwen1 matrix shapes require a union of 17 nonzero BSGS/norm rotations. The current per-feature single-token case needs none. At current N131072/Q84/P11/dnum8 these 17 raw rotation-key pairs occupy 12.62 GiB, before formatted keys, NTT tables and encoded diagonals. This key cost is a major tradeoff, and no speedup is established. The proposed dnum3 profile must regenerate and revalidate all keys and P coverage; it is not a metadata-only change.

## Pair layout for larger toy cases

Flatten `(padded_batch, padded_query, padded_key, feature)` with feature fastest. Choose power-of-two F large enough for hidden, MLP, vocabulary, and padded attention heads. Put each head in a Dp=next_power_of_two(head_dim) segment. Fuse GQA replication into K/V projection rows. Query/key transposition is a masked sum of rotations `(key−query)*(S−1)*F`.

Head dot products reduce over Dp with rotate/add trees and an anchor mask, then broadcast within that segment. Scale by the actual head dimension, not Dp. Softmax key reductions use rotations F*2^j, a key-zero anchor mask and reverse broadcasts. Preserve the current mean-centered polynomial contract. Inactive heads need a public in-domain denominator, such as 1, before reciprocal evaluation; masking its result after an out-of-domain evaluation is insufficient. Public key/causal masks remain explicit.

Normalization-energy bounds alone do not certify every exponential input lies in `[-4, 4]`, including masked slots evaluated before their output mask. [The reproducible BERT bound review](review_bert_bound_contracts.py) derives conditional ideal-arithmetic matrix/reduction bounds and constructs norm-contract counterexamples using public weights; [its report](review_bert_bound_contracts.json) records exact rational checks and source hashes. These witnesses are not public-fixture failures: fixture-specific domain checks, CKKS error, and encrypted end-to-end gates remain separate evidence.

RoPE rotates between the two actual head halves, using ±head_dim/2 feature offsets and masks for the negative first half and positive second half. Cosine/sine coefficients depend on query position. Apply K's RoPE before q/k transposition. Padding never changes the actual head dimension or positional rule.

For RMSNorm, keep unused feature slots zero, sum squares over F, divide by actual hidden size, add epsilon, evaluate inverse sqrt and multiply by gamma padded with zero. For LayerNorm, a mean operator should write the mean only into active hidden-feature rows. Naively subtracting a full broadcast mean from zero padding makes inactive values nonzero and contaminates the variance. Fold final activation masks into the polynomial weighted sum so approximation constant terms do not leak into unused features. Any unused inverse-sqrt input must be assigned an in-domain public value before the polynomial.

All twelve public toy shapes fit one activation tensor into 65536 slots under this uniform pair layout. BERT cases use F=32,32,32,64,64,64 and periods 512,2048,2048,8192,32768,65536. Qwen uses F32 and periods 32,128,512,2048,1024,8192. Feature packing increases the sparse bootstrap period by F: fewer ciphertexts do not imply proportionally cheaper refresh. A token-feature layout between attention phases reduces period further, but expansion/compression needs explicit masks, permutations, levels and keys.

The staged-transpose NumPy proof also verifies a useful alternative: S16 can swap two 2-bit groups with 12 rotations at two mask stages, versus 30 rotations at one stage. S8 uses 8 versus14 rotations with the same extra stage. Do not apply the extra level to presently unrefreshed BERT1/2 without replanning its complete circuit.

## Batch-axis alternative and memory limits

A less invasive prototype stacks 2 feature ciphertexts on Polynomial's batch axis, runs a common nonlinear circuit or refresh, then splits back to individual wrappers for linears. Shape is `(batch, 2, r, c, Q)`. Inputs must share key identity, Q prefix, layout, scale, noise-scale degree and device. The batch-one plaintext API broadcasts common polynomial coefficients correctly. Feature-dependent gamma/weights require splitting or a separately specified per-lane plaintext API. Use a separately initialized batch context; do not mutate a live context's batch or shallow-copy mutable facade caches.

Bootstrap's canonical wrappers, monomial operations and modulus raising use initialized cache.batch, so the layout is structurally compatible. Actual batched refresh remains unproved. It must preserve the keyless refresh restrictions, including disabling the client-encrypted DC-subtraction path.

`plan_transformer_batch_memory.py` generates actual secure Q/P sets and traces the core NTT over abstract arrays; it never allocates full-ring ciphertext/key tables. At N131072/Q84, dnum3 requires P27 and stores a 333 MiB raw key pair / 666 MiB formatted uint64 pair. Dnum8 requires P11 and stores 760 / 1520 MiB respectively. Key-switch partitions run serially; reducing their count does not remove the per-part BAT workspace. The last dnum8 partition is smaller, giving an 88-limb largest complement versus83 for dnum3.

| Batch | One ciphertext | Largest logical NTT temporary, dnum3 | Sum of all NTT SSA outputs, dnum3 |
| --- | ---: | ---: | ---: |
| 1 | 84 MiB | 336 MiB | 5.65 GiB |
| 2 | 168 MiB | 672 MiB | 11.14 GiB |
| 4 | 336 MiB | 1344 MiB | 22.11 GiB |

The SSA sum is a deliberately loose upper bound on explicitly represented arrays in one NTT if all intermediate outputs are retained. It is not measured peak memory: XLA may fuse/reuse arrays, while internal dot workspaces, public tables, other live values and allocator fragmentation add memory. A full HBM upper bound is not established. Even batch2 needs a measured full-ring primitive probe before integrating refresh. Batch4 should wait for that evidence; the existing CPU proof does not establish TPU batch behavior.

## Full-checkpoint path

Uniform pair packing is not sufficient for full checkpoints. For BERT S128 and H padded1024, the pair-feature tensor has 16,777,216 slots (256 ciphertexts), versus two ciphertexts for canonical token features. MLP width padded4096 needs eight canonical ciphertexts. A practical path tiles query/key/head/features: a 16-query ×16-key ×4-head ×64-feature tile fits 65536 slots. Reduce feature products promptly into a compact score layout and stream tiles instead of retaining every pair-feature tensor. Twelve-head S128 scores themselves occupy three ciphertexts. This needs a layout-aware level/refresh plan, matrix-free BSGS operators, and bounded diagonal/key caches. No full-checkpoint runtime or numerical result is claimed.

Reproduce on CPU:

```sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python cadathlon/validation/check_packed_transformer_layout.py
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python cadathlon/validation/check_batched_transformer_features.py
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python cadathlon/validation/plan_transformer_batch_memory.py
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python cadathlon/validation/review_bert_bound_contracts.py
```
