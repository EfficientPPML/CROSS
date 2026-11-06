# Full-size public Qwen native workload audit

**Native/exact semantics passed; the existing fixed arithmetic domains failed coverage.** This audit uses the pinned full28-layer Qwen3-0.6B checkpoint and48 public prefill inputs (16 each at8,16,32 tokens), plus8 public16-token prompts with four teacher-forced and four independent exact-oracle free-running decisions each. It is not encrypted execution, approximate-circuit validation, hidden-suite scoring, or language-task accuracy.

The56 public prose fixtures were fixed before model measurements. The pinned tokenizer produces at least32 tokens from every raw paragraph; the audit takes exactlyS tokens without a chat template, padding or repeated-token filler. Actual token-ID sequences are unique within every bucket. Each call recomputes full prefill with position IDs0..S−1 and no KV cache; decode inputs have lengths16,17,18,19.

Native Transformers4.51.3 runs in float32/eager mode. The unchanged NumPy semantic oracle uses float64 and the same pinned checkpoint tensors. Every one of the151936 final logits is checked per context, preserving the earlier exact-semantic gates: maximum absolute error≤1e-3, nRMSE≤1e-5 with floor0.01, and exact argmax agreement. The native eager-attention observer delegates the original operation unchanged; actual-shape checks and tiny-model regression confirm bitwise-inert observation. The final norm domain covers only the required last-position vocabulary input; earlier RMS/QK/SiLU contracts cover all needed token positions.

## Results

- All112 native/oracle contexts passed:48 prefill,32 teacher-forced,32 free-running-context comparisons. Worst absolute error3.2645516060e-5; worst nRMSE2.0930257831e-6.
- All32 teacher-forced decisions and8 complete four-token exact-oracle sequences matched native. Teacher contexts use native preceding tokens; free-running contexts use the NumPy oracle's preceding tokens. These are exact-semantic agreement checks, not the approximate or encrypted Q6 benchmark gate.
- Every one of112 contexts violated at least one unchanged arithmetic domain. Across the aggregate,156 of197 contracted sites failed: input RMS18/28, post-attention RMS19/28, Q norm15/28, K norm25/28, SiLU25/28, centered scores28/28, range-reduced exponential inputs26/28. The final RMS domain had no violation.
- Centered native scores reached[-17.4494934,21.9976768]. Layer0 alone reached[-15.2066813,21.8084240] against its fixed[-9,11] contract. Range-reduced exponential inputs reached[-0.83423436,0.87580693], outside[-0.5,0.5].
- The frozen softmax manifests contain iteration tables only for K1–4. The audit records784 missing(layer,K) entries covering K5–32. It does not silently extend those tables or change any interval.
- Native learned Q/K-normalized outputs reached absolute486.7954407. This is an observed magnitude, not a certified bound; BERT bound64 or toy bound16 cannot be assumed to cover Qwen refresh inputs.

`full-v1/report.json` separates `passed` (successful audit and exact-semantic checks) from `fixed_domain_coverage_passed:false`. The aggregate and per-context failures remain available, including all public token IDs and generated choices. Native ranges do not certify the approximate or CKKS trajectory. The old three-prompt planning result remains unchanged and is not relabeled as full-workload coverage.

## Provenance and lifecycle

Session14761 completed and was authoritatively reaped with exit0. Shell1035557, timing wrapper1035562 and Python1035563 were independently observed absent. The audit took100.230 s internally; `/usr/bin/time` recorded101.14 s and7581764 KiB peak process RSS. These are native audit resource measurements, not inference calibration or encrypted performance.

All21 selected source/evidence files match before/archive/after/current verification. The exact source archive SHA256 is `78c443a38d8f8daf20653273f09f4c23d5bbdda45d3a6515ce3e28ba602be6c8`. The seven pinned model/tokenizer assets were rehashed after process completion and still match. No JAX/HE import, accelerator, network download, runtime edit, domain retuning or validation clipping occurred. The source archive captures the audit/reference/manifest and selected relevant Transformers modules, with package versions; it is not a complete OS or binary dependency image.

Five focused tests passed, including actual token lengths/uniqueness, bitwise-inert hooks and cancellation restoration, causal future-key exclusion, distinct teacher/free contexts, and fixed semantic gates. `test-evidence/` records unchanged eight-file local source hashes around the final test run. The preserved small `smoke-v1/` passed11 contexts but already failed domain coverage; its session72632 exited0. The full run used the same instrumentation sources.

No private key, evaluation key, ciphertext, hidden activation array, downloaded dataset or model checkpoint is archived. The prompts/token IDs are intentionally public validation fixtures. Run `python verify_archive.py` here to verify exact archived sources and workload/context bindings without importing torch, JAX or crypto.

Reproduction from a fresh extracted source directory uses the separately installed pinned assets and environment:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
taskset -c 140-143 /tmp/cross-pretrained-validation-env/bin/python \
  audit.py --output fresh-output
```
