# Full-width BERT encrypted linear components on CPU

All three projections using actual pinned BERT layer-zero weights passed the fixed **1e-6** absolute-error gate. Query768×768, intermediate3072×768 and output768×3072 each had maximum error below **3.93e-11** across all8192 slots, including output padding. Each operation consumed one level and retained the expected scale and noise-scale degree.

The genuine CKKS run used N16384, Q4/P3 and dnum2, with actual log2(QP)206.997 under the128-bit security limit. The public server had no secret key; key-generation and server encryption/decryption entry points were blocked after client setup. Each projection used one deterministic public input vector repeated in two4096-slot blocks. These are three independent component checks, not a chained BERT block, independent token rows, nonlinear operations, refresh, MRPC or full-model inference.

The CPU process completed and was reaped with exit0 after **594.91 seconds**, including23.12seconds setup. Server CPU times were106.45,232.53 and232.17seconds. Peak host RSS was922904KiB. These are component observations on CPU, not TPU competition baselines or a controlled speed comparison.

[report.json](report.json) records the ring, fixed gates, exact software versions, input contract and cleanup. The checker verifies the pinned checkpoint SHA256 and revision, exporter source, exported inventory/dtypes/shapes/hashes, and each tensor's original safetensors byte range before converting to float64. All100 measured source/provenance files stayed unchanged and their exact archived bytes were independently checked. The owned key directory was removed; cleanup reported no errors.

[sources.tar.gz](sources.tar.gz) preserves the exact runtime/checker/exporter/manifest bytes; [archive_validation.json](archive_validation.json) binds them to the successful run. Checkpoint and exported weight payloads are not included. Reproduction requires the pinned public local checkpoint and exported arrays, explicit `JAX_PLATFORMS=cpu`, and `CROSS_ROOT` pointing to the matching runtime sources. Use `check_bert_fullwidth_linear_cpu.py --help` for required paths.
