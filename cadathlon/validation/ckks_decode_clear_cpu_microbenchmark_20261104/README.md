# Clear CPU decoder microbenchmark

One bounded run compared the original decoder with the isolated scaled-integer-conversion candidate. Both used real CRT, the unchanged 1/32 noise gate, and real FFT at degree131072/all65536 slots. All **20 calls**—one warmup per decoder followed by four alternating pairs for each of two cases—produced exactly identical binary64 outputs within each case.

The median observed paired candidate/old ratio was **0.9996217118740975 at Q6** and **1.0022803426557707 at Q40**. These are four-pair observations on a shared host, not a statistically supported speed improvement or regression. There is no performance acceptance gate and no optimization was made.

| Case | Pair/order | Old seconds | Candidate seconds |
|---|---|---:|---:|
| Q6 | warmup old→candidate | 0.413563064 | 0.447307864 |
| Q6 | 0 old→candidate | 0.409267684 | 0.407107534 |
| Q6 | 1 candidate→old | 0.417213284 | 0.405355575 |
| Q6 | 2 old→candidate | 0.400895815 | 0.402749464 |
| Q6 | 3 candidate→old | 0.403269955 | 0.405093344 |
| Q40 | warmup old→candidate | 1.263969982 | 1.269905373 |
| Q40 | 0 old→candidate | 1.276880772 | 1.280304342 |
| Q40 | 1 candidate→old | 1.268564472 | 1.270948723 |
| Q40 | 2 old→candidate | 1.266138783 | 1.284278162 |
| Q40 | 3 candidate→old | 1.282702973 | 1.271782472 |

Q6 uses the first six primes of the frozen commonp60 profile with its representative canonical scale. It models the modulus-count boundary, not a pending ciphertext or its actual returned scale. Q40 uses the recorded public Toy5 output modulus/scale with newly generated synthetic coefficients. Seed20261104 generates bounded real slots; the unchanged inverse FFT and nearest-integer rule produce ordinary coefficients, then independent integer modulo constructs genuine uint64 residues. No overflow fallback is exercised by these ordinary inputs.

Only `_fast_decode` is timed. Dataset construction, source checks, imports, output comparisons and hashing are outside measured intervals; warmups are reported separately. CPU affinity was112–115; default garbage collection remained enabled. The exact original before-file AST and actual candidate module come from an unchanged copy of the immutable207-source proof. Its pre-import CPU/libtpu/model guards are reused. CPU-only backend, unloaded TPU plugin at completion, and no device audit events were recorded. No key/context construction, encryption, decryption, model execution or TPU computation occurred. This is a clear decoder microbenchmark, not full HE or calibration performance.

All210 benchmark sources match before/after/current/archive, including every207 predecessor byte. The six synthetic arrays (51,905,280 serialized bytes) remain in `/tmp/ckks-decode-clear-cpu-microbenchmark-20261104/run-v1/synthetic_datasets`; only their seed, generator, hashes, types and shapes are archived. No `.npy` bodies are included here. Root independently recomputed every timing summary, source binding, dataset hash and process-terminal check without rerunning the benchmark.

Session28130 was authoritatively reaped with exit0 (`41e03a`). A live external row recorded PID3011310, PPID3011309 and actual PGID3011309; PID/group absence was checked afterward. Parent role is not inferred from those numbers. The original207 proof, installed runtime, frozen199, queued diagnostics and actual failure classifications are unchanged.

Run `python -B verify.py` for portable metadata/source checks. `--current` additionally checks the frozen benchmark tree, unchanged207 source tree and retained synthetic dataset file hashes. Neither mode imports a numerical backend or reruns timing work.
