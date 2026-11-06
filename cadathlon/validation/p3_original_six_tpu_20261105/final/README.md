# Original P3 six-case TPU validation — all six passed

The six original public synthetic fixtures passed on the pinned baseline source: LeNet and AlexNet-tiny, each at batches 1, 2 and 4. Root's original session81235 returned exit0 at tool chunk84c601; the final parent report records3369.480570556014s. This capsule archives that completed run and does not launch or rerun anything.

| Case | Model | Batch | Recorded max abs error | Recorded NRMSE | Top-1 agreement | Runner total (s) |
|---|---|---:|---:|---:|---:|---:|
| test1.in | LeNet | 1 | 9.1697483e-14 | 4.192558e-12 | 1.0 | 538.134082 |
| test2.in | LeNet | 2 | 1.9729314e-13 | 1.1524011e-11 | 1.0 | 616.144903 |
| test3.in | LeNet | 4 | 9.141108e-13 | 3.2013201e-11 | 1.0 | 635.264810 |
| test4.in | AlexNet-tiny | 1 | 5.187465e-13 | 2.0665839e-11 | 1.0 | 436.045436 |
| test5.in | AlexNet-tiny | 2 | 2.9998616e-13 | 1.1783458e-11 | 1.0 | 512.773272 |
| test6.in | AlexNet-tiny | 4 | 4.3929964e-13 | 1.8574621e-11 | 1.0 | 532.460897 |

Exact metrics and setup, request, server, external wall-time and GNU RSS values are in `summary.json` and the unmodified case files. Each fresh process used zero warmups and one cold request. Runner total includes mapping release; parent elapsed includes the sequential sweep. These measurements do not establish the separate compact five-minute schedule or ranked performance. GNU time RSS is not a peak-HBM or aggregate-memory measurement.

The original source gates every logit with `abs(output-reference) <= 0.02 + 0.01*abs(reference)` and requires top-1 agreement at least0.99. Decrypted output arrays were not retained: verification is of recorded, source-bound evidence, not independent recomputation. All cases report actual N32768/Q16/P4 classical128; the exact public QP product has bit length605, within the pinned inclusive881-bit ceiling. This is synthetic arithmetic validation, not trained-model accuracy or a new security-estimator result.

All30 completed-case files are retained exactly, with final report and source-start/after/end joins. The actual archive contains128 measured source/fixture files. The preparation archive contains those same bytes plus one separately hashed API_REFERENCE sentinel (129total). The historical preparation and15 fake-process collector tests retain their original statuses; those counts are not added to the six actual cases. No dynamic-constants experiment replaced the original baseline.

The original collector records each exit0/reaping, child key-directory removal and handle closure. Root separately records original terminal authority, completed wrapper/source-bound group absence, original key-parent identity and emptiness before removal, then final removal admission. Later-case inner PIDs are not inferred. The initial live receipt covers its original parent/test1 observation only. The packager neither polls the original P3 session nor restats/removes key paths; portable verification only joins these public receipts. The archived root cleanup script is provenance text and must not be executed by the portable verifier.

Raw stderr is preserved, including captured-constants warnings and any hugepage alignment diagnostics, without causal attribution. Earlier test1–3 partial checkpoints remain unchanged and remain partial historical snapshots.

Run `python -I -S -B verify.py` for portable stdlib-only evidence verification. `--current` additionally hashes only the declared public source/evidence origins. Neither mode imports a candidate/backend, reruns numerical work, polls processes, reads key/CT bodies, extracts archives, or invokes the archived root cleanup program. No repository edit, promotion or grant is part of this capsule.
