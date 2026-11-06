# P3 original test1 checkpoint — 1 of 6

The original LeNet, batch-1 synthetic encrypted TPU case passed its pinned runner gates. This is a public checkpoint for test1 only; the original six-case parent was still running when its report was captured.

| Recorded result | Value |
|---|---:|
| Maximum absolute error | 9.169748294013402e-14 |
| NRMSE | 4.192557960553601e-12 |
| Top-1 agreement | 1.0 |
| Runner total, including setup and release | 538.1340822290222 s |
| External process elapsed | 554.561725527019 s |

The source-bound runner checks every logit against `0.02 + 0.01 * abs(reference)` and top-1 agreement >= 0.99. Raw decrypted logits are not archived, so this is verification of recorded, source-bound evidence rather than independent recomputation. The reported actual ring is N=32768, Q16/P4, classical 128; the exact modulus product has bit length 605, within the pinned inclusive 881-bit ceiling.

All five completed-case files are retained exactly. The run archive has 128 measured files; the preparation archive has those same files plus the separately hashed API_REFERENCE sentinel. Sources match the start/after maps and current prepared/original files at capture. The parent report is an exact timestamped partial snapshot, not a terminal sweep report.

The original collector records exit 0, reaping, child key-directory removal and handle closure. Independent late checks found both recorded child PIDs and their process group absent, and the child directory absent. The live parent and root-owned key parent were not stopped, removed or traversed. The packager did not inherit direct-wait or live-observation authority.

Original stderr includes a 12.36GB captured-constants warning. It is preserved without a causal or performance interpretation. This checkpoint is not final sweep success, ranked scoring, trained-model accuracy, or proof of the separate compact five-minute schedule.

Run `python -I -S verify.py` for portable standard-library verification. Optional `--current` additionally hashes only the declared public source and stable evidence origins; it never polls processes or reads the live parent report. The verifier imports no candidate or numerical backend and extracts no archive files.
