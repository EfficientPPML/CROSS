# P3 original test2 checkpoint — second of six

The original LeNet, batch-2 synthetic encrypted TPU case passed its pinned runner gates. This is a public checkpoint for test2 only (one case admitted here, not a two-case aggregate); the original six-case parent was still running when its report was captured.

| Recorded result | Value |
|---|---:|
| Maximum absolute error | 1.9729313668892523e-13 |
| NRMSE | 1.1524011247437687e-11 |
| Top-1 agreement | 1.0 |
| Runner total, including setup and release | 616.1449033009994 s |
| External process elapsed | 632.5870987679809 s |

The source-bound runner checks every logit against `0.02 + 0.01 * abs(reference)` and top-1 agreement >= 0.99. Raw decrypted logits are not archived, so this is verification of recorded, source-bound evidence rather than independent recomputation. The reported actual ring is N=32768, Q16/P4, classical 128; the exact modulus product has bit length 605, within the pinned inclusive 881-bit ceiling.

All five completed-case files are retained exactly. The run archive has 128 measured files; the preparation archive has those same files plus the separately hashed API_REFERENCE sentinel. Sources match the start/after maps and current prepared/original files at capture. The parent report is an exact timestamped partial snapshot, not a terminal sweep report.

The original collector records exit 0, reaping, child key-directory removal and handle closure. Independent late checks found the recorded wrapper PID, its source-bound process group, and the exact completed child directory absent. No test2 inner PID was recorded or independently observed. The retained initial launch/live receipt describes the parent and test1, not test2 live identity. Wrapper group ownership is derived from the exact original collector's `start_new_session=True` call. The live parent and key-parent metadata were not newly statted. The live parent and root-owned key parent were not stopped, removed or traversed. The packager did not inherit direct-wait or live-observation authority.

Original stderr includes a 12.36GB captured-constants warning. It is preserved without a causal or performance interpretation. This checkpoint is not final sweep success, ranked scoring, trained-model accuracy, or proof of the separate compact five-minute schedule.

Run `python -I -S verify.py` for portable standard-library verification. Optional `--current` additionally hashes only the declared public source and stable evidence origins; it never polls processes or reads the live parent report. The verifier imports no candidate or numerical backend and extracts no archive files.
