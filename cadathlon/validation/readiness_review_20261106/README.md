# Readiness review — November 6, 2026

**The competition-readiness goal remains incomplete.** This review preserves
current evidence gathered within the user's requested 15-minute work window.
It does not turn setup success, plaintext calibration, source review or a missing
tool session into a successful encrypted evaluation.

GitHub still reports [PR #6](https://github.com/EfficientPPML/CROSS/pull/6) merged
at `98ca6833397069e4228c265d75fef64884218a28`; the connected GitHub read is retained
in [pr6-metadata.json](pr6-metadata.json). The PR explicitly leaves encrypted
transformer references and the five-minute evaluation target unfinished.

| Requirement | Verified disposition |
| --- | --- |
| P1 shared client/server keys | Fix implemented; archived P1 student-v2 run remains passing. |
| P2/P4 encrypted references and calibration | Unresolved. The original twelve-case TPU sweep has seven passes and five failures: BERT4/5/6 and Qwen5/6. |
| Actual-ring 128-bit security checks | Implemented in runners and P1 stage3; previous tests retain their stated scope. This is not isolation from arbitrary contestant Python. |
| P3 batch2/4 TPU abort | All six original LeNet/AlexNetTiny cases passed in3369.48s. This review rechecked all56 indexed final artifacts. |
| Approximately five-minute evaluation | Reduced P1 student-v2 schedule passed in257.56s and P3 compact-v1 in145.00s. These timings exclude the original six-case cold schedule and pretrained transformer workloads. |
| Full pretrained endpoints | Full BERT12, all five file stages, ranked100/200/408 for seeds42/43/44, and full Qwen28 with genuine HE/TPU evidence remain incomplete. |

[issue-evidence-readback.json](issue-evidence-readback.json) records65 artifact
pins and the scoped assertions. Its [reader](issue-evidence-readback.py) performs
only public JSON/hash checks; no archived author/verifier or candidate was run.

## Current execution evidence

[report.json](report.json) records the November6 observation. All five former
root-owned session handles returned `Unknown process id`. OS inspection still
found the original BERT stage4 worker1688839, P60 worker1179041 and prepared-BSGS
BERT worker2613514 with matching command/start identities. Their parent PID is1.
No healthy worker was cancelled or restarted; no success is inferred from their
continued presence. Root did not poll the CNN-owned session46674.

A [follow-up at22:44:24 UTC](worker-progress-followup.json) confirmed the same
three command/start identities and positive CPU-time increases for every worker.
This establishes continued work, not a terminal or numerical pass.

The BERT4 diagnostic's collector, stage and worker public reports are mutually
consistent and report successful setup. The worker's direct exit0/reaping is
recorded by the collector; both original process groups are now absent. Root
checked all260 source files in both trees and all three source archives, actual
Q/P security metadata, TPU device metadata and the two retained workspace
directory identities. The collector reports3151.16s elapsed. Exact evidence is
retained in the `toy4-*` files here.

The original outer-tool terminal receipt cannot be reconstructed from a successful
report and process absence. Strict terminal admission therefore remains false.
Server and validator were not launched. The workspace remains retained; no key,
model or ciphertext bodies were opened. The setup does not establish numerical
accuracy or a successful model inference.

P56's original worker is absent, but its original key directory remains present;
its outer/process metadata is unfinished. Its numerical result was not opened
or admitted, and no cleanup was attempted without its required ownership review.

The [Qwen calibration public readback](qwen-calibration-public-readback.json)
verified all139 execution sources and their archive against the unchanged
start/end manifest. The public report records3200/3200 complete calibration
entries,3200 semantic passes and16/16 complete free sequences, after170999.79s.
Its worker is absent. These are its fixed calibration inputs only; validation
inputs were not read and encrypted execution was false. Required independent
exit0/reap evidence is unavailable, so the result is not admitted and private
outcome bodies remain unopened. No held-out results were used to choose
coefficients. The exact public report, sources and readback script are retained
in the `qwen-calibration-*` files.

## Verification and limits

[review.py](review.py) is the root-owned readback implementation. It verified533
source/public files, report hash joins, source archives, security arithmetic and
process/workspace metadata without importing candidate code. Its file pins are
in [verified-source-and-public-pins.json](verified-source-and-public-pins.json).
No new numerical tests were run in this review. Historical numerical results
retain their original source snapshots and are not relabeled as current runs.

Remaining work is successful encrypted transformer execution and calibration,
completion of the full pretrained/file-stage/ranked requirements, and recovery
or replacement of missing terminal evidence under the existing execution gates.
The current evidence does not support a competition-ready release.
