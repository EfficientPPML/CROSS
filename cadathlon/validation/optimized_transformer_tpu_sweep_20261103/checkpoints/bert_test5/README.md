# Optimized toy BERT test 5 failure

The original case **failed** with `OverflowError: int too large to convert to
float`. The child exited 1 after 19,834.479544978007 seconds (about 5 h 30 min).
The saved result contains no decoded logits, maximum error, nRMSE, top1 agreement
or successful runner timing. The external process duration is not a passing
benchmark measurement.

This is the ninth completed case in the live twelve-case sweep: seven passes
and two failures, BERT tests 4 and 5. Root retains session `12657` and its final
reaping responsibility. Qwen test 5 was active at this checkpoint; the overall
sweep remains incomplete. This archive neither polls that handle nor launches
a retry.

The public synthetic fixture has batch 2, sequence length 12, hidden width 24,
MLP width 48, four heads, three layers and two labels; valid-key counts are
8 and 7. It is not full pretrained BERT, MRPC accuracy or ranked execution.

The stderr log records all three layer-output messages and eight matching
refresh-start/refresh-complete pairs. Both input and output metadata boundaries
were recorded on TPU v6 lite. The unchanged actual ring is degree 131072,
Q100/P24, with evaluator decomposition count 4 and classical 128-bit security
metadata; the independently summed log2(QP) is approximately 3490.161581 under
the recorded limit 3523. Output metadata has 40 Q limbs, finite scale
`6.8994011570896904e16`, and NSD 1. These metadata observations do not establish
numerical correctness.

The saved stderr contains no exception traceback identifying the failing source
line. The error text and completed refresh logs alone do not localize the first
numerical fault. Source diagnosis is separate; no intermediate ciphertext or
decrypted trajectory was inspected or copied for this archive.

The case's security report remains `passed:false` with `SystemExit: 1`, despite
capturing both boundaries. The original collector records child `2610682`
reaped, its owned public-key directory removed, and the owned-directory handle
closed. This reviewer independently observed the child PID/group and directory
absent, without opening key material or reaping the child.

All 199 current source files match the existing enclosing source archive and
manifest. The six case files were copied unchanged and checked before/after.
The shared archive SHA256 is
`bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b`.
Root independently verified the same source/case closure, process retirement,
metadata and nine-case checkpoint. Its `review_passed:true` means the retained
failure evidence is consistent; `case_passed` remains false. The exact review,
executed script and its separately captured parent checkpoint are included.
An independent public security/device review also checks the source order of
the metadata gates, actual Q/P inventory and partition coverage, device records,
resource samples and process absence. Its initial reviewer-script whitespace
syntax error occurred before execution and is preserved with the corrected
successful review. This is reviewer history, not a new case failure or rerun.

There are 18,957 resource samples at a nominal one-second interval. Sampled
aggregate process-tree RSS peaks at 151,212,969,984 bytes; shared pages can be
double counted and brief peaks missed. Output-device checkpoint
`peak_bytes_in_use` is 29,580,395,008 bytes. This is not continuous peak-HBM
measurement or an independent memory-limit guarantee.

Only public source, logs, results, security/resource metadata and the incomplete
parent checkpoint are retained. No key or ciphertext bodies are included. No
profile, coefficient, gate or source was changed, and no numerical rerun occurred.

Run `python -I -B verify.py` for portable public-evidence verification, or add
`--current` to compare the still-frozen source and completed case files. The
verifier imports no backend and follows no payload or key descriptors.
