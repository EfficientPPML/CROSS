Key-context cleanup v2: preserve the original mask before mutation
================================================================

This isolated successor queries the original thread mask with `SIG_BLOCK,()`
before separately attempting the mutating block, both when installing and when
restoring cancellation handlers. A setter that changes the mask and then raises
can no longer discard the prior mask. Handler tracking, resource cleanup,
quarantine, authentication, and arithmetic otherwise retain the reviewed v1
behavior. The full patch applies to the original `ee796798bc20…` loader and
produces `7eb793deecca…`; `incremental.patch` shows only the v1-to-v2 change.

Four regressions fail on the preserved v1 bytes: install/restore block failure,
each with and without an existing primary exception. V2 passed127 source-bound
tests:104 existing checks,10 deterministic signal checks, nine sampled actual
loader/single-fire-wrapper subprocess checks, and the four new mask faults.
The exact wrapper remains `1798455cdd88…`. Session15402 exited0 and was reaped;
pytest5.90s/collector6.939s,77 sources unchanged, temporary fake files removed.
The full public archive contains duplicated source dependencies, including the
6.46MB public plan; bounded readback allows8MiB/member and32MiB uncompressed.

No actual keys, CKKS contexts, ciphertexts, model execution, or live runtime
edits occurred. Existing security predicates use public CPU constants; all
loader contexts/key files are tiny test doubles. Only source, public profiles,
logs and aggregate lifecycle metadata are archived.

The original process-signal limitation remains explicit. Main-thread masking
does not prevent Python signal handling after delivery to an unblocked worker.
The prior two handler-install failures, late process-directed cancellation,
and deterministic worker-delivery counterexample are preserved under
`prior-v1`. V1's123-test pass was real but did not cover the newly reproduced
mask mutation. V2 does not claim process-signal atomicity or preservation of a
primary exception for arbitrary delivery outside an entered guard. Nine
wrapper integration samples are evidence for their tested cases, not that
stronger guarantee.

`python verify_archive.py` checks both exact source sets, counts, retained
negative evidence, terminal records, and applicability of both patches in
temporary directories. It imports no runtime and runs no model or HE.
