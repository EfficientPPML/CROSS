# BERT6: detection point and localization limits

The actual negative is the original `ApproximationErrorTooHigh` result,
estimated standard deviation `1.036e26`, limit `0.03125`, with child exit 1.
The saved stderr has no exception traceback. Nothing in this checkpoint
identifies the first corrupted ciphertext or measures plaintext-logit error.

The following is a **frozen-source inference**, pinned by complete file hashes
and selected definition AST hashes in [source_refs.json](source_refs.json).
The definitions are in the enclosing 199-file archive; none was imported or
executed to produce this diagnosis.

1. `bert_optimized_reference.create_session` selects the existing packed-pair,
   bounded-refresh, NTT-subset and dispatch-cache-retention options.
   `PackedTransformer._bert` emits the two layer milestones after attention
   normalization and after MLP/output normalization. After layer 2's output
   milestone it still computes CLS pooling, the tanh polynomial and classifier.
   Therefore the eighth refresh belongs to the post-layer suffix by log order;
   the log does not identify its exact internal operation or prove its accuracy.
2. `BoundedArithmetic._ensure_capacity` emits a refresh-complete message only
   after the refresher returns, the minimum level is checked, relations are
   preserved and a public bound is declared. It does not check a decrypted
   numerical value. Eight complete messages are not eight correctness checks.
3. `benchmark_session` evaluates and synchronizes the result, then invokes the
   original ciphertext/security gate before `session.decrypt`. The instrumented
   wrapper records each boundary only after that security gate returns. Both
   boundary records are present. This supports completion of evaluation and
   its output metadata/security boundary, not a successful numerical decode.
4. `ToyReferenceSession.decrypt` calls `client.decrypt_slots(value)` with the
   default approximation validation enabled. The unchanged `CKKSContext`
   facade resolves the ciphertext's scale and calls `_fast_decrypt_decode`.
   That route converts to RNS coefficients and calls `_fast_decode`.
5. `_fast_decode` CRT-combines the coefficients, centers and divides them by
   scale, forms the complex vector and its conjugate counterpart, and calls
   `_validate_decode_approximation` before averaging and `FFTSpecial`.
   `_check_decode_stddev` raises the exact recorded message for a nonfinite
   estimate or an estimate above the fixed `2**-5` limit. The reported
   `1.036e26` estimate is finite but far above that limit. It is an internal
   conjugate-symmetry estimate at message scale, not an observed logit error.
6. The runner calls `metrics` and its accuracy gate only after decrypt returns.
   Its exception handler publishes `case/passed/error/error_type/notes` and
   returns 1. B6 has exactly those five fields and no successful metric or
   runner timing fields. The common runner preserves the primary error across
   session cleanup; the saved notes are empty.

The observed output scale and remaining 40 Q limbs neither explain nor excuse
the failed approximation gate. Unlike BERT5's recorded integer-to-float
overflow, this BERT6 path reached the approximation check; that distinction
does not establish why either encrypted computation diverged. The four JAX
constant-capture warnings also do not identify a numerical cause.

An exact first-divergence diagnosis would require separately authorized
intermediate evidence with original stage/refresh metadata and unchanged
gates. This checkpoint neither proposes a retry nor selects an optional
decoder or changes the queued work. It preserves the genuine failure and the
limits of the available evidence.
