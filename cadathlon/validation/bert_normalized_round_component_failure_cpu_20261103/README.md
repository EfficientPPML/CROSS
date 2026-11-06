# Preserved genuine normalized-round failure

The corrected common-ring component completed all 54 ciphertext checkpoints and client validation, then **failed** the unchanged 1e-6 absolute gate. Six K128 inverse estimates y08–y13 failed, with worst error 1.9912012305667304e-6; the final true-inverse diagnostic also failed. All 54 finite/metadata checks and all tracked residual/invariant checks passed. K15 passed every checkpoint and derived check. Both final probability checks passed (K128 1.2224062213850306e-10, K15 9.74606229053876e-11); these do not override the inverse failures.

This is one frozen normalized-squaring round on each of two full 65,536-slot fixtures, K128/all16 rows and K15/partial9 rows, using N131072/CD2/depth49/p60/d8 Q100/P14. No refresh, full attention, block, or model was evaluated. All server outputs preceded any client decoding. The run took 49,719.543 seconds; this is uncalibrated component timing.

Session17163 exited1 and was reaped. The producer and its process group are absent, the owned key directory is removed, and cleanup reported no errors. Independent readback matched all114 startup/archive/final/current source hashes. Original arithmetic, gates, reports and source bytes remain unchanged. No failure diagnosis or revised arithmetic is claimed here. The retained archive contains public code and metadata only, no key/ciphertext/plaintext-array payloads.

Run `python verify.py` for portable verification of the preserved failure and recorded terminal receipts. Security independently recomputed the public gate accounting; that review is separate from the actual run.
