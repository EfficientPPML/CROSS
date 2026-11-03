# Separate held-out wrapper cleanup hardening

The measured candidate1 held-out run remains unchanged. This separate wrapper fixes a finalization edge: an exception/cancellation during model cleanup or `gc.collect()` now marks the attempt failed, preserves an existing primary exception and still attempts provenance checks and report publication.

An isolated regression reproduces the original behavior: a cleanup KeyboardInterrupt replaced a prior ValueError and left only the initial report. Two fixed-path tests verify primary error/cancellation identity and final failed-report persistence. The unchanged calibration prerequisite tests also pass. Source-bound session80714 exited0/reaped:20tests in11.80s,40sourcefiles unchanged, all31 calibrated runtime files identical. No candidate coefficient/domain, arithmetic, model or validation execution changed.

`cleanup.patch` applies only to the frozen earlier `validate_heldout.py`; its base and result hashes are in `verification.json`. This is wrapper robustness evidence, not a rerun or repair of candidate1's held-out SiLU domain failures. The complete source snapshot is preserved in `cleanup-preflight-v2/sources.tar.gz`.
