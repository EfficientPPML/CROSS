The five reviewed candidates passed together in a fresh isolated checkout:
**1131 tests and 93 subtests passed**, with **11 skips**, in
357.04 seconds (360.00 seconds externally).
All 320 recorded source and fixture files were unchanged; every exact
source-archive member was independently checked against both manifests.

The 28-file combined patch contains bounded giant-step queuing, native key
allocation, optional HERot/HEMul NTT reuse, guarded dispatch-cache retention,
experimental model opt-in wiring, and owned-key-directory collector cleanup.
It reconstructs every changed source from the archived base bytes. The five
prerequisite patches are preserved exactly; the only normalization removes
one trailing empty line in a test file.

The suite covers all CADathlon tests plus CKKS contexts, codec preparation,
uniform encoding, Mapping compilation, BSGS, packing, prime generation,
bootstrap, key generation, rotation and multiplication regressions. Exact
commands, package versions, raw logs, JUnit output, resources and individual
skip reasons are retained. Explicitly opt-in large/secure checks and the
optional second-device guard are outside this run; their earlier separate
evidence remains tied to its original snapshots.

This candidate has not been applied to the live checkout. Existing ring
profiles and core Bootstrap stage clears are unchanged. This CPU regression
does not validate a complete encrypted transformer, TPU memory/performance,
full pretrained file stages, or a scoring baseline.
