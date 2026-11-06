All three reviewed follow-ups compose with the earlier five-candidate base:
scoped rescale-facade controls, nested/public HEMul rescale controls, and
serialization-safe lazy public BConv construction. The resulting 32-file patch
passed **362 tests and 9 subtests**, with
**4 skips**, in 199.51 seconds
(200.89 seconds externally). All 325 recorded
source and fixture files stayed unchanged; every exact archive member and
every reconstructed changed source was independently verified.

The run covers real secure tensor/multiplication/rescale and lazy-versus-eager
parity, control ownership and retirement, serialization, model opt-in wiring,
all twelve public ideal graph contracts, staged-helper/release packaging and
collector cancellation/cleanup. Secure component errors were at most
8.1156797e-08; each test retained its original gate. The ordinary one-device
run skipped two optional second-device guards and two opt-in large secure
refresh tests. Both device guards subsequently passed in a fresh two-device
process. The large refreshes were not run by this checkpoint.

The exact prerequisite patches, combined patch, changed base bytes, source
archives, commands, raw output, JUnit, resource records and numerical records
are retained. This combined candidate has not been applied to the live runtime.
It preserves existing ring profiles and core Bootstrap arithmetic/stage clears.
This is not a complete encrypted-model, pretrained file-stage, TPU-memory or
scoring-calibration result. The live Qwen run keeps its original frozen sources.
