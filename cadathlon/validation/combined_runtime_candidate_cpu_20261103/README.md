# Combined runtime candidate: isolated CPU validation

**Ready for review; not applied to the live runtime.** The [single patch](combined_runtime.patch) combines bounded giant-step scheduling, native NumPy key-generation allocation changes, and explicit owner-scoped NTT subset reuse. It restores the linear method's documentation/source-mask comment, adds the NTT helper to the release dependency manifest, and includes portable queue/NTT tests. NTT reuse remains opt-in and defaults false.

The final isolated sweep passed **326 tests and 68 subtests**, with **7 skips**, in 167.18 seconds (168.54 seconds including the wrapper). All 171 captured source/fixture files stayed unchanged. It covers queue ownership, actual encrypted linear parity, all twelve ideal packed and bounded graph configurations, packaging/P2 staging, and key/rotation/context regressions. Six secure arithmetic/codec/refresh tests require separate opt-in flags; the optional second-device guard skipped in the ordinary one-device process and then passed in a fresh two-device process.

Real component evidence remains separate from symbolic graph checks:

- Four ordinary/bounded N8192, Q4/P3 linear cases, including masks and non-default input scale, matched the baseline ciphertext payload and metadata exactly. Scoped NTT reuse was enabled. Maximum client-decrypted error was **4.57e-12**.
- A separate secure N8192, Q8/P2, dnum4, two-lane probe checked rotations +3/-5 at levels 3/1. Payloads and metadata matched eager rotations exactly; maximum decrypted error was **6.07e-8**. Executables survived cache clear/rebuild, and owner contexts were collected.
- All twelve complete-graph planners use singleton pending queues because symbolic values have no physical payload shape. The real multi-pending path is established by the separate encrypted and queue-lifetime checks, not by those ideal graph results.

The initial run is preserved under `initial_one_device_failure/`: 326 passes, 68 subtests, 6 skips and one assertion that two CPU devices existed. Only the optional-device test was made portable before the final sweep; runtime code was unchanged. The explicit two-device rerun actually covers that guard.

[summary.json](summary.json) records scope, counts and checksums. `combined/` and `ntt_followup/` contain commands, raw logs/reports, JUnit results and exact source archives; every archived source hash was verified. [patch_sources.json](patch_sources.json) binds the patch's unchanged live base and tested candidate files. `git apply --check` passed against that live base; no mutation was performed there.

These are CPU development checks, not performance calibration, full encrypted transformers, large-ring bootstrap accuracy or TPU memory evidence. The queue's extra payload budget excludes baby ciphertexts, the accumulator, plaintexts, NTT/key/control tables and primitive workspaces. No total HBM bound follows from it. Existing published TPU runs retain their original source snapshots.
