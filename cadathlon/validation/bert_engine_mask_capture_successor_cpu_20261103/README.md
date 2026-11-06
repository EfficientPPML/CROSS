Engine cleanup mask-capture successor

The isolated final engine is SHA256 `a5384484269139f67db18ea90b3a99f2952265321d84bc04964433d9dede3f53`. It changes the prior v2 engine only at the two signal-mask transitions: first query the old mask using SIG_BLOCK with an empty set, then attempt the mutating block, and restore the separately captured old mask after all handler/resource cleanup attempts. A failed query does not authorize a mutating block with unknown restoration state. Existing primary/cleanup-cancellation precedence is unchanged.

Eight negative cases against unchanged v2 `2c68a0d7125bbdc314845553162cc0f9b07ea7fec283481d4b143f2130ddf129` failed as expected: installation/restoration block calls changed the mask then raised OSError or KeyboardInterrupt, with/without a primary body failure. All owners still retired but the old mask leaked. Their exact nine-file source archive and failed log remain under negative-v1.

The repaired source-bound run passed all 37 tests (25 unchanged previous tests plus 12 mask fault cases), 0.05 seconds pytest / 0.19269 seconds wrapper. Nine exact files matched before/archive/after and the original v2 source files remained unchanged. The new cases also reject an unobserved original mask after query failure and retain primary errors. No encryption, model or numerical backend was executed.

The unchanged historical tests include real signals in a single-thread pytest process with plugins disabled. Their old test names do not establish process-wide atomicity: masking the main thread cannot prevent delivery through native worker threads. The separately preserved worker-delivery counterexample and combined loader/termination-wrapper stress remain separate evidence. No arbitrary post-scope cancellation timing guarantee is made.

`full_original_engine.patch` is a complete one-file patch against original engine `841fb5a5f635a10a5e2f3f84b04361816deda11ab1869f0472e1acc00507f009`; it includes earlier cleanup hardening and this successor. `v2_successor.patch` contains only the new mask-capture change. These patches are reviewable handoff artifacts; no live/frozen runtime was modified.

Run `python verify.py` for offline exact-source/artifact verification. Full stage4/file-handler integration, actual encrypted inference and terminal reporting remain separate obligations.
