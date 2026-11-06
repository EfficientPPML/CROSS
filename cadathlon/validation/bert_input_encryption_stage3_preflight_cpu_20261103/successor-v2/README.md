# Stage-three gate successor

The actual inherited public `twelve_layer_plan.json` is **6,461,050 bytes**. The first preflight's synthetic predecessor archives did not expose its old 2 MiB member cap. `history/negative-plan-size-v1` preserves the reproduced failure against the old model gate; no keys, context or model ran.

The successor permits up to **8 MiB per source member and 16 MiB in total**, while preserving regular members, unique complete inventory, external report hashes and exact archive/member digests. The key gate additionally pins the original 58-file producer source manifest. Both that manifest and the actual public source archive fixture are now captured by the collector. Oversized member/total cases still fail.

Source-bound verification passed **120 mocked/source tests plus 21 actual installed entrypoint denials**, with all **79 proof files (78 candidate closure files plus the external driver)** identical before, current, execution snapshot and archive. All original 59 copied runtime files remain unchanged. The encryption worker and phase/retention behavior are unchanged. Actual stage-three execution still requires successful caller-pinned key and model predecessor reports and explicit authorization; this checkpoint did not encrypt anything.

No private inputs, key arrays, model tensors or ciphertexts are included. V1 measured sources and proof are retained unchanged in the parent directory.
