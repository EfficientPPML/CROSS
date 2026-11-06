Stage5 client decryption preparation only
========================================

The isolated scaffold passed169 mocked checks on CPU168–171, with all88
before/after/snapshot/archive sources equal. Session58588 exited0 and was
reaped; child1717805 was independently absent. Final source archive is
3943ad416e8e8f6ad4e4161a605c613d2be2d79d9a11dea5e59960f54504e9d2.
All59 copied runtime files, including the existing client_decrypt_decode body,
remain unchanged from stage3-v3. No actual client decrypt or numerical gate
was executed. No real key, ciphertext, private reference or input payload was
read by these tests; synthetic cases exercise the actual public metadata and
independent response-scale contracts.

The exact completed-stage3 gate is copied from the reviewed stage4 predecessor.
The stage4 gate binds an externally pinned successful report, all217 reviewed
sources, worker/collector completion and reaping, ordered request/response
metadata, actual response scale, and retained directory identity. Incomplete
stage4 evidence fails before any client-private reads. The execution dependency
flag is still false. Root review, completed actualstage4 evidence and a separate
launch authorization remain required.

The mandatory size0 gate uses official FP32 native logits on the RHS of
abs(HE-native)<=0.02+0.01*abs(native), and requires the same class prediction
(class1 on ties). HE-versus-frozen-approx absolute error and nRMSE remain
separate diagnostics. One sample cannot establish ranked accuracy. Raw decoded
and reference values, target association and completion receipt stay in private
0600 files under0700 roots; public evidence contains only aggregates/hashes.

The collector retains external key/model/upload/download/reference paths and
new private results even on failure. It validates the completed receipt without
retiring any request. Tests include wrong targets/types, complex/object data,
finite subtraction overflow, stale source and completion flags, nested artifact
replacement, private permissions, pending prerequisites, cleanup-only
cancellation precedence, scope-exit republication, and intentional main exits.
These process/entrypoint guards are regression checks, not an OS sandbox;
calling-thread signal masking does not imply process-wide atomicity.

Historical successful preflight-v1 is retained unchanged. Its worker/gate/runtime
bytes equal v2; only the preflight collector's cleanup-error precedence and
source-origin metadata changed. The final fixture/source closure is frozen in
/tmp/bert-client-decryption-stage5-component-20261103/source-bound-preflight-v2.

Run the portable verifier with: python verify_archive.py
