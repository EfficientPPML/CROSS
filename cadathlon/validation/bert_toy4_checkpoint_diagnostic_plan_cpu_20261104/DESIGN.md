# Frozen toy BERT4 checkpoint diagnostic plan

This is a public plan and a source-bound clear/synthetic preflight. It does not contain a launchable HE diagnostic worker and does not authorize a TPU run. The original optimized sweep and its 199 files remain unchanged. The original toy BERT4 failure, its failed decoder, seven earlier successful cases, and the parent sweep's continuing ownership remain separate evidence.

The preserved case uses two examples, eight positions, hidden width24, intermediate48, four heads of actual width6, two layers and two labels. Its public seeded weights and input are reconstructed from the exact upstream fixture. The packing has feature width64 and period8192 repeated eight times in65536 slots. This is the original sparse-period circuit, not a full-independent-slot proof or a pretrained full BERT model.

The fixed profile is N131072, CD2, depth49, scale bits56, dnum4, ordered Q100/P24, scale68994011570896900. Its actual log2(QP) is3490.161581036245 under3523, and P covers all four25-Q partitions. The public preflight checks exact integer products and the original source/key inventory; a future actual worker must also call the exact production qp_is_secure and P-coverage checks on the actual context before any key generation. No security checking here imports a backend.

## Preserved failure and planned checkpoints

The original process exited1 after two layer callbacks and five refresh pairs. Client decoding rejected an estimated standard deviation of infinity at the unchanged0.03125 limit. No decoded logits, accuracy or final operator errors were produced. Output metadata showed L11/NSD1; this does not establish numerical validity or locate the first failure. The public checkpoint archive remains the authoritative failure, and a future diagnostic is a new run with new random encryption/key material, not a replay of identical ciphertext bytes.

The plan records96 ordered ciphertext checkpoints: original input; before/after18 linears, two attention operations, four row norms and11 polynomials; five actual refresh inputs/outputs; the existing refresh/layer progress checkpoints; final classifier. Some are deliberately duplicate checkpoints, preserving the original phase ordering and allowing independent envelope authentication. All65536 physical slots are checked, including replica, feature/head padding, invalid-key lanes, and pooler source-mask/bias effects. Expected arrays are reconstructed on the client from the pinned public fixture; only hashes and scalar summaries enter evidence archives.

The five unchanged refreshes are:

| Event | Value | Entry | Required next levels | Fixed bound | Output |
|---|---|---:|---:|---:|---:|
|0|Second-layer masked scores before centering|2|1|16|25|
|1|Second-layer attended heads before output dense|2|1|8|25|
|2|Second-layer attention variance before inverse sqrt|3|3|2|25|
|3|Second-layer GELU input|2|5|4|25|
|4|Second-layer MLP centered values before square|2|2|8|25|

The independent dense NumPy reference uses original fixed polynomial coefficients, actual rounded Chebyshev-to-power coefficients and explicit dense attention/MLP operations. It does not invoke packed graph, diagonal or rotation methods. The second reference executes the exact frozen graph/evaluator/BSGS loops with named NumPy primitive doubles and identity refresh. All96 checkpoints agree within the predeclared clear self-check1e-10. Eleven declared polynomial domains and all five ideal refresh amplitude assumptions pass this fixed public example. This provides reference consistency, not a CKKS perturbation bound.

The original final tolerance remains abs(error)<=0.02+0.01*abs(reference), plus the exact upstream binary prediction tie rule. The decoder stddev limit remains0.03125. A fixed1e-6 absolute all-slot checkpoint comparison is an additional diagnostic; failure never changes or replaces the original gate. NaN, infinity, wrong shape, complex outputs and overflowing metrics must be reported as failed/null rather than prevent failure JSON publication. Client readback should preserve each independent decode failure and continue other saved checkpoints, then require every original gate and all declared diagnostics for a diagnostic pass.

## Actual scale and phase obligations

The public schedule replays ordinary binary64 metadata operation ordering. It gives nominal scales for an identity refresher returning canonical scale. Tests also inject adjacent binary64 refresh scales and require those actual scales to flow forward. A future server must record the actual scale at each refresh and independently replay following operations with that value; it must not assign a canonical label or compare only to the returned wrapper's own metadata. Every checkpoint requires exact typed L/NSD/modulus layout and finite positive scale. No scalar metadata check substitutes for payload validity.

A future execution has separate client setup, keyless server and client readback processes. Setup creates fresh owned keys and encrypts the original fixture; public graph/key inventory is fixed before setup. The server receives evaluation keys and opaque ciphertext only. It blocks both keygen module aliases and all seven encode/decode entrypoints before initialization, has no secret/public encryption key or client reference path, and records actual context security/scale/layout. Actual before/after wrappers must retain original array owners through callbacks, independently of mutable ciphertext wrappers, and verify supplied context/helpers/keys/source at callbacks.

Each checkpoint is synchronized and immediately saved to bounded authenticated opaque transport; the server keeps no additional all-checkpoint device list. The transport must bind context digest, source/case/profile/coefficient identity, ordered checkpoint label/index, actual L/scale/NSD, exact uint32 RNS dimensions/residue ranges and payload SHA. Header bounds precede allocation. Owner-pinned directories and held descriptors must detect replacement and preserve unexpected entries. These transport/runtime pieces are deferred, not supplied by the synthetic PhaseLedger.

All server computation, all96 checkpoint saves and final source readback must finish before any private decode. The client accepts an externally authenticated terminal server manifest only after direct-child wait, owned-group absence, complete keyless/codec flags and complete ordered inventory. PhaseLedger tests the proposed order with opaque digests; it is not process isolation or a ciphertext implementation. A server failure forbids early private decode in this planned run; failure evidence remains public and the saved files may be retained for a separately authorized investigation.

## Resource and lifecycle limits

The actual graph rotations number41; the bootstrap/conjugation inventory172; the union199. Each raw evaluation-key pair is520093696 bytes. Rotation keys alone require103498645504 bytes; including one relin pair, bounded header allowances and64MiB reserve yields104087486464 bytes (about96.94GiB). Public encryption key104857600 bytes and client secret1048576 bytes are additional. A future capacity gate must account for the actual filesystem, existing live jobs and separate public/private/checkpoint locations; this plan makes no current capacity or accelerator lease assertion.

All96 raw checkpoint payloads total3758096384 bytes (3.5GiB). Header allowances plus1GiB reserve give4832624640 bytes. Largest checkpoint is100MiB. The intended incremental save adds at most one host transport payload and one temporary device checkpoint owner beyond graph-owned live values. These are storage/extra-owner bounds, not a measured memory peak. JAX compilation, tables, graph temporaries, mmap/file cache and synchronization failures are excluded from those numbers; the original failed run's sampled RSS is historical evidence only.

Future execution must use a dedicated OwnedChild group, record direct-child wait separately from group absence, retain the owner on failed waits, and shield attempt-all cleanup/publication. Any uncertain asynchronous readiness requires process-lifetime quarantine of compiled owners/provider/arrays and public mappings, preserving the original failure; never close backing maps merely because a wrapper was discarded. Terminal/outer restoration errors must republish failure, and SystemExit must occur after guards exit. The public preflight exercises the reviewed process/terminal helpers synthetically and with a tiny clean child; no healthy cryptographic deadline is introduced. Main-thread signal masking is not a process-wide atomicity guarantee.

## Remaining implementation and launch gate

Implement the exact immutable instrumentation adapter, actual metadata replay, authenticated96-checkpoint transport and separate three-role worker after this plan is reviewed. Then source-bind and review its real context/security/keyless/codec/lifetime/capacity preflight. Any actual accelerator launch requires a separate root grant and an available lease; root retains the optimized sweep's session12657. No code in this lab changes the live199 tree, retunes coefficients/domains/gates, or retries the failed encrypted case.
