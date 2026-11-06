# Prepared engine + frozen BERT BSGS v2: public compatibility plan

The authentic74-matrix audit closes the actual-key gap. The frozen BSGS graph requires62 distinct keys; unchanged full-slot refresh requires106. Their union has118 keys, all present in the actual retained130-key stage1 public manifest. Missing keys: none. No key array was read, generated, rewritten, deleted or repacked.

## Exact public matrix inventory

Every selected public NPY file was authenticated against the completed stage2 report (5b9589b5…), manifest and fixed tensor pins, and against its actual original safetensors byte offsets. The original checkpoint hash e1f8085… and all74 NPY hashes matched before/after. Exact copied `MatrixDiagonals` and chained gap-planner definitions were executed on those authentic public F32 matrices. No model forward, private input/reference, native model, ciphertext or key payload was read. NumPy alone was imported; no JAX/Torch/Transformers backend.

| Maps | Shape | n1 | Active diagonals each | Chained keys |
|---:|---|---:|---:|---|
|49|768×768|64|1535|1,64,64064|
|12|3072×768|64|3839|1,64,61760|
|12|768×3072|64|3839|1,64,61760|
|1|2×768|32|769|1,32,64768|

The union is six keys, not four. Classifier keys32 and64768 are already supplied for bootstrap. All74 matrices have their full signed diagonal support; that is measured here rather than assumed from shapes.

BSGS route36, attention assembly59 and normalization22 are evaluated from exact public source definitions. Combining attention, normalization and the six dense keys gives62. The106 refresh keys include special index262143; it is preserved literally and must not be reduced modulo65536. Graph plus refresh gives118. The12 unused existing keys are315,378,441,567,630,693,64843,64906,64969,65095,65158,65221. Keep the actual130-key store untouched.

Actual existing public NPY storage remains125,380,363,136B. A hypothetical118-key subset plus the same public/relinearization arrays would occupy113,904,744,320B including the recorded headers, but no subset store is created or proposed for this integration. Two prepared graph/linear owners retain their fixed2entries/2GiB caps (4GiB aggregate logical); physical/transient/compiler/master-backing exclusions remain exactly as in the separately reviewed engine candidate.

## Source and import seams

Start from the reviewed prepared-engine251-source snapshot, not from the active217 runtime. Apply only the frozen BSGS v2 top-level substitutions to its `joined` directory:

1. Add `encrypted_bsgs_score_routing.py` and `encrypted_bert_bsgs_attention.py` from the frozen BSGS v2 lab.
2. Replace `encrypted_bert_layer_attention.py` and `encrypted_bert_full_model.py` with their v2 versions. Their imports select `BertBSGSAttentionTile` under the existing owner role.
3. Keep the prepared engine/controller/loader and optional `StreamingLinear.arithmetic_factory` constructor from the251-source candidate. Blindly copying v2's whole common directory would overwrite that optional constructor with its ordinary predecessor. Other shared runtime sources compared byte-identically; test/collector-only files are not runtime substitutions.
4. Bind import search order explicitly: isolated runtime/stage, joined top-level, joined common and core package roots. Verify resolved paths/hashes for the two new routing modules, both owners, StreamingLinear/CompiledArithmetic, prepared cache/loader, ordinary refresh, and full_model_engine before any payload/context use. No ambient main source or active217 module may satisfy an import.

A future file-handler successor can add an exact-bool, default-false `prepared_rotation_cache` keyword to `server_encrypted_compute`. The explicit prepared branch selects `load_server_prepared(..., dedicated_no_write=True)` and passes its yielded handle as `prepared_handle` to `server_engine`; the ordinary branch remains identical. The seven-executable wrapper must call the prepared branch literally under a reviewed mode/source contract, rather than changing behavior through an inherited environment flag. This plan does not implement that branch or change any wrapper.

The model weights, fixed nonlinear manifest and profile are unchanged. That does not make old source-closure gates authorize the new engine: producer/consumer manifests, installed-entrypoint probes and phase reports must bind the complete new runtime closure. Existing actual stage1/2 metadata can authenticate reusable public data; existing size0/ranked readiness and timing evidence cannot be relabeled as a prepared+BSGS run.

## Borrowing, clear and quarantine contract

Authenticate the existing public key store before constructing either prepared owner, and preserve a dedicated no-write lifetime. Readonly mmap/stat checks alone are insufficient. The prepared graph and linear arithmetic must exist before BSGS constructors capture their helper objects, methods and accessor identities. Both use the same keyless context and exact ordered Q100/P14 profile. No active helper, factory, arithmetic owner or refresh closure is swapped afterward.

BSGS binds rotate/weighted_sum/add and level/encode/clear accessors, and the rotation/plaintext/rescale helper objects and methods. Existing low-memory clears remove entries/compiled controls, not these owner/helper identities. Thus ordinary checkpoint/level transitions are compatible with the fixed bindings. They may erase reuse: repeated key1 babies can hit, different BSGS keys can evict, and giant/input-level alternation clears. This is no speed or hit-rate guarantee.

BSGS owns its retained baby/source/output payload references only for its serial operation. Before successful return, normal arithmetic completion/barriers apply; on failure the routing/attention owners retain uncertain arrays through their own failure state/quarantine. A file-handler exception must propagate through engine and loader contexts without a caller swallowing it and continuing work. The engine attempts captured original prepared owner cleanup before provider disposal. The loader's separate original registrations must complete before lease retirement and public mappings close. Any readiness failure, owner substitution or failed entered server stage retains mappings/quarantine and prevents reuse. Original input/upload/model/key workspaces remain externally owned.

Refresh remains ordinary: its evaluator and compiled primitives are captured during construction. They are not added to the4GiB prepared cap. The ordinary-group/hoisted implementation is not wired or treated as a pass. A future grouped path would require its own source/digit/output barriers before the same retirement boundary.

The inherited main-thread signal-mask limitation remains. The prepared engine's three separate groups passed37+49+9 checks; the earlier combined67 result was66pass/1failure and is retained. Nine native-thread samples under the actual single-fire wrapper passed, but provide no process-wide atomicity guarantee.

## Next checks before any genuine run

After source-seam review, build a new isolated combined closure and test constructor ordering/default behavior, BSGS capture of prepared helpers, no helper swap across checkpoint clears, same-context static metadata and exact key inventory, both-owner cleanup including partial construction/readiness failure, BSGS retained-array failure propagation, and quarantine before mapping retirement. Keep existing BSGS v2 full ideal run untouched until terminal. Then review the concrete handler/installed guards and immutable phase gates. Genuine combined encrypted accuracy, repeated-refresh behavior, file-stage correctness and ranked/storage authorization are still pending; this public plan grants none of them.

## Evidence boundaries

`actual-v2` is the successful38-source authentic public audit (session25855, exit0/reaped). `final-public-tests` contains15 synthetic/public checks with39 proof sources (38 audit plus its external driver). `actual-v1` is retained as a provenance failure: all74 numerical/public checks completed, but two generated gate `.pyc` files entered an overly broad inventory, so the report stayedfalse and exited1. The v2 inventory explicitly lists measured sources; no gate/runtime byte was changed. Initial syntax and AST-extraction recursion development failures are also retained. No public tensor values or arrays belong in the evidence archive.
