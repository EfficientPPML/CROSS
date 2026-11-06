# Qwen initial expanded-prefix observations

The frozen 61-source observer completed all **3,200** ordered calibration contexts: 3,072 prefills (96 at each length 1–32), 64 teacher continuations and 64 free continuations. All 3,200 recorded semantic checks passed. Native FP32 forward counts were 3,232 started and completed, including one additional uninstrumented invariance call per length. The independent exact reference used FP64. The frozen candidate3 baseline generated 64 free-trace contexts; its ranges are excluded from initial selection bounds.

The producer recorded 113 normalization **input** variance sites, 28 SiLU gate-input sites and valid per-query/per-K raw and centered causal score rows for each native/exact context. K1 remains the direct bypass. This archive does not claim new candidate accuracy, heldout validation, coefficient selection or encrypted execution. The observer did not execute a tokenizer; its public report distinguishes permitted transitive tokenizer imports from tokenizer execution.

Root independently reauthenticated and reconstructed all 9,666 private files, the full ordered context/prefix/choice ledger, per-context coverage and aggregate native/exact observations. The public reconstruction receipt is bound to the reviewed script and a separate successful outer-process receipt. The 4,611,984-byte reconstructed snapshot remains private. This archive contains its descriptor only; the portable verifier checks that descriptor's public bindings without opening the snapshot or rerunning private reconstruction.

The actual observer owner, root, reaped session **2619**, exit 0 (tool chunk `00c5fe`); its parent and worker were absent. Root separately reaped reconstruction session **64028**, exit 0 (`ee9aa2`), with its PID/process group absent. These are the owner's authoritative receipts, not this archiver's tool sessions. Both producer reports record completed cleanup, source stability, signal restoration and zero audit denials. Parent `model_forward=false` describes the collecting process; the nested worker records `model_loaded=true`, `model_forward=true` and the completed call counts. The initial observer took 2,551.071 seconds in its worker and 2,552.282 seconds in its collector; these are component observations, not benchmark timing.

`public_whitelist.json` lists every copied artifact. The exact source archive, manifests, aggregate parent/worker reports, public logs, launch/preflight linkage, root reconstruction review and exact review script are included. No private journals, snapshot, tokens, calibration bodies, model tensors, ciphertexts or expanded execution-source copies were opened or copied by this archiver. Source tar members include public implementation code and pinned public manifests/reports; they contain no private model/input payloads. Private paths in descriptors are metadata, not included files.

The linked preparation passed 274 synthetic checks, eight installed backend denials and a separate genuine Qwen-class import-only smoke. Earlier failed observer/import attempts remain preserved in the [initial preflight archive](../qwen_expanded_prefix_observer_preflight_cpu_20261104/README.md) and [import successor archive](../qwen_expanded_prefix_observer_import_successor_cpu_20261104/README.md). This actual proof neither replaces that history nor promotes the source audit into an OS sandbox or a binary-reproducibility claim.

Run `python -I -B verify.py` for portable public verification. Add `--current` to compare the 61 current and execution-source files, the frozen review script and current process absence. Neither mode imports producer/model code or opens private descriptor paths. The included reconstruction script is reviewable provenance; **do not execute it as part of portable verification**.

Public pins:

- Actual report: `bc77c79e104af3f2f7c0da709bbf5c9319483ce7ad9a1dbb7cb3b8002510afb3`.
- Actual source archive: `d4ccebcca8691b78a00eb3af58cbb67e213e286038c112d7f85550484303791c`.
- Root reconstruction review: `edc9a06ef42108d8ccdefdf5a805c262c0a0f85c5d4ce900b3ec8b13a7feccad`.
- Root reconstruction outer receipt: `120e6df818f6ea2207fa8c7b40aeaa34dcc709376294de909ba3438256c90644`.

Actual coefficient selection requires its own externally pinned grant and terminal evidence. Nothing in this archive grants that next stage.

Security independently inspected the public verifier and ran portable/current readback successfully. Its unchanged receipt is retained under `peer-review/security.json`; it did not follow private descriptors or rerun reconstruction/certificates.
