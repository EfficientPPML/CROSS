# Separate I/O cleanup successors

This is preparation only. No frozen source, producer report, key store, input,
ciphertext, private reference or result is changed. The original stage5 parent
contains 89 sources; gain4 fixture/server use the separately frozen 163-source
producer. The new client-v2 descriptor repair is a prospective dependency until
its complete source-bound preflight and independent review finish. No execution
permission follows from this plan.

## Original stage5 client

Create a new lab from the exact 89-source stage5-ready archive. Keep its original
217-source server gate, key/model/stage3 chain, original IO paths, size0 input and
target binding, official FP32 reference, P2 predicate
`abs(error) <= .02 + .01*abs(official)` and prediction agreement. Frozen-approx
absolute error and nRMSE remain diagnostics. The new 286-source combined-server
gate is not a substitute for this producer.

Apply four explicit runtime overlays:

| File | Permitted change |
| --- | --- |
| `client_keys.py` | Descriptor-only bounded `_read`, borrowed `_sha_fd`, shared `_CloseState` and non-owning `_stream`; original key-generation/profile/serialization definitions remain exact. |
| `key_contexts.py` | `_map_array` stream/descriptor/mapping cleanup only. Keep NPY format, dtype, shape, byte-count, digest, residue, permissions, context, guard, transfer and quarantine requirements. |
| `file_stage_io.py` | Shared cleanup-origin state across `read_manifest` and `_read`; protect `new_file` parent snapshot and preserve write-body/cancellation priority through stream/file/parent cleanup. |
| `request_lifecycle.py` | Only descriptor/primary handling in `_tree` and `_observe`, plus the shared helper import. Preserve the original `record_completed`, original receipt destination and retirement code. |

Do not copy the entire separate-client lifecycle module: it also contains the
optional `completion_directory` feature, which is not part of this original
stage5 successor. Its unchanged handler must still write its original default
completion ledger. Do not copy the separate response-root contract or accept a
combined-server response. Fresh original result/ledger guards stay mandatory.

The actual source manifest, explicit overlay hashes, parent archive/report pins
and loaded-module origin checks must describe the successor. Prior key/model/
stage3/stage4 producer inventories remain pinned independently; changing a
consumer must not rewrite any producer's expected source map. In particular,
the existing key manifest keeps its measured key-generator source identity;
the current loader module hash never replaces that provenance.

The old worker publishes before its own standalone output exclusion checks and
the old CLI publishes its result before the outer termination scope exits.
Do not silently carry those semantics into a new actual attempt. Introduce
reviewed successor entry wrappers: authenticate bounded public fixture/reference
paths before any publication, acquire a new output inode before failure reports
can be written, protect result/evidence paths, and publish final success only
after cleanup and outer termination restoration. Keep direct-child wait and
owned-group absence distinct. These are explicit collector/worker overlays,
not changes to the copied decrypt arithmetic. Main `SystemExit` remains outside
the protected body. Numeric failure is complete evidence with `passed:false`.

## Gain4 validator consuming the old server

A new validator can soundly consume the old 163-source fixture/server outputs.
The current obstacle is a caller-level source equality assumption, not a wire
contract mismatch: both `collect_component.preceding_config` and
`component_worker.prior_gates` pass the *current consumer* source map to
`terminal_gate(expected_sources=...)`. Once a loader changes, those calls must
use an independently pinned producer map instead.

Use a separate validator-only lab. Keep `component_contract.py`, numerical plan,
gain4 owner, all fixture/oracle definitions, ciphertext transport, profile,
candidate, checkpoint order, actual scales, domains and gates byte-identical.
Only `client_keys.py` and `key_contexts.py` require the descriptor overlays on
this role's active loader path. The component uses its own already-reviewed
transport/protocol readers; it does not invoke stage `read_manifest`, completion
receipt creation, key generation, encryption or server arithmetic. Any extra
runtime overlay needs its own stated reason and tests.

Freeze a public producer-binding record containing:

- the exact 163-entry producer source map and its canonical file hash;
- the reviewed ready proof report `6fbf898fbcb8a884e2f2c7f1e2308dfd04171be990bfde82a20200d20f00e19c`
  and archive `382933c982162e6d83c7d5cc366e59b7abfd674ef408efcc529c9ba13db912ce`;
- the completed fixture report
  `aca358a292ca8cc867241e09abec18889b20b31ccc4111b4fa66b2830ff271ad`;
- the completed server report and authoritative exit/reap review, which are
  **pending while server session27546 runs**. Do not fabricate a successful
  server pin or derive expected producer sources from its untrusted report.

Authenticate the public binding against its externally fixed hash before any
payload or private access. Authenticate the ready proof/archive against the
frozen producer map. Reuse the existing strict `terminal_gate` with that map for
both old phases: their parent/worker source maps, archives, terminal/cleanup/
phase flags, key context, nonce, IO inode, request contract and complete ordered
artifact manifests must still match exactly. Require matching fixture/server
input digests. The validator's own startup/import/end/source-archive checks use
its *different, explicitly overlaid* consumer map. Report both identities.

The request and ciphertext contracts remain unchanged: two fixed L49 inputs,
54 ordered outputs, the original actual level/scale and input digest for every
output. Before client key/private loading, reread and validate both requests and
all 54 response headers/residues/digests against the pinned server result. Keep
that complete readback before the first decode and repeat it afterward. Preserve
the independent oracle hashes, all 65,536 slots per checkpoint, original
`1e-6` checkpoint and derived inverse/residual/invariant/quotient gates, finite
failure reporting, and the original failed normalization proof. No metadata
scale relabel, new iteration, refreshed ciphertext or numerical tolerance is
introduced.

Make the successor CLI/config explicitly validator-only. Existing fixture and
server modes must reject before payload/context work. Prepare it with execution
disabled; after the server terminates, independently verify the final persisted
parent report and all artifacts, bind that report, and request root's separate
validator grant. The collector's stdout hash is not a terminal identity.

## Required preparation checks

1. Run the preserved FD-handoff and body-primary failures against old source,
   then the descriptor and receipt successor suites. Include direct files,
   directory batch reads, NPY header/memmap creation, hashing, stream failures,
   malformed shape/dtype/order/length/digest/residues, all nested close orders,
   uncertain-close no-retry, private write rollback/replacement and initial
   parent snapshot failure. Preserve negative capsules and exact source hashes.
2. Verify all unchanged definition bodies and the allowed source overlays.
   Test missing/extra/stale/wrong-path imports and wrong overlay digests before
   private reads. Check original key-generation provenance and output format.
3. For original stage5, reject a 286-source combined producer and preserve the
   217-source gate. Verify exact default receipt location and no optional-path
   drift. Cover output collision/private-path aliases, failed bootstrap/import,
   outer restoration, completed numeric failure and no retirement.
4. For gain4, use synthetic old-163 reports with a distinct consumer source map:
   exact old source map passes the *producer* gate; substituted consumer hashes,
   omitted/extra producer files, worker/archive mismatch, false terminal flags,
   bool/int aliases, wrong nonce/inode/profile/request/input digest, or any
   missing/duplicate/reordered checkpoint reject before private access. A valid
   consumer archive alone must never authenticate an old producer.
5. Run fresh installed boundary/import probes: no keygen, encrypt, native model,
   server computation or retirement entrypoint can execute; allowed client
   decode remains unchanged. Probe only denied functions, never create keys,
   contexts, ciphertexts or private results in preparation.
6. Freeze complete explicit inventories and archives before final synthetic
   suites. Read back current/execution/archive sources and exact XML counts,
   direct wait/group-empty records, cleanup and outer terminal flags. Preserve
   any failed preparation instead of changing measured bytes.

Measure the expanded source archive before choosing bounds. Keep the old
producer archive verifier's existing exact inventory and size limits. If copied
public predecessor archives make the *new consumer* closure larger, declare a
separate bounded consumer limit with its exact inventory; never broaden the old
producer gate implicitly. Both future roles retain borrowed stores and owned
results. Gain4 validation writes no new ciphertext: reserve its source copies,
reports and a declared disk margin separately. Neither this plan nor the ideal
proof supplies a cryptographic peak-memory or elapsed-time bound. No healthy-job
deadline is introduced.

Source snapshots in `inputs/` and `source_pins.json` make this review portable.
They contain public code/reports only. Final client-v2 preflight and old gain4
server completion remain external prerequisites; this plan does not claim either.
