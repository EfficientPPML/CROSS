# Remaining Qwen client input/output glue

This is a source-only plan. Existing reader43, publisher28, file adapter238,
layout109 and composition328 proofs remain immutable. The new selection337
introduces explicit parser/finalizer selection but supplies no actual role grant.
No selected modules, tests, backend contexts, or actual model/token/embedding/
key/ciphertext/private bodies are executed or read by this plan.

## Selected interfaces and public admission

Keep the exact166/335 schema and concrete readers from reader43. Use the selected
`qwen_public_parse` parser, `qwen_finalizer` cleanup policy, and exact shared
Toy4v4 `cleanup_origin`4543b26b. Do not fall back to the older `Finalizer` through
`reused_client_io` or a predecessor import root. Source pins are recorded beside
this plan; the four-overlay selection review does not establish a complete
role-level integration or an actual startup proof.

Both client stages take a **full `FilesAdmission` in client role**, not merely
the structural `Admission`. Before any actual input or key body read, the future
entry authenticates its external source/grant and public proof bytes, checks
`FilesAdmission.check()`, and compares its identity with the exact public request.
Required independent identities include context ID, public-manifest digest,
private-manifest digest, expected key-generation source, model/checkpoint/candidate
pins, and producer/consumer source identities. Neither self-consistent documents
nor a directly constructed dataclass provide those external expectations.

Public input/reference/model metadata does not authorize reading the underlying
bodies. Original embeddings must arrive through the separately admitted original
client preprocessing/embedding boundary. This glue performs no tokenizer/model
lookup, substitution of selected hidden states, or coefficient selection.

Suggested internal future APIs are a request-publication function and a
response-consumption function with explicit full client admission, original
request bytes, public/private key paths and immutable source-bound dependencies.
The public factories stay immediate-denial until separately reviewed active
source/producer/resource/role gates exist. No implementation is added here.

## Client input transaction

1. Validate the full client admission and exact public identity before the input
   reader, key reader or output creation. Reconstruct the request through the
   selected `prepare_client_embeddings(public, identity, request_id, embeddings,
   positions)` interface. Treat externally constructed `PreparedEmbeddings` as
   data to revalidate, not as a capability.
2. Require exact finite float64 `(1,S,1024)` original embeddings, `S=1..32`, and
   an exact tuple of strictly increasing original positions in `[0,40960)`.
   Physical rows use token order, not the position number. Keep positions in the
   immutable request separately. The reviewed layout places16 rows per tile,
   row pitch4096, active1024 features, zero feature/unused-row padding and no
   replication: one tile for S<=16, two for S>=17. Retain independent immutable
   full65536-slot vectors; do not normalize or alter them in this stage.
3. Enter one client loader scope. It authenticates every335 public key body and
   the private secret under the exact existing schema; it keeps PKE+secret alive
   and forbids key generation. No server loader or second client scope overlaps.
4. For each ordered tile invoke the **default selected** context facade
   `ctx.encrypt_slots(tile, scale=DELTA)`. Use separate batch-one calls rather
   than changing the transport to a ciphertext batch of two. Capture each
   returned wrapper and original payload/moduli arrays immediately, synchronize,
   then check exact `[1,2,256,512,100]`, ordered full Q100, layout256×512,
   precision32, NSD1, fresh level49 and exact DELTA against the request binding.
   The genuine49→25 prefix step remains exclusively the server preparation.
5. Through a fresh owned publisher write ordered `tile00.npz` and, if needed,
   `tile01.npz`. Its artifact bindings must exactly match the request, context,
   sources, positions and payload hashes. Call original `finish()` only after
   all tiles. Reauthenticate the returned canonical upload manifest and input
   digest; close the original publisher before key-loader exit.
6. Synchronize and retire all retained client ciphertext consumers while the
   context/key owner remains live, then exit the loader. Only afterward can a
   separately implemented producer publish a terminal report. The upload
   manifest alone does not certify final close, source checks or process exit.

The future stage must capture original loader/publisher objects and close/finish/
write callables immediately, maintain an independent retained-array inventory,
and check the returned context/handle after callbacks, finish and cleanup. This
matches the server ownership repair being prepared separately; mutable wrapper
attributes must not redirect retirement. A failed encryption may have pending
work without returning an array: propagate that failure to the loader, retain the
whole owner graph and require process retirement rather than pretending all work
was synchronized.

## Client output transaction

1. Authenticate exact independently pinned producer report and outer receipt
   bytes against the original request and input digest. Existing
   `ProducerTerminal.authenticate` requires all38 outputs, complete body/source/
   parameter/cleanup/termination flags, exit0, authoritative reap and observed
   group absence. A response manifest or caller-generated receipt is insufficient.
2. Construct and fully authenticate `ResponseFiles` **before private loader
   invocation**. It binds the canonical response, exact `response.json` plus38
   ordered block files, hashes, sizes, exact six-field NPZ containers, dtype/
   residues, same request/context/model/sources/input digest, original last
   position and each actual positive finite scale. All38 are L2 with the
   original six-Q prefix and physical `[1,2,256,512,6]`; block37 carries the final
   384 vocabulary entries. Reject any missing/extra/reordered file before keys.
   Recheck held identities/content around private consumption.
3. Validate the separate **full client `FilesAdmission`** against that same
   response/request identity and enter a fresh, nonoverlapping client loader.
   `ResponseFiles.run_client` currently accepts trusted callbacks, so the future
   stage must bind its private-loader callback to this admission and source-bound
   key paths; public schema admission alone must not choose private paths.
4. Materialize each immutable admitted uint32 payload with the literal default
   Polynomial constructor and metadata assignment already used in the selected
   transport/composition. Do not cast malformed data, relabel a level, normalize
   scale, or insert an independent computation-format conversion. The exact
   admitted backend/profile must agree across producer and consumer. Retain the
   original wrapper/payload/moduli arrays through barriers and decode.
5. Use the default **bc978cb7… `CKKSContext.decrypt_slots` facade**, with the
   authenticated actual scale and `slots_to_decode=65536`. Preserve its default
   `validate_approximation=True`; do not disable the existing check or substitute
   the optional9675 decoder proposal. The facade produces one real float64 slot
   vector. Require exact ndarray/float64/shape65536 and finiteness of **every
   physical slot**, including padding, before reading any vocabulary slice.
6. Copy each complete decoded vector into independent immutable bytes **before
   the next decoder call**, as existing `run_client` does. A decoder may reuse
   its scratch array; slicing first or copying only the selected aliases is not
   equivalent. After all38 snapshots, use exact ordered descriptors and
   `read_vocabulary`: concatenate37×4096+384 values from physical row0 into an
   immutable `(1,151936)` array. No alias averaging, argmax-only shortcut, new
   tolerance or new accuracy gate is introduced.
7. Retire materialized ciphertext/decoder consumers before exiting the private
   loader; close the original response owner after private scope retirement.
   If keeping existing `run_client`, its `private_loader` should be a composite
   scope that owns these consumers and retires them before its nested key scope
   exits. This closes the ordering seam: `run_client` itself controls loader exit
   and cannot infer that an arbitrary decode callback has released device work.
   Any decode, readback or retirement failure must propagate into the key scope
   and preserve the original owner graph; never return partial logits as success.

## Schema and ownership decisions still required

No change to candidate, profile,166 keys,335 descriptors,38-block vocabulary,
positions, levels, coefficients or numerical policy is needed for this glue.
It must select the Qwen schemas; BERT's130-key/eight-input/one-output transport
and BERT-specific stage receipts are incompatible. Literal65535 and262143 keys
remain distinct. The private key directory remains exactly `secret.npy` plus
`private_manifest.json`: do not add or reuse BERT request/completion ledgers there.
Any owned client request/result evidence belongs in separate disjoint roots and
needs an explicit schema; no old ledger behavior is silently inherited.

The public identity currently has one producer-source and one consumer-source
hash. Using one reviewed client source closure for both input and output stages
fits it. If future input/output executables have different source closures, bind
both explicitly in a reviewed successor receipt or common client closure; do not
overload an existing field or silently change expected IDs. The actual producer
of new Qwen keys also remains a prerequisite; key-generation source identity
alone proves no production or retained-key provenance.

The stage result can report a canonical upload manifest or complete immutable
logits only after its own lifecycle succeeds. Actual process startup, externally
pinned reports, source rechecks, owned-output publication, exit/reap/group proof,
and any required numerical evaluation remain separate concrete work. Existing
public/recording proofs do not establish joined real codecs, full-size key-body
loading, security registration, native signal behavior, device memory/latency,
or whole-model correctness. No actual grant follows from this plan.
