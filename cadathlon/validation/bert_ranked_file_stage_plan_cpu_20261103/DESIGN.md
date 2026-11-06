# Ranked BERT file-stage orchestration proposal

This is a public-source design, not an implementation or launch authorization.
No private fixture/reference bodies, key arrays, ciphertext payloads or model
assets were opened. No model, JAX, CKKS context or encrypted operation ran.
The current size0 ledger, retained keys, 54 original input files and 11 private
reference files remain unchanged. Source snapshots are in `sources/` with
original paths/digests in `source_origins.json`.

## Existing capability and exact gaps

The isolated `file_stages.py` already accepts size1/2/3 with100/200/408 records.
Its input handler reads one ordered batch, constructs original-position
embeddings before compaction, encrypts eight fresh L49 ciphertexts per record,
then publishes the complete batch manifest and private request binding. The
server loads one key context/model/engine per batch and serially calls the real
12-layer encoder plus pooler/classifier for each ordered request. The client
validates the response list, decrypts N responses and writes N two-logit records.
Here “batch” means N file requests; ciphertext batch dimension remains1. There
is no cross-record SIMD, deduplication or reuse of an encrypted output.

Public batch contracts already cover100/200/408 synthetically. The historical
five-handler lifecycle tests are size0 mocks; size0 cryptographic stages and
per-record component checks do not establish that these batched handlers work.
In particular, repeated graph construction on a shared engine, complete-batch
publication, cache release between records and large-file lifecycle need their
own evidence.

The current stage3 gate fixes one record, K32, eight input tiles and a scalar
input digest. Stage4 readback and stage5 gate/worker/readback also fix one nonce,
one response and one target, often indexing `[0]`. Their hardcoded source
inventories bind those exact size0 producers. The stage5 `ready` flag authenticates
reviewed code only; it is not a completed stage4 computation or decryption proof.

The main P2 staged adapter still must be wired to this actual isolated handler
closure. Copying the current P2 starter or passing synthetic runner checks does
not select the full model. Preserve the seven upstream entrypoint names and
CLI `<size>`; add an explicitly packaged backend adapter and complete helper/
model-profile closure with provenance. No plaintext-oracle forwarding, missing
hook substitution or silently selected experimental default is acceptable.

## Inputs, reference gates and ordering

The externally pinned completed fixture report is
`ad1458f5183c6ccf2631c2a8771d0da92196f5f9c690016e2a0b12fa7b88f4d7`;
the completed reference report is
`77f93c90e03609bdaf12441cc8a8d07e08b5e1a475c1480f283b4f592668ae74`.
The fixed schedule is each size100/200/408 with seeds42/43/44 (2124 case records).
Authenticate all nine public entries/source archives/assets before granting
access to any private case. Then select by exact public `(size,seed,count)` and
authenticate its owned original input bytes, target-order hash and corresponding
private reference file. The three408 cases have identical original order but
must still be freshly encrypted and executed three times. The reference
producer's reuse of408 unique native examples is not an encrypted-work shortcut.

An immutable `RankedCase` descriptor should bind fixture/reference report hashes,
size/seed/count, input-file/order hashes, pinned model revision/assets/candidate,
and a new opaque attempt ID. Server proofs receive only public case/count/context/
model/request digests and authorized public masks/positions, never target IDs,
labels, original tokens or private references. A client attestation of private
binding is not permission for server gates to reopen those files.

Replace scalar proof fields with exactly N ordered records. Every record binds
its nonce, eight tile indices0..7, ordered transport-manifest digest, actual Q100
prefix/scale/NSD/shape, and one L2 response digest/independently accepted actual
scale. Reject missing/extra/duplicate/reordered records, a bad final record,
cross-seed/case/context response, and unrecognized inventory. Canonical aggregate
digests commit order; dictionary/set membership alone does not. Source/archive/
worker/report links, observed exit/reaping, terminal/cleanup success and all
producer phase prohibitions remain mandatory.

The client must require exactly N unique original targets and finite real
two-logit results, restore the original order, and validate emitted binary
predictions. Class0 iff logit0>logit1; an exact tie selects class1. Official ranked
acceptance is `4*HE_correct >= 3*N` and
`100*(official_correct-HE_correct) <= 2*N`. Native prediction agreement is a
diagnostic. The reference producer's logit tolerance, strict prediction agreement
and exact FP64/NumPy parity remain separate diagnostics; do not impose size0's
same-prediction/logit screen as a new ranked rule. The unchanged strict
`check_result.py` still runs after each official harness invocation. Harness
exit0 alone does not enforce all count/quality requirements.

The reference report's complete publication/retention/error-absence/source/
asset/input/cleanup/terminal flags must all pass before private references are
accepted. All11 private files and their original inode/hash/mode bindings remain
read-only. Raw logits/labels/IDs stay0600 beneath0700; publish only aggregate
diagnostics and hashes. Preserve an explicit distinction between official
reference availability and optional approximation diagnostics.

## Two modes that must remain distinct

**Official default:** use nine separate fresh staged run roots and unchanged
`python3 harness/run_submission.py SIZE --num_runs 1 --seed SEED`. Provision the
exact offline model/tokenizer/dataset identities first; upstream unpinned `main`
must resolve to the approved checkpoint. The original harness starts by deleting
`io/{instance}`, then generates fresh keys and context-bound model preprocessing.
It must never run inside the original fixture root or point that root at retained
keys, the live size0 ledger, or original reference files. Authenticate generated
selection/tokenization bytes against the corresponding prepared case; rerunning
these timed input stages cannot be replaced by copying a prepared output. Pin
the interpreter that upstream resolves as literal `python3` and actual device.
The unchanged cleartext step still runs outside the online boundary; cached
private references strengthen checking but do not replace an official stage.

**Optional retained-key functional batch proof:** a separate successor may reuse
the completed key/model store under its authenticated no-write ownership lease.
It must introduce an independently owned0700 request-state directory. Existing
`client_private` remains the immutable SK/manifest location; a new explicit
`client_request_state` defaults to old behavior for source compatibility but
ranked functional adapters must supply a distinct path. Only requests.json and
completed.json move there. `file_stages`, request_lifecycle and client path
validation need coordinated changes; the server path allowlist must omit it.
An exclusive identity-bound lock protects the whole encode→compute→decode→archive
transaction. Do not clone or mutate SK manifests, symlink ledger files, or share
the current size0 ledger. This is not the unchanged fresh-key harness mode and
earns no official timing claim without an explicit common organizer cache policy.

## Timing and acceptance lifecycle

`--num_runs` defaults to1 in the pinned harness. Steps5–9 are five stage numbers,
not five-to-nine repetitions. The ranked handout asks three fresh one-run
invocations per size with published seeds42/43/44. For each accepted invocation,
sum its original `Timing` fields Input preprocessing, Input encryption, Encrypted
computation, Result decryption and Result postprocessing. Take the median of
those three sums; throughput is N/median. Do not sum stagewise medians, average
per-record times, use server-reported timers or use harness Total. Preserve
original four-decimal seconds strings and auxiliary monotonic outer/process-tree
evidence separately. Original `datetime` stage deltas also include intervening
size-reporting/dispatch work; do not silently subtract it.

`--num_runs>1` is a separate mode: it shares setup and reinitializes RNG from the
same base seed each iteration, so selection repeats. It is not the nine-case
fresh-invocation recipe. Reject mixed repeat/cache/source/hardware policies in a
median. Archive every failed attempt and keep the predeclared case schedule;
never silently replace failures or select the fastest successful suffix.

Setup, data generation, native reference, communication, device peak and total
process-tree RSS are separate. Handout targets are up to8 TPUv6e chips,512GiB
aggregate RSS,1TiB working storage,2h setup/configuration and24h online/instance,
subject to organizer calibration. CPU component passes confer no TPU credit.
Actual hardware, affinity, concurrency and cache policy must be recorded. This
proposal does not impose those future limits on any healthy currently running
component job.

Completion has separate states: authenticated prerequisites → encode committed →
all N server responses committed → decrypt/receipt committed → postprocess and
strict/ranked checks passed → public evidence/private results archived → eligible
for explicit retirement. A completion receipt is currently issued after decryption,
before postprocessing/accuracy checks; it does not mean a ranked attempt passed.
The orchestrator must prevent another encode from triggering automatic retirement
until evidence/checking is complete. Failed or uncertain attempts retain their
owned artifacts; no new batch starts if this leaves insufficient capacity.

Receipt validation already supports up to408 nonces and covers each8-tile upload,
response, manifest, server report, decrypted file and client request mapping using
inode/metadata/hash checks. Extend tests to N=408, last/nested replacement,
same-inode edits, partial retirement retry, concurrent state reuse and uncertain
descriptor closure. Never delete foreign replacements, original fixture/reference
roots, active borrowed uploads, shared keys or public model caches. Final-case
retirement must be an explicit owned action, not a fabricated next request.

## Storage and execution plan

At N131072/Q100/CD2/uint32 one L49 input is100MiB; eight inputs consume800MiB per
record even when K<128. Each L2/Q6 response is6MiB. Current stage boundaries
require the complete upload before server invocation and retain it through
client verification; streaming internal compute does not eliminate this storage.

| Records | Upload payload GiB | Response payload GiB | Combined payload GiB | Header/reserve planning GiB |
| --- | ---: | ---: | ---: | ---: |
|100|78.125|0.5859375|78.7109375|80.765869140625|
|200|156.25|1.171875|157.421875|159.53173828125|
|408|318.75|2.390625|321.140625|323.36474609375|

Last column adds current64KiB/container allowances and two1GiB stage reserves;
it is not measured filesystem overhead or a reservation. There are27N encrypted
container files,8N request ciphertexts and N response ciphertexts. Keeping all
nine uploads requires1659.375GiB plus12.4453125GiB responses, exceeding the1TiB
work-storage target even before keys. Execute and retire one fully archived
successful case at a time; stop safely on retained failure/capacity exhaustion.

The existing public key store occupies125,380,363,136 NPY bytes separately
(roughly116.77GiB), plus private material, model cache and runtime buffers.
Nine official fresh contexts cannot all be retained in constrained storage;
archive public metadata/source proof and retire only each new owned context
after terminal client verification. The original retained context is excluded.
Per-record execution retains eight inputs plus graph/keys/refresh temporaries;
disk accounting cannot predict device/host peak. Full public keys and formatted
runtime states may dominate RAM; no estimate here certifies512GiB compliance.

At this audit, `/tmp` is a roughly96.7GiB root filesystem with about53GiB free:
not enough even for100 inputs. `/dev/shm` may have nominal room for one408 upload,
but is shared RAM, currently occupied by live work; free bytes are a transient
observation, not permission or a capacity guarantee. Require separate approved
storage or a fully checked resource lease; do not steal live key/tmpfs capacity.

Synthetic maximal-position public batches serialize below1.50MB at408; response
batch≈105KB and private request mapping≈1.50MB. The actual ordered proof report,
completion receipt and private reference bounds still need full-count tests;
do not inherit the one-reference64KiB bound. No payload files were created here.

## Minimal implementation and review sequence

1. New isolated `ranked_case_contract`/`ranked_reference_gate` authenticate all nine
   existing producer reports, exact case/asset/source identities and complete
   private publication before any authorized private read. Add immutable per-case
   bindings, full-count bounded readbacks and threshold/order regressions.
2. Generalize stage3/4/5 workers, gates and readbacks to one complete named ranked
   case: ordered arrays/aggregate digests,8N request and N response counts,
   terminal/source/reaping/error gates, stage-specific forbidden entrypoints.
   Copy reviewed terminal/cleanup/engine/key-context successors exactly. Keep
   runtime arithmetic/candidate/profile unchanged; preserve all size0 archives.
3. Add batched mock five-handler tests for100/200/408, varied K/positions, last
   record faults and lifecycle failures. Verify per-record real graph calls and
   output objects in order; no native or surrogate graph in any actual worker.
   Add source-closed installed-boundary probes before real payload/context use.
4. Wire an explicitly named full-model seven-entrypoint staged backend plus
   verified closure into a fresh upstream staging tree. Test exact unchanged
   harness hashes and literal-interpreter/assets/device resolution. Keep the
   retained-key functional adapter, if built, separately named and reported.
5. Add an outer case scheduler/timing/acceptance collector: immutable execution
   snapshot, exclusive workspace ownership, process groups, graceful cancellation,
   descendant reaping, primary-error-preserving attempt-all cleanup and terminal
   publication. Archive raw measurements/checker results immediately; expose no
   private values in logs/errors. Preserve partial/failed attempts and originals.
6. After review and explicit resource/hardware authorization, run actual full
   batch handlers. Only those results can establish ranked HE correctness. Only
   the nine complete official invocations under calibrated matching conditions
   can establish per-size median timing. No launch is proposed by this artifact.

Decisions before implementation/launch: official fresh-key mode versus a labelled
functional retained-key phase; approved storage and failure-retention budget;
explicit baseline/cache/device policy; completed size0 stage4/5 status; and review
of final batched source closure/negative tests. These are not requests to alter
current frozen jobs or to treat existing ideal/component evidence as ranked HE.
