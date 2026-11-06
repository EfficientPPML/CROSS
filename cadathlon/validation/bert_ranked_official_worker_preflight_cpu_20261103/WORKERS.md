# Fresh official ranked setup and online worker preparation

This successor preserves every one of the parent's 241 files byte-for-byte,
including all 217 stage4 predecessor dependencies. It adds new wrappers and
public proof gates; it does not change CKKS arithmetic, model coefficients,
transport layouts, file-handler record loops, or original accuracy gates.
No ranked key generation, model preprocessing, ciphertext operation, private
fixture/reference read, or official performance measurement is part of this
preflight. Synthetic tests explicitly replace crypto/model operations. The
installed probes import the real closure and call only denied entrypoints.

## Concrete fresh setup and handoff

`ranked_execution.initialize_attempt(root, authenticated_case)` creates a new
0700 root and 0600 attempt record before any setup. Existing roots and retained
fixture/reference descendants are rejected. The caller pins the returned
manifest SHA256. Original upstream selection must run into this new root;
the worker checks its `test_input.txt` hash against the authenticated case.
It never copies a prepared tokenization or native reference into that path.

The original directory layout is preserved:

- `io/{small,medium,large}/public_keys` and `client_private` hold newly generated
  keys and that attempt's request/receipt ledger.
- `io/server_data/{context_id}` holds that key context's full public model.
- `datasets/{instance}/intermediate/test_input.txt` is the fresh upstream input.
- `io/{instance}/intermediate/client_preprocessed_input`, `ciphertexts_upload`,
  `ciphertexts_download`, decrypted results and predictions use original names.

`ranked_worker.py` and `ranked_collect.py` accept an externally hashed JSON
request. Setup stage1 accepts no key/model fallback, runs the unchanged
`file_stages.client_key_generation`, and checks the generated public/private
binding and permissions. Setup stage2 requires the completed new stage1 proof,
runs unchanged `server_preprocess_model`, and verifies all196 immutable FP64
public tensor lifts and74 matrices. Both publish `cross-bert-ranked-setup-proof-v1`
with exact source/archive/process/terminal and attempt/case binding.

Stage3 authenticates the completed fresh setup chain, calls the exact copied
upstream preprocessing body, checks its fresh128-token output hash, then calls
the unchanged all-N encryption handler. It checks all8N payloads and the private
request/input/order binding. Existing request/receipt ledgers are rejected.
Stage4 requires complete stage3 evidence and receives only public paths; it calls
the full twelve-layer/pooler/head server handler for every ordered record and
checks all N responses plus unchanged original uploads. Stage5 authenticates the
complete setup/stage3/stage4 chain before client reads, decrypts every response,
runs the original postprocessor, validates the all-N private receipt, and applies
`4*HE_correct >= 3*N` and `100*(official_correct-HE_correct) <= 2*N`. Ties select
class1. Logit tolerance and prediction agreement are not additional ranked HE
gates. Original54 fixture files,11 reference files, size0 keys and ledger are
never owned, retired, or rewritten by this successor.

Each collector makes an immutable execution copy and exact source archive.
The subprocess helper owns cancellation, process-group reaping and its own
small temporary key directory; it never owns external attempt keys. There is no
healthy deadline. Unknown process cleanup fails closed and retains artifacts.
Both worker and collector publish completion only after cleanup **and** the
outer termination scope exit; primary cancellation survives secondary cleanup
and publication failures. Public failures omit private exception messages.

## Seven entrypoints and timing

`ranked_entrypoints.py` resolves all seven literal bodies: five complete backend
handlers plus copied upstream preprocess/postprocess. It preserves the server
path filter. This supplies concrete call interfaces for a future source-bound
staged runner; it does not claim that the original harness has been run.

The proof workers combine preprocessing/encryption and decryption/postprocessing
for strict provenance and readback. Their call durations are explicitly
**diagnostics**, not original online subprocess timing. `ranked_timing.py`
independently parses original steps5–9, preserves the harness's rounded strings,
sums all five stages per complete run, then takes the configured repeat median.
The original default is `num_runs=1`. Ranked three-seed medians require separate
fresh invocations for seeds42/43/44; there is no inferred5–9 repetition count.
Official time must include real input preprocessing and original subprocess
execution, never substitute precomputed references, server diagnostic time, or
collector elapsed time. Source/cache/hardware policy and producer identity still
require external authentication before timing records may be accepted.

## Remaining gates before an actual ranked launch

A complete original-harness staged invocation and its seven executable wrappers
must be source-bound to this successor. Fresh setup reports, ordered all-N
artifacts and original timers must then be produced by real execution; preflight
reports cannot stand in for them. Size0 encrypted completion/accuracy and an
explicit storage/memory allocation remain unresolved launch prerequisites.
N100/200/408 still requires800MiB upload per record plus6MiB response per record,
retained concurrently until complete authenticated receipt and acceptance.
The collector reserves full new-stage storage and never deletes prior attempts
to make room. No retained-key functional mode, private-ledger redirection,
inter-record SIMD batching, or reduced model is enabled.

The inherited `README.md` and `run_preflight.py` describe the earlier141-test
contract-only capsule. The new `worker_preflight.py` is the authoritative driver
for this successor. Earlier fixture/import/cleanup negatives remain separate
history; they are not overwritten or counted as passing evidence.
