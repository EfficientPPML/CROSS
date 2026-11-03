# Reusable streaming-linear QKV component

The isolated `encrypted_streaming_linear.py` candidate passed a genuine CPU file-boundary computation of the pinned BERT layer 0 query, key and value projections on two **distinct actual embedding rows**. Every full 768×768 matrix and bias is unchanged. This is a reusable linear component; it does not implement a complete attention layer, full BERT/Qwen model, ranked file-stage submission, MRPC accuracy evaluation or calibration.

The four-file [candidate patch](streaming_linear.patch) adds the common module, its package-manifest entry and two portable test files. It remains a reviewable candidate; this evidence archive does not apply it to production. Its SHA-256 is `1e4eb8b5fbb15a4bd845a3c39dcf1687a1b558656e05fb2224a21fe283651ea5`. Review copies under `reusable_candidate/` use `.py.txt` so evidence is not accidentally collected as live tests.

## Reusable contract

`RowTileLayout` packs consecutive distinct client rows into F4096 feature blocks. It streams tiles without repeating rows, zeroes feature padding and inactive final-tile rows, and reports explicit input/output widths, logical row count, tile count/order and final-tile occupancy. Client unpacking rejects missing/extra tiles and optionally checks decoded padding against a caller-supplied error tolerance. Importing the codec does not import JAX or the CKKS backend.

`StreamingLinear` borrows an initialized batch-one, full-slot, secure CD2 context; it never closes that context, encrypts, decrypts, generates keys or refreshes. The constructor and each input call reject a retained secret key. The ciphertext must be a canonical two-component `Polynomial` with the declared ring degree and one remaining level. Public finite matrices fit within F4096, including 768→768,768→3072 and 3072→768. The module delegates to the existing `MatrixDiagonals`/`CompiledArithmetic` BSGS path. Its public source mask is applied in the source coordinate of every signed diagonal; padding and inactive input rows are excluded even if their ideal values are hostile. Public bias is zero in inactive rows. The ideal circuit consumes one rescale level, with zero output feature/row padding. CKKS error remains subject to actual numerical gates.

`iter_evaluate` retains only its current input tile across a yield. Consumers must exhaust the iterator to establish that no extra trailing tiles exist. Public weights are supplied one matrix at a time; this module does not yet schedule whole-model layers, attention tiles or a vocabulary head.

## Actual encrypted result

All 24,576 physical output slots, including feature padding, passed the unchanged absolute-error gate `1e-6`:

| Projection | Maximum absolute error |
| --- | ---: |
| Query | 3.0063004999e-11 |
| Key | 3.0369303856e-11 |
| Value | 3.0743818937e-11 |

The exact ring is in [summary.json](summary.json): N=16384, 8192 slots, F=4096, CD2, Q4/P3, dnum=2,uniform-ternary secret,default sigma,actual log2(QP)=206.9965209854861, level 1→0. The same 86 public rotation keys and 1535 signed diagonals per dense matrix are used as in the prior standalone QKV proof. There is no ciphertext×ciphertext multiplication or refresh. This run does **not** establish encrypted 3072-width or partial-final-tile accuracy through the new wrapper; those are covered here by independent ideal arithmetic tests.

Trusted native preprocessing verifies the pinned checkpoint/tokenizer/config hashes and exports the complete exact embedding module's first two rows, unchanged full Q/K/V weights/biases, and private native references. It runs separately from three fresh cryptographic subprocesses: client encryption, keyless server, client decryption. The server receives only public model/key files and the encrypted request. Context, nonce, public layout, pinned weights, request digest and output role bindings are checked. The v2 public manifest also binds the reusable module SHA-256 `83e12b9cf00f1393a47aadddffaaa1ff6b4d0745cb68d9393d2dd38a2a165ecb`.

The server has no secret in context, parameters or cache; its client codecs and key-generation entry points are disabled. Python file-access instrumentation recorded no access to the private client subtree. These are inspected local regression guards, not an OS sandbox. The client comparison/decryption process starts only after the server has finalized all three responses and exited.

The corrected collector pins its owned directory with an open descriptor and fails closed if an active child has missing or corrupt cleanup evidence. All four physical children (one preprocessing plus three cryptographic stages) and the collector were reaped; all temporary client secrets, embeddings, public keys and ciphertext files were removed. No such payloads are included in this archive.

## Source and test evidence

The startup [source archive](measured/sources.tar.gz) contains 132 files whose exact bytes were unchanged through the run; SHA-256 `31686c4a0a7fe7e58fd423b2d1006da15cfa08d866e33e2f72182e1a6622316d`. Raw process records, stdout, stderr, public manifests and source hashes are under `measured/`. [archive_candidate.py](archive_candidate.py) independently verified archived/current source bytes, process absence and owned-file removal before copying artifacts.

The pre-encryption suite passed 90 tests in 3.16 s: real delegated BSGS algebra using NumPy primitives, both 8192/65536 slot capacities, distinct and partial rows, all three rectangular widths, hostile padding, signed source masks, dense asymmetric maps, zero-matrix bias, exact tile counts/order, backend-free imports, keyless boundaries, strict transport and collector failure cleanup. The proposed portable repository tests passed 55 tests in 0.97 s against the same runtime after encryption; they overlap the 90-test suite and are not additive.

Initial failures are retained rather than relabeled: the inherited Python-only checkout lacked the declared `Makefile.problem` dependency, and the first independent ideal-test file omitted the checkout root from its import path. Neither failure attempted encryption. Both corrected test suites and the numerical proof are separately source-bound.

## Timing and remaining work

CPU affinity 168–171, explicit `JAX_PLATFORMS=cpu`, OMP/BLAS threads 1. Native preprocessing took 2.6606 s internally (3.4014 s subprocess wall time); three cryptographic subprocesses took 211.7769 s combined; total collector wall time 215.4505 s. Per-projection server observations were 66.9212/64.6426/64.5040 s. These are cold development observations on a shared host, not calibrated performance or a controlled speed comparison. Peak memory was not measured.

Next required work includes streaming attention score/value routing, all-round encrypted full-domain softmax and normalization/nonlinear circuits, their actual refresh periods/bounds/error budgets, complete layer/model scheduling, full output heads and the real file-stage/scoring contracts. The separate full-width and routing component proofs do not establish those complete paths.

## Package and staging closure for local promotion

Apply the [combined six-file patch](combined_streaming_linear.patch) for the complete reviewable change; SHA-256 `c091da61fce22eee606b06bcdaa2f63edd6a5c1ba423ada88561a4ab6d96eeb9`. It is the unchanged four-file module/test candidate plus a two-file test-only follow-up: the existing package test's explicit helper inventory and P2 staged helper inventory/import/tamper checks must include the new module. Both patches pass `git apply --check` on the current live checkout; neither has been applied by this work.

The separate isolated [closure follow-up](closure_followup/summary.json) passed 75 tests in 6.43 s with all 315 captured source/fixture files unchanged. It covers archive extraction, backend-free fixture/preflight imports, exact P2 staged helper bytes, changed/missing helper rejection, and import from the staged helper directory. Its tests overlap the 55 portable tests above. The exact follow-up source archive and logs are preserved. Three auxiliary setup failures are also retained: copying into an existing partial checkout, a missing checkout-marker document, and a missing P3 demo dependency plus accidentally tracked mutable pytest cache. These failures did not change the numerical runtime or attempt encryption.

The [read-only calibration identity check](closure_followup/calibration_source_check.json) verifies all 103 measured P1 files and all 119 measured P3 files still match their recorded bytes. None of the six proposed paths overlaps either measured inventory. This supports preserving the existing matching calibrations; it is not a fresh baseline measurement.

Local promotion review checklist:

- Confirm the common module equals the source hash bound in the actual encrypted run; preserve borrowed-context ownership and per-call secret/ciphertext checks.
- Keep source-coordinate masks and zero inactive-row bias in the delegated one-level BSGS call; the numerical proof is specifically the full 768×768 QKV case.
- Include both package/staging test changes with the module manifest entry. Rebuild any release archive only after promotion.
- Keep the corrected file-component collector and source/public-layout/role bindings when reusing its prototype; raw successful evidence remains tied to its exact archived files.
- Preserve the explicit component-only scope and leave full model, full file stages, unseen-input contracts and calibration requirements unresolved.
