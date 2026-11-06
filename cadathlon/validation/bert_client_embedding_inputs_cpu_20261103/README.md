# Pinned client embeddings and eight-tile packing

The isolated client module passed 45 focused tests in 4.81 seconds, including native float64 `BertEmbeddings` against independent NumPy word, original-position, segment, and LayerNorm arithmetic. Maximum error was 8.89e-16 under a fixed 1e-12 gate. The previously fixed K15 and K128 public examples match the prior native embedding bytes exactly. A third case keeps fixed noncontiguous positions `(0,1,4,7,15,16,31,63,64,95,127)` from the K128 example. No input was selected by activation ranges.

The module accepts only the four-field preprocessed JSONL schema, exact 128-element integer lists, unique nonnegative client target IDs, and valid original CLS token 101 at position zero. It loads only five pinned embedding tensors into the native embedding module in CPU float64 eval mode. Tests prohibit construction of the full model, encoder, pooler, and classifier. Original learned positions are applied before public-mask compaction. All eight fresh 65,536-slot tiles are checked for distinct-row ordering, width-768 values, zero feature padding, zero unused rows, and immutable independent buffers.

`PreparedInput.public_metadata()` constructs a fresh packing/mask/position schema without target IDs or token IDs. The target association remains client-only. This is a client preparation component: no encryption, server evaluation, five-handler integration, full-model output, or ranked accuracy claim is made here.

The collector's three source files and six checkpoint/tokenizer/prior-fixture assets were hashed before and after the run. Every archive member was independently read back. Source archive SHA-256: `2b675675fb04babb7d84e5bf53419a01784c8ac3595ac89b8db702fae418d2cb`. Runtime module SHA-256: `ea026bcadb56281762535daa0168ecaa3abf18df29f20fa81f4e7adb183b1e72`. The successful subprocess was reaped. Independent read-only review found no concrete semantic or identity defect; callers must validate externally supplied public metadata rather than trust manually constructed result objects.

This evidence contains source bytes, public metadata, tensor/file hashes, logs, and test results. It contains no checkpoint tensors, plaintext embedding rows, target mapping, or client NPZ arrays. Local arrays remain outside this archive. The verifier checks provenance and the recorded gates; it does not recompute numerical outputs without those local assets.

Run `python verify.py` from this directory for offline verification.
