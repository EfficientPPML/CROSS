# Original embedding placement and full vocabulary readout

Pure NumPy helpers for the reviewed Qwen batch-one file contract. This isolated
tree changes no frozen runtime, arithmetic, normalization, polynomial, refresh,
decoder, or numerical acceptance rule. It performs no I/O, model work or HE.

`place_embeddings(embeddings, positions)` requires an exact float64 ndarray of
shape `(1,S,1024)`, S1–32, and an exact tuple of strictly increasing original
integer positions in `[0,40960)`. It returns one or two immutable independent
float64 `(65536,)` arrays: sixteen rows at pitch4096, each token in its ordinal
row's first1024 features. Remaining features and unused rows are zero. Positions
remain separate metadata; noncontiguous positions do not insert physical rows.

`read_vocabulary(decoded_blocks, descriptors)` requires a tuple of38 exact
float64 `(65536,)` arrays and a tuple of38 descriptor dictionaries exactly
matching the file contract's index/start/stop/physical_row/level fields. Every
slot must be finite. It returns immutable independent float64 `(1,151936)`:
37 complete4096-feature slices and384 final features, all from physicalrow0.
It never averages aliases or reads other physical rows into the logits.

The reviewed `decrypt_slots` facade returns a float64 vector; the lower-level
batch decoder returns `(batch,slots)` and requires explicit caller indexing.
No implicit squeeze or conversion is added here. Source/request/actual-scale
authentication, original position metadata persistence, and decoder guards
remain caller obligations. The helper does not introduce a closeness threshold.

Output storage is new immutable `bytes` with NumPy views. Thus `OWNDATA` is
false, but outputs do not alias caller storage and cannot be made writable.
No concurrent input mutation or arbitrary Python monkeypatch guarantee is made.

`provenance/` contains only exact public source copies, read as source text in
tests; these modules are never imported. `source_provenance.json` records their
origins and hashes. Focused tests use synthetic arrays only, including all32
lengths, independent slot indices, noncontiguous positions/views, immutable
ownership, malformed schemas, nonfinite padding and the partial final block.
No actual tokens, embeddings, model weights, keys or ciphertext bodies are read.
