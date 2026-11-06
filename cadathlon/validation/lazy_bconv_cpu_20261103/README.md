The fixed isolated candidate passed **1146 tests and 93 subtests**,
with **12 skips**, in 370.22 seconds.
All 323 recorded source and fixture files stayed unchanged;
every member of the exact source archive was independently verified. The
second-device delayed-read check also passed in a fresh process.

Ordinary finite-field, NTT and psi lookups no longer build unused cached BConv
controls in deterministic lazy mode. Direct `LevelParams.bconv_params` reads,
the public getter, dataclass helpers and serialization materialize the same
controls on demand. The stored specification contains only immutable public
metadata and the construction device, so retained parameters can outlive their
key-owning cache. Eager and synthetic performance-test generation order stays
unchanged. An AST comparison confirms unchanged control arithmetic/partitions.

A genuine secure N8192 batch-two affine-then-square circuit exactly matched
the eager ciphertext payloads and metadata while cached BConv construction,
secret-dependent key generation and server encryption/decryption were forbidden.
All 8,192 client output slots were checked; maximum error was
3.968195e-08 against the unchanged 2e-6 gate. This covers
that component only; it adds no complete model, refresh or TPU timing claim.

The initial candidate passed 1142 broad tests but failed a separate
pickle reproducer: its deferred JAX Device was not serializable. That version
is retained under `v1_incomplete` and is not promotion-ready. The fixed version
materializes public controls before serialization; protocol 4/5, shallow/deep
copy, direct reads, manual replacement, retry, cache eviction, both reduction
backends and retained-owner lifetime are covered. The original failure and
its fixed rerun remain separate, with exact source hashes.

`candidate.patch` is a two-file follow-up after the archived five-candidate
integration prerequisite. It reconstructs the tested files from archived base
bytes and remains unapplied to the live checkout. No running proof or published
calibration was changed, and no model speedup is inferred from avoiding this work.
