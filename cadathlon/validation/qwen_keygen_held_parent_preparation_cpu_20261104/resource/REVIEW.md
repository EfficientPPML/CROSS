# Future Qwen key generation: source-derived resource review

This is source/public-metadata analysis, not a capacity acquisition, benchmark, implementation, permission or launch. No candidate/backend/NumPy import, RNG, current store inventory or body read occurred. The admitted71 producer/writer and pure28/39 authority checkpoints remain unchanged. Exact source paths and SHA256 values are in `source_refs.json`.

The conditional arithmetic assumes the original fixed N=131072, Q100/P14, dnum8, all114 moduli below2^31, `CROSS_NP_KEYGEN=1`, no supplied sampler/secret overrides, and the original Python-integer secret passed to every primitive. The genuine `_correct_check` and actual selected security predicates still have to run after a separately admitted grant. An unsupported scalar fallback is outside this budget.

## Live allocations

`key_gen.gen_pke_pair` retains transformed secret, a, e and the growing b lists. During an earlier transform, the old polynomial and new list can overlap. The returned secret has13,107,200 Python integers; its52,428,800B persisted uint32 file is not its live size. Evaluation adds a full squared-secret list. Rotation instead temporarily creates two100×N object-pointer arrays around advanced indexing, with pointers to existing secret integers, plus one permutation list.

`key_switch_gen` retains all eight uint64 a/b parts while constructing both uint32 output arrays. That overlap, not the stored output pair alone, dominates the evaluation phase. `s_qp_eva` holds114 uint64 rows; the last error plane remains live, and a new `np.stack` can overlap its row list. Per-row modular/NTT/sampler temporaries must be included. No caller may retain previous generated pairs after their complete storage/readback.

| Source-derived payload | Bytes |
|---|---:|
| Both sets of eight uint64 switch parts |1,912,602,624|
| Both uint32 switch outputs |956,301,312|
| One114×N uint64 plane |119,537,664|
| All114 native NTT tables, `(5*N-2)*8` each |597,686,496|

The small additional caches are the one-N bit-reversal Python list,114 scalar roots, and one fixed-sigma Gaussian CDF. Root search builds at most N primitive-root candidates transiently. No HEParameterCache or CKKSContext is selected. JAX/NumPy import and native allocator state remain a separate baseline, not an omitted zero-cost component.

For a conservative *conditional live-object allowance*, use C=100*N*(64+16)+4096*(100+1)=1,048,989,696B per Python residue tower: at most64B per integer,16B per list slot and4096B per row/outer container. These deliberately generous interpreter assumptions must be checked against the selected64-bit CPython build before using them in an actual resource policy. With T=597,686,496, U=119,537,664 and256MiB explicit algorithm scratch:

| Phase | Formula | Allowance |
|---|---|---:|
| PKE construction |4C+T+scratch |4.714GiB|
| Evaluation construction/compaction |2C+T+16U+8U+3U+scratch |5.766GiB|
| Rotation construction/compaction |C+2*100*N*8+T+16U+8U+3U+scratch |4.985GiB|
| PKE serialization |3C+T+2*100*N*(8+4)+64MiB |3.843GiB|
| Switch serialization |C+T+8U+64MiB |2.487GiB|

The256MiB scratch allowance deliberately covers transient root/permutation lists, current stack assembly, NTT butterflies and sampling/row arithmetic; the three extra U planes already cover lifted secret, error plane and stack-row overlap. The formulas overcount some mutually exclusive allocations. The8GiB rounded working-object allowance is conditional, not a proven process RSS ceiling. Python/NumPy allocator arenas, native fragmentation, import state and concurrent host activity cannot be bounded by array shapes alone.

The writer retains values only during one write and clears them after its seal/readback. PKE lists need transient int64 `np.asarray` and uint32 conversion; native switch results already have uint32 dtype. The pinned NumPy `write_array` uses16MiB iterator chunks and materialized chunk bytes; `_NpySink` itself holds no pending byte buffer. Readback uses at most one1MiB byte chunk (a real limb is512KiB), a view and a bounded comparison. All335 outputs are retained on storage and as descriptor/hash metadata, not as335 in-memory arrays. On failure, quarantine may retain the current call's objects/traceback until process exit; no automatic retry or continued key accumulation is allowed.

## Recommended future admission policy

One conservative policy proposal is32GiB additional working headroom plus8GiB host reserve **after** measuring the admitted CPU-import baseline. This is a4x margin over the conditional8GiB object allowance, not a measured peak or formal RSS guarantee. Bind the exact interpreter/dependencies, native branch, import baseline and remaining process/cgroup headroom. MemAvailable after imports already reflects the resident baseline; do not count it twice. No RLIMIT_AS, forced memory reservation or current-capacity fact is proposed here.

Storage must remain a separate budget. For disk-backed roots, require the unchanged160,935,772,160B shared-filesystem allowance plus evidence space; clean file page cache is reclaimable after the existing fsync path. **For tmpfs roots, retained key files also consume memory/cgroup capacity:** add approximately149.883GiB, making this proposed additional memory-plus-reserve requirement approximately189.883GiB. Root placement and shared-device relationships have not been inspected. Quotas, filesystem metadata, inodes and concurrent consumption still require actual admission.

Descriptor admission is measured baseline +342 retained +8 existing temporary allowance + a separately justified runtime/evidence margin. Soft limit350 alone is inadequate. The old logical disk/descriptor facts are neither RAM evidence nor a resource grant.

The actual entry must verify the plan's source/import/security gates before RNG and preserve complete readback/close plus independent normal-zero/reap/root acceptance. Scientific calibration/freeze and downstream role readiness remain separate.

`HELD_PARENT_PROPOSAL.md` describes the minimal remaining parent-identity acquisition overlay. It is an unapplied API/ordering proposal only.
