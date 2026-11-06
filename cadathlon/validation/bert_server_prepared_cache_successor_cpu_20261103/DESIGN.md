# Prepared rotation cache: isolated full-BERT engine successor

This candidate adds two explicitly prepared arithmetic owners to the existing full-BERT engine. The graph and chained-linear owners each have a two-entry, 2 GiB logical LRU cap. Refresh uses its original ordinary evaluator and five compiled helpers. No rotation, polynomial, rescale, level, mask, scale, checkpoint or key inventory is changed. There is no new HE run, model run, speed measurement or physical-memory claim.

The base is the exact 231-source completed prepared-rotation parity closure. Its genuine proof compared 12 ordinary/prepared pairs (24 calls), all 239,075,328 payload words and metadata exactly, at levels49/25. That establishes the scoped rotation representation parity; it does not validate this complete engine. The pending ordinary-group result is not evidence for this candidate.

## Opt-in construction

The existing file handler remains ordinary. Future reviewed wiring must use the authenticated `load_server_prepared(..., dedicated_no_write=True)` loader and pass that same yielded handle into `server_engine(handle.ctx, manifest_bytes=..., prepared_handle=handle)`. The loader authenticates the public source bytes and owns read-only mappings; a dedicated no-write lifetime remains an explicit caller obligation. Read-only flags or inode/stat equality alone do not prove immutable file contents.

`PreparedEngineOwners` checks the actual bound loader factory and common live keyless context. It constructs the graph first and the linear arithmetic second through the registered factory. `StreamingLinear` has one optional constructor seam; its default still constructs ordinary arithmetic, and all nonconstructor method bodies are identical to the tested baseline. No active helper is replaced. Both original owner registrations, rotation objects, and clear/close methods are retained independently of the public list. List mutation, factory/context/map substitution, and budget aliases fail closed. The engine also binds graph/linear identities and the ordinary refresh helper/evaluator objects.

The refresh rotation is captured by accessor closures during refresh construction. This candidate does not change those closures or swap a helper afterward. Doing so would require a separate implementation and proof.

## Retirement and failures

Existing low-memory level transitions and checkpoints still clear each prepared owner. The unchanged prepared rotation waits completed device arrays before publication and barriers every retained entry before clearing. Entries contain explicit arrays, not Polynomial wrappers/registry keys. Context, level, device, representation and authenticated source checks remain in the proved implementation.

The engine attempts all original prepared owners before NTT provider disposal. The outer loader retains its own original registrations and closes them before lease retirement, context/mapping disposal. A sticky readiness failure remains the primary exception ahead of later inventory failures; cleanup-only cancellation still outranks ordinary cleanup errors. Failed readiness or changed ownership keeps original references and forces loader quarantine. A failed entered server stage also triggers the loader's existing quarantine. Normal engine close does not independently retire the loader lease.

This is a dedicated serial evaluator. No outstanding same-input group operation is introduced; ordinary-group/hoisting is not wired. Any future grouping integration must synchronize its own retained source/digit/output arrays before cache and loader retirement.

## Logical storage

For the unchanged N131072/Q100/P14/d8 profile, alpha=13 and active digits=ceil(2(L+1)/13). One prepared entry is two uint32 operands of `digits × N × (2(L+1)+14)` plus an N-element int32 map. At L49 this is956,825,600B; two owners holding two entries each total3,827,302,400B. At L25 the same total is1,109,393,408B. Each map is charged logically per entry even when its physical backing is shared. The public plan covers all50 levels.

The 4GiB combined cap excludes the125,380,363,136B original public NPY store, full backing references retained by views, mmap virtual/resident/page-cache memory, conversion/in-flight arrays, ciphertexts, ordinary refresh/NTT controls, JAX executable/compiler/allocator memory, and other arithmetic controls. It is not a process RSS bound. Any requested index can enter the two-entry LRU;1/64 are not an exclusive key allowlist.

## Schedule and BSGS interaction

Repeated chained baby key1 rotations at one level can reuse a prepared entry. Giant accumulation rotates at a lower level while other operations revisit the input level, so existing low-memory transitions can clear the cache frequently. Different BSGS baby indices can evict one another. No hit rate or end-to-end improvement follows from the cap. Construct prepared arithmetic before BSGS child guards capture owner/helper/accessor identities; active owner replacement correctly fails. This lab does not alter root's separate BSGS integration.

## Validation scope and retained failures

Tests use fabricated contexts, payload owners and tiny public bookkeeping; no real key loading, context construction or HE. They execute the actual constructors, actual low-memory prepare/checkpoint method bodies, and actual loader registration/close methods with stubs. An AST comparison confirms every streamed-linear method other than its constructor is unchanged; source hashes confirm229 of231 parent files are identical.

The original37 engine signal/lifecycle tests run in a fresh process separately from NumPy-importing constructor tests. A combined earlier67-test process recorded66pass/1failure in a process-directed signal-restoration test; that exact failure is preserved. A separate observation shows NumPy created three native worker threads whose signal masks stayed unblocked when the main thread blocked INT/TERM/HUP. This explains a possible delivery route but does not trace that earlier signal. Main-thread masking is not process-wide atomicity, and the passing separate suites do not erase this limitation or establish native-thread cancellation safety for a full engine.

The owner-list mutation failure and the later readiness-primary ordering failure are preserved separately. No original proof, main runtime, live QK source or active217-source engine was edited. Root review is required before any future genuine combined-engine validation.

A separate follow-up executes the actual reviewed single-fire `termination_signals` wrapper after four NumPy/native threads exist. Nine isolated mock-engine samples cover first cancellation during cleanup, first cancellation during handler restoration, and a body cancellation followed by restoration signal. Every sample exits nonzero, attempts all original prepared owner/provider cleanup, and checks final handler/mask restoration. These finite samples do not establish arbitrary native-thread signal atomicity. An initial test-script syntax error was preserved separately.
