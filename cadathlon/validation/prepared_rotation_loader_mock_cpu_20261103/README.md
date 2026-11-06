# Prepared rotation cache: authenticated-loader wiring candidate

This separate candidate connects the isolated bounded cache to the actual
authenticated key-loader code. It preserves the original key-loader/client
default API. Only `load_server_prepared(..., dedicated_no_write=True)` creates
the opt-in factory. Existing key-file digest/schema/residue verification runs
before a lease is exposed. No live stage4/QK/main files were changed.

The caller explicitly attests sole no-write ownership of the authenticated
producer's public key directory during this dedicated server process. The lease
binds the authenticated manifest/context identity, original readonly map objects,
coefficient maps, held directory identity and per-file inode/size/timestamps.
This does not defeat an arbitrary external writer: a preserved earlier test
showed a rapid same-inode write with unchanged timestamps. Neither READONLY nor
metadata proves immutable contents. The explicit no-write ownership assumption
is required, and native/ctypes arbitrary mutation is outside this contract.

The handle's `prepared_arithmetic(...)` factory creates a fresh owner and records
its original rotation and bound cleanup callables for cleanup. Replacements
fail closed and cannot hide the original pending buffers. Every valid registered
owner's existing cache-clear method runs before
loader context invalidation and mmap closure; then its rotation owner closes and
the source lease retires/drops references. This remains true if the handler
already cleared/closed a rotation owner. Every owner is attempted after a failure.
An uncertain preparation/execution/readiness or caught factory failure prevents
successful retirement, retains the manager and backing in the existing loader
quarantine, rejects future loaders, and preserves the original exception object.
The existing outer single-fire termination policy remains required; this adds
no process-wide asynchronous cancellation guarantee.

The cache itself is the prior reviewed candidate with two small lifecycle
successors: successful lease retirement releases its source references, and a
successfully closed rotation owner accepts idempotent cleanup. The arithmetic,
state conversion, key-switch sequence and default runtime remain unchanged.
Entries are bounded to two and 2GiB by default. No input batching, hoisting,
rescale change, key-generation change or model-layout change is included.
That cap is per registered arithmetic owner; two owners may retain up to4GiB of
logical prepared arrays. The loader does not claim an aggregate2GiB cap. A future
driver must declare/count its owners and their separate budgets explicitly.

The final source-bound suite covers116 checks: the original49 loader regressions,
38 cache checks, and29 integrated lifecycle/provenance cases. It uses the actual
loader's regular-file/manifest/hash/residue verification on tiny files produced
by a fake producer, followed by fake contexts/device arrays. It exercises cache
hits, key replacement, changed metadata, explicit no-write policy, actual clear
hook ordering, all-owner cleanup, caught failure, primary exception identity,
quarantine and no post-retirement reuse. The copied test helper's public profile
paths are localized; its security/backend test is not collected.

This is mocked integration evidence only. No actual JAX backend, CKKS context,
key generation, encrypted rotation or model is executed. A future genuine small
secure comparison still requires separate approval, source-bound inputs and
exact payload/scale/NSD parity gates. There is no measured performance claim.

Historical `source-bound-v1` preserves108 passing checks before the independent
cleanup reviews. `lookup-negative-v1` and `replacement-negative-v1` preserve the
two synthetic counterexamples and exact affected sources. The final successor
fixes both without changing arithmetic or the old proof bytes.
