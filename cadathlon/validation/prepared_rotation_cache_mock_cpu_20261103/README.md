# Bounded prepared rotation operands — isolated proposal

This candidate keeps at most two completed immutable device key states, with a
2 GiB logical-array budget. It changes no key-switch kernel, coefficient-map
order, rescale, plaintext weight, ciphertext layout, or gate. The normal
`CompiledRotation` remains byte-identical in `base/`; the new class is an
explicit opt-in. `create_prepared_arithmetic` constructs a fresh compiled owner
and uses its existing clear hook, which clears rotations before other controls.
It never replaces a working owner or modifies a global runtime class.

The source-only review at `/tmp/bert-runtime-cost-review-20261103/report.json`
finds 830,976 attention rotations among the ideal K128 graph's 906,829, excluding
bootstrap internals. Dense chained baby rotations use the same key 63 times.
The baseline drops formatted keys after each rotation and prepares them again
on a compiled-binding hit. One common Q100/P14/d8 prepared pair plus coefficient
map is 956,825,600 bytes; two total 1,913,651,200 bytes (1.782 GiB). The entry budget
excludes authenticated host maps, compiler controls, allocator backing and
conversion temporaries. Completed entries are evicted before a new conversion.
No wall-time benefit has been measured. The older direct QK projection mostly
uses distinct keys, so this proposal does not explain its current long runtime.

## Trust and lifecycle boundary

The source lease binds the actual context/cache, ordered Q/P, partition/layout,
field conversion implementation, raw key/map identities and metadata, device,
and public context identity. It requires a caller-owned authenticated backing
guard. That guard must retain the original public key mappings, verify held
file/directory identities and state, enforce the dedicated loader's no-write
ownership contract, and reject retirement. **The real loader does not yet expose
this API; real-loader wiring and review remain required before a genuine run.**
The cache itself neither authenticates files nor infers immutability from an
array identity. Writable/custom host backing is never cached.

The first development test preserved in `development-v1/` observed a same-inode
write with unchanged nanosecond timestamps. Thus inode/stat checks alone do not
prove unchanged contents. A readonly mmap still requires the explicit no-write
ownership/lifetime assumption. This is a protected dedicated-process contract,
not protection against arbitrary native writes or an adversarial file writer.

Entries retain explicit arrays, never Polynomial wrappers. Publication follows
result readiness and an explicit key-array barrier; source bindings are checked
before and afterward. Clear/eviction barriers precede release. An uncertain
preparation, execution or late-readiness failure quarantines the owner, source
references and traceback until process exit, rejects reuse, and prevents a
successful clear. A source validation failure preserves its primary exception
if cleanup also fails. The loader must not retire mappings after failed cleanup.
The existing loader's exception/quarantine policy must be connected and tested
with the new lease before any encrypted claim.

`source-bound-v1` runs 38 mocked checks, including exact output/metadata parity
against the unchanged baseline through lightweight payload stubs; it does not
establish genuine CKKS numerical parity. Tests execute the actual unchanged
`CompiledArithmetic` class's clear method with a fake superclass to verify
retirement ordering. No backend, model, key or ciphertext is constructed.

No live source or active QK/stage4 dependency was edited. Separate future work
may assess exact-order decomposition reuse, but existing QP hoisting cannot be
substituted without checking its ModDown placement and residue parity.
