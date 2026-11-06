# Unapplied held-parent identity overlay proposal

The original71 writer2e60908c is preserved. Its constructor currently opens the parent, derives an identity from that newly opened object, compares the named parent to it, then creates the child. This provides held-versus-named consistency but does not compare the object to the independently granted parent identity. Checking only in an outer caller leaves a replacement window before this open. No runtime failure or exploit was executed in this review.

## Constructor contract

In a separately reviewed future writer successor, propose one required keyword:

`KeyStore(root, specs, *, expected_parent, private=False, shield=cleanup_signals)`

`expected_parent` is immutable canonical JSON bytes for the existing closed grant parent object: exactly `path`, `device`, `inode`, `mode`. Suggested bound8192B; path is an exact canonical absolute non-root string of at most4096 characters; device is an exact int>=0, inode an exact int>0, mode an exact int in0..4095. Bool must not pass as int. Reject unknown/duplicate fields and noncanonical bytes. No I/O occurs in this parsing step. The actual authenticated launch supplies `canonical(grant['parents']['public_root' or 'private_root'])`; never construct these expectations from observations made inside the writer.

There is no `None` or observe-current fallback in this future constructor. Existing71 recording APIs/tests remain frozen; a future recording adapter may pass explicit synthetic parent bytes for its own files. This signature change must be disclosed in any newly selected tests. Do not silently activate `_generate_for_recording` or alter its synthetic completion schema.

## Exact acquisition and later checks

1. Authenticate the actual ModelAdmission/intent/grant and source closure outside this component, before backend imports, filesystem work or RNG. Require the fresh leaf to be canonical and two-way disjoint from all protected roots. Its existing trusted parent may contain protected siblings such as other `/tmp` trees.
2. Parse and capture the exact immutable expected-parent bytes/tuple and process effective UID. Require expected path to equal `root.parent`. Include these captured values in the store's construction binding; mutable public mirrors cannot redefine the expectation.
3. Resolve/check the existing parent path as currently required, then open the original parent FD with `O_DIRECTORY|O_NOFOLLOW`. **Before the first `mkdir`**, compare both `fstat(original_parent_fd)` and the named-parent `lstat` against the external path/device/inode/mode. Both must be directories and agree on UID. Require UID to be0 or the captured effective UID; if either group or other write bit is set, require the sticky bit. Retain the original admitted UID as well as the expected tuple.
4. Only after these checks create the fresh leaf relative to that same original FD. Keep the existing exclusive child/file creation, modes, held descriptors, source/identity/readback/fsync and uncertain-close semantics.
5. At every later `_directory_check`, repeat the held/named exact parent comparison, original UID equality and trusted UID/sticky policy, before accepting file operations. Capture original parent-check/close callables for final cleanup in the actual orchestration. Final checks must run before retiring the original parent descriptor; a failed check does not skip remaining owned closes.
6. A before-mkdir mismatch raises with no child creation and no RNG. The already acquired parent FD is retired exactly once using the existing shared4543 failure policy. Partial creation failure retains partial owned evidence; no borrowed deletion or uncertain-close retry is introduced.

This proposal changes parent acquisition/binding only. It does not add RNG, change `_NpySink`, the read header path, array/canonical key schemas, cryptographic primitives, original secret operands, capacities or automatic cleanup deletion. Any additional owner-policy expansion must be separately declared rather than inferred from this note.

## Schema and source-pin impact

The pure28 grant already carries exactly path/device/inode/mode for each parent; no grant/report field addition is required. Trusted-UID/sticky enforcement is a source-bound policy, and the observed original UID is an internal lifetime invariant, not a newly invented historical attestation. Existing completion `stores[*].parent` must equal the original external tuple. The separately reviewed actual process supplies the existing `external_parent_identity_checked_before_creation` check truthfully; recording callbacks cannot produce that actual claim.

The future overlay changes `qwen_key_store.py` and therefore the actual generation source map/archive. A separately reviewed authority successor must update **only that fixed writer prerequisite** in `REQUIRED_ACTUAL_SOURCES` (plus separately declared real entry/runner additions), preserving core34, key_gen742e6, producerf852, ModelAdmission60bb and other unchanged dependency pins. The old pure28/39 gate correctly rejects an overlaid writer under its literal2e609 requirement; it must not be weakened or retroactively relabeled.

The new generation map requires a new external launch/prelaunch binding and a ModelAdmission expected-consumer pin for that exact generation map. Content model_id, Qwen key schema, literal166 inventory, core keygen_source_sha256742e6 and future request/response role maps do not change merely because parent acquisition changes. No map/hash for an unimplemented overlay is invented here. `source_pin_impact.json` records the pending change explicitly.

No patch has been applied, no constructor/helper imported or tested, no current parent/store metadata acquired and no actual grant issued by this proposal.
