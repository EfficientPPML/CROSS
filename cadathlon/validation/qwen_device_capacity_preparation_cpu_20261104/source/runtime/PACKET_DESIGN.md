# Pure server public packet adapter

This source-only subset adds `qwen_server_authority_packet.py` around exact scientific20, the snapshot of root-owned server-phase source a096e5, and three unchanged helper ASTs from client464. It does not select a complete actual server runtime, activate a role, read proof paths, inspect capacity, or claim any scientific/HE result. No actual proof bundle, policy instance, grant, test or backend execution is included.

## Preliminary inspection

`inspect_packet(public, raw, *, expected_sha256)` authenticates a canonical compact JSON-plus-newline packet of at most16MiB against an independently supplied digest. Packet expected-pin bytes inherit that external authority. The packet has closed fields and only bounded public UTF-8 strings for generation/model/grant/policy evidence. Scientific statement and ExpectedScientificPins are small strings; scientific reports are never embedded as escaped large strings.

The exact25 `scientific_proofs` mapping entries are descriptors `{name,path,sha256,limit}`. Names and per-name limits are the selected scientific contract's exact inventory. Paths are canonical absolute strings, never accessed by this module; hashes equal the externally pinned scientific evidence fields. The same path may represent two roles only with identical hash and limit. All original proof bytes must later be supplied unchanged.

Inspection validates exact phase source-map/archive/grant joins, request/upload/input bindings, scientific statement and source joins, and independently pinned policy hashes. It also authenticates the small actual generation ModelAdmission, launch/readiness and server ModelAdmission before returning any descriptor. Both consumers require the same four actual model body-validation proof hashes. Generated-key identity must match the request public identity. The model/store digest and generated public-key store are checked against the phase grant.

The returned frozen `ServerPacketInspection` retains original packet/expectation and immutable descriptors. It exposes `source_root`, `model_roots`, both public/private `key_roots`, `request_root`, `output_roots`, and independently pinned dependency/role/resource/review bytes. These path scopes are derived from authenticated public evidence, not guessed from suffixes. Proof paths overlapping these roots are rejected lexically. `check()` reconstructs inspection and compares all retained fields. `forbidden_proof_roots` returns the complete derived tuple after reauthentication. `proof_reads_authorized` and `execution_authorized` remain false.

The separate actual process must compare source/archive/entry bindings, admit an independently reviewed metadata-memory policy BEFORE the25 proof reads, and enforce lexical/resolved path separation against all returned roots PLUS static private/token/calibration roots, installed libraries, and its own source/output scopes. It must authenticate bounded regular-file reads, preserve original held descriptor ownership and postchecks, and pass the original bytes. A suffix denial alone is insufficient. The209453056-byte proof parser ceiling is not a sufficient memory policy: parsing, retained evidence, reauthentication and packet overhead require a separately justified actual allowance. This module deliberately does not invent that allowance.

## Full authentication

`authenticate_packet(public, raw, *, expected_sha256, scientific_proofs)` requires an exact dict of25 original immutable byte strings. Each hash and length is checked against the preliminary descriptor before scientific parsing. It constructs exact ModelAdmission and GeneratedKeyReadiness, then ScientificAdmission, then ServerPhaseAdmission. Scientific v2 requires explicit post-client embedding provenance and its nested request-client terminal evidence hashes; this draft supplies no such evidence.

`ServerPacket` retains an immutable sorted tuple of original proof bytes, the packet, selected policies and exact phase. `check()` reauthenticates the packet and supplied proofs, validates the retained phase, and compares its full public/model/generation/scientific byte identities with the reconstructed phase. No expected pins are derived from those outcomes. The phase, packet and immediate-deny constructor confer no filesystem/backend/process authority.

## Selection and limits

`packet_selection.json` records exact scientific parent and the developmental server-phase snapshot. `packet_import_closure.json` supplies the complete local pure import list for bootstrap selection. Historical source copies remain unimported. `_bytes`, `_model` and `public_from_sources` match their original client helper ASTs exactly.

`packet_schema.json` is a schema/field/bound description only. No successful packet, synthetic proof instance, actual grant, model/key manifest body, or private evidence is present. Source AST parsing, hashes, copy comparisons and tar inspection are the only preparation checks performed.
