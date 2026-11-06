# Aggregate Qwen model-stage admission plan

This is a public-source and metadata proposal for reusing the existing Qwen exports. No parameter/model/token/key/ciphertext body was opened, no candidate module or backend was imported, and no test, model forward, re-export or actual role was run. The existing366 file-engine sources and every historical export remain unchanged. Root review precedes implementation and any separately granted current-store validation.

## Existing evidence and selected consumer

The selected `ParameterCatalog.from_public` in `/tmp/qwen-organizer-file-engine-integration-v2-20261104/qwen_public_parameters.py` rederives84 QKV matrices and114 decoder names from exact source-bound public metadata. The tied embedding and head share one physical file, yielding197 physical NPY files. The gamma report supplies113 exact vector names/shapes/hashes and one bounded NPZ container. The physical matrix raw payload is4,767,875,072bytes; gamma raw payload is524,288bytes. The catalog preserves exact source BF16 hashes, promoted payload and NPY-file hashes, dimensions and the original tied alias.

The historical QKV export has its actual4-source report/archive and independent documented readback. Decoder export has10-source parent/worker/archive, outer terminal and root readback. Gamma extraction has its6-source report/archive and asset/source before/after bindings. These are distinct original evidence schemas; this plan does not manufacture missing historical worker, outer, grant or current-store fields. `evidence_pins.json` records the exact existing public files and selected source dependencies. No body paths from those descriptors were followed.

`open_source_bound_parameters` already rederives the catalog before opening directories. `PublicParameterFiles` holds original directory/file descriptors, bounded NPY headers, readonly mappings, file and payload hashes, finite checks and the complete bounded113-member NPZ readback. Its matrix loading is lazy: `readback()` covers only opened leases. A complete aggregate validator must explicitly visit every unique physical matrix descriptor and the gamma inventory; merely constructing the owner and calling `readback()` is insufficient.

## Two identities, without a hash cycle

The pure **model_id** is SHA256 of canonical `qwen-learned-model-content-v1` bytes: original checkpoint/config hashes; exact198 logical matrix names, shapes, little-endian float64 payload hashes and original BF16 source hashes; explicit embedding/head alias; and113 named gamma shapes/payload hashes. Canonical means UTF-8, sorted keys, compact separators, finite JSON numbers, and one trailing newline, matching the selected `canonical` helper.

This content body excludes file paths, NPY/NPZ container encodings, producer or consumer source maps, grants, process IDs, timestamps, candidate coefficients and terminal reports. Moving the same bytes, changing a source closure or changing a container encoding does not silently redefine learned content. The actual selected catalog still requires the exact currently reviewed NPY/NPZ file bytes and provenance. Any future catalog/source change needs its own reviewed successor.

The aggregate manifest separately includes that content body/model_id, the exact catalog and catalog digest, all12 source-bound public metadata pins, the three historical export evidence bindings, and the existing protected applicability bridge to candidate344954/full owner208. Old exports retain candidate7f6478 provenance. The bridge is not a calibration PASS or a new export claim.

`aggregate_manifest.schema.json` is a closed fixed-profile schema whose fields are pinned to the public-derived preview. It requires a fresh reviewed successor for a different export or candidate. A future authenticator must still recompute content/catalog digests and compare canonical original bytes; schema matching alone is not authority.

The producer source inventory contains code, static policy and declared public metadata, not its later process reports/store observations. Generated evidence may bind that fixed source inventory. No source digest enters the content body, and no generated report is recursively hashed into its own source inventory. A copied preview may be a static source input, but it carries no actual validation or readiness claim.

## Public derivation already possible, not admitted

`derive_public_metadata.py` is a standalone stdlib-only derivation that reads the12 exact public metadata files and prior public review pin map. It does not import the selected consumer or NumPy and never opens a tensor descriptor path. It emits the four selected ParameterCatalog fields as canonical JSON, the proposed content body and aggregate manifest. Its logic is a public derivation, not an implementation or test of the future admission gate.

The resulting **unadmitted** model_id is `ee3daf8b07141134466d02a6e1c141365ac4e0f0117d62f887c416a050b04bfb`; catalog digest is `6faa4630169f055b853a62a64088952e2ead48dc2ace5b61b6aa438c0106a0c0`; aggregate-manifest digest is `18189e5f2f2de3a4939c6afca19d79822465d917caf06c4f6e36668bd37c1f55`. These identify what future authorized validation must authenticate. They do not establish current body integrity, calibration success or permission to execute. `derivation.json` explicitly keeps all such authority false. No actual validation/grant/outer receipt placeholders are fabricated.

## Minimal admission API and exact evidence contract

Proposed pure API:

`authenticate_model(public_contract, manifest_bytes, validation_report_bytes, validation_outer_bytes, root_review_bytes, *, expected_pins) -> ModelAdmission`

`expected_pins` comes from the independently reviewed caller/grant, never from the supplied reports. It pins the model/manifest/catalog digests, producer source map/archive, validation grant/report/outer/root-review digests, and selected consumer source map. `admission.schema.json` gives closed positive-admission schemas for this external argument and each evidence document. It is a specification, not executable validation. Each input must be immutable exact bytes, canonical and duplicate-key-free; retain them inside the returned frozen binding and reauthenticate them at the body-opening boundary.

Bounds: aggregate manifest1MiB; validation report2MiB; outer receipt64KiB; root review64KiB. Trusted expected source map has at most512 explicitly reviewed relative ordinary-file entries. Paths must be absolute canonical no-alias strings, flat names must reject dots/separators/NUL, integers must be exact integers rather than bool, and all hashes lowercase64-hex. The future implementation must enforce these semantic rules in addition to JSON Schema structure.

Cross-document checks are mandatory:

1. Recompute model_id from content and catalog digest from the exact selected catalog representation. Re-form the catalog from the fixed PublicContract and require exact equality. Validate the original exporter evidence through its own known schema and externally pinned digests; preserve the old/new candidate bridge explicitly.
2. Require the report's source before/after maps to equal the externally reviewed producer map. Recompute the canonical map digest; authenticate exact ordinary archive members against it. Distinguish this producer map from the historical exporter maps and selected consumer map.
3. Require complete successful current validation of all197 physical matrices/198 logical names and113 gammas, exact aggregate/model/catalog/grant bindings, all required integrity/cleanup/source flags true, no error_type and no audit denials. The report contains three held-root identities and198 sorted file-state records:197 matrix files plus the gamma container. Match every record to the catalog; enforce exact inventory, actual extents, hashes, modes and held identities. The gamma record has no aggregate raw-payload hash; its container and113 member hashes remain distinct.
4. Require the independently pinned outer receipt to match report/manifest/source/archive/grant/model hashes and actual process identifiers, normal return0, direct reap and known group absence. A self-published report cannot certify its last close. The future entry must distinguish every raised infrastructure BaseException from normal return0; no acceptance based on absent stderr.
5. Require the externally trusted root review to bind those exact artifacts and the canonical store-binding digest. It is model-integrity admission only and has `actual_server_execution_authorized=False`. Passing this gate does not authorize model execution; an actual server/client role still needs its own source, resource and execution grant.

A failure/incomplete report remains preserved evidence but cannot satisfy this positive admission schema. No old-body or historical timing claim is upgraded by generating the content manifest.

## Current-store validation and ownership

The smallest actual successor is a readonly reuse validator, not a new exporter. Before any parameter open, its fresh source/bootstrap/role/resource gate authenticates this fixed manifest and the old public evidence and excludes output/publication paths from all borrowed stores. It then uses the selected catalog and held consumer, explicitly authenticates the complete physical inventory, performs final readback and closes every owned mapping/descriptor with the shared body/cleanup-origin policy. It retains borrowed stores and never removes, rewrites, chmods or re-exports them. The NPZ's existing public mode is not silently changed to a private-key mode.

Root/file identity binding must use held descriptors before body reads. A path-stat followed by the existing factory is not a sufficient atomic substitute. The new admitted factory may require a narrow expected-identity hook at held-directory/file acquisition; such an overlay must be explicit and separately reviewed. The existing source-bound factory remains a component API until that binding exists.

Capture the original parameter owner, its close/readback/check callbacks and held-root identities immediately after acquisition. Enumerate all198 logical names against the rederived catalog while opening each of the197 physical `(family,file)` pairs once; verify the tied alias without keeping an extra array view. Visit all113 gamma names even though the first gamma access authenticates the complete container. Returned matrix/gamma views must stay inside each short validation operation and be dropped before final owner retirement; retain no array collection in reports or cleanup closures. Invoke the captured original final readback and close, preserve any genuine body failure, then complete source postchecks before reporting terminal success. An unresolved mapping/descriptor close is a failed validation, not a complete model admission.

Proposed new consumer API:

`open_admitted_parameters(public_contract, model_admission) -> PublicParameterFiles`

It reauthenticates the immutable admission, rederives the catalog, takes only admitted store descriptors and checks held identities before hashing/mapping bodies. Existing readonly finite/header/extent/hash checks and consumer-retirement rules remain. If the server fails after consumers enter, the ordinary engine and backing owner must stay quarantined until its containing key-loader scope observes that failure. This proposal does not remove the established stage lifetime guards.

Because existing bodies are reused readonly, no fsync claim is required for borrowed parameter data. Only new public report/manifest publication needs its held-output write/fsync/readback/final-close and independent terminal proof. Current resource checks must cover readonly mappings/descriptors and output evidence; the4.768GB raw size is neither a fresh disk-copy requirement nor a peak-resident-RAM measurement.

## Downstream changes and original embedding boundary

`PublicContract.identity/check_identity` currently accepts a syntactically valid externally supplied model_id; it does not authenticate a model-stage producer. Future actual key-generation intent, key FilesAdmission, client request identity and server parameter factory must receive the same ModelAdmission-derived model_id. Keep model-stage producer sources separate from request/response source maps and historical exporters. Preserve all frozen366/373/core sources; these interface changes belong to a new selected successor.

The official session already accepts authenticated original `(batch,S,1024)` embeddings and original positions. Existing `qwen_layout` copies those rows into one/two65536-slot tiles; it does not tokenize or alter learned embedding content. This API does not acquire an extra tokenizer or token-lookup prerequisite. If a later token-facing wrapper is explicitly selected, authenticate that upstream input provenance and perform lookup through the admitted original tied embedding/head descriptor. That optional wrapper is separate from model-stage admission and does not license post-heldout selection or feedback.

The next decision is root review of this source-only plan and schemas. Implementation, synthetic tests, current parameter-body validation, actual role grants and any model/HE run are separate steps. Existing full-model/ranked work remains open.
