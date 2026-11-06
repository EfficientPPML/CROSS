This is a disabled successor preparation, not an actual diagnostic launch. Original246/200 and all negative reproductions remain unchanged in the predecessor/review labs. Original199 encrypted numerical sources, profile, coefficient arrays, checkpoint definitions and gates are copied unchanged.

Actual entry uses a clean exact source-only tree and externally supplied canonical source-manifest SHA256:

    python -I -B execution_source/collect.py --source-manifest SOURCE_MANIFEST --source-manifest-sha256 EXTERNAL_SHA --output FRESH_OUTPUT

The command above performs public/mock preflight only. The source tree may contain no unlisted files, symlinks, bytecode or caches. Source authentication and static/preloaded import checks precede local imports, publication and backend access. The external invoker remains responsible for trusting the initial script/interpreter and external digest; this is not an OS sandbox.

Actual execution_authorization.json remains exactly {"ready":false}. Future authorization requires a separately reviewed closure; no gate should be edited in the current measured source. Actual roles are one per invocation:

- Setup: --execute --stage setup --workspace-parent OWNED_PARENT creates fresh keys/public weights/one input in a fresh workspace. It publishes complete source/device/security/storage/cleanup metadata and retains the workspace.
- Server: --execute --stage server --config CONFIG --config-sha256 EXTERNAL_SHA consumes the externally pinned completed setup receipt. It saves all96 opaque checkpoints, authenticates them after all server work, and publishes direct-child/group terminal receipts. No private path is permitted.
- Validate: --execute --stage validate --config CONFIG --config-sha256 EXTERNAL_SHA additionally requires the separately produced external server root-review path and exact SHA256. No review or expected digest is inferred from a server report. All96 current opaque responses and their independently replayed actual scales are authenticated before private loading, then each consumed response is authenticated again.

Exact role configuration fields are stage, workspace, workspace_identity, nonce, setup_receipt, setup_sha256, server_receipt, server_sha256, server_root_review, server_root_review_sha256. Setup has no prior receipt/review; server has setup only; validator has all. Existing workspaces use authenticated exact dev/inode pairs. The review schema helper response_admission.review_fields documents exact bindings to setup/server receipts, source map/server archive, original profile/plan, workspace, request/input and ordered96 manifest/metadata rows. Constructing this schema is not review approval; an external reviewer supplies its hash only after reviewing completed server evidence.

Output publication is bound to the originally acquired directory and each published file inode, including failure publication. Existing output collisions, subsequent output/parent replacement and report replacement reject while preserving foreign data. Outputs must be separate from sources, workspace, external config and prerequisite evidence.

The unchanged synthetic tests and new admission/ownership/bootstrap tests run without actual backend/model/key/private access. Three earlier exact246 failures (response order and two replacement cases) plus two independent pre-import canaries are preserved separately. Separate development failures here were test-only assumptions about synthetic config completeness and cleanup-note wording; their source/log histories remain outside the execution tree. Any later genuine decoder failure remains a failure, with original0.03125 gate and additional1e-6 diagnostic intact.
