# Optional decoder-only BERT client proposals

These source-only proposals are optional robustness successors. The completed
original122 and combined125 clients remain the launch defaults. No new full
preflight, installed-backend probe, context/key construction, ciphertext/private
read, or client execution has been performed here. No launch is authorized.

Both old clients use decoder SHA
`bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6`,
exactly the baseline of the separately reviewed 207-source decoder proof. Both
proposals copy its complete replacement SHA
`9675b300d507fd9c6a3d03656c9df325d8da68a9331b4c92def0512784d0dc26`.
The entire module AST is unchanged after reversing one new helper and its two
call substitutions. No transplant or unrelated runtime change is involved.

The actual chain is `file_stages.client_decrypt_decode` →
`key_contexts.load_client` / `_crypto` → `jaxite_word.CKKSContext.decrypt_slots`
→ `_fast_decrypt_decode` → `_fast_decode`. The package facade aliases flat
`ckks_ctx` and qualified `jaxite_word.ckks_ctx` to the same module object. The
existing fresh source guard resolves and binds both spellings to
`runtime/jaxite_word/ckks_ctx.py`, rejects preload substitution, and rechecks
origins. The proposed guard changes only its explicit decoder overlay hash.

Successful integer conversion still evaluates the same normal expression:
`raw = float(value)` followed by `raw * scale_inv`. Only overflowing conversion
uses the exact binary64 scale's integer ratio. The helper does not change FFT,
conjugate/approximation checks, finite-real output validation, private readback,
P2 gates, key format, profile, decryption arithmetic, caches or ownership.

The authenticated final full-BERT response is L2 with six Q towers. Their fixed
product is 181 bits; every centered coefficient has at most 180 bits. Therefore
the initial integer-to-float conversion cannot overflow at this valid final
boundary. The proposal does not claim a BERT repair, accuracy result, toy5
recovery or performance benefit. A separate bounded CPU timing comparison is
owned by Transformer; it is not part of this source-only evidence.

| Proposal | Sources | Changed from clean parent | Guard overlays vs89 | New manifest SHA256 |
|---|---:|---:|---:|---|
| original122 | 122 | 4 | 8 | `6e3e30ad643921e9546be144e7da0dce485e9dd1be088f31f62f35cd2aa4c5f9` |
| combined125 | 125 | 4 | 9 | `6b2b1985c1f9e7298bf47cf8559fd8a2122b0ad854289fed8674c2c05e8475a0` |

Each child directory contains `source/`, exact old/new manifests,
`candidate.patch` and `sources.tar.gz`. Four changed files are the decoder,
`client_source_guard.py`, the existing guard test's overlay-count and decoder
alias assertions, and `README_CLIENT.md`. No new test case was added or run.
`report.json` records the complete changed and protected-unchanged inventories;
`prepare.py` is the executed stdlib-only construction/static-check script.

All original58/65/80/217 producer pins, the reviewed286 combined producer
binding, public source archives, candidate/profile, numerical comparator,
original or optional receipt semantics, worker/collector entrypoints and
bootstrap file-name lists stay exact predecessor bytes. They are intentionally
not rewritten to the new consumer decoder hash. Historical source-origin maps
describe the inherited producer/base; the current local overlay and external
manifest describe this proposal. The unselected historical `dependencies.check`
and wrappers are not substituted into the modern entry path.

Before any optional adoption: independently review these four-file deltas and
the normal-path timing result, obtain authorization for a new complete client
source-bound synthetic/installed proof, and retain its exact new source hashes.
An actual client still needs its corresponding completed217 or completed286
producer, independently authenticated public payload/receipt chain, protected
private roots, clean no-bytecode source tree and separate launch grant. Existing
server46674, original217/286, downloads and request/receipt ledgers are untouched.
