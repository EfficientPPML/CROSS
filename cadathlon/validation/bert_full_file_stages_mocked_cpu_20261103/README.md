# Full BERT five-handler integration: mocked boundary checks

The isolated five file-stage functions are implemented in
[file_stages.py](source/file_stages.py). The final frozen suite passes **112
checks** over **24 unchanged source/public-input files**. It exercises all five
handlers and two consecutive runs through the upstream harness's reused paths.
Keys, codecs and numerical graph work are explicitly mocked in this suite.
This is not encrypted file transport, an actual key-generation run, a joined
numerical model result, MRPC accuracy evidence, or a runtime calibration.

Production handler bodies call the actual client key helper, pinned public
model loader, client embedding preparer, strict ciphertext transport, context
loaders and joined twelve-layer owner. No native encoder or plaintext full-model
fallback is implemented. The numerical owner/runtime and key-loader checks are
separate prerequisites; this archive does not promote a complete release.

The batch protocol binds each ordered eight-tile request to authenticated
context/model identities, original positions and an opaque nonce. Private
target IDs and the original input-byte digest remain on the client. Every
response binds the original request digest, and the client independently checks
the fixed public output-scale enclosure before decoding. Reordered, duplicated,
foreign, malformed and stale requests/responses are rejected in the tests.

The upstream harness reuses upload/download/result paths between iterations.
Successful client decryption publishes a private completion receipt. The next
input stage retires only those completed artifacts after checking their root
and nested inode identities, file size/timestamps, inventory and metadata byte
digests. Interrupted requests remain for review. Partial retirement can resume;
keys remain in place. These are concurrency/ownership checks, not an OS sandbox.

Independent review reproduced two defects and verified the fixes. First, the
initial completion receipt could delete a replaced nested ciphertext file.
Receipts now bind all nested identities and reject the original replacement
reproducer while preserving the new file. Second, archive-digest calculation
could mask a snapshot-creation failure. Final publication now preserves that
primary error and records the failed attempt. Additional checks cover cleanup
cancellation, repeated signals and a worker descendant surviving its leader.
Original source captures and reproducers are retained, with their capture scope
explicitly distinguished from startup archives.

`development-v1/` preserves the first passing 112-test mocked checkpoint. It
used the earlier key-context helper, before review identified lazy asynchronous
rotation-transfer cleanup. `source-bound-v2/` repeats the same tests with the
corrected helper: an exception from an entered server stage retains public
mapping/transfer owners until process exit and rejects loader reuse. Successful
stages still close mappings normally. The mocked handler pass does not itself
validate DMA behavior; that policy is tested separately in the loader suite.

Both checkpoint processes exited0 and were reaped. Owned test trees containing
tiny fake artifacts were removed; no keys, ciphertext payloads, model weights,
or activation arrays are archived. The final source archive, report, test log
and XML are retained alongside directly reviewable source files.

Run `python verify.py` to audit recorded evidence without executing tests or
HE. To rerun the mocked suite, extract the final `sources.tar.gz` into a fresh
directory and run the six `test_*.py` files with pytest, NumPy and Python3.13.
The collector records uncalibrated test time only. Real common-ring encryption,
size0, ranked100/200/408 runs and full-model TPU measurements remain required.
