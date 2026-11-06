# Toy BERT4 diagnostic plan: public preflight only

The fixed public graph and independent dense oracle agree over all96 checkpoints/all65536 slots. The61-test preflight authenticates222 sources (including the exact original199), five refreshes, key/storage inventory, synthetic post-server decode admission, and owned-group/terminal cleanup. No HE, TPU or model framework was imported or run. The future actual diagnostic worker/transport remains unimplemented and unlaunched; see DESIGN.md.

Original BERT4 encrypted failure remains a failure: decoder stddev=inf, no decoded accuracy result. The extra1e-6 checkpoint diagnostic is predeclared and does not replace the original gates. Identity-refresh and public domain checks do not certify CKKS error or refresh range preservation.

`preflight/` is the source-bound final evidence. `development/` retains earlier successful graph/reference and44/61-test logs; those exploratory runs did not create startup source archives, so only their observed outcomes are claimed. No development failure was observed. The final source archive includes tests and lifecycle helpers. No private input, learned-model tensor, key, ciphertext or derived slot-array bodies are archived. The plan and authenticated199-source archive include the original public toy fixture token IDs, token types and attention masks, plus its seeded reference source.

Run `python -B verify.py` for portable public readback. The verifier checks immutable recorded evidence, not current machine resources or live jobs.

Subsequent implementation review corrected the private-storage assumption in
the preserved plan. Its1MiB `secret_i64_bytes` field does not bound the actual
serialized RNS secret: uint32[100,131072] requires52,428,800 bytes (50MiB), or
53,481,472 bytes including the declared header/reserve allowance. The separate
[correction receipt](resource-correction/receipt.json) pins the original plan and
key-generation source; root independently verified the dimensions/arithmetic.
Public-key, rotation-key and checkpoint payload totals stay unchanged. Frozen222
sources/reports remain exact; future capacity gates must use the corrected
private total. Python key-generation buffers and process memory remain excluded.
