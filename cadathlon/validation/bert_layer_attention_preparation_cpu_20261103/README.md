# Layer-indexed attention preparation

These are separate preparation records for the isolated layer-attention owner. The complete24-case proof is a separate later artifact.

- Public contract:128 tests,1536 layer/K schema combinations, all96 pinned projection tensors and strict descriptor identities;9 source files, no backend execution.
- Trusted native fixtures:12 helper tests and a pinned float64 model preparation for the same fixed K15/K128 pairs. Every layer's compact input and native attention output is recorded by hash; full/compact native parity is checked. The NPZ/model arrays remain private to the local validation directory and are excluded here.
- Development smoke:66 tests covering two complete K15 component graphs (fresh layer0 and later layer1) and64 ownership guards, with137 captured source files. This is ideal arithmetic with identity refresh, not CKKS or a complete twelve-layer model.

The initial contract test collection typo is retained separately as a test-authoring failure. No numerical gate was relaxed. Each source tar was read back against its measured before/after hashes. The smoke source snapshot is historical; the final collector/provenance metadata was subsequently extended before the full component proof. No private activations, model tensors, keys, ciphertexts or per-image labels are included.
