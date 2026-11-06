# Isolated twelve-layer BERT softmax owner

This validation-only owner preserves the frozen BERT-v2 manifest, selected coefficient bytes, polynomial bases, static initial inverse, and tracked normalization inverse counts. It makes no claim of encrypted execution or actual refresh accuracy.

Layer 0 keeps the existing fresh-score path L45→L23 without an extra initial refresh. Layers 1–11 use the frozen split path: score L21→mean L19→normalized t L18, explicit B2 full-slot refresh to L25, then the unchanged masked 19-level initial tail to L6. The per-layer normalization divisors are 16/64/64/64/32/64/64/32/32/32/32/32. Each layer executes its original 5/6/7 rounds. A public capacity schedule refreshes probabilities at B2 only when needed, followed by an unconditional final B2 continuation refresh and real mask L25→L24. The required post-mask tail reserve is17 or18 levels according to the selected attention normalization.

Tests execute all physical65536 slots for every public K across all three distinct round counts, plus all twelve exact endpoint contracts. Independent dense NumPy arithmetic supplies the oracle; actual helper loops and ordered scale recurrences supply the tested graph. Identity refresh can inject nonzero invalid-slot values. Callback/retained branch tests cover ownership, cancellation and metadata changes. A public-only inventory checks1536 layer/K schedules with canonical and adjacent-ULP input scales.

The B2 range/error contract, propagated denominator bounds, and correlated probability errors remain unproved for actual CKKS and repeated refresh. This is not full attention, a file-stage handler, or a twelve-layer encrypted model. Source archives contain no model arrays, keys, ciphertexts, or private inputs.

Measured validation: 521 tests passed; exact 127 source files remained unchanged before/after and match the source archive. Collector duration 1503.206s, a validation duration rather than an implementation performance measurement. The collector was authoritatively reaped with exit0.
