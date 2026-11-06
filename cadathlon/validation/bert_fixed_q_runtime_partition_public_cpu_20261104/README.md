# Fixed-Q registered-validator and frozen-partition audit

The source-bound public/abstract audit completed successfully:21 tests,43 exact candidate sources/44 including wrapper,150 physical-prefix shape traces and450 malformed-shape rejections. Session83053 exited0 and was authoritatively reaped; both children were absent. This is an audit pass, not registered-profile acceptance. CPU backend initialization and abstract JAX tracing occurred; no CKKS context, parameter-cache or Polynomial instance, keys, model, ciphertexts or encrypted arithmetic was created or executed. All32 forbidden-state/HE guards remained untouched and were restored.

## Registered outcomes

| Fixed profile | Actual P | Registered requirement | Actual outcome |
|---|---:|---:|---|
| Current d8/aux30 |14|15|Reject|
| d6/aux30 |18|19|Reject|
| d6/aux31 |17|18|Reject|

These exact ValueError results came from the copied, imported `validate_runtime_config`, with explicit128-classical target. The transparent public configuration uses fixed100Q, N131072, CD2, first61/scaling60, exact scale integer and default sigma/distribution; the validator couples aux bits to register_word_size−1. All three still pass the separate actual-QP security check. The registered validator partitions50 logical composites; the current ordinary-key path fixes partitions from100 physical limbs. No Q/P/count/validator was changed to force a pass, and no lower-level view was incorrectly treated as a newly registered independent ring. The current d8 baseline rejection is retained, rather than hidden or blamed on d6.

## Frozen max-Q structural checks

The actual imported HEParameterCache.get_rot_key and level helpers ran under abstract shape tracing. Raw keys `(digits,100+P,131072)` transpose to formatted-full `(digits,131072,100+P)`, then use the actual ordered Q-prefix/P-suffix slice, concatenation and reshape. All50CD2levels per profile retain maxQ100: alpha13 for d8 or17 for d6. The real imported `_HERotAtLevel.__init__` dispatched these max-Q/active-digit parameters, with its kernel/control constructor replaced by a clearly labeled recording double. The actual imported HERot paired-key method performed structural shape validation on the resulting abstract tracers. Too-few-digit, wrong-tail and unpaired cases rejected at every level. Independent exact-integer partitions/P coverage and all public JAX slice/transpose/output shapes are recorded.

The recording double generated no controls. Full control_gen, BConv/NTT Polynomial state, compiled rotation/key-switch execution, error/noise, refresh and model composition remain untested here. Abstract acceptance is not a genuine compiled-kernel or encrypted proof. No active profile, key store, source, file handler or numerical job changed. Any d6 deployment would require an explicit parameter-contract decision and new keys/bindings/genuine proofs; no speed, accuracy or memory-peak claim follows.

The initial9-test plan mislabeled raw axes as the formatted-full axes. Root caught this before the imported audit. History/plan-v1 preserves those exact mistaken bytes/results; the corrected source-anchored transpose regression and final21-test proof supersede that preparation only. No prior39-source public audit was edited.

Run `python verify.py .` for portable source/report/shape verification without numerical imports. Add `--current /tmp/bert-fixed-q-runtime-partition-audit-20261104` to verify retained exact sources and the external driver. Only public metadata, reports, logs and source bytes are archived.
