# Two-level routing with fewer public keys

This separate candidate factors each key placement into a permutation within a16-key tile and a whole-tile rotation. All128 tile/offset mappings retain two forward levels, two inverse levels and the same rotation counts as the direct two-level route. The route-wide public key union falls from276 to66:22 head shifts,30 local-key shifts and14 tile shifts. Nonzero tiles still use27 rotations per direction; tile0 uses26.

At the pinned actual depth49/p60/dnum8 N131072 Q100/P14 profile, raw routing-key storage is58.78125GiB instead of245.8125GiB. Adding the prospective QK/PV row alignment and head sum/broadcast keys yields89 keys; adding the106-key refresh union yields128 keys/114GiB. These are public key inventories and raw payload calculations, excluding dense projections, normalizations, headers, context tables, ciphertexts, compiler buffers and peak-memory overhead. They are not full-model resource or timing results.

The candidate accepts finite positive noncanonical input scales, as produced by a real QK product. Its first weighted sum performs the actual scale alignment while applying the public mask. Weighted outputs must be canonical within the arithmetic owner's tolerance, and rotations must preserve their actual scale exactly. No ciphertext metadata is relabeled and no alignment level is added. Underlying plaintext-encoding range and coefficient-round-away guards still apply.

The source-bound public test wrapper passed421 tests in3.39seconds, with all10 sources unchanged and every archive member verified. Tests cover every mapping and inverse against independent gather/scatter references, hostile padding, exact key inventories, input preservation, callback mutations, noncanonical input scales and inherited metadata guards. Session45911 exited0 and was reaped; the wrapper took3.6867seconds. Independent read-only review found no algebra or masking blocker.

No encrypted operation ran for this version. The previous direct-routing proof and subsequent guard repair remain immutable, separate evidence. Actual key-switch/alignment error, accumulated full-attention error and common-profile performance require new encrypted measurements. The candidate is outside the main code and contestant release.

`python verify.py` checks exact artifacts, the recorded test result and independent key-union/resource arithmetic. It does not repeat encrypted execution.
