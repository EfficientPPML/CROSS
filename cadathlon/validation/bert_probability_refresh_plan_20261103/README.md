# Probability-only refresh: separate public follow-up

This report checks exact public Q/P profiles and levels while allowing refresh only on probabilities before a whole normalization round. It preserves the earlier common-ring report unchanged. No context, key generation, backend or encrypted operation was invoked. The planner exited0;108 before/archive/after/current source bytes agree, and the independent offline verifier checks all640 K/profile schedules and the intentional insecure-profile rejection.

Output31 supplies mask1 + square1 + anchor1 + thirteen-update static inverse25 + numerator product1 =29 levels, leaving2 for the next bounded refresh. The integer32 restoration consumes no level. The saved numerator and denominator are formed from the same refreshed probability values; there is no denominator refresh inside the inverse. This repairs the ideal same-squares relation, but does **not** establish the earlier probability mass, ratio or positivity assumptions after refresh.

| Profile | Q/P | Actual partitions | log2(QP) | Refresh output | Raw224 direct keys | Prospective116 keys |
|---|---:|---:|---:|---:|---:|---:|
| depth53/p60/d14 |108/9|14|3492.282217|29|358.3125 GiB|185.5546875 GiB|
| depth55/p60/d28 |112/5|28|3492.733952|31|716.625 GiB|371.109375 GiB|
| depth56/p58/d20 |114/6|19|3521.326550|32|498.7500 GiB|258.281250 GiB|
| depth56/p58/d28 |114/5|23|3491.355365|32|598.71875 GiB|310.05078125 GiB|
| depth56/p56/d14 |114/9|13|3417.977497|32|349.78125 GiB|181.13671875 GiB|

All accepted profiles pass actual128-bit-classical QP security and P coverage at N131072/CD2/full65536 slots, layout256×512, first61 and budget4/4. The requested dnum is not assumed to equal the actual partition count. Depth57/p58/d29 is rejected by the security guard; the rejection is retained.

The output29 comparison supports only K1–74 under the probability-only schedule. Output31 and32 support all K1–128 conditionally. For K128, the fixed counts8/9/12/13/13 require four probability refreshes: output31 ends normalization at level2; output32 ends at3. Every refresh input and final product remains at least level2, and the same later attention/FFN/LayerNorm continuation checks are applied. The level costs still refer to the unchanged layer0 five-squaring graph; the new training-only BERT candidate has other degrees and round counts.

The prospective dense-linear key set is `{1,64,61760,64064}` for the three declared full-width matrix shapes. Combining it with the unchanged head, row, routing, reduction, normalization and full-slot bootstrap keys gives116 distinct keys. This is an ideal-tested chained/Horner candidate, not a measured secure execution. It changes the key-switch noise history, and general sparse matrices require their actual gaps. Raw key storage excludes generation temporaries, controls, NTT/FFT tables, compiler/executables, allocator and ciphertext live sets; it is not an HBM fit claim. In particular,716.625 GiB of raw direct keys already exceeds the709 GiB tmpfs size before other users or overhead.

## Conditions still required

Probability refresh may introduce signed errors at small probabilities, change total mass, and enlarge the maximum/minimum ratio. An observed component1e-3 gate is not a uniform future-error certificate. The existing tighter stage intervals and iteration counts therefore remain conditional; no domains were expanded in this report. New p60/p58/p56 chains, full-slot bounds16/64 and same-ring full-width components require their own numerical proofs. The prior failed full-MRPC domain coverage also remains unresolved.

A separate mass-only observation is retained for future work. If original probabilities are nonnegative with mass in[1−1e−12,1] and every refreshed valid slot has error at mostε, then before square/reduction error:

- `sum(p+η)^2 >= max(0,1−1e−12−Kε)^2/K`.
- `sum(p+η)^2 <= (1+sqrt(K)ε)^2`.

For K128 andε=1e-3 these lie within the broad interval[0.5/K,2]. This does not prove the tighter ratio-derived stage intervals, establish a uniform encrypted error bound, or include actual mask/square/reduction error. It suggests a separate possible broad-domain certificate, which would require its own public seeds/counts and precision validation. Nothing in this archive silently substitutes that certificate for the existing schedule.

`report.json` contains exact primes, every branch, scale planning and conditional bounds. `sources.tar.gz` has SHA256 `9db87f381b30e1459076f99080472a28dc781a87c21ad0e9fbe70a19d7a09992`. Run `python verify_plan.py` for offline integrity/level checks. `attempt1/` preserves a preparation failure before parameter generation: the source glob included a `__pycache__` directory; the final planner filters regular files. No failed encrypted run or changed numerical gate is hidden by that correction.
