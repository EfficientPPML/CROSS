The public stage schedule passed all 640 (K, round) contracts for K=1…128 and
all seven pinned BERT plaintext cases. Accuracy gates, s=4 scaling, exponential
coefficients, five squarings, and input score domains remain unchanged. The
seven-case run took 45.50 seconds; all nine tracked source hashes were unchanged.
This is conditional arithmetic planning, not an encrypted inverse or model run.

The whole-interval polynomial error bound is E≤3.407326e-8. It is derived from
exact rational differences between the stored binary64 coefficients and the
degree-eight Taylor polynomial, plus an analytic Taylor remainder. A positive
exponential series with a geometric tail bounds exp(1/2), whose reciprocal also
bounds exp(-1/2). The existing independent grid error gate remains 1e-10.

For positive active probabilities with mass m≤1 and ratio max/min≤R,
sum(p²)≤m²R/(K−1+R); Cauchy's inequality gives sum(p²)≥m²/K. Common positive
normalizers preserve ratios, while squaring squares them. Consequently the
public round contracts use R_j=R_0^(2^j), a=.5/K, and
b=min(2,2R_j/(K−1+R_j)), rounded outward. Assuming ideal normalization mass in
[1−1e-12,1] establishes the next contract. Directed 100-digit arithmetic certifies
the smallest Newton count meeting the unchanged 1e-12 relative contraction gate.
This proof does not certify binary64 or CKKS rounding errors.

The K=128 counts are [8,9,12,13,13], giving abstract depth 202 versus 232 for
fixed thirteen-iteration s=4 normalization. K=11,15,16 use [8,9,10,10,10], depth
178; K=1 uses [6,6,6,6,6], depth 127. These estimates include the separately
identified mean anchor-mask depth and are not complete homomorphic schedules.
The legacy public-affine counters in raw reports omit that one anchor stage.

Inactive groups require a public midpoint denominator for each new interval.
For K=128 in round zero, b≈.04191: the old dummy denominator 1 is outside the
interval and gives a first Newton output of approximately −56.82. The midpoint
replacement depends only on public K and round; initial mean-exponential dummy
1 remains unchanged. No values are clipped and no activations select constants.

The maximum probability difference from fixed-thirteen-iteration s=4 was
1.555e-15. Native context and block errors were at most 5.260e-10 and 2.496e-10.
All 10,212 intermediate diagnostics pass the existing binary64 allowance;
strict last-bit excursions remain recorded. Observed intermediates lie in
[-3.2,4]. Full-slot refresh, actual ciphertext precision, and unseen-input score
domains remain unvalidated. Correlated positivity and mass assumptions cannot
be replaced by independent loose magnitude bounds.

`summary.json` records commands, cases, source provenance, and artifact hashes.
Compressed raw reports preserve the full public contracts and observations;
`tracked_sources.tar.gz` preserves the exact tracked files. Model weights are
not included. Replay in a new copy after adapting the original source paths.
Earlier fixed-iteration and factored-map archives remain unchanged.
