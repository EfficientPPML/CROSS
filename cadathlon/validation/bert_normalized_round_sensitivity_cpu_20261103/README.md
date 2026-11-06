# Normalized-round reciprocal sensitivity

The failed encrypted run remains failed. This public-only analysis explains why
an accurate final probability does not imply that every raw inverse estimate
passes the same absolute tolerance. It reads the unchanged report and the exact
client metric source; it reads no keys, ciphertexts or decoded arrays.

Writing the decoded denominator as d_tilde, its recorded minimum as m, its
maximum error as E, the invariant discrepancy as I, and the recorded tracked
residual maximum as R gives the conditional estimate
`|y - 1/d| <= E/[m(m-E)] + (I+R)/m`.
This follows by adding reciprocal perturbation and residual-over-denominator.
Published metrics and the client invariant use floating arithmetic, so this
calculation is a sensitivity diagnosis, not an outward-rounded interval proof.

For K128, denominator error1.2153951e-10 near0.00781315 gives reciprocal
perturbation1.9909727e-6. The remaining invariant/residual term is1.7564e-9;
their sum1.9927291e-6 covers the observed true-inverse error1.9912011e-6.
For K15, the analogous terms sum2.7080096e-8 and the observed error is2.7044040e-8.
This is consistent with amplification of denominator error. Aggregate maxima
cannot establish exact slotwise causation. More reciprocal iterations alone
do not address error in the denominator being inverted.

For K128, the denominator contribution alone would require roughly6.1045e-11
error to stay below1e-6 at this recorded minimum, before allocating any error
to the remaining computation. This sufficient estimate is not a revised gate
or a guarantee for another run. No arithmetic, profile, coefficients, domains,
iteration counts or frozen evidence were changed, and no HE was run here.

Run `python verify.py` for artifact and deterministic analysis readback.
