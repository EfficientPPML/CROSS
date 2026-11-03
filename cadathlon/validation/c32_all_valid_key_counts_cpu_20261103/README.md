# Fixed integer C32 inverse: all public key counts

This is a public conditional arithmetic certificate and NumPy diagnostic, not an
encrypted, refresh, composed softmax, model, or throughput result. All640
contracts (K1–128, five squared-probability rounds) retain the original public
schedule's domains and update counts. The exact four sources are archived.

Use C=32, initial estimate z0=2/(32(a+b)), initial affine z1=(-32*z0*z0)d+2*z0,
then z_next=z*(2-32*d*z). The target is z=1/(32d), equivalent to the old notation
s=K/32. Neither K/4 nor a per-input convergence test is used. K128 retains
[8,9,12,13,13] updates; K15 retains[8,9,10,10,10]. The initial mean-exp reciprocal
remains the previous separate six-update C1 circuit on[.5,2].

The ideal seed cancels C in the relative residual, retaining the original
rho=(b-a)/(b+a) certificate. This check also treats the ACTUAL rounded initial
affine weight and bias as exact rational coefficients, finds the quadratic
first-residual extrema at both endpoints and the vertex, and propagates its
absolute bound by directed100-digit Decimal squaring. Every declared count
still certifies final relative residual≤1e-12. This is a real-arithmetic
certificate for those public coefficients; it does not certify rounded CKKS
operations or bootstrap error.

1,312,173 endpoint/interior/dummy samples pass the separate binary64 diagnostic.
Worst sampled relative residual is9.797e-13; worst z error is6.919e-12 and restored
reciprocal error2.215e-10. The exact maximum z bound8 holds across all updates.
There are two rounding details: outward a may make1/(32a) slightly greater than
K/16 (600contracts), and the rounded first affine may have a slightly negative
relative residual near its vertex (342contracts). Thus the first d*z bound is
0.03125000000000001 and first following correction lower bound is
0.9999999999999999. From update2 onward, squared residual is nonnegative;
d*z≤1/32 and subsequent corrections lie in[1,2]. These are public exact bounds,
not a noise allowance.

All640 declared inactive midpoints are checked for exact closed-domain
membership. Dummy1 is outside318 tightened domains, including early K128 rounds.
The backend-free validate_public_dummy checks finite positive public values and
domain membership before any ciphertext operation;5120 rejection cases cover
nonfinite, boolean, nonpositive and immediately out-of-domain values. A future
layout helper must validate each public row/stage dummy before placing it in
inactive anchors as a folded public bias. The older59-case reduction evidence
is unchanged; its dummy1 was explicitly conditional on domains containing1.

The common normalizer preserves ratios. Under the existing ideal assumption
that d is the sum of the SAME squared probabilities, form p_i_squared*z first
(≤1/32), then restore with integer32, yielding mass1-residual. Do not materialize
32*z as a purported bound16 intermediate: that reciprocal can reach256. The
public domain, positive base exponential and incoming mass assumptions remain
conditional, and no CKKS/refresh or independent numerator/denominator guarantee
is added. Existing integer-affine helper paths would use2n-1 levels through each
nth inverse update; this planning count is not a new ciphertext measurement.

Run `python check_all_k.py` beside the three supplied source/manifest files.
The source snapshot is portable and imports no cryptographic library. Its
recorded run used CPU148–151 with BLAS/OMP1. The copied reviewed helper is included
only to document its generic positive-integer C API, not executed by this check.

`initial-wording/` preserves the first draft's report/log/exact source snapshot.
Its global first-update product/correction metadata was too strict because it
overlooked the rounded initial affine's tiny negative residual. Its sampled
results did not change; the current report fixes the exact bound statements.
Do not use that superseded draft as a valid bound certificate.
