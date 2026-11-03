# Independent final-v2 uniform norm certificate review

All eight frozen polynomial post-attention norm certificates passed an independent source/readback and exact-algebra review. The four source files matched before/after/current/tar identities. The audit checked actual rounded affine constants and their exact normalized-domain pullbacks, coefficient-byte hashes, an independent exact Clenshaw/Horner midpoint sign, the rational error bound and directed binary64 multiplier endpoints.

The exact Chebyshev square uses T_iT_j=(T_(i+j)+T_|i-j|)/2 with correct diagonal/cross factors. If H=v*p(v)^2−1 has coefficientL1 delta<1, then p has no zero; positive midpoint gives positivity throughout the connected domain. Thus |sqrt(v)p(v)−1|≤delta/(sqrt(1−delta)+1)≤delta/(2−delta). V2 avoids numerical square-root lower bounds entirely. Lower/upper multiplier endpoints were compared against exact rational1−bound and1+bound.

All eight final relative bounds are below1e-6; the maximum is5.029028745383948e-7 at layer27. The review took0.40s and did not rerun the expensive exact high-degree convolution. Its implementation and source-bound results were reviewed. Subsequent floating arithmetic, CKKS/refresh error and membership in the declared variance domains remain separate obligations. No candidate coefficients/domains changed and no model ran.

The exact final certificate report/archive are copied here so `python verify_archive.py` works without the original scratch directory. Historical v1 limitations remain in the certificate author's separate archive.
