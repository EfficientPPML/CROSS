# Exact public embedding and Q/K bounds

This audit reads only the pinned public Qwen checkpoint/configuration and prior public-weight audit. It evaluates no model, token sequence, prompt, calibration or validation item; it selects no coefficients.

Each BF16 word represents an exact dyadic value. For a normal word its unsigned significand is128+fraction and exponent is exponent_bits−134; a subnormal uses fraction×2^-133. Squaring and summing uses Python integers shifted to a shared exact exponent per row. The interval is the minimum/maximum over **all151936 rows**, divided by1024, plus the exact configured binary64 epsilon. Sixty-four fixed evenly spaced public rows and the extrema are independently re-summed using `Fraction` of their exactly promoted FP32 values. A hash commits to the complete ordered row-sum stream without publishing the tensor.

The report separately records epsilon rounded to FP32 and an interval using that epsilon with otherwise exact sums. Neither is a claim about native FP32 reduction/addition order. A check of whether stored BF16 squares happen to be exactly representable in FP32 is observational; it does not certify the native reduction. The only tightening comparison is against the existing public coordinatewise envelope for the first embedding normalization. Later residual sites remain unchanged.

For post-Q/K-norm scores, positive-epsilon exact RMSNorm implies pre-gamma head L2≤sqrt128. RoPE is orthogonal and preserves the64 half-split coordinate pairs `(i,i+64)`. For arbitrary positions, the operator norm of `diag(gamma_q) Rq^T Rk diag(gamma_k)` is bounded by the maximum over pairs of `max|gamma_q(pair)|*max|gamma_k(pair)|`. This is no larger than the product of the global gamma maxima. For equal positions only, the rotations cancel and `max_i|gamma_q[i]*gamma_k[i]|` suffices. The score divides the dot product bysqrt128; an outward dyadic endpoint encloses the resulting sqrt128 factor.

The separate conditional approximate score bound multiplies by `(1+1e-6)^2`. It requires a uniform relative multiplier certificate at both norms and the same feature/variance relation. It is not granted by this exact-native audit. Native floating RoPE and CKKS/refresh errors remain separate.

All seven pinned asset hashes, relevant tensor raw/promoted hashes and source bytes are checked before/after. No public weights, raw embedding rows, hidden activations, keys or ciphertexts are archived.
