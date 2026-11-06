This exact public certificate covers the ideal arithmetic of all24 frozen BERT-v2 normalization sites. It keeps manifest `4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9`, every selected power/balanced method, and every coefficient unchanged. No model, tokenizer, input sequence, inference, key generation or ciphertext was used. Public learned affine tensors were read from the pinned checkpoint solely for output bounds; no tensor arrays are archived.

All24 conditional outputs fit B128 with the fixed9/8 margin. The largest squared-relative polynomial envelope is1.5732220830514083e-5; the largest affine output bound is76.2970645530664, and its guarded bound85.8341976221997. These bounds assume the represented variance lies in the site's declared domain and shares the exact ideal moment relation with its normalized coordinates. They do not certify domain propagation, public plaintext encoding, floating-point trajectory error, CKKS noise, independent feature/variance perturbations, or refresh preservation.

For the actual rounded public map t=A*v+B, the proof forms exactly

    H(t) = ((t-B)/A) * P(t)^2 - 1.

Balanced sites use their original binary64 Chebyshev coefficients as exact rational values. Power sites use the actual NumPy binary64 cheb2poly output, then an exact rational power-to-Chebyshev conversion; no rounded conversion back is used. For M=max(1,|A*lo+B|,|A*hi+B|), the certificate bounds |H| by sum |h_n|*T_n(M), including any one-ULP domain-map extension.

The variance uses the actual coefficient alpha=Fraction(float(1/768)), so v=alpha*sum(u_i^2)+epsilon with positive epsilon. Therefore |u_i*P(v)|≤sqrt((1+delta)/alpha). This argument does not assume that rounded centering produces an exactly zero coordinate sum, or that P is positive. Exact learned gamma/beta products and outward square roots give each site's final bound. Negative polynomial sign would still be an accuracy issue outside this absolute-magnitude certificate.

| Site | Selected method / degree | Squared-relative envelope | Affine absolute bound | Bound with9/8, rounded up to power of two |
|---|---|---:|---:|---:|
| 0.attention | power / 12 | 5.3479435e-06 | 76.29706455 | 128 |
| 0.mlp | power / 16 | 1.2574621e-05 | 26.2791088 | 32 |
| 1.attention | power / 16 | 6.8393909e-07 | 74.45564554 | 128 |
| 1.mlp | balanced_chebyshev / 48 | 6.8128464e-07 | 27.67538903 | 32 |
| 2.attention | power / 12 | 1.5732221e-05 | 57.59488566 | 128 |
| 2.mlp | balanced_chebyshev / 48 | 3.661345e-07 | 27.60189343 | 32 |
| 3.attention | power / 16 | 1.7633171e-06 | 61.2866567 | 128 |
| 3.mlp | power / 32 | 6.3628625e-06 | 27.42807049 | 32 |
| 4.attention | power / 16 | 1.0858409e-06 | 71.85351563 | 128 |
| 4.mlp | power / 32 | 1.5050006e-05 | 28.5688307 | 64 |
| 5.attention | power / 16 | 1.4333325e-05 | 68.2582184 | 128 |
| 5.mlp | balanced_chebyshev / 48 | 1.1316368e-07 | 28.95419481 | 64 |
| 6.attention | power / 24 | 4.8376936e-07 | 72.95968585 | 128 |
| 6.mlp | power / 32 | 1.3735513e-05 | 28.28230971 | 32 |
| 7.attention | power / 24 | 1.6412596e-07 | 68.02856532 | 128 |
| 7.mlp | balanced_chebyshev / 48 | 7.8893592e-06 | 28.64101873 | 64 |
| 8.attention | power / 24 | 1.3678302e-06 | 74.80326924 | 128 |
| 8.mlp | balanced_chebyshev / 96 | 1.1978981e-05 | 28.97579941 | 64 |
| 9.attention | power / 24 | 1.947878e-07 | 54.76465305 | 64 |
| 9.mlp | balanced_chebyshev / 128 | 1.4073185e-06 | 31.71923386 | 64 |
| 10.attention | power / 24 | 3.0659879e-07 | 75.99700699 | 128 |
| 10.mlp | balanced_chebyshev / 128 | 6.7080831e-06 | 41.40771924 | 64 |
| 11.attention | power / 32 | 4.0507021e-06 | 51.05737938 | 64 |
| 11.mlp | power / 24 | 1.0697647e-06 | 24.47769433 | 32 |

31 source-bound tests passed in1.09s, including exact conversion/product identities, domain endpoints and extended Chebyshev bounds, and tiny/large outward-square-root cases. A separate verifier converts the certificate back to ordinary powers and multiplies there, then checks exact Clenshaw extension envelopes and all48 public affine tensor hashes/coordinate maxima. All24 identities and envelopes passed; the complete verification wrapper took3.11s with eight unchanged source files.

The public derivation preserves six exact sources, archive SHA256 `97bb85bb50637320f544fad0e3caeeaa4203c468ae52f585be324c66c41015c9`. The independent verification archive is `1f052900b03cd36f8af6d941d6b09fc552bd1225e9eb2284a0324e456b4650f0`. The first attempt failed only while serializing an exact fraction exceeding Python's4300-digit limit. Its post-terminal source/log copy is retained in `failed-public-v1`; it was not a model or arithmetic-gate failure. Version2 increases the trusted public-integer serialization limit without changing the mathematical computation.

Run `python verification/verify.py public-v2` for offline source/polynomial/envelope checks. Add `--checkpoint /path/to/pinned/model.safetensors` to independently re-read all48 affine tensors and check every coordinate. Without that argument the verifier explicitly reports that learned tensor bytes were not re-read. The public asset SHA256 is recorded in `public-v2/report.json`.

This supplies the conditional ideal-circuit B128 obligation left open by the earlier [twelve-layer continuation plan](../bert_twelve_layer_continuation_plan_cpu_20261103/README.md). It does not prove that actual encrypted layer outputs, full-slot p60 refreshes, or all12-layer composition satisfy these bounds or the final benchmark gates.
