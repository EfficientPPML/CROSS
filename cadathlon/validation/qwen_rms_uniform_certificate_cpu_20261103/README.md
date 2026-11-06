# Frozen Qwen RMS polynomial certificates

All eight selected post-attention inverse-square-root polynomials have a certified uniform exact-real relative error below `1e-6` on their unchanged declared positive domains. The largest outward upper bound is `5.029028745383948e-7` (layer27). The source-bound run exited0 in298.12s;14 pure regression checks passed. No model, JAX, keys, encryption, or refresh ran.

The certificate uses the actual frozen `evaluation_coefficients` and binary64 affine constants, pulled back exactly with rational arithmetic into Chebyshev coefficients. It forms `H(v)=v*p(v)^2−1` and bounds its magnitude uniformly by the exact coefficient L1 sum δ. A positive midpoint and δ<1 establish positivity throughout the interval; the relative error is then bounded by `δ/(2−δ)`. Directed binary64 multiplier endpoints enclose the exact rational result. Point checks only validate algebraic identities; the uniform bound does not rely on a grid.

Subsequent binary64 arithmetic rounding, CKKS encoding/evaluation noise, refresh error, and propagated model-domain membership are excluded. This is a certificate for eight fixed public functions, not arbitrary inputs or model/HE readiness.

`report.json`, `sources.tar.gz`, `run-v2.log`, and `verification.json` are authoritative. The archive preserves all four exact inputs (script, test, pinned manifest, pinned evaluator); their hashes match before/after/current bytes. The pinned manifest SHA is `d16cfe0eaf4079b1e782790dcff277ed6901aa7ea1376ee7877b40b54dde93a2`.

`v1_crude` preserves the earlier valid but looser δ upper bounds and an ancillary nearest-rounded lower multiplier endpoint, which is not authoritative for directed enclosure. V2 uses the tighter rational formula and directed endpoints. `development_failures` records a corrected test expectation, not a numerical-certificate failure. `affinity_change.json` records moving the earlier run away from the subsequently started encrypted job; the final run used CPU152–155.

Reproduction: restore the pinned manifest/evaluator from `sources.tar.gz`, set `QWEN_FROZEN_MANIFEST` and `QWEN_FROZEN_ARITHMETIC` to those files, place `certify.py` beside `test_certify.py`, run the test and then `python certify.py`. The script writes its new report/archive in its own directory; preserve this evidence directory unchanged.
