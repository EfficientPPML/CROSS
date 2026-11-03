Two genuine CKKS CPU runs evaluated one fixed scaled Newton update on the same
105 public (K, x, initial-error) combinations, repeated across all slots. The
keyless server computed x*z, then 2−(K/4)*(x*z), then z times that correction.
Client decoding occurred only after the fixed server circuit completed.

| Profile | Ring / actual log2(QP) | Product error | Correction error | Update error | Fixed 1e-6 gate |
|---|---|---:|---:|---:|---|
| Lower precision: scale40, first48 | N8192, Q8/P1, dnum8 / 196.49564 | 6.9303e-7 | 1.9322e-5 | 1.15975e-4 | Failed |
| Higher precision: scale56, first61 | N16384, Q8/P3, dnum4 / 318.99194 | 1.6847e-11 | 5.2588e-10 | 4.1656e-9 | Passed |

The respective QP integer bit lengths, 197 and 319, are below the archived
128-bit classical ceilings of 218 and 438. Exact P coverage was independently
rechecked. All 4,096 lower-profile and 8,192 higher-profile slots were checked;
expected output magnitudes reached 7.9992. Scale, level, and NSD metadata checks
passed in both runs. The same 1e-6 gate was retained throughout.

The lower run completed in 9.04 seconds and its exit code 1 was observed by the
parent orchestrator. The higher run's report records success in 10.65 seconds;
its numeric exit code was not recoverable after the exec session closed. Both
driver processes were independently confirmed absent. No exit-zero claim is
inferred from the higher run's report.

Both reports record successful cleanup and guards against server encryption,
decryption, and secret-dependent key generation through both module spellings.
Each exact 98-file source archive was checked against its start/end hashes and
copied driver. The 97 runtime sources and the public fixture/circuit source
fragments match between runs; only the ring, precision, and layout settings
changed. `driver_profile_difference.patch` records those changes. The original
reports, archives, and raw logs remain byte-for-byte unchanged.

The failed profile is concrete evidence of precision amplification. The passing
profile establishes one update only, not a full reciprocal, thirteen iterations,
refresh, normalized softmax, or BERT. Fresh keys and different ring profiles
also prevent a controlled performance comparison. The client and server are
separate objects in one process, not an OS sandbox.

`verification.json` records archive SHA-256 hashes, security arithmetic,
provenance checks, process absence, and the exit-status distinction.
`verify_archives.py` repeats the read-only archive/report audit without
re-executing encryption. It checks recorded error/gate consistency, not the
original ciphertext values, which are not included in this evidence.
