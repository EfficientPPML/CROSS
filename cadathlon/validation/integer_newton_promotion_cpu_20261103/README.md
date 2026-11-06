# Integer affine / scaled Newton promotion candidate (CPU)

This is the six-file, isolated promotion candidate for `encrypted_integer_affine` and `encrypted_scaled_newton`, plus the explicit package manifest and release/P2-staging tests. No live runtime was edited by this validation. `candidate.patch` applies to the recorded main-tree base; its SHA256 is `98bbd58136f47e3078296fc92d1247fdc4c765aa3139b2f1d4f369732199f515`.

The helpers perform a public integer affine transformation with existing HE add/subtract/plaintext-add operations and a static, caller-planned scaled Newton sequence. They retain actual scale arithmetic; the first affine genuinely encodes and rescales. They reject bound-aware evaluators, implicit refresh, non-real/Boolean estimates, inadequate public encoding precision, and secret keys in the context, cache or parameter dictionary. The generator repeats the keyless/static checks before each subsequent primitive. It strips stale public semantic attributes from its own affine outputs and requires the caller to exhaust the generator. It does not derive a valid denominator domain or convergence certificate.

The API accepts an explicit positive integer correction coefficient. The measured inverse experiment uses K=128, s=4, C=32; it does not round K/s or imply encrypted coverage for other K. A separate public-only all-K certificate is not an encrypted test of this API.

## Final evidence

- `focused.json`: 88 API, packaging, extracted-release import, and staged-helper provenance tests passed, exit 0, in 5.923 s. All 434 exact source/fixture bytes match before/archive/after/current verification. The tests include rejection of bound-aware owners and secret-key injection between yields before another multiply occurs.
- `secure-v2/report.json`: actual N=16384, CD2, Q8/P3, dnum=4; log2(QP)=318.991937, integer bound 319 <= 438 for the declared 128-bit classical profile. Eighteen checks passed in 14.070 s on CPU cores 152–155. Six integer-affine variants match the residue oracle exactly; the two helper-generated updates match the direct circuit exactly. All 8192 slots passed the unchanged 1e-6 gate, worst absolute error 9.1255359e-10. Input immutability, scale/level/NSD checks, keyless server guards and cleanup passed. Session 46835 was authoritatively reaped with exit 0. All 101 source bytes match before/archive/after/current verification.
- `measured_source_intersection.json`: the six proposed paths have no intersection with the recorded 103-file P1 or 119-file P3 measured source inventories, checked before candidate edits.

This final source was tested for **two Newton updates**, not a converged full inverse. The reported true-inverse error after two steps is diagnostic and is intentionally not a success gate. The earlier successful 13-update K=128 experiment is preserved separately in `../integer_newton_thirteen_cpu_20261103`; it used earlier helper sources. This archive establishes no complete softmax, refreshed inverse, full model, TPU performance, or speed comparison.

## Preserved earlier candidate checks

`tests-v1.*`, `pre-guard-focused/`, and `secure-v1/` preserve the preceding 81-test / two-update candidate before the final owner/key-injection guards. That secure session exited 0, but its sources are not substituted for the final candidate. `tests-v2.*` is the preliminary 88-test result; `focused.*` is the final source-bound repetition after documentation was finalized.

The complete focused source archive is a copy of the tracked public test closure. It includes existing public fixtures, goldens and a public P1 model checkpoint needed by packaging tests; it is not a contestant release. No newly generated private input, secret key, evaluation key or ciphertext artifact is included. The secure archive contains exact source files only. The release and staging tests independently check their exported inventories and exclusions.

Run `python verify_archive.py` here to validate the recorded archives and candidate hashes without importing the crypto runtime or requiring a device. `audit.json` records the archive-time comparison against the original immutable scratch files. The focused command is retained in `focused.json`; the secure command was `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 152-155 /home/jianming_gatech/miniconda3/envs/jaxite/bin/python check_component.py --output secure-v2`, run from the scratch root.
