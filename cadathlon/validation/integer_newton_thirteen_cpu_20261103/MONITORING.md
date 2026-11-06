# Active isolated integer-correction inverse CPU proof

Authorized by root after actual Q54/P7 security/P-coverage, public scale/storage
preflight and 14 source-bound contract tests. No TPU use.

- Security-review agent owns session30428 and authoritative reaping.
- Shell PID969114; child PID969120; affinity152–155; explicit CPU backend,
  BLAS/OMP caps1. No healthy-work deadline.
- Output `encrypted/report.json`, `encrypted/sources.tar.gz`, `encrypted.log`.
- All99 source files in ready.json/actual startup archive are frozen until exit.
- Existing root-owned original proof session24289 and TPU sweep session12657
  are separate and must not be interrupted or modified.
- Scope: thirteen updates on four public domains in four8192-slot segments;
  declared results at updates9/12/13/13. Every32768-slot checkpoint must meet
  fixed1e-6 against its float circuit, then declared segments against4/(128x).
- Initial scale is realized by actual affine PTCT/rescale; integer corrections
  use validated add/sub/plaintext-add. No metadata scale relabeling, refresh,
  composed normalization, fullsoftmax or model claim.
- Different ring/layout and uncontrolled shared-host load preclude a controlled
  speedup comparison to the original proof.

Numeric exit and terminal cleanup are not yet known. Do not infer exit0 from
process absence; reap the owned session and audit the final evidence.

Terminal audit: session30428 returned exit0 and was reaped. Shell969114/child969120 absent. Final gates passed; all99 source hashes verified, cleanup empty. Archive `/home/jianming_gatech/CROSS/cadathlon/validation/integer_newton_thirteen_cpu_20261103`.
