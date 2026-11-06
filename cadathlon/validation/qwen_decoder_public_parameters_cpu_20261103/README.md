# Remaining public Qwen decoder parameters

**PASS, public parameter preparation:**114 named matrices in113 physical files,
including all28 O projections,84 gate/up/down projections, and the complete
151936x1024 embedding/vocabulary matrix. The pinned checkpoint stores embedding
and vocabulary weights separately. Every original BF16 byte matched before the
exporter shared one output body; configuration alone was not accepted as proof.

The actual export completed in26.21499 seconds from ten unchanged source files.
Session81225 exited0 and was reaped. Its direct child exited0, was waited, and
its owned process group was absent. Root independently verified source/archive
and terminal evidence, all file/payload hashes, and every saved value against
the checkpoint. This covered478,543,872 unique FP64 values and634,126,336 named
values when the two tied tensor names are counted separately. Both checkpoint
hash passes matched. No model construction/forward, backend, keys, codecs, HE
or full-model numerical evaluation occurred.

The public parameter manifest SHA256 is
`60c556376e42358f227a2978edb6b2addf59dc9e694b65e17265b15e813d65c3`.
The completed parent report SHA256 is
`0f9c089cc4e4e63b626f264ae5bdc6e5c0b1c8fa2c0af6c85fee01a81e888be6`;
its source archive SHA256 is
`1b921363993f5a06eced87ffa5f44b284b51100a46af43dd58e6788cb806a175`.
Bodies remain in `/tmp/qwen-decoder-parameters-20261103/actual-v1/public_matrices`
and are excluded from this archive. The prior84 QKV matrices and113 norm gammas
remain separate authenticated artifacts; this export does not replace them.

Final preparation passed53 synthetic format/lifecycle tests, including all
finite BF16 bit patterns, bounded chunks, truncated/corrupt data, exact tied
alias checks, FD handoff and cleanup priorities, and actual CLI report paths.
An independent eleven-source capsule contains the same ten candidate files plus
its review tests: nine checks passed, including two real first-SIGTERM injections
at synthetic collector cleanup boundaries. These tests used no real model or
child computation. Main-thread cleanup shielding is not process-wide atomicity.

History is retained: the initial eight-source preflight passed47 checks, then
two independent expected failures exposed unshielded collector finalization.
The first ten-source successor passed52 checks and failed one new malformed-
return test because failure publication masked its original exception. The
final successor fixes both issues and republishes outer termination failures.
No earlier failed result is relabeled. The intermediate independent successor-v1
is retained; successor-v2 binds the authoritative full ten-file candidate.

Run `python verify.py` for an offline metadata/source check. It reads no matrix
body or checkpoint and imports no model, NumPy or encryption library. Export
time is preprocessing time, not encrypted execution or ranked benchmark time.
