# Expanded-prefix input freeze: preparation

This separate stage prepares the reviewed Qwen prefix protocol. Execution is
disabled until source-bound public tests and independent review finish. No
tokenizer asset, old token/corpus body, completed model report, learned tensor,
coefficient selection, model forward or HE operation has been read or executed
to prepare this design. Only pinned public protocol/code/hash metadata and the
new authored validation text recipe have been copied.

The eventual worker authenticates the five externally pinned input files and
the exact tokenizer assets before deriving input data. Calibration is the same
96 original prompt IDs in original order: each original S32 token record supplies
prefixes1..32, with exact original S8/S16 equality and original16 decode seed IDs
verified. No calibration prompt is retokenized to choose a replacement input.
All original288 records and all96 natural lengths remain mandatory.

The worker constructs tokenizers.Tokenizer.from_str from the authenticated local
JSON bytes, avoiding a second unbound path read, with add_special_tokens=False
for the32 new authored validation prompts. Every natural token sequence must
have at least32 tokens and every token must be an exact nonnegative integer
below151936. It creates1024 ordered prefills and eight explicit16-token seeds.
Repeated short prefixes remain in their original entry denominators; entry and
unique sequence counts are reported separately for every length.

The3200/1088 calibration/validation counts are workload entries. A tokenizer
cannot produce future teacher or free-running suffixes. It freezes3072/1024
prefill entries plus16/8 seed bodies, and the declared four-step append-only
decode schedule. Actual generated contexts belong to later model executions.

Disjointness compares every known new validation prefill/seed to the expanded
calibration prefixes, all three old validation input sets and row0/tokenID0 S1.
It additionally proves disjointness of future append-only decode families:

* Every new prefill of length>=16 must have a first16-token prefix different
  from every old/calibration decode seed.
* Every new decode seed must differ from the first16 tokens of every excluded
  known context of length>=16 and from every old/calibration decode seed.
* The same new seeds must differ from every expanded calibration seed. Any
  future extension retains its seed, so it cannot equal an excluded context
  or any excluded append-only family when their first16 tokens differ.

This stronger input-only condition is fixed before reading token bodies. It
avoids opening completed model reports or inspecting old logits/ranges/token
choices. Old decode seeds must be derived from authenticated source input
records and their pinned append-only workload definitions, not assumed from
generic S8 provenance. Unknown/missing schemas or decode-seed mappings fail
preparation. The future evaluator must preserve append-only semantics and
record actual token contexts; a changed decode policy invalidates this proof.

The worker never replaces prompts, drops duplicates or relaxes overlap checks.
Insufficient length, tokenizer mismatch or overlap leaves a complete failed
input-freeze report and requires a separately reviewed design revision before
any model run. Equality/prefix disjointness is not semantic independence or a
generalization guarantee.

All token bodies remain in a fresh owned0700 output directory with0600 files.
Public evidence contains source archives, input/tokenizer digests, ordered
mapping descriptors, counts and overlap summaries; it excludes original text,
token bodies, model metrics and derived slot arrays. It retains partial owned
outputs on failure and never edits or removes external inputs/assets.

The fresh worker binds an external manifest digest and exact source closure
before helper/runtime imports or output publication. Input readers use bounded
descriptor reads, regular-file/identity checks and primary-preserving cleanup.
Output publication is tied to the created inode and must finish after cleanup
and outer termination restoration. Public preflight uses synthetic schemas,
sentinel readers and tokenizers, plus negative import/data-access guards. No
actual input or tokenizer access follows from preparation alone.

Next steps are implementation, source-bound synthetic preflight, independent
review, then a concrete tokenizer-only launch. Expanded native/exact observation,
coefficient selection, approximate calibration and validation remain separate.

The implementation uses one isolated Python -I -B entry. An externally pinned
manifest authenticates all source bytes and rejects extra files/directories,
symlinks, preloaded local helpers and local aliases before helper imports.
Authenticated bytes supply local imports directly, preventing a later path
replacement from substituting helper code. Model/backend imports and networking
are denied. Preflight and collector audit guards deny original input/model body
reads; the actual worker permits only the five pinned inputs and four tokenizer
assets, read-only. Dependency versions are checked before tokenization.

Body cleanup shields HUP/INT/TERM. Genuine body exceptions retain identity;
cleanup-only cancellation takes priority over earlier cleanup-only ordinary
errors. Healthy child waits have no deadline. Exceptional retirement addresses
the owned process group even if its direct leader has already exited. Public
and private output paths must be canonical, disjoint, and outside source/input
parents and tokenizer assets. The original output directory inode is retained
across body cleanup and must match the final publication reopen.

A report cannot certify its own final publication close or interpreter exit.
Acceptance therefore requires the exact persisted worker report hash, independent
parent wait/exit0, and owned-group absence. Collector acceptance additionally
requires root's authoritative outer terminal/reap observation. Report publication
failures remain nonzero; a self-reported success field alone is insufficient.

This separate successor preserves the32-source preflight and its two follow-up
negatives. Local module origins resolve through symlinks before admission.
Cleanup-only exceptions carry a shared internal origin marker across helper
boundaries, preserving later cleanup cancellation priority without replacing
a genuine body exception.
