# Ranked BERT private references: actual CPU preparation

PASS, exit0/reaped. The frozen candidate evaluated408 unique original examples
and produced ordered private references for all nine ranked cases: sizes
100/200/408 × seeds42/43/44, totaling2124 case records by reuse. The three408
cases have the same original ordering; they are not independent datasets.

| Size | Seed42 correct | Seed43 correct | Seed44 correct |
| --- | --- | --- | --- |
| 100 | 82 | 79 | 84 |
| 200 | 162 | 172 | 166 |
| 408 | 345 | 345 | 345 |

Official FP32/default-SDPA and frozen approximate predictions agree for every
unique example. All nine reference-preparation logit-tolerance and strict
prediction-agreement checks passed; these are additional plaintext diagnostics,
not extra ranked HE acceptance requirements. The official ranked gate remains
at least75% label accuracy and at most2percentage points below the official
cleartext accuracy, with prediction agreement reported as a diagnostic.
Class1 wins an exact logit tie, and emitted binary labels must match the
decrypted logits in original input order. All nine accuracy gates passed;
the408-record accuracy is84.5588%.
Zero approximation execution/domain failures were recorded. Independent exact
NumPy computation uses only original-position compact embeddings; every valid
row at all12 native FP64/eager attention/block checkpoints, pooler and logits
passed the unchanged1e-12 absolute gate, and embedding parity passed gate0.
Native inputs retain the original full128 tokens/mask. No padded-query or
ciphertext/noise claim is made.

All41 measured source files, six public assets and54 original private fixture
files remained unchanged. Actual default-SDPA and FP64/eager implementation
classes were checked and recorded. Complete private publication/readback,
retention, source/asset/input/cleanup/terminal/error-absence flags all passed.
GNUtime PID1879300 and native collector1879301 were absent after session2764
was reaped with exit0. Sources.tar.gz SHA:
`ad8b6c4a84c1ba6277dde34da6a27009c4ed0e97661d7140a461e4fb0ce64ee2`.
Report SHA:
`77f93c90e03609bdaf12441cc8a8d07e08b5e1a475c1480f283b4f592668ae74`.

The0700 retained workspace and its11 private0600 files are identified in the
public report. It contains the master, nine ordered case references and a
manifest written last. Raw IDs/labels/logits/diagnostics remain there, outside
this archive. The archive contains only aggregate reports, source bytes,
launch/terminal readback and process logs. The archiver did not read any private
reference or original input bodies. Consumers must require every explicit
reference_acceptance_requires flag in the final report, not merely an
in-memory evaluation-complete flag.

This is client-only plaintext reference/approximation validation, not HE,
file-stage completion, encrypted ranked accuracy, release readiness or a
benchmark/per-case native timing result. Reusing two native models and computing
unique examples once was solely for reference production. The352.90163s internal
elapsed time is retained as process evidence. Active encrypted jobs on separate
CPU affinities continued concurrently.

Preparation evidence,69 mocked tests and12 installed denials are separately
preserved in ../bert_ranked_client_reference_preflight_cpu_20261103.
No main/runtime/release files changed. Run `python verify.py` offline.

Root independently reread all11 produced private files and54 original files,
verified0700/0600/inode/hash/order/positions/labels, and recomputed every gate
without repeated inference (session9977, exit0/reaped, as reported by root).
All63,648 domain observations stayed within frozen domains. The largest exact
native64 checkpoint error was1.2789769243681803e-13; the largest approximate
versus officialFP32 logit error was0.0033541579154104384. The public aggregate
and executed verifier source are included; no private bodies are included.
This review is separate post-terminal evidence. Its JSON was written after the
41-source current-file check; it does not redefine that measured inventory.

The original9977 audit wrote its aggregate JSON at the lab top level, where the
dynamic source enumerator would include it in a future check. Root moved only
that post-terminal output under root-readback/ and reran the full private
readback (session47924, exit0/reaped): identical aggregates, original41-source
inventory restored, no model rerun/runtime edit. root_readback_successor/
contains the corrected verifier/report; the original9977 files above remain
unchanged historical evidence. Offline verify.py checks both reports without
opening private data. The optional private verifier should use the successor.
