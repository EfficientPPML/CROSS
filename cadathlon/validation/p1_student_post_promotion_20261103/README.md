# P1 student-v2 calibration after runtime promotion — 2026-11-03

The complete C3/C4 encrypted harness schedule passed: three runs per case,
100 images on one/eight TPU chips, including setup, case archival, numerical
checks, owned worker/daemon cleanup and calibration publication.

- External evaluator process: **257.561762272 seconds**; internal evaluator: **257.467303654 seconds** (300-second budget).
- Evaluation ID: `9250e83535a84d98a3918c629d64380c`.
- Both cases achieved 96/100 accuracy and 100% agreement with the frozen ranked model in cleartext.
- All **103** measured source files were unchanged, with exact bytes under `sources/` and SHA256 inventory in `source_inventory.json`.
- Self-score: **30 execution points, zero speed points**; remaining P1 evidence points are judge-assessed.
- Root authoritatively reaped collector tool session 68201 with exit 0. Wrapper/collector/evaluator PIDs 919898/919904/919905 were absent; no matching harness or resident-service process remained. This archiver independently checked recorded collector/evaluator PID absence and the evaluator's cleanup-success report.

`verification.json` records independent exact-source, archive-hash, schedule,
identity, security-metadata, budget and self-score checks. `verify_archive.py.txt`
is the exact verifier source used for this copy. It performs no cryptographic
execution. `report.json` and `calibration/report.json` are byte-identical;
`baseline_manifest_student.json` points to that evaluation ID and binds the
same case identities, software versions and hardware allocations. The eight
aggregate case/run JSON files are in `archives/`. The prior report and baseline
manifest are retained as startup snapshots, without changing the old archives.
Only explicit aggregate JSON, source snapshots and raw diagnostic logs were
copied. No key bytes, ciphertext arrays, images, per-image labels or predictions
were copied. Filenames and hashes describing such inputs remain public evidence.

This run used CPU affinity **0–127** and the exclusive TPU lease while an
independent CPU routing proof continued on **144–147** (PID 735609, approximately
14.9 GiB RSS and four busy cores). The prior 260.38-second measurement used a
different CPU allocation; this result is a fresh complete-schedule validation,
not an isolated optimization speedup. Repeated runs use the same 100 images and
measure timing variation, not additional accuracy samples. It does not establish
legacy six-case evaluation latency or performance on another machine.

The collector's offline snapshot/copy and self-score phases are outside its
separately measured evaluator process wall time. The evaluator's entire 300-second
boundary includes setup, inference, artifact creation, cleanup and publication.
The parent owns updates to public aggregates, handouts and release packaging;
this directory preserves the observed attempt and does not itself publish a
new competition baseline.
