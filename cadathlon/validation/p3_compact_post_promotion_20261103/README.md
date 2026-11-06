# P3 compact-v1 calibration after runtime promotion — 2026-11-03

The complete two-model compact-v1 encrypted schedule passed, including cold
setup, one warmup and three measured requests per model, fixed numerical and
128-bit actual-ring checks, owned Mapping release and process cleanup.

- External evaluator process: **145.001872417 seconds**; internal evaluator: **144.906202598 seconds** (300-second budget).
- Fixed-fixture maximum absolute errors: `student1.in`: 1.39457131e-08, `student2.in`: 1.22242128e-08; top-1 agreement 1.0 for both.
- All **119** measured source/fixture files were unchanged; exact bytes are in `sources.tar.gz`, checked against both source manifests.
- Baseline self-score: **10/20**, comprising 10 correctness points and zero speed points.
- Root authoritatively reaped collector tool session 96737 with exit 0. Wrapper/collector/evaluator/worker PIDs 940846/940852/940853/940854 were absent. The archiver independently checked collector/evaluator PID absence and the final worker/evaluator success records; no timeout, cancellation or forced kill was recorded.

`verification.json` binds every copied artifact by SHA256 and records the
independent archive/source, model/fixture, sample-count, actual-ring metadata,
budget and self-score checks. `verify_archive.py.txt` is the exact verifier used;
it performs no numerical execution. `process.json`, `evaluator-process.json`,
`stdout.json`, `stderr.log`, `evaluation/worker.json`, `evaluation/worker.log`
and `evaluation/report.json` retain the complete observed process evidence.
Only an explicit whitelist of source snapshots, aggregate reports and raw
logs was copied. No keys, ciphertext payloads, images or per-image output arrays
were copied. The older 148.35-second archive remains unchanged.

This run used CPU affinity **0–127** with the exclusive TPU lease while a
separate CPU routing proof continued on **144–147** (PID 735609, about 15 GiB RSS
and four busy cores). The earlier run used a different CPU allocation. This is
a fresh complete-schedule validation, not an isolated optimization speedup.
The fixtures use the two published compact CNNs; this does not establish trained
model accuracy, the legacy six-case schedule's timing, or performance elsewhere.
Source copying and offline self-scoring are outside the separately reported
whole-evaluator wall time. Setup, inference, artifact creation and cleanup are
inside the evaluator's 300-second boundary.

The parent owns updates to public aggregates, handouts and release packaging.
This directory preserves the exact observed attempt and its verification.
