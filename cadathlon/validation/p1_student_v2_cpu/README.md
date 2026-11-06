# P1 student-v2 CPU verification

This checkpoint verifies evaluator, scoring, identity and release behavior
using CPU tests and synthetic report fixtures. It is not a new TPU run or
performance calibration. The existing 262.14-second historical run remains
unchanged and predates `student-v2` evidence requirements.

`report.json` records the exact command, versions, return code and before/after
source hashes. The pytest log/XML preserve 100 passing tests. Coverage includes
legacy-profile rejection, complete-run budget/cleanup requirements, missing or
changed reports/archives, model/input/software/device and schedule identities, publication
rollback after final I/O failures or deadline expiry, early environment
failures, process-group cancellation, and release dependencies.

`legacy_acceptance_before_fix.json` preserves the read-only reproduction that
awarded 30 student correctness points to legacy C3/C4 archives. The same
command's post-fix output is `legacy_rejection_after_fix.json`: nonzero exit,
zero points, and an explicit missing complete-evaluation error. No TPU ran
during either scoring command.

After the root agent grants the TPU lease and shared runtime integration is
finished, collect the new calibration from the repository root with a fresh
output directory:

```sh
env -u JAX_PLATFORMS CROSS_ROOT="$PWD" \
  python cadathlon/validation/collect_p1_student_evaluation.py \
  --output /tmp/p1-student-v2-baseline-20261102
```

The wrapper saves exact measured source bytes before timing starts, invokes
`evaluate.py --profile student --baseline --budget-s 300`, records the full
evaluator process wall time, and retains its raw log/report, aggregate archives,
immutable calibration report, and offline self-score. Its `collection_elapsed_s`
also includes evidence copying and offline scoring; these are separate from
`external_evaluation_elapsed_s`. The evaluator requires actual TPU devices and
publishes an immutable aggregate calibration only after the complete run and
final reporting pass. The wrapper forwards cancellation to the evaluator and
allows its owned worker/daemon cleanup to finish. `collector_preflight.json`
records CPU-only snapshot and cancellation checks; it is not a TPU result.

The equivalent standalone self-score command is:

```sh
python cadathlon/non_agent_pset/p1_benchmark/src/suite/score.py \
  --evaluation-profile student --label student-baseline
```

No historical source hash or report identity should be rewritten to substitute
for that new run. The root agent owns the accelerator queue.
