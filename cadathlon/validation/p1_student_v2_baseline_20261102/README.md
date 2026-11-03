# Complete P1 student-v2 baseline — 2026-11-02

The real upstream C3/C4 harness schedule passed with three repeats per case,
including cold setup, archiving, correctness checks, daemon cleanup and
calibration publication. The collector and evaluator both exited zero.

- External evaluator process: **260.375644706 seconds**; evaluator: **260.301572016 seconds**.
- Both cases processed the same 100-image public dataset on one/eight TPU v6 lite chips.
- Both achieved 96/100 accuracy and 100% agreement with the frozen ranked model in cleartext.
- Baseline self-score passed: 30 correctness points, zero speed points; report evidence points remain judge-assessed.
- Evaluation ID: `ad4afd10e6624268818234a4805bdab1`.
- All **103** measured source files were unchanged; exact bytes are stored below `sources/` with a `.txt` suffix to prevent test discovery.

`collection.json` records the command, environment, external timing, source
hashes, process status and offline self-score status. `source_inventory.json`
maps measured source names to exact archived bytes. `collector.py.txt` is the
collector snapshot. Source copying happens before the externally timed command;
offline scoring and evidence copying happen afterward. `collection_elapsed_s`
is therefore a separate collector duration.

`report.json` is the exact canonical evaluator report, byte-identical to
`calibration/report.json`. `baseline_manifest_student.json` and `archives/`
retain the published calibration and the aggregate case/run JSON it binds.
`evaluator.log`, `C3.log` and `C4.log` preserve raw output. `self_score.json` and
its stdout/stderr preserve the offline check. No private keys or per-image
labels, pixels or predictions were copied. The published proof in the problem
folder is the same evaluation and survives runtime-scratch cleanup.

The startup snapshot of the previous manifest is retained separately. The
older 262.14-second pre-v2 report, source provenance and aggregate archives
remain unchanged in `../p1_student_pre_v2/`; they are not relabeled as this run.
The current public aggregate is `../p1_student_suite.json`.

This establishes the complete two-case student schedule on the recorded host
and software. It does not establish a five-minute six-case legacy evaluation,
new-sample model accuracy, or performance on other machines. Repeating the
same images three times measures timing variability, not additional validation
examples. Child-process RSS is not a bound on the total peak memory of detached
resident services. Organizers must recalibrate and rerun candidates on their
judging hardware.
