The isolated collector follow-up fixes two reproduced outer-sweep failures:
an initial source-inspection error previously left an empty attempt directory,
and a final report-write error could replace the primary cancellation. It now
writes an explicitly failed/incomplete record before implementation, source or
package inspection and preserves primary errors through final inspection and
publication. A completed initial source hash is retained even when subsequent
package inspection fails. An unpublished success never returns success.

The final three-suite CPU checkpoint passed **90 tests**. It includes the
existing real subprocess cancellation, surviving-descendant, owned-directory
replacement, retained-parent and cleanup tests, plus twelve new metadata and
cancellation regressions. All **183 captured files** remained unchanged during
the final run; exact archive members were independently checked against the
start/end hashes. No encryption or TPU was executed by this checkpoint.

`incremental.patch` changes only the collector and its tests. It applies after
the 32-file composed candidate at
`/tmp/runtime-followup-integration-20261103/checkout`; all other copied Python
files match that base. The separate tested copy is
`/tmp/transformer-collector-outer-failure-20261103/checkout`. Neither measured
base nor live runtime was changed.

`initial_reproducer/` retains the original two controlled failures against the
unchanged composed candidate. `initial_regression_failure/` preserves an
incorrect test expectation: the last prior sweep snapshot contains five cases,
because it was written before case6; case6 has its separate completed process
record. The test was corrected without changing runtime. `intermediate_90pass/`
then passed. A subsequent metadata-consistency review split the recorded
initial-hash assignment from fallible package inspection, and the final
`result/` reran all90 tests with that additional regression assertion.

`before/` and `candidate/` contain exact changed-file bytes; the source archives
contain their full tested dependencies. `run_tests.py`, raw logs, JUnit and
`summary.json` describe every checkpoint separately. Transformer-agent
independent review found no further blocker. Consumers must still require
overall `passed` and `source_files_unchanged`: individual passing case records
or `all_six_passed` alone do not certify final provenance/publication.

The related [prepared sweep plan](../transformer_sweep_plan_20261103/PLAN.md)
distinguishes six toy fixtures from full pretrained/file-stage calibration.
No baseline, TPU result or full-model completion is claimed here.
