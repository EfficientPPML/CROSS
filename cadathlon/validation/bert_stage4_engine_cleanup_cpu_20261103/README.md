This isolated candidate hardens cleanup in the stage4 joined server engine. It is not promoted into the original file handlers, current key-generation source closure, or any active model proof.

The original engine `841fb5a5…` reproduced five failing regressions: an ordinary cleanup exception masked a later `KeyboardInterrupt` or `SystemExit`, and repeated TERM/HUP/INT interrupted an individual cleanup operation. Four existing-behavior checks passed in the same negative run. Exact original sources and the raw negative log are preserved.

The candidate defers all three cancellation signals on the main thread until executable owners and the NTT provider have been retired, restores prior handlers, then propagates cancellation. An existing body/setup exception remains primary. Without a primary exception, a cleanup cancellation takes precedence over an earlier ordinary failure, with the earlier failure retained as a note. Attribute lookup failures cannot skip later owners. The outer key loader still owns context disposal and its conservative failure quarantine; this change does not assert device readiness or release quarantined mappings.

All21 mocked lifecycle checks passed, including process-local repeated signals, handler restoration, both interruption classes, preserved primary errors and continued cleanup after attribute lookup failures. Eight source files match before/current/archive/after; the original two files remain unchanged. Commands terminated and were observed reaped (negative exit1, final exit0). No crypto, encoder/model forward, array transfer or numerical accuracy test ran.

`candidate.patch` applies to the original engine and adds the regression test file. Applicability was tested in another scratch directory, and resulting runtime/test bytes equal the validated candidate. Candidate engine SHA256 is `57b3e365a66da8bc999c577a88bcefef481c68b77afce0818a0fcf13b7b227a4`; patch SHA256 is `ff1a6704d723000eecc52589450937759c38d1d8b26582a5e1c306852a7c64c8`.

Signal deferral follows the existing dedicated-process key-loader policy. It is a lifecycle guard, not an operating-system sandbox or a general multithreaded cancellation guarantee.
