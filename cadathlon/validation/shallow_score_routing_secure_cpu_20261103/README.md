# Genuine shallow score-routing component

The frozen candidate passed eight encrypted checkpoints covering 524,288 slot comparisons at a fixed absolute-error gate of 1e-6. The worst error was 1.1303527003292402e-9. A full 65,536-slot input has 192 distinct live anchors and independent hostile padding; the public route uses key tile7, offset15, and gains1 and1/8. Forward routing consumes two levels, and its inverse consumes another two, producing checkpoint levels4,3,2,1 from input level5.

The actual N131072, Q12/P3, dnum4 profile has log2(QP)=430.5911 and passed the core classical128-bit security and P-coverage checks. The run used54 rotation keys,108 rotations and112 public multiplications across both gains. The dedicated CPU process took9259.62 seconds internally and peaked at9,721,912KiB host RSS. No independent external timing was collected. Session16776 exited0 and was reaped; its process and owned key directory are absent.

All99 source files stayed unchanged and match the archived bytes. Both key-generation aliases were blocked before server initialization. Six class codec entrypoints were blocked during the complete server phase; the separate `decode` entrypoint was not patched in this historical collector. Source review confirms decoding happened only after all eight encrypted outputs were complete. The server context was keyless, and the input payload and metadata were preserved. These are separate client/server objects in one process, not an OS sandbox or the requested file-stage integration.

This demonstrates one tile/offset and two gains. It does not establish all-key score accumulation, complete attention, a common deep-ring precision result, refresh, full-model execution or a calibrated speedup. The experimental helper remains outside the contestant release; callback and metadata hardening require separate review before reuse.

Run `python verify.py` for an offline integrity and recorded-gate check. The verifier does not repeat encrypted arithmetic. `report.json`, `sources.tar.gz` and `run.log` preserve the terminal collector output; `terminal.json` records root's process observation and session reap.
