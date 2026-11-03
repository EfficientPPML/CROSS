Final candidate2 collector dtype hardening preflight. The prior 66-test source/proof remains unchanged in `qwen_candidate2_collector_preflight_cpu_20261103`.

Root identified that a complex finite array could be converted to real float64 after discarding its imaginary part and falsely pass the metric. `v1_complex_reproducer.json` invokes only the exact archived v1 metric function and preserves that false pass. The new metric rejects both operands unless their dtype is floating, signed integer, or unsigned integer before any finite check or conversion. Eight regressions cover nonzero-imaginary complex, object, string and boolean arrays on either side. Existing overflow/cleanup/cancellation/prerequisite checks remain.

Only `check_approximate.py` and `test_collector.py` differ from v1. All selected coefficients, arithmetic/numeric functions and native capture remain byte-identical. The source-bound v2 checkpoint passed 74 tests plus 19 subtests in 5.63 seconds, outer 7.84 seconds, on CPU140–143. Session98228 was observed exit0 and reaped. All53 before/archive/after sources match. This preflight ran no checkpoint model and read no validation outputs. The authorized full416 calibration was subsequently launched with these frozen sources; its result will be archived separately and is not implied by this preflight.

Offline verification: `python verify_archive.py`.
