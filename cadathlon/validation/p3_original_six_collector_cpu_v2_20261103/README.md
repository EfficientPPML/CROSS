# Proposed original six-case P3 collector v2 — CPU lifecycle checks

This validation-only successor is prepared for a future exclusive TPU lease.
**No original P3 encrypted case was run by this evidence.** The existing six
original-case passes span historical source snapshots; this artifact does not
replace them with a current-source sweep or establish compact-v1 timing.

The candidate invokes each of `test1.in` through `test6.in` in a fresh process
with `--baseline --warmups 0 --repeats 1`. Its child shim verifies the actual
TPU backend before invoking the normal runner. The runner retains its genuine
ciphertext, live-ring security, numerical and owned Mapping-release gates.
There is no healthy-work deadline. Cancellation gives the owned process group
30 seconds to finish before escalation and reaping. A failed case is preserved;
source changes or incomplete cleanup prevent another case from starting.

Process-group and owned-directory cleanup are reused byte-for-byte from the
reviewed `check_transformer_suite.py` helper, with its checkout root configured
for this collector. The helper's captured SHA256 equals the live reviewed file.
The wrapper adds a small cancellation-aware P3 child shim and shields repeated
Ctrl-C while using the helper's existing TERM/HUP handling. It does not copy
fragile cleanup logic from the old four-case driver.

At actual launch the collector archives exact core, demos, P3 Python sources,
shared support, its own/helper sources and all twelve public fixture files.
Global and per-case hashes detect changes; raw JSON/stdout, stderr and GNU-time
RSS are retained for each child. All six passing runner reports, complete
cleanup and final source equality are required for overall success. Output
must be a fresh directory. No calibration or score is published.

The final isolated CPU suite passed **15 tests in 0.63 seconds**. All four
captured source files were unchanged and exact archive bytes were verified.
Tests use fake subprocesses and NumPy-free JSON records: complete sequential
collection, source mutation, failed cases, unknown cleanup, metadata errors,
publication failure preserving primary cancellation, wrong mode/profile/CPU
rejection, backend-free planning, and outer SIGTERM with child cleanup/reaping.
These are collector/lifecycle checks, not cryptographic or TPU passes.

To reproduce the tests, extract `sources.tar.gz` into a fresh directory and run
`python -m pytest -q test_collector.py`. The proposed collector currently lives
at `/tmp/p3-original-six-collector-v2-20261103/check_p3_original_six.py`; this archive
preserves the tested sources as `.py.txt` files to avoid accidental discovery.
Review is required before the future lease. The parent controls launch order;
do not interrupt the active transformer sweep or the separate CPU routing proof.

After explicit TPU authorization, a proposed command is:

```sh
CROSS_ROOT=/home/jianming_gatech/CROSS taskset -c 0-127 \
  /home/jianming_gatech/miniconda3/envs/jaxite/bin/python \
  /tmp/p3-original-six-collector-v2-20261103/check_p3_original_six.py \
  --repo /home/jianming_gatech/CROSS \
  --output /tmp/p3-original-six-post-promotion-20261103 \
  > /tmp/p3-original-six-post-promotion-20261103.collector.log 2>&1
```

The child environment explicitly selects TPU and the measured checkout.
Record actual CPU affinity, concurrent load and software at launch; no old-to-new
speedup should be attributed to these diagnostic timings. Keep failure evidence
and use a new output directory for every attempt.

V2 moves the initial attempt-report publication inside the protected try/finally.
A transient first write failure is now retained as failed attempt metadata by
the final publication path; no child starts. The added regression verifies that
the original OSError propagates while the recovered report remains failed.
The original 14-test candidate and archive are retained byte-for-byte in the
sibling `p3_original_six_collector_cpu_20261103` directory. `incremental.patch`
records the exact two-file change from that version. This v2 candidate remains
subject to parent review and the future exclusive TPU lease.
