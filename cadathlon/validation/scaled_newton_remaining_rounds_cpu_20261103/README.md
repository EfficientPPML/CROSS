# Remaining scaled-Newton inverse components, secure CPU

The genuine encrypted run passed all thirteen update checkpoints over all
65,536 slots. Worst error against the exact binary64 circuit was
**5.3213987e-8** at the fixed **1e-6** gate. Four independent input segments
represent the historical K128/s4 remaining-round contracts; their outputs at
updates 9/12/13/13 differed from `4/(128*x)` by at most **5.3186645e-8**.

This preserves the older weighted, three-level Newton update. It does not
validate the newer integer-affine or tracked-residual schedule, compose four
normalization rounds, execute refresh or softmax, or establish model accuracy.

The actual N131072/CD2/depth41/p56/dnum3 ring has Q84/P27 and
log2(QP)=3147.9536016394, below the archived classical 128-bit ceiling of 3523.
Actual P coverage and execution rings were checked. No rotation keys were
needed. Each of four 16,384-slot segments repeats 258 distinct public endpoint,
adjacent, linear-grid, geometric-grid and inactive-midpoint probes. These are
finite sampled checks, not an all-input CKKS error bound.

All thirteen encrypted outputs were retained before any client decoding.
Levels were 40,37,34,31,28,25,22,19,16,13,10,7,4; actual scale metadata was
checked against the modulus-drop recurrence. The small positive excursions
above the ideal inverse bound 8 are preserved in the report. Server context
and cache contained no secret; both key-generation module aliases and six
server codec methods were guarded. Client and server were separate objects in
one process, not a file boundary or OS sandbox.

The internal run took **26194.746 seconds**, with **31,263,132 KiB** peak host
RSS on CPU cores136–139. Root observed session24289 exit0 and reaped it;
child954738 was independently absent. Cleanup reported no errors. These are
uncalibrated CPU development observations with concurrent work.

[The measured report](encrypted/report.json), exact 97-source archive,
process log and separate preflights are retained. All measured current source
bytes matched the before/after hashes and archive at root readback. The final
preflight used the same source closure. Earlier preflight/test records remain
separate. No keys or ciphertext payloads are archived.

Run `python verify.py` for a read-only archive, parameter, metadata and report
consistency audit. It does not rerun encrypted arithmetic or modify evidence.
To reproduce HE, extract `encrypted/sources.tar.gz` into a new directory and
run `check_remaining_rounds.py --output <fresh-directory>` with the recorded
CPU environment and dependencies. Fresh randomness can change measured
errors; the gate stays 1e-6.
