# Isolated routing guard repair

The measured routing helper accepted boolean noise-scale metadata and could return a result after a progress callback altered the input or output metadata. `baseline_reproduction.json` records these gaps against the exact immutable module from the successful encrypted component proof.

A separate candidate now checks exact integer noise-scale degree, levels and layout dimensions; finite positive representable scales; borrowed arithmetic/context/cache identities and public parameters; and fixed public tile/offset values. Every stage validates its result before invoking progress. After a callback, it checks both the result and original input metadata/payload identities and rejects changes before another primitive executes. It also rejects key injection, implicit refresh, newly bound-aware owners and reentry. The active-operation state clears even when the callback raises.

The source-bound test run passed252 checks, including all128 tile/offset mappings, hostile padding, both callback positions and route directions, input/output metadata substitutions and failures before subsequent arithmetic. The baseline reproduction and exact archive readback also passed. Session77636 exited0 and was reaped; the wrapper took1.8443 seconds. Independent read-only review found no concrete blocker.

No encrypted rerun occurred and no main or release module changed. The earlier genuine result remains evidence for its exact original source bytes. The new guards preserve the routing arithmetic but remain a separately tested candidate for later composition. These callback guards help catch accidental state changes; they do not isolate arbitrary Python code.

`python verify.py` checks the stored reports and source archive without running encrypted operations. The source tar includes the original module, repaired module, independent tests and source-bound test wrapper.
