# Terminal-exit successor preparation

This isolated source copy starts from all 139 files in the reviewed candidate
preflight archive `7909d32548c7eea8974a9fdc21a508028d231716ef5c8cf6ed474b40a0d72901`.
Only runtime `entry.py` changes. Preparation metadata
`preflight_expectations.json` changes from 738 to 781 synthetic tests, adding the
43 focused boundary tests. The six numerical files, protocol, selected manifest,
worker/collector, lifecycle helper and all existing proof bodies remain exact.
The two additions are this scope document and `test_terminal_exit.py`, yielding
141 files: 137 exact parent files, two parent overlays, and two additions.

The old worker and collector publish their reports before the final publication
handle closes. A late ordinary exception exits 1, which collides with a normal
completed scientific failure. The report can therefore retain an eligible
failure flag although final publication cleanup raised. Empty stderr does not
generally prove a normal return: an injected `SystemExit(1)` is silent. No
ordinary unmodified `SystemExit(0)` path was identified; that injection is broad
boundary coverage, not an observed or known reachable production failure.

The new entrypoint parses command-line help/usage normally. Every exception
escaping subsequent source authentication, import or dispatch maps to status 70,
including `SystemExit(0)`, `SystemExit(1)` and diagnostic-output failure. Only a
normal exact `int` return of 0 or 1 yields a scientific status. The final intended
`SystemExit` remains outside that boundary. Error diagnostics contain the class
name only and are best effort. Status 2 remains argparse usage error.

Focused tests use the real original/successor CLI with an authenticated synthetic
helper containing the exact unchanged `supervise` AST. They cover both worker
and collector mode strings and verify that both production role functions end
by calling that helper. The identical 43-test probe yielded 36 expected failures
and seven passing controls against the preserved original entrypoint, then 43
passes against the new entrypoint. The focused 141-source snapshot and its
terminal receipt are retained separately before the test-count metadata update.
The planned full preparation checks all 781 synthetic tests plus ten installed
import denials. No probe runs a model, replays actual journals, or launches the
actual candidate.

This successor does not repair or reclassify any old result. The running original
139-source candidate, 44-source reconstructor, six-source issuer and 167-source
heldout preparation remain untouched. A future launch requires new externally
reviewed source/consumer pins and a separate grant; no old grant applies here.
