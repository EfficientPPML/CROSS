# Qwen candidate terminal-exit successor — public preparation

**PASS: 141 frozen sources, 781 synthetic tests and ten installed import
denials.** Owner session `29701` exited 0 and was reaped. Current, execution,
start/end and archive hashes agree; the observed collector and worker process
groups are absent. Root independently verified the proof. This is preparation
only: no actual candidate, model, calibration, validation or private-journal
work was performed, and no execution grant is included.

The original candidate entrypoint could conflate a late publication-close error
with a normal scientific failure: both exit 1 after a failure report was saved.
The preserved exact-function subprocess negatives demonstrate ordinary
`OSError` and silent injected `SystemExit(1)` collisions. An injected
`SystemExit(0)` also tests the wider PASS boundary; source review found no
ordinary unmodified late `SystemExit(0)` path. None of these tests establishes
an actual failure in the original run.

The successor allows scientific statuses 0/1 only after a normal exact-integer
return from authenticated work. Escaped `BaseException` and malformed returns
produce infrastructure status 70. Diagnostic failures cannot change that status,
and diagnostics omit arbitrary exception text. Argparse help/usage retain 0/2.
The final deliberate `SystemExit` remains outside the protected work.

Relative to the original 139-source preparation, 137 files remain exact.
`entry.py` is the only runtime overlay; `preflight_expectations.json` changes
738 to 781. The additions are the 43-test boundary probe and its scope document.
The six numerical files, protocol, selected candidate manifest, original
worker/collector and cleanup helpers are unchanged. See [SCOPE.md](SCOPE.md).

The identical real-entry probe had **36 expected failures and seven passing
controls** against the preserved predecessor, then **43 passes** against the
successor. It invokes the actual CLI with authenticated synthetic helpers
containing the unchanged `supervise` function, for worker and collector mode
strings. It does not execute a full numerical worker/collector. All 42 probe
children were waited on and their groups checked absent. Earlier development
41-test results and all three narrow negative capsules remain separate history.

The old running 139-source candidate, 44-source reconstructor, six-source freeze
issuer and heldout preparation remain unchanged. This successor cannot repair
or reclassify an old outcome. A failed original139 result does not gain revision
authority from this proof. Future use needs independently reviewed source and
consumer bindings plus a separate grant.

Evidence pins:

- Source manifest: `45cd92473d9ba558288651d59d4f089c7fb9ba37650802848cad24687beaa437`
- Report: `c847b64d3cada0add4071fafe5c6ce79f2b70b0db6da7abcde41adf48f853ab9`
- Source archive: `04edc1012ef8d503b878979c21cdeb997c2dd6e098a9407e8d59d9328ddb6c12`
- Worker report: `316b887ae7acdb1c00db66e927371542cfffcab3358e620a0a4b7d478f28a155`

Observed collector PID/PGID was `2988850/2988850`; worker was
`2988851/2988851`. A separate wrapper PID was unavailable; parent `16918` is
tool infrastructure and is not claimed as owned or retired. Exact observations,
owner terminal receipt, root review and independent source/terminal reviews are retained.
The peer terminal review initially counted four pytest `current` directory aliases
twice; its original review script and disclosed count correction are preserved.
No production or test source changed for that reviewer-only correction.

The whitelist includes public source archives, aggregate reports, XML/logs and
review scripts only. It excludes private input/journal/model bodies, expanded
source copies and temporary test trees. Historical synthetic report files and
stderr are explicitly synthetic. Public prerequisite assessments remain inside
the exact source archive, with their original 64 MiB exceptional member bound.

Run `python -I -B verify.py` for portable verification, or add `--current` to
compare the frozen lab's current bytes. Neither command follows private
descriptors or executes model code, reconstruction or tests.
