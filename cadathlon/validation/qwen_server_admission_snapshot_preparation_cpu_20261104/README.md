# Qwen server admission snapshot: recording-state preparation

The corrected preparation passed all 18 focused cases on the exact 38-file
test closure, including 28 unchanged runtime files. Root owned session 91616,
observed normal exit 0 and reaped it (launch `e197dd`, terminal `70406f`). The
reported elapsed time was 1.698390556 seconds for this small recording-state
suite. This is not a performance measurement. The independent observation was
after exit; no independent live-process observation is claimed.

The tests construct the exact 13 admission types using intentionally invalid
synthetic marker bytes. Only PublicContract construction bypasses its real
validator. ServerPacket.check and the snapshot's saved validator temporarily
use the same disclosed recording function, and every successful child verifies
restoration of the original validator identities. The cases cover retained
identity, mutation rejection, callback identity, caught and uncaught reentry,
sticky failed capture, wrong thread/packet and clean retirement. Full scientific
admission is untested. No NumPy/backend, model, key generation, capacity, private
proof, calibration, heldout, row0 or HE execution occurred in this suite. The
guard was not integrated into the actual server by this evidence.

- [Measured report](measured/report.json), [XML](measured/tests.xml) and
  [source manifest](measured/sources-before.json) retain the successful result.
- [Root terminal review](root-review/passed/terminal-review.json) and
  [direct owner receipt](root-review/passed/direct-owner.json) bind the terminal
  result. The [independent terminal readback](peer-review/passed-terminal/review.json)
  has its own post-exit observations and does not claim root's wait authority.
- [Owned child records](measured/test-child-terminals.json) join 18 early parent
  receipts, 18 positive child terminals, 18 results and 18 logs. All 72 regular
  files are retained under `measured/owned`. The
  [metadata manifest](measured/owned-pytest-metadata.json) records 17 owned pytest
  aliases without following or copying them. Two validator cases share pytest's
  truncated prefix, so 17 aliases is expected for these 18 cases.

The historical guard v1 had two source-reviewed gaps: caught reentry could lose
its sticky failure and a failed capture could lose original graph retention.
[Guard v2](history/guard-v2/guard.patch) addresses those gaps. Both source
archives and the [scoped peer findings](peer-review/guard/review.json) and
[successor review](peer-review/guard/review-v2.json) remain unchanged. No runtime
negative reproduction is claimed for guard v1.

Test source history is also preserved. The original 36-file preparation gained
one unconditional child PGID assertion in the 37-file successor. That run failed
all 18 cases in PublicContract.__post_init__, before the recording seam or guard,
then the collector failed because child terminal records were missing. The
[failed 44-file retention index](failed/files.json),
[failed XML](failed/evidence/tests.xml), logs and
[root failed-terminal review](root-review/failed/failed-terminal-review.json)
preserve this outcome. It supplies no guard result or child PID/PGID absence
proof. The 38-file successor explicitly bypasses only that synthetic constructor
and publishes an early process receipt before asserting a child's normal exit or
reading its result. Runtime28 and the intended 18 guard assertions are unchanged.

Historical handoffs and source-only readiness flags describe their capture time;
they are not rewritten after execution. The root reviewer’s original whitespace
assertion failure and corrected script are retained as reviewer history, with no
product source change. Preparation v1 used a compact manifest; the held v3 source
and preparation v2 use the exact sorted indent=2 manifest. Both byte encodings
and their original hashes remain recorded.

`whitelist.json` closes the capsule contents. `origins.json` binds every copied
public artifact to its original path and digest; `files.json` hashes every other
capsule file. To verify copied evidence without importing measured code:

```sh
python -I -B verify.py
```

`python -I -B verify.py --current` additionally compares the explicitly listed
original public source/evidence files where they remain available. Neither mode
reruns tests, imports candidate modules, follows the recorded aliases or probes
processes. The portable verifier authenticates recorded process evidence rather
than claiming a new process observation. This capsule issues no execution grant.
