# Process-local retained admission guard — source only

`capture_admitted(packet, *, check_bindings)` accepts only the exact selected
`ServerPacket`. It invokes the unchanged complete `ServerPacket.check()` at this
boundary. The original graph is captured before that check and compared after
the full check and the caller's final binding callback; a post-validation
callback cannot substitute unvalidated proof bytes before capture. No successful
packet, proof, actual process or grant is supplied by this component.

The returned `AdmissionSnapshot.check(original_packet)` performs no JSON parsing,
proof hashing, public evidence reconstruction or scientific evaluation. It checks
the exact original process, thread object and thread ID, packet object, selected
module/class/function bindings, and the complete retained instance-field graph.
All 13 current admission types have explicit closed fields. Their graph contains
only exact admitted types, immutable tuples, exact bytes and strings. Unknown
types, bytearray/list/dict payloads, extra instance fields and replaced original
instance dictionaries reject. Tuples and proof bytes are retained by identity;
they cannot change in place. It neither accepts an equal replacement object nor
keeps a copied report outcome independently of its original proof objects.

The guard captures all selected pure helper exports that are Python functions or
classes, every defined class descriptor, and original function code/defaults.
Class descriptors are checked before reading admission instance fields. Module
objects and their two retained file/spec origins are checked by identity. The
caller must independently authenticate this source, its complete helper source
closure and actual origins before construction, and provide the original
`check_bindings` Python function or method. That callback is captured with its
function/code/default and bound-owner identities and invoked before each hot
check. It must check source/import/audit/live-role state; it must NOT call the
snapshot, full packet/phase authentication, or a RuntimeAdmission.check that
calls this snapshot. Reentrant use rejects. A caller-supplied no-op does not
constitute source or execution authority.

The guard is opaque and has no writable instance fields. A private process-local
registry retains its original graph and bindings. Exactly one capture attempt
is allowed in a process, and failure is sticky. Clean `retire(original_packet)`
checks once and releases its registry references; failures retain the original
graph until process exit. No guard can be transferred to a fork or another
thread. The registry is not an on-disk, global outcome, or cross-run cache.
Within the existing trusted-source callback contract, the caller retains the
original guard and check/retire methods and rejects replacements before calling
them. This is not a defense against arbitrary mutation of the guard module's
private globals or interpreter internals; those were not the original callback
contract either.

The actual process still performs complete entry and terminal pure admission,
independent source/archive/grant validation and original external-proof file
postchecks. This component does not weaken any public check method. A future
separately reviewed integration must replace hot full-phase calls with this
guard explicitly, including the indirect `phase.expected_ids()` and
`phase.input_digest` properties where applicable. Those values may be derived
once under full admission and retained by the original process closure; this
component does not silently alter those properties or issue a new capability.
Current physical store checks, original held descriptors, array ownership,
backend/security guards, sticky audit state and cleanup/quarantine remain live
checks. They must not be replaced with historical proof-cache results.

`execution_authorized` remains false, and `create_actual_snapshot` immediately
denies. No actual server source map, runtime process, scientific proof instance,
memory allowance or execution grant is invented here. No candidate import,
collection, test, backend call, body read or runtime measurement occurred while
authoring this source. Focused tests and integration are still unperformed.

The v2 successor keeps the original first-capture packet before any callback and
retains the completed graph before full validation; failures leave that attempt
retained until process exit. A reentrant check marks the registry failed before
raising, and the outer check verifies its original in-progress registry identity
before restoring active state, even when the callback catches the inner error.
These two fixes are source-derived; no runtime failure or successful test is claimed.
