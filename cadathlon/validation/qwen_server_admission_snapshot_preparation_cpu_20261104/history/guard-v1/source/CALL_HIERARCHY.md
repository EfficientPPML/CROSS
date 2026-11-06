# Source-derived repeated work, not a runtime measurement

At the initial source trace, `qwen_server_process.py` was absent in the root
server draft. It appeared during this component's authoring and was not read
here. Therefore RuntimeAdmission.check → packet.check is a proposed integration
edge in this review, awaiting that separate held process source. The following
edges are present in exact source:

1. `ServerBackend.check()` calls `_runtime(self.runtime)`, which invokes the
   future RuntimeAdmission.check, then separately calls `self.phase.check()`.
2. The bound key `map_array` callback calls `backend.check()` and the runtime's
   public-key scope check for each of the 335 canonical public array descriptors.
   The loader visits those descriptors in its original `specs.items()` loop.
3. `ServerPhaseAdmission.check()` calls full authenticate_server_phase. That
   calls scientific.check, then scientific.bindings; bindings calls check again.
   Thus one phase check invokes full authenticate_scientific twice.
4. `ServerPacket.check()` first checks its retained phase (two scientific calls)
   and then calls authenticate_packet. That function directly authenticates
   scientific evidence once and authenticates the phase (two more). Total: five
   full scientific authentications per packet.check, with additional small
   model/readiness reconstruction. This is finite repeated work, not an infinite
   recursive cycle. Scientific authentication does not call the packet checker.
5. The first loop of each scientific authentication parses all 25 bounded proof
   byte strings, up to 209,453,056 raw bytes in aggregate, before several later
   phase/provenance helpers parse selected objects again. This ceiling is not the
   actual input size, resident allocation, elapsed time or an adequate RAM grant.
   The explicit backend phase check alone means at least 670 full scientific
   invocations over 335 mappings. If RuntimeAdmission.check calls packet.check
   once, backend.check contributes seven per mapping before any further scope
   check calls. Actual multiplicity depends on the still-missing process code.
6. `ActualServerDependencies.check()` also calls phase.expected_ids, whose
   property path performs another phase check. The response publisher obtains
   phase.input_digest with the same full-check behavior. These are additional
   integration call sites to review.

The frequent BorrowedServerEngine `_check`/run-local callbacks use the distinct
FilesAdmission/Admission key/file chain. Current source does not automatically
route those checks back into full scientific admission. No blanket claim of a
scientific parse per arithmetic operator or engine callback is made.

Recommendation: fully authenticate at declared process boundaries, then use a
source-fixed single-process immutable graph guard for authority identity at hot
callbacks. Retain live source/import/audit, store FD/state, backend and ownership
checks separately. Recompute full admission at terminal as root requires. Never
reuse a PASS across runs or cache a failed/mutated object graph. No speedup or
resource sufficiency has been measured or granted.
