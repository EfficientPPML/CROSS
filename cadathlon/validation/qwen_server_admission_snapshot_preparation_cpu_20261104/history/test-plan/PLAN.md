# Snapshot v2 focused guard-state preparation

Source-only proposal. No candidate import, collection, test, public proof instance,
backend, capacity observation or process integration is included.

The successor will preserve exact snapshot v2 `8790711a…` and all27 files from
the reviewed pure packet closure. Tests select only the14 pure Python helpers
and this guard. The prospective source tree is
`/tmp/qwen-server-admission-snapshot-tests-20261104`; actual tests run only after
root review and a separately bound harness/launch authorization.

Each selected case owns one fresh `-I -B` Python child. No test resets `_ISSUED`,
`_REGISTRY` or `_CAPTURE_ATTEMPT`. The parent records the child PID and observed
PGID, waits/reaps and verifies exit; exceptional parent cleanup retires only its
owned child/group. Each child reports its self-observed PID/PGID, source hashes,
guard assertions and explicit seam flags. Original parent/child environment
imports cannot silently select another local helper. NumPy/JAX/jaxlib/libtpu,
jax_plugins/jaxite_word and other model backends are forbidden before helper
imports, checked again at completion. No full runner or actual mode is added.

The test graph uses the exact13 admitted dataclass types, their declared fields,
immutable synthetic bytes/strings/tuples and separate model/launch nesting.
The25 scientific proof names are preserved, but their byte values are explicitly
non-proof synthetic markers. No valid scientific PASS fixture is fabricated.
Only `ServerPacket.check` and the guard's saved `_FULL_PACKET_CHECK` pointer are
temporarily replaced with the same recording validator. It checks the original
graph object and records its invocations; it does not authenticate evidence.
All other public validator bytes and functions remain unchanged. Child cleanup
restores both exact original validator function identities on every path and
asserts restoration, including failed capture. Any additional deliberate
function/default/descriptor mutation is likewise restored in reverse order.

The proposed ordered selection is18 fresh-child cases:

1. Complete13-type graph, one recorded full check, repeated hot checks with no
   additional full checks, source-binding callback observations, clean retire,
   denied second retire/check/capture.
2. Equal-content replacement of nested model report bytes; failure stays sticky
   after the original field is restored.
3. Equal-content replacement of a nested immutable public-blob tuple.
4. An extra field on a retained expected-pins instance.
5. Equal-value replacement of an original instance `__dict__`.
6. Unknown mutable graph payload rejected during initial capture.
7. Selected helper code replacement detected without invoking that helper.
8. Selected helper export replacement detected.
9. In-place selected keyword-default mutation detected.
10. Class descriptor replacement rejected before the replacement property runs.
11. Equal-content alternate packet rejected by original identity.
12. Wrong thread rejected, then original-thread use remains failed.
13. Uncaught reentry rejects and remains failed.
14. Caught reentry still causes the outer check to fail and remain failed.
15. Recording full-validator exception preserves exact exception identity and
    the original captured graph; retry capture denied.
16. Initial source-binding callback exception retains the original packet before
    graph capture and preserves exception identity; full validator never called.
17. Final source-binding callback exception after validation retains the graph;
    retry capture denied.
18. Recording validator replaces a graph field then returns; pre-validation
    graph comparison rejects and retains the original field references.

Assertions observe returned-object identity, invocation counts, replacement
non-invocation, original graph retention, registry status and clean retirement.
They do not merely assert an exception message. Registry inspection is read-only.
No fork/PID spoofing, arbitrary interpreter/private-global mutation, scientific
numerical checks or inherited suite is added. Mutable helper globals and closure
contents remain outside this guard's trusted callback contract.

The source handoff will include a compact exact map/archive, origin links to the
v1 findings and v2 source review, ordered static node IDs, closed selected files,
backend guard source and explicit unexecuted status. Static AST parsing and hash
readback are the only preparation operations. No positive runtime result will
be claimed until a separately authorized run completes.
