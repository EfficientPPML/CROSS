# BSGS integration into full128-key BERT attention

This isolated successor connects the genuine-component-verified BSGS routing
owner to the original 16-query/128-key score assembly and value mixing loops.
Both paths retain128 ciphertext products,768 head rotations and120 row rotations;
the routing calls contribute1520 rotations and896 actual rescales, with the
same three critical levels. Original public Q/K/V interfaces and head/key
packing remain unchanged. No new refresh or approximation is introduced.

The score oracle is an independent full12-head QK dot product divided by8.
The value oracle is an independent full128-key PV sum. Six complete65536-slot
fixtures cover actualL49 andL25, canonical, adjacent and noncanonical scales,
with hostile unused feature lanes. All returned metadata follows literal
binary64 recurrence; finite all-slot error must be at most1e-8. Sixteen further
binding/callback/retention checks accompany the six numerical checks.

This evidence uses explicit NumPy payload, security and backend doubles. It
reads no model, key, ciphertext or held-out data. The module is not yet joined
to the complete12-layer BERT owner, and this is not HE accuracy or a speedup
measurement. Source files of the active encrypted full-model run are unchanged.

Both input wrappers and original payload storage survive entered failures.
An entered failure quarantines the owner, arithmetic, original payloads and
exception/traceback until dedicated-process exit. Complete attention key/map
identity, multiplication accessor and callback identity are construction-bound.
Callback-added output annotations are stripped while borrowed annotations remain.
A future genuine outer loader must preserve backing storage on failed readiness.

collect_join.py snapshots17 exact source files and uses unchanged OwnedChild,
cleanup, terminal and finish helpers from the reviewed public exporter. Only
its public IO helpers are used; its parameter export functions are never called.
The child has no healthy deadline; direct wait and owned-group absence are
separate required terminal checks. Existing files under development-v1/v2/v3
preserve earlier development outcomes, not final source-bound evidence. v1
failed an incorrect test count-field name after the numeric assertion; v2 had18
passes; v3 had22 after explicit metadata and payload/annotation retention checks.
