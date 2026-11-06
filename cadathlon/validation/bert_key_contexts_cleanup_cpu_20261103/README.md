Key-context cleanup successor — scoped public/mock evidence
==========================================================

The isolated successor records each original signal handler before attempting
to change it, blocks main-thread INT/TERM/HUP delivery during installation and
restoration, attempts every recorded restoration, and preserves cleanup-error
precedence at the final mask restoration. Resource retirement, readiness
quarantine, key authentication and arithmetic are unchanged. The patch targets
only `runtime/stage/key_contexts.py`; its base SHA is `ee796798bc20…` and its
result SHA is `0abf32280ab5…`. No original or active runtime was modified.

The final source-bound run passed123 tests:104 existing producer/context/scale
checks,10 new deterministic signal/fault checks, and nine fresh subprocess
checks of the actual loader under the separately reviewed single-fire
termination wrapper. Those nine runs used tiny mocked key files and contexts;
signals arrived during install, body, or restore (three samples each). Every
sample attempted all cleanup and exited nonzero. No actual key generation,
CKKS context, encryption, ciphertext or model execution occurred. The inherited
public security test initialized CPU constants and used real security predicates.

All65 source files matched before, archive readback, and after. Session84909
exited0 and was reaped; pytest took5.58s, collector6.493s. The owned fake-test
directory was removed. The exact wrapper tested is SHA
`1798455cdd88d4bbac5e1769e6b3226676ccb9901e7be6faf383e7ad063c1ef6`.
Adopting the loader alone does not establish the wrapper integration evidence.

Important retained limitation
----------------------------

`pthread_sigmask` only blocks delivery to the calling thread. It does not make
multi-handler installation/restoration process-signal atomic in a threaded
NumPy/JAX process. A separate deterministic probe sends SIGTERM to an unblocked
worker while the main thread masks SIGTERM: Python still invokes the main
handler with that mask set. The original process-directed stress also produced
a late uncaught KeyboardInterrupt after52 passing tests. Both are preserved.
The final checks make no guarantee about arbitrary signal timing or preservation
of an earlier exception's identity outside an entered guard.

History is not relabeled as passing: the exact original loader fails both
handler-tracking regressions; development-v1 passes59 bodies but has one test
fixture teardown error; development-v2 is the process-directed cancellation
failure. The final deterministic pending-signal tests use `pthread_kill` to the
main thread; separate cleanup and wrapper stress retain real process signals.
The copied source archive includes every historical source variant used here.

Run `python verify_archive.py` for portable hash/count/patch readback. It executes
no runtime or tests. This archive contains source, public profiles, logs and
aggregate lifecycle records only; tiny mocked key arrays are not archived.
