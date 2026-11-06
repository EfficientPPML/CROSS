Stage3 preparation v3 — signal cleanup and terminal publication
===============================================================

The frozen v3 checkpoint passed135 mocked checks plus21 real installed
entrypoint denial/restoration probes in4.889s. Its81 proof sources include80
candidate files and the preflight driver. All current/snapshot/archive/before/
after bytes match. Root authoritatively reaped session79107 exit0; PID absence
was independently checked. No actual encryption, CKKS context, key generation,
or model execution took place in this preparation checkpoint.

The sole handler-runtime overlay is the reviewed key loader SHA7eb793deecca…;
58 other runtime files remain unchanged. The top-level cleanup wrapper is the
reviewed single-fire1798455c… successor. `terminal_scope` keeps the completion
flag false until the shield has exited, then publishes the completed record.
Scope/publication exceptions trigger best-effort failure republication, clear
cleanup completion, and preserve primary/cancellation precedence. Worker
success is accepted only with the completed terminal flag. Intentional main
SystemExit occurs outside the termination guard so a zero exit cannot mask its
restoration failure. Main-thread signal masks are not a process-wide atomicity
guarantee for arbitrary worker-thread delivery.

All earlier artifacts remain untouched. The archived negative terminal-scope
v1 interrupted its runner (exit2); v2 demonstrates the scope defects (exit1).
The separate main-exit reproduction has2 failures and2 passes. These outcomes
are retained unchanged with their exact source archives. The actual stage3
attempt is a separate root-owned run and is not part of this archive.

Run this directory's `verify_archive.py` for portable readback. The parent
verifier also verifies all three generations. Evidence contains source/public
metadata and mocked-test records only, without actual input/key/ciphertext
payloads or private target associations.
