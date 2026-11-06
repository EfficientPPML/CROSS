# Remaining public Qwen matrices

This isolated exporter reads only the pinned public Qwen checkpoint/config and
candidate manifest. It extracts28 full O projections and84 full gate/up/down
matrices, plus the151936x1024 vocabulary and embedding weights. The checkpoint
stores both tied matrices separately. One output body is shared only after
comparing every actual BF16 byte from those two checkpoint regions; configuration
alone does not establish equality. The manifest retains114 named descriptors
for113 physical NPY files. The previous84 QKV matrices and113 norm gammas stay
in their existing authenticated workspaces.

BF16 bits are promoted exactly through binary32 to binary64 in chunks of at
most4096 rows. No model/library construction, forward, backend, key generation,
codec or encrypted work is permitted. All output bodies remain local and are
excluded from public evidence archives. Output storage is fresh and retained
on failure. No other export, key, ledger or live source is modified or retired.

The collector snapshots an explicit ten-file inventory, archives it, runs a
dedicated owned child from that copy, verifies direct-child wait and process-group
absence, and verifies source bytes again. Healthy waits have no deadline.
Completion additionally requires successful outer termination-scope restoration.
The inherited process/cleanup helpers are byte-identical to the independently
reviewed BSGS ready preparation. Synthetic format/FD tests open no actual model
file. Successful public export is parameter preparation, not HE or full-model
evidence. Only the measured report can establish which phase actually ran.

The original eight-source preflight passed47 checks, then independent injected
SIGTERM regressions exposed an unshielded collector finalization. That history
remains unchanged. The successor shields complete finalization with the reviewed
terminal scope and republishes failed outer termination restoration. This is a
main-thread signal/cancellation boundary, not process-wide atomicity.
