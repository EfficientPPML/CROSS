# Secure CPU ciphertext file-stage component

A genuine three-process client → public-only server → client round trip passed
on a 128-bit-secure N=8192, Q8/P2, composite-degree-two ring. The server performs
an affine map, encrypted square, positive/negative rotations and a weighted
sum. All 8,192 decoded values (two independent 4,096-slot inputs) matched the
client-only oracle within **1.1266e-7**, below the fixed 1e-6 gate. All three
processes exited zero, all 100 recorded sources were unchanged, and the owned
temporary keys, private inputs and ciphertext artifacts were deleted.

`successful_run/` preserves exact source bytes, public parameter/key digests,
request/response identity manifests and raw logs. The 12.5449-second duration
is internal collector wall time, not a TPU or BERT performance baseline.
No generated secret key, private input array or ciphertext payload is archived.
The server's file-access audit is a regression guard, not an OS sandbox.

`initial_failure/` retains the first failed attempt, which stopped before the
server because the fresh codec did not attach an explicit noise-scale-degree
tag. The successful client adapter records the known NSD=1 construction
metadata. Transport validation was not weakened. Its prototype source snapshot
is separate; unchanged core/helper hashes match the successful source archive.

`boundary_checks/` preserves **67 passing CPU tests** for strict transport and
public/private manifest boundaries. These use synthetic arrays and do not
establish encryption by themselves; the genuine encrypted run is recorded
separately. Source copies are stored as `.txt` to avoid accidental execution.

This is isolated validation code, with no production runtime hooks. It does
not implement pretrained BERT, MRPC evaluation, full-width transformer packing,
homomorphic refresh, or a calibrated competition baseline.
