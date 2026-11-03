# Secure CPU file-stage component proof

The isolated client/setup → public evaluator → client/decrypt protocol passed
in three fresh processes (all exit 0). Total collector wall time was 12.5449 s;
this is a component diagnostic, not a BERT/MRPC or TPU benchmark.

The actual secure profile is N=8192, composite degree 2, Q8/P2, dnum=4, with
log2(QP)=212.45574163467 under the 128-bit classical rule. One Polynomial carries
two independent 4096-slot random real inputs. The server evaluates
`a=.75*x+.125; z=a*a; y=z+.5*rot(z,+3)-.25*rot(z,-5)-.2`.
Client-side comparison checks all 8,192 output slots; maximum absolute error
was 1.1265914393043808e-7 against a fixed 1e-6 gate. Conjugation is untested.

The server loads only public/evaluation/rotation keys and ciphertext files.
It holds no secret in its context/cache/parameters; both historical `key_gen`
and `jaxite_word.key_gen` creation APIs are forbidden, as are client decrypt
methods. A read-audit regression guard records no client-only file accesses;
it is not an OS sandbox. Inputs, private sample IDs and the oracle stay in the
client process. A public context ID binds actual ring/layout parameters and
key-file digests. Responses bind the validated request manifest, and the
client retains the initial request digest in its private manifest.

`report.json`, per-stage stdout/stderr and public/transport manifests are raw
results. `source_inventory.json` maps all 100 unchanged measured sources to
exact byte copies below `sources/`. Prototype code is preserved there as .txt
files so it cannot be accidentally collected as active tests. No secret key,
private manifest, real input array or ciphertext payload was copied to this
evidence directory. The owned temporary key/input/ciphertext workspace was
removed after all subprocesses were reaped.

`test_transport.py` and `transport_tests.log` preserve 47 separate strict-format
regressions, including context/request mismatches, unexpected schema or files,
invalid payload metadata/residues and response binding. Those format tests do
not replace the real encrypted round trip.

The earlier result1 directory preserves an initial failed attempt: the strict
transport rejected a fresh ciphertext lacking an explicit noise-scale-degree
tag, before any server process ran. The successful client adapter initializes
the known fresh NSD=1 metadata from its explicit codec construction; it does not
weaken transport validation or alter ciphertext arithmetic.

No repository runtime was changed, and the five full BERT file stages remain
unimplemented. Full-width tiling, the chosen pretrained nonlinear circuit,
refresh/error analysis and actual MRPC harness runs are separate work.
