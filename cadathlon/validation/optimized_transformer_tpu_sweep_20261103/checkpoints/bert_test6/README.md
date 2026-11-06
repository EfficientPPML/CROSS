# Optimized toy BERT test 6 failure

The original case **failed** with `ApproximationErrorTooHigh`: the decoder
reported estimated standard deviation `1.036e26` against the unchanged limit
`3.125e-2`. Child `897099` exited 1 after `41889.48500001902` seconds. Its saved
result contains no decoded logits, maximum error, nRMSE, top1 agreement or
successful runner timings. The external process duration is not a passing
benchmark measurement.

This completes the original twelve-case public toy sweep: **7 PASS, 5 FAIL**.
The included root receipt records session `12657` terminal exit 1 at tool chunk
`27218b`, original parent/child PID and process-group absence, owned key-path
absence and successful cleanup. Those are root's independent observations;
this archival reviewer did not poll that session, inspect key paths or reap it.
The receipt's separate TPU-occupancy observation is retained verbatim, with no
authority to operate the observed job or change the queue.

The fixed public fixture uses seed 9506, batch 3, sequence length 16, hidden
width 24, MLP width 48, four heads, three layers and two labels. Its valid-key
counts are 15, 14 and 13. It is synthetic BERT, not pretrained BERT/MRPC or
ranked accuracy.

The exact stderr records **8 refresh starts, 8 completes and 6 layer
milestones**. All three attention/normalization and output milestones occur;
the final layer output is logged at 37269.2 seconds. The last refresh starts
at 37289.6 and completes at 41795.6 seconds, a 4506.0-second difference between
rounded log timestamps. The ordered entries and original line numbers are in
[milestones.json](milestones.json). These messages establish progress, not
numerical correctness or the first location of divergence.

Both security-boundary events record TPU v6 lite payload placement. Their
ring metadata is degree 131072, Q100/P24, evaluator decomposition 4, with
classical 128-bit security metadata and log2(QP) approximately 3490.161581
under the recorded degree limit 3523. Output metadata has 40 Q limbs, finite
scale `6.8994011570896904e16` and NSD 1. The security report still ends
`passed:false`, `SystemExit: 1`; metadata does not establish output accuracy.

[DIAGNOSIS.md](DIAGNOSIS.md) separates the recorded failure from the frozen
source inference: the unchanged fast decode path checks conjugate-symmetry
error before its forward FFT, feature readout and logit metrics. No traceback
or intermediate encrypted/plaintext trajectory localizes the first fault.
No decoder gate, arithmetic, coefficients or frozen source were changed.

The 40,088 nominal one-second resource samples have peak aggregate
process-tree RSS 119,110,234,112 bytes. Shared pages can be counted twice and
brief peaks missed. Output-device checkpoint `peak_bytes_in_use` is
23,503,338,496 bytes; it is not a continuous peak-HBM measurement or a memory
capacity guarantee. Four constant-capture warnings are retained without
assigning them a causal role in the numerical failure.

All six case files, the final parent report and the root terminal receipt were
copied byte-for-byte. Their exact origins are in
[copy_origins.json](copy_origins.json); [artifact_hashes.json](artifact_hashes.json)
is the closed local whitelist. The unchanged enclosing
[199-source archive](../../sources.tar.gz) and
[manifest](../../snapshot_manifest.json) supply the complete original sources;
this checkpoint does not duplicate that archive. Its SHA256 is
`bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b`.

Run `python -I -B verify.py` for portable verification with those two enclosing
files, or add `--current` to rehash the declared frozen source/public evidence.
The verifier uses only the standard library and does not import subjects,
decode ciphertexts, read key/model bodies, observe processes or follow key
descriptors. No numerical rerun or new actual launch was performed.
