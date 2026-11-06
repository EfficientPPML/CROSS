# Gain4 encrypted normalization: completed server phase

The genuine keyless server completed all 54 checkpoints for the unchanged public
K128/16-row and K15/9-row fixtures. It used the existing authenticated p60/d8,
N131072, Q100/P14 context and borrowed keys. The server took 36,769.57 seconds;
this is component execution time, not full-model timing or a speedup claim.

The work includes two genuine level49-to25 reductions, six plaintext products,
46 production ciphertext products, two diagnostic ciphertext squares,
54 rescales and 28 rotations. No refresh, key generation or server codec call
occurred. All outputs were synchronized and serialized; the loader and compiled
owners retired successfully. The original inputs, keys and outputs remain
external and retained.

Root independently verified the final persisted parent and worker reports,
all 163 original/execution/start/end/archive source hashes, fixture/server
public terminal gates, unchanged input binding and process retirement.
Session27546 exited0 and was reaped; worker group2594585 is absent. The inherited
stdout digest was emitted before outer signal restoration, so it is not used
as the terminal identity. The final parent digest is
`897fa13c99005dff41d10e192d87417a6d31a155995c3e65805922fc5326d68a`.

This capsule records server completion before numerical validation. The
[separate client result](../bert_normalized_round_gain4_validation_cpu_20261104/README.md)
now records all checkpoint and derived-error gates. This server evidence alone
does not establish full BERT, ranked accuracy, calibration or competition
readiness. The prepared185-source validator remains immutable; its separately
reviewed successor performed the client validation.

`verify.py` checks this public capsule offline with the standard library. It
does not rerun HE, inspect external key/ciphertext bodies, or observe historical
processes. `root-review-driver.py.txt` preserves the executed root review;
its external paths are historical provenance.

```sh
python verify.py
```
