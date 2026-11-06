# Pinned layer-0 first-attention ideal composition

The source-bound final suite passed 108 tests in 225.08 s (225.62 s collector),
with 130 exact source files unchanged. Tool session 67199 exited 0 and its child
was reaped. No ciphertext arithmetic or real refresh was executed.

Both K15 and K128 fixed public fixtures execute the actual chained BSGS and
QK/PV routing loops, the frozen initial+five-round softmax, output projection,
residual and attention LayerNorm. Every physical output slot is checked at the
fixed 1e-8 ideal gate against an independent dense polynomial reference. The
maximum differences are 3.55e-14 and 2.84e-14. Exact NumPy/native first-attention
parity is below 2.5e-14. Polynomial/native differences 4.57e-5 and 7.27e-5 are
reported diagnostics, not a newly invented block-accuracy gate. Propagated
valid-query domains are checked. There is no padded-query domain claim.

Refresh calls are identity slot doubles with the public actual level 25 and
adjacent binary64 scales: 5 calls for K15 and 40 for K128. They validate the
unchanged public bootstrap inventory and five supplied helper identities, but
prove no refresh range/error or repeated-refresh stability. Output level 9 is
an actual metadata plan, without scale relabeling.

The isolated public schema binds eight fresh input descriptors, original token
positions, context/request identity, the four pinned projections and caller-
supplied actual output scales. It is not a ciphertext serializer or any of the
five full BERT file-stage handlers. Model/activation arrays, keys and ciphertexts
are excluded. The trusted fixture preparation and hashes are included.

`measured/sources.tar.gz` is the complete tested source closure; `source/` makes
new owner/test code directly readable. The historical development-v1 proof
predates the final boundary hardening and remains unchanged. Negative guard
checkpoints preserve their exact owner/test copies and available safe short
logs; detailed local pytest failure logs that displayed fixture arrays are
intentionally excluded. No historical source record is relabeled as final.

Whole layer0 joining, actual encrypted execution, B2 full-slot refresh and a
full pretrained/file-stage workload remain separate evidence requirements.
