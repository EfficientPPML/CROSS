# Prepared full-slot score reductions: public checks only

V4 passed75 ideal/contract/lifecycle checks with104 unchanged source files and
matching actual-ring preflight. Four exact source archives (test+ring, V3+V4)
were verified byte-for-byte. No archived run generated keys or evaluated a
ciphertext. The separate authorized encrypted-v1 run is pending and is not
represented as a pass here.

The isolated borrowed-CompiledArithmetic helper evaluates masked means in2levels
and explicitly masked squared denominators in3levels over65536 distinct slots,
16query rows at pitch4096,128keys and12real/16padded heads. Both paths use14
rotations; inactive denominator values must be finite positive public values
inside the declared domain, and are folded into the anchor bias. Ciphertext
domain membership is an external contract; the helper cannot inspect it.

V4 prepares four encrypted inputs and four outputs: two varied signedscore
vectors for means, and two separate normalized probability vectors for squared
denominators. The fixtures have K128/16requestedrows and K15/7requestedlogicalrows
with noncontiguous keys, hostile inactive padding, unique varied active means,
and declared stage0 midpoint fills. Actual gates are fixed1e-6 across all65536
slots/output. All requested BERT padding queries must remain requested rows;
this component's7-row fixture declares the other9rows outside its logical input.

The ring-only preflight checks N131072/CD2/depth4/d4/Q10/P3, actual log2QP
374.75191856669466, P coverage,14rotationkeys,790MiB raw totalpublickeys and an
additional4GiB disk margin. Candidate outputlevels are mean4→2 and denominator
4→1, with no refresh. Allclient decoding is deferred until all4serveroutputs
complete. Separate client/server objects in one dedicated process are not an
OS sandbox or a file transport proof.

Historical V3 is preserved unchanged: its proposed encrypted mean inputs were
normalized probabilities, yielding constant1/K means. V4 fixes that coverage
weakness and rejects bool/float/NumPy-int NSD tags, requiring exactPythonint1.
V3 was preparation only; there was no encrypted V3 attempt. No core or live
student runtime was changed, and no softmax/model/performance claim is made.
