# Full28 Qwen owner: fixed-profile and protected-setup successor

All150 focused checks pass on87 exact sources. Tests took484.91s and the
collector548.4760s; session9542 exited0 and was reaped. Both recorded test/plan
child PIDs and owned groups are absent. Before/end/current/tar source bytes match.
The ordinary-Python audits record zero parameter/input-body or backend/model
attempts. No full learned28-layer trajectory, native forward, encryption or HE
ran in this proof.

Runtime f4e8f93a170cd8c9c7570f23eb1db4159c7d6da315560e34691eb5986520ff17
adds exact ordered common Q100/P14/p60 Delta/dnum8/layout checks and puts all
activation/setup under try/finally. Six preserved negative tests show why those
changes were needed. Arithmetic, coefficients, domains, iteration counts and
actual returned-scale equations remain unchanged. The earlier84/144 proof is
preserved separately in `qwen_full_model_focused_runtime_cpu_20261103`.

The actual parent connects attention outputs to the tail, tail outputs through
explicit next-layer refreshes, then last-valid-row mask4→3/rotation/B2048refresh25,
final RMS7 and all38 vocabulary maps6. Focused tests replace the child component
arithmetic with explicit synthetic stand-ins; only the parent masks/rotation,
metadata, object flow and ownership are exercised numerically here. The independent
dense reference has17 tiny arithmetic checks against the frozen evaluator; it has
not yet consumed actual embeddings/weights. Wholegraph execution requires the
next review. Resource/domain/refresh-error obligations remain as in the public plan.

Source reviews by Security found no further concrete blocker in the narrow
successor; CNN found no dense-reference semantic blocker. Those are source reviews,
not numerical evidence. No private/tensor/key/ciphertext bodies are archived.
Run `PYTHONDONTWRITEBYTECODE=1 python verify.py` for portable readback.
