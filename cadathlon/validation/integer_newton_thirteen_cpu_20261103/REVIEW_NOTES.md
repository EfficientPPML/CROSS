# Read-only review notes during the frozen numerical run

Root reviewed the driver and composition after launch: public schedule,
initial affine operation order, all13 retained ciphertext checkpoints, exact
binary64 scale recurrence, post-server-only decode, declared9/12/13/13 gates
and cleanup are consistent. No current-proof blocker was found.

Future reusable API work, only after this numerical proof is preserved:
- Reject bool/complex initial estimates before float64 coercion. Active inputs
  are explicitly constructed finite real float64 vectors.
- Initial affine should reject nonzero public weights that would round away
  at a computed alignment scale, matching weighted_sum. Active scales stay
  around56bits and active coefficients are safely representable.
- A static backward scale plan assumes no mid-circuit refresh. Integrating
  with actual refresh requires an explicit capacity/scale schedule and new
  numerical evidence; this component supplies neither refresh nor softmax.

These observations have not changed the active99-source snapshot.
