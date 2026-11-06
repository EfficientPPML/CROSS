# Qwen public RMS repair feasibility

The public plan completed successfully; a usable numerical repair is **not** established. The 113 exact-native intervals come from the previously archived pinned public-weight audit. Frozen candidate2 coefficients and arithmetic are inputs only; no new coefficients, prompts, model outputs, HE or TPU computations were used.

Exact dyadic scaled/tracked Newton needs **18–60 updates** for a uniform exact-real relative error at most 1e-6. The stated genuinely aligned macro consumes **55–181 inverse-state levels**, or **59–185 levels including variance and learned-normalization output**. At refreshed capacity25 it requires **2–14 state refresh calls** from a fresh level49 feature, or **4–16** from a refreshed level25 feature, conditional on preservation of state accuracy/correlation. Every branch and level2 continuation reserve is recorded.

The naive binary64 tracked implementation fails the fixed lower-endpoint 1e-6 check at **66/113 sites**; **49** initialize the residual at its fixed point. The smallest initial scaled residual gap is approximately 3.55e-21. These are failures of the prospective scalar implementation, not additional measurements of candidate2. The artifact preserves them alongside the ideal certificate. Existing full-slot refresh error gates do not resolve this precision obligation.

The frozen exact-real exp/reciprocal circuit has a conditional positivity and row-mass proof on its declared domains, including the rounded initial affine and public float(1/K). Coarse public Q/K norm bounds would require **10–16** power-of-two exp reductions, while the frozen circuit uses 5–6. This is a bound-certification gap, not an observed trajectory failure. Approximate residual-domain induction, global score coverage, SiLU output bounds, and floating/CKKS/refresh error remain open.

`DESIGN.md` gives the derivation, actual branch scheduling, limitations and recommended next public evidence. `public-v2/` is the final nine-source report/archive; `source-bound-v2/` contains 40 focused tests and the plan, both exit0 with unchanged sources. `historical-public-v1/` retains the initial planning pass before documentation/provenance additions; its arithmetic rows are not relabeled as the later source set. No raw tensors, ciphertexts, keys or hidden activations are included.

Run `python verify_archive.py` for offline exact-rational residual, archive and level-schedule checks. The verifier does not reread model weights or evaluate any input. No live runtime change or promotion is proposed.
