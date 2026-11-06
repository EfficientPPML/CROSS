# Correction to the public Qwen RMS feasibility plan

This v3 supersedes only the initial softmax mass bookkeeping in the independently preserved v2 artifact. The actual frozen circuit uses the same rounded `alpha=1/K` in the denominator `alpha*sum(exp)` and the final probability `alpha*exp*inverse`. Consequently its exact-real mass is `denominator*inverse=1-r`; multiplying by `beta=K*alpha` a second time was incorrect. At K3 the actual mass exceeds the old beta upper bound by approximately5.55e-17. That counterexample is preserved as a regression.

The beta factor belongs in the initial exp-mean denominator interval. The analytic exponential margins still place that denominator in[.5,2]. Initial and later normalized masses lie in[1-1e-12,1], so the existing [.5/K,2] squared-denominator induction remains valid under the stated exact-real, domain and shared-square assumptions.

All 113 inverse certificates, 904 level schedules, the 66 binary64 endpoint failures and 49 fixed-point initializations, public score bounds and conclusions are unchanged. No new candidate, coefficients, model input/output, HE or runtime changes occurred. The predecessor remains at `../qwen_rms_repair_feasibility_public_20261103`; its original report/source hashes are recorded in `superseded_reference.json`.

49 tests pass, including actual exact-Fraction execution of the frozen denominator/seed/Newton/final-probability chain for eight K values and the K3 negative reproducer. All nine sources are unchanged across tests and planning. `python verify_archive.py` independently checks archive bytes, interval recurrences, all level branches, all33 reciprocal certificates, and the shared-alpha mass identity for all K1..32. This verifies public math only, not floating/CKKS/refresh accuracy.
