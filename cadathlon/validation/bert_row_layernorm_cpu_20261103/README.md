# Full-width LayerNorm encrypted component attempts

`attempt1/` is a **failed** genuine CPU run, retained unchanged. Attention LayerNorm output error was **3.6375236547e-6**, above the fixed absolute **1e-6** gate. The other seven checkpoint gates passed, including MLP output at1.38524e-7. Every checkpoint was checked over16,384 slots, for131,072 slot comparisons in total. There is no passing encrypted LayerNorm claim for this attempt.

The original profile was N32768, Q26/P4, depth12, dnum8 with seven actual switching partitions, scale7.198723871526093e16 and22 rotations. Actual log2(QP)=853.2333533 satisfies the881-bit classical128-bit library ceiling and P coverage. Two fixed native full-width layer-zero inputs, with predeclared hostile padding, crossed strict ciphertext file boundaries into a separate keyless server. The fixed public inverse polynomials, input data and thresholds were not changed after seeing this failure.

| Checkpoint | Attention max absolute error | MLP max absolute error |
|---|---:|---:|
| Mean | 1.84408e-10 | 1.72825e-10 |
| Variance | 3.28110e-7 | 3.30196e-7 |
| Inverse polynomial | 2.99110e-7 | 1.77687e-8 |
| Learned affine output | **3.63752e-6 — failed** | 1.38524e-7 |

Attention output errors occur in the two active rows (3.63752e-6 and1.09422e-6). Its output padding error is only5.99721e-13. The observed pattern is consistent with variance/inverse error amplified by the learned affine transform; this is a diagnosis hypothesis, not proof of a sole cause. It is distinct from final output-padding leakage. Native polynomial approximation errors are separate diagnostics: the encrypted gate compares against the fixed polynomial circuit.

The collector finished after1152.49745s with exit1. Root owned and reaped session27678; the independent audit verified that wrapper990394 and all five stage children were absent, the owned component directory was removed, every stage cleanup field was true, and all **111 source files** matched startup, the exact source archive, final verification and the permanently frozen original checkout. The server stage succeeded; client validation returned exit1 for the numerical gate. Source archive SHA256 is`c43128efa0b047d2543a5118ac2aa38b7cf70eaab5989cbe7e86434d0564dfc8`.

The archive contains only the explicit report/process/stdout/stderr log whitelist, public model/key descriptors, request manifests, preprocessing aggregates, source bytes and independent audit. It contains no secret keys, model/reference arrays or ciphertext payloads. The failed client stage prevented publication of aggregate response manifests; none are invented here. This is an isolated component, not a full BERT layer/model, MRPC evaluation, accelerator benchmark or scoring calibration. Different-ring FFN/QK/LayerNorm components are not directly composable.

`precision60_prelaunch/` records a separate candidate with a larger canonical scale, 1.1513033521e18, and thirteen switching partitions. The actual N32768/Q26/P3 profile has log2(QP)=870.8806, below the same881-bit ceiling, and P coverage passes. The helper arithmetic, native fixtures, coefficients, role/level schedule and1e-6 gate are unchanged. Only profile parameters, corresponding metadata-test shapes and provenance change. All155 prelaunch tests passed against111 unchanged sources; the separate genuine file-stage CPU run completed successfully as recorded below.


`precision60/` is the passing repair experiment. It retains the same helper arithmetic, native fixtures, inverse polynomials, role levels and absolute1e-6 gate as the failed attempt; only the secure precision profile and its matching metadata changed. All eight checkpoints passed over131,072 slot comparisons. Attention output error is3.2229642954e-8 and MLP output error is2.0229293796e-9. The highest mean/variance/inverse errors are8.29003e-12,1.13969e-8 and1.27636e-8 respectively. Levels remain12→11(mean)→9(variance)→3(inverse)→1(learned affine).

Root reaped session90252 with exit0 after1642.08s internally (GNU time1642.15s, peak9,725,380KiB). The separate keyless server completed in1559.52s internally; all five stage processes exited0, were reaped and no longer exist. Class-wide codecs and both key-generation aliases were prohibited during server evaluation, both encrypted inputs stayed unchanged, and all owned key/private directories and handles were removed. All111 source files match startup, the archive, end-of-run hashes and the frozen checkout. Source archive SHA256 is`7a5016d4539b7dc40e54a3310e6bf02268c20feecdeb3da4550407ebb5bf20e4`. `precision60/verification.json` records independent archive/cleanup/numerical verification and hashes of the33 whitelisted artifacts.

This passing result covers two fixed layer-zero LayerNorm inputs, including hostile padding. The error gate compares encrypted evaluation with its fixed polynomial circuit; it does not certify approximation error versus native LayerNorm. The original failed attempt remains unchanged. No source promotion, common-ring integration, full model/MRPC accuracy, refresh or TPU performance claim follows from this component result. In particular, an independent full408 native audit later found failures in the earlier full-model domain contracts; the passing fixed-input component does not repair those contracts.
