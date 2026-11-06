This is a genuine CPU proof of a **historical layer-0 FFN component**, using the pinned BERT checkpoint's full 768→3072 and 3072→768 matrices and the earlier fixed degree-48 GELU on [-19, 11]. It is separate from the later v2 degree-64 GELU and common N131072 design. The preselected private fixture is native attention-output rows 0 and 1 of one fixed 128-token public pair; native attention is trusted fixture preparation, not an implemented encrypted attention stage.

The client encrypted one F4096 row tile. A fresh keyless server reconstructed public context/keys and emitted all three encrypted checkpoints before a fresh client decoded any result. Server codec, runtime-key generation, and client-file access guards remained active. Actual N65536/CD2/depth17/dnum2 used Q36/P17, log2(QP)=1522.139617 (128-bit classical ceiling 1747), with checked P coverage. Levels were 17→16→2→1; no refresh ran.

| Checkpoint | Slots checked | Worst error against independent fixed polynomial |
|---|---:|---:|
| First full-width dense map | 32768 | 4.3013042017e-9 |
| Balanced Chebyshev GELU48 | 32768 | 2.7138638004e-9 |
| Final full-width dense map | 32768 | 2.1143587159e-9 |

All unchanged 1e-6 gates passed. Raw nonlinear padding was compared with P(0)=0.00012006456190527537; the final source mask removed it, with final padding error 1.8166770230e-9. The final diagnostic difference from the native FFN was 0.0024848781453 (nRMSE 0.0007941203451), mostly polynomial approximation; this is not a P2 logit or ranked-accuracy result.

Native preprocessing took 2.7602 s internally (3.4763 s subprocess wall). The three cryptographic subprocesses took 18,595.3725 s together; complete collector wall time was 18,600.4167 s. Server phases took 9,485.5126 / 5,663.1804 / 3,049.9454 s. These are a single CPU component observation with four pinned cores and BLAS/OMP=1, not calibration or a TPU projection.

The collector exited 0, its session was reaped, every owned child exited, and temporary component/key directories were removed. All 135 initial/end/current source bytes matched the independently read-back archive, SHA256 `650adcd2a63ca1d76e7608c2e0f1d64527a536b13edbfef9c70ae613247ed5a0`. `run-v1` preserves raw records, including stack-only observations. Earlier metadata/native preflights are retained separately; model tensors, private row arrays, secret keys, and ciphertext payload files are not included.

Run `python verify.py` for offline source/evidence checks. No residual, LayerNorm, encrypted attention, complete block, all 12 layers, pooler/classifier, MRPC accuracy, full file-stage submission, or scoring calibration is established by this component.
