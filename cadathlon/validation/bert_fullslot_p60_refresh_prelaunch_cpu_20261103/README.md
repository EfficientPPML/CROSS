# Prepared p60 full-slot refresh follow-up

This isolated candidate copies the active p56 experiment's core/common runtime without edits, then changes only the experiment driver to depth49/p60/dnum8 and two independently encrypted probes. It has not run encrypted arithmetic and requires root review before launch. The active p56 experiment and the QK experiment remain unchanged.

The actual ring is N131072/CD2, Q100/P14, dnum8, layout256×512, canonical scale1.1467305003699405e18. Actual log2(QP)=3402.0803 fits the classical128-bit ceiling3523; every Q partition is covered by P. Full65536-slot budget4/4 refresh has106 public rotation keys and output25. No sparse-period averaging is available.

Separate fixed seeds20261103/20261104 generate65536 distinct slots at B16/B128, with13 scaled signed endpoint/interior probes. Both inputs are encrypted independently and lowered to level2. One keyless public context and refresher is reused with the explicit bound for each call. Every refresh uses meta10 and runs both real/imaginary ApproxMod branches twice, followed by a genuine square at level24. Both refreshes and both squares finish before any client decoding. All four checkpoints have fixed strict1e-3 absolute gates over every slot. Normalized-error summaries are diagnostics and cannot replace the restored-value gates. B128 square sensitivity is deliberately retained; failures will not be weakened.

Client/private ownership and the keyless server are separate objects in one process; this is not an OS sandbox or cross-process inference. Both key-generation module spellings are guarded across four entrypoints before server initialization. All seven class codecs are forbidden during server work, including on the client instance. Inputs and retained outputs are fingerprinted; actual output security/level/scale are checked before decoding. Cancellation preserves the primary error, attempts every release, records cleanup failure truthfully, and the parent process helper removes only its fd-owned key child directory. Source bytes are captured/read back and bound across collector and child.

106×912MiB raw rotation pairs require94.40625GiB. Preflight requires headers plus an additional4GiB free and rechecks actual free space before key generation. Public storage figures exclude transient controls, compiler memory, and run time; no peak-memory or speed claim is made. Two level2 input payloads total12MiB; the four retained outputs total204MiB.

Final preparation is `final-preflight-v3/report.json` plus its exact source archive. Earlier42/44-test checkpoints remain under `final-preflight` and `final-preflight-v2`. The final47-test checkpoint adds binary64 conversion/finite-JSON guards for longdouble diagnostics. Fake orchestration tests prove the four-output-before-decode ordering and that an injected second-square failure cannot borrow the first probe's pass; they are not cryptographic numerical evidence.

Proposed command, only after review/lease:

```sh
/usr/bin/time -v env JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CROSS_ROOT=/tmp/bert-fullslot-p60-refresh-candidate-20261103/checkout taskset -c 128-131 /home/jianming_gatech/miniconda3/envs/jaxite/bin/python /tmp/bert-fullslot-p60-refresh-candidate-20261103/prototype/collect.py --output /tmp/bert-fullslot-p60-refresh-candidate-20261103/encrypted-v1
```

There is no healthy-job deadline. No full-model, softmax, LayerNorm or calibration readiness follows from this component.
