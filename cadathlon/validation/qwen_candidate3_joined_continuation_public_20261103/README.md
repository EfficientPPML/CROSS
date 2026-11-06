# Public candidate3 Qwen join plan

The source-bound result passed113 public tests (6.432s;13.294s collector), with27 exact before/after/current/archive sources. Both children exited0 and were reaped; outer session13721 exited0 and was reaped. No model, private input, raw weight, key, ciphertext or HE backend was read/run.

The conditional schedule joins unchanged113 RMS rules, post-RoPE refresh, existing causal softmax/PV, and unchanged SiLU/vocabulary metadata. Nine L2 Q/K norms require an extra pre-RoPE refresh. Natural V23 yields PV20/O19; all block tails areL4, final norm7 and all38 vocabulary blocks6. Conditional refresh totals are638/3270/6278/8043/12931 forS1/8/16/19/32. Stored-coefficient paired-coordinate post-RoPE bounds areB64..2048. Raw norm input bounds still reachB32768.

None of868 generic centered-score outer envelopes proves the frozen exp domain. Exact-real same-input and identity-refresh premises are explicit; no CKKS, refresh perturbation, domain coverage or practical full-model readiness claim is made. Four scale examples do not replace actual runtime metadata replay. The preliminary166-key dense-case union is not an authenticated final matrix inventory.

See [DESIGN.md](DESIGN.md) for derivation and missing owners. Exact numerical sources and all public dependencies are in `result-v1/sources.tar.gz`; no raw trigonometric/gamma arrays are archived. Development logs are preserved and are not additive test evidence. Verify offline with `python verify.py result-v1`.
