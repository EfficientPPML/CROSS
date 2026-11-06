# Fixed-Q public dnum/P audit

Completed public-only parameter/resource audit on 2026-11-04. The existing ordered 100 Q primes, N=131072, depth49 and p60 scale remain fixed. Actual copied `he_params` functions generated P and checked the kernel modulus constraints, primality, hybrid partition coverage and the 128-bit classical uniform-ternary table ceiling for every one of 100 Q prefixes in each of 12 profiles (1,200 checks; 50 CD2 levels per profile). CPU backend initialization occurred. No parameter cache, CKKS context, Polynomial, keys, ciphertexts, model or HE operation was constructed.

Eight focused tests passed, including direct denial probes of all 28 registered package/flat state, codec and key entrypoints. Actual audit invoked none of them. The source-bound wrapper and audit both exited0; session82594 was authoritatively reaped and both child PIDs were absent. The 39 candidate sources and 40 proof sources (including the external wrapper) match before/after/archive. Original source bytes are in the two source tars. The first five-test development checkpoint is retained separately and was superseded by the final eight tests.

| dnum | aux bits | P count | actual QP bitlength | full security | BERT118 raw rotation GiB | Qwen166 raw rotation GiB |
|---:|---:|---:|---:|:---:|---:|---:|
| 5 | 29 | 21 | 3595 | FAIL | 69.71679688 | 98.07617188 |
| 5 | 30 | 21 | 3610 | FAIL | 69.71679688 | 98.07617188 |
| 5 | 31 | 20 | 3607 | FAIL | 69.14062500 | 97.26562500 |
| 6 | 29 | 18 | 3508 | pass | 81.58593750 | 114.77343750 |
| 6 | 30 | 18 | 3521 | pass | 81.58593750 | 114.77343750 |
| 6 | 31 | 17 | 3514 | pass | 80.89453125 | 113.80078125 |
| 7 | 29 | 16 | 3450 | pass | 93.57031250 | 131.63281250 |
| 7 | 30 | 16 | 3462 | pass | 93.57031250 | 131.63281250 |
| 7 | 31 | 15 | 3452 | pass | 92.76367188 | 130.49804688 |
| 8 | 29 | 14 | 3393 | pass | 105.09375000 | 147.84375000 |
| 8 | 30 | 14 | 3403 | pass | 105.09375000 | 147.84375000 |
| 8 | 31 | 13 | 3390 | pass | 104.17187500 | 146.54687500 |

All profiles pass the public kernel/P-coverage/NTT checks. All dnum5 profiles fail the full QP security ceiling of3523; passing smaller Q prefixes does not rescue an insecure full profile. The current dnum8 default aux30 produces exactly the existing14 P primes. dnum6/aux30 gives18 P, actual log2(QP)=3520.5261964180486 and bitlength3521; dnum6/aux31 gives17 P and bitlength3514. The unchanged output levels, input/output scales and Q-prefix ciphertext sizes follow from the fixed Q sequence. At L49 one two-component batch1 ciphertext is104,857,600 raw bytes; at L25 it is54,525,952.

The BERT118 count is checked against the copied completed authentic74-matrix/BSGS/bootstrap inventory. Qwen166 is an explicitly requested prospective count scenario; this audit does not derive its inventory. The raw rotation-pair formula is `2*dnum*N*(Q_count+P_count)*4`. The report separately includes one relinearization pair, the public key, raw ciphertext sizes, and selected L0/L25/L49 counts. Physical memory excludes NPY headers, secret keys, mapped-page residency, duplicate/transient/device buffers, the ciphertext inventory, refresh/NTT controls, JAX/compiler/allocator memory. No128GiB peak-memory, latency or accuracy claim is made.

This verifies actual public parameter functions, not registered `validate_runtime_config` acceptance, a frozen-maxQ compiled partition acceptance check, a compiled kernel execution or encrypted noise behavior. No active profile, file handler, key store or frozen job changed. Any new dnum/P profile would require fresh keys, new context/model/source bindings and genuine rotation, refresh and composed numerical proofs before use. Smaller raw arrays alone do not establish a successful runtime choice.

Run `python verify.py .` for portable artifact/source/arithmetic verification, or add `--current /tmp/bert-fixed-q-dnum-public-audit-20261103` to compare the retained exact candidate and external wrapper. The verifier imports no JAX/HE code and reads no private/model/key/ciphertext payloads.
