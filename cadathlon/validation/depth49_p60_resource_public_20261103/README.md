# Depth49 p60 output25: public parameter and resource check

The final audit uses the actual copied production prime generator, P-tower generator, security/P-coverage checks and RefreshProfile parameter method. It constructs no CKKS context, keys, ciphertext or HE operation. Broad package import initializes a CPU backend; this is recorded truthfully. Context/key-generation calls and subsequent backend-entry calls were guarded and none occurred. All105 captured sources/input reports match before/after and exact archive bytes. The final public calculation completed in0.4814s.

| Precision / dnum | Q/P limbs | log2(QP) | Inclusive integer bits /3523 |128-bit profile|116 raw rotation keys|
|---|---:|---:|---:|---|---:|
| p56/d4 baseline |100/24|3490.1616|3491|Pass|56.1875GiB|
| p60/d4 |100/26|3757.2399|3758|Reject|57.09375GiB|
| p60/d5 |100/21|3609.3213|3610|Reject|68.5352GiB|
| p60/d6 |100/18|3520.5262|3521|Pass|80.203125GiB|
| p60/d7 |100/16|3461.3055|3462|Pass|91.984375GiB|
| p60/d8 requested |100/14|3402.0803|3403|Pass|103.3125GiB|
| p60/d9 |100/13|3372.4619|3373|Pass|115.2070GiB|
| p60/d10 |100/11|3313.2219|3314|Pass|125.7422GiB|

Every generated P covers its largest exact Q partition. Dnum6 is the lowest accepted choice among the checked4–10 and uses six actual partitions of at most17 Q limbs. Its P product has534bits versus a510bit maximum partition. Dnum8 uses eight partitions of at most13 Q limbs, with415bit P versus390bit maximum partition. The security check uses the integer bit length, not a rounded logarithm. These are parameter checks under the standard classical128-bit uniform-ternary security target, not a new attack analysis.

The116 indices are reconstructed as a set, not assumed from a count: replace the dense linear key group with{1,64,61760,64064}, retain the other56-key graph union, and union106 bootstrap rotations. The resulting list matches the independent archived prospective inventory. The direct graph/bootstrap alternative remains224 keys: p56/d4 costs108.5GiB, p60/d6 costs154.875GiB, and p60/d8 costs199.5GiB. The prospective chained route still needs same-ring numerical evidence; its smaller-ring pass does not establish this full-width noise or performance behavior.

For p60/d6 each raw switch-key pair is708MiB; d8 is912MiB; baseline p56/d4 is496MiB. Add one relinearization pair and100MiB public encryption key to the rotation store. Top level49 ciphertexts remain100MiB, output25 ciphertexts52MiB, eight top row tiles800MiB, and X/Q/K/V with eight tiles each3.125GiB before temporaries. Raw sizes exclude formatted controls, NTT/FFT tables, compiler/JIT, key-generation workspace, allocator overhead and concurrent live values.

At measurement time, /tmp had49,798,008,832 free bytes and /dev/shm650,139,103,232 free bytes. None of the proposed rotation stores fits the current /tmp free space; each fits the raw-key-only tmpfs minimum. This records live free space, not post-cleanup capacity or peak-memory feasibility.

The normalized full-slot output is still25 at depth49 with[4,4] transform budgets, meta10, period65536 and no aliases. Actual p60 Q/P primes/scales differ from the active p56/d4 ring. No refresh accuracy, CKKS noise, common-ring joined graph, HBM or speed result transfers between them. The tracked-inverse and GELU-refresh level plan is conditional; propagated domains, attention-normalized refresh range and composed softmax errors remain separate obligations.

`attempt1` preserves the overly strict preparation guard that rejected CPU backend initialization during import before any profile calculation. `attempt2` preserves a missing typing-symbol error in an optional pure-source cross-check. `pure_check` contains the successful exact-function AST cross-check, whose Q/P/security/resource values match the final production-function run (`pure_crosscheck.json`). Those preparation paths are separate from the final report, and no encrypted job was launched by any of them.

Follow-up: [actual normalization and base-capacity accounting](NORMALIZATION_FOLLOWUP.md) explains why the smaller p60 baseQ/scale ratio alone does not require a largerB: the adapter performs an additional real adjustment before reaching base level0. The original report/source archive above is unchanged.
