**This is public parameter and conditional ideal-circuit planning, not an encrypted result or a recommendation to switch all models.** The prospective depth-50/dnum-5 schedule requests refresh bound **32 for BERT cases 4–6**, beyond the bound-16 amplitude of the earlier component proof. Fewer refreshes do not establish accuracy, memory use or speed.

The current depth-49/dnum-4 Qwen test2 proof continues unchanged and requires all three actual refreshes plus its final numerical gates. A separate Qwen2-specific candidate uses depth 50/dnum 5, reducing its ideal schedule from three refreshes to two with bounds **[0.5, 4]**. Its actual generated N=131072 ring has Q102/P20 and log2(QP)=3424.72985, below the 128-bit classical ceiling of 3523. The earlier BERT test1 run used that secure ring without refresh; it does not validate bootstrap at this profile. Any candidate needs a complete refresh at its actual packed period, input amplitude and restored capacity, followed by complete encrypted model gates.

The [original Qwen2 sweep](report.json) preserves all depth-49–54/dnum-4,5,6,8 generated rings, security outcomes, schedules and key sizes. More initial levels are not monotonic in this greedy schedule: Qwen2 needs two ideal refreshes at depth 52 but three at depth 53; both depths have generated secure rings at dnum 6 or 8. The script, raw log and 97 exact source files are archived alongside the report.

The [all-public extension](all_public/report.json) retains the existing leveled profiles for the three shallow fixtures and compares the two deep profiles on the other nine. All 21 conditional ideal graph evaluations pass their original fixture gates and declared polynomial-domain checks. This excludes CKKS rounding and refresh error. Every refresh's level, period, public bound, bound provenance and observed fixture range is recorded; the observed ranges do not replace the public contracts. Its script, log and 120 exact source files are separately archived.

| Deep fixtures | Refresh counts: 49/d4 → 50/d5 | Largest public bound: 49/d4 → 50/d5 |
| --- | --- | --- |
| BERT3 | 5 → 3 | 8 → 16 |
| BERT4 | 5 → 3 | 16 → **32** |
| BERT5–6 | 8 → 7 | 16 → **32** |
| Qwen2 | 3 → 2 | 4 → 4 |
| Qwen3–4 | 11 → 5 | 4 → 16 |
| Qwen5 | 19 → 8 | 4 → 16 |
| Qwen6 | 19 → 8 | 8 → 16 |

Larger key storage is another tradeoff. Qwen2's 57 raw rotation-key pairs grow from 27.61 to 33.96 GiB; BERT5 grows from 109.47 to 134.63 GiB and Qwen6 from 102.69 to 126.29 GiB. These figures exclude evaluation-key arrays, NTT controls and ciphertext payloads. No keys were generated or device execution performed in this planning comparison. No runtime profile was changed.
