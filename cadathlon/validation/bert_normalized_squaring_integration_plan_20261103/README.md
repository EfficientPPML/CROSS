# Conditional common-ring normalized-squaring plan

This is public-only parameter, level, scale and payload accounting. No CKKS context, key generation, encrypted operation or device backend was invoked. The source-bound planner and five public regression checks passed; all 108 source/evidence bytes match before/archive/after/current verification. The independent transformer-agent review found no concrete level/scale algebra error. The existing full-slot refresh run was left unchanged.

**The present depth49 refresh cannot sustain all of these static inverses.** Its normalized output is level25. Thirteen updates consume25 levels and leave no level for the retained numerator product. Even twelve updates leave an output product at level1, below the bounded refresher's input minimum2. With the current public counts and at least2 levels after each product, K1–37 are schedulable; K38–128 are not. This does not rule out a separately designed inverse that refreshes internally; the current static helper explicitly prohibits that.

## Actual generated profiles

All rows use N131072, CD2, full65536 slots, layout256×512, precision32, first61, uniform ternary, and bootstrap budget4/4. Actual Q/P primes and partition coverage are recorded, with the same 128-bit classical ceiling3523. The raw-key count is224 for the dense-diagonal first-block graph plus106-key full-slot bootstrap union.

| Profile | Q/P | Actual partitions | log2(QP) | Refresh output | Raw rotation store |
|---|---:|---:|---:|---:|---:|
| depth49 / p56 / d4 | 100/24 | 4 | 3490.161581 | 25 | 108.5000 GiB |
| depth52 / p56 / d6 | 106/17 | 6 | 3443.441041 | 28 | 161.4375 GiB |
| depth52 / p60 / d12 | 106/10 | 12 | 3462.663831 | 28 | 304.5000 GiB |
| depth53 / p60 / d14 | 108/9 | 14 | 3492.282217 | 29 | 358.3125 GiB |

The keys alone are not an HBM or peak host-memory estimate. The report separately lists public/evaluation keys, retained ciphertext payloads and legacy BSGS stacks; controls, tables, compiler/executables, key-generation temporaries and allocator peaks are excluded. The224 count covers the described Q/K/V, attention output, FFN, row LayerNorm, routing and reductions, not a complete pooler/classifier interface. Dense diagonal support is a public upper contract, to be checked against the chosen implementation.

## Explicit branch schedule

The fused QK path consumes11 levels: projections1, ciphertext product1, head reduction with gain folded into the existing nine-level routing filter. Initial softmax consumes22 further levels: mean2, masked centering/scaling1, balanced degree8 powers plus masked coefficient sum4, explicit masked exp mean2, six-update C1 inverse11, product1, shared public1/K scale1. Reusing the same binary64 factor1/K in mean and output is required for ideal common-factor normalization. The report adds an exact rounded-initial-affine C1 contraction certificate; it excludes CKKS error.

Each squaring round retains its numerator: input mask1 → square1 → anchor/dummy reduction1 → C32 inverse(2n−1) → retained-square product1 → integer32 restoration0. The14 reduction/broadcast rotations consume no level. The initial probability and/or denominator may be refreshed, but every refresh input must already be at level>=2. The dynamic program minimizes refreshes only within those two allowed refresh locations and requires every final product at level>=2. It does not claim a global optimum over other circuits.

The inverse's final scale target is computed from the saved square's scale and the actual pair dropped by the following product. The first affine genuinely encodes that target; the reported binary64 recurrence never relabels ciphertext scale. The examples assume nominal canonical refresh/mask outputs. Integration must use each actual public metadata scale and recheck the recurrence; the largest planned product deviation from canonical is7.863e-13, below the existing1e-12 tolerance. No new encrypted proof of this composition is claimed.

For K128 the counts are8/9/12/13/13. Output28 needs seven normalization refreshes and ends at level2; output29 needs six and ends at level3. K15 uses8/9/10/10/10, five normalization refreshes for either profile, ending at5 or6. Every K1–128 has a conditional schedule with output28 or29.

From a refreshed final probability, inverse routing9 → V product1 → attention output linear1 → attention LayerNorm11 leaves6 with output28 or7 with output29. That learned normalization result needs a refresh before FFN16. FFN16 plus MLP LayerNorm11 then leaves1 with output28, so an additional pre-LayerNorm refresh is necessary; output29 leaves2 and can refresh for the next block. For K128 this is ten versus eight refreshes per query tile for the first block, including normalization and these continuations. There are eight query tiles at128 tokens. These are operation counts, not runtime estimates; meta10 performs two genuine bootstraps per refresh.

## Unresolved conditions

- The exact C32 inverse relation is z≈1/(32d), where d is the sum of the **same** retained squared probabilities. Multiplying those squares by z before integer32 keeps the ideal product≤1/32. Materializing32z would require a bound up to256 and is not this plan. A denominator refresh perturbs the equality and needs an explicit correlated error budget.
- A1e-3 denominator error at ideal d=1/128 gives a0.128 relative perturbation; ideal normalized mass would become about0.8865 or1.1468 for the two signs. Thus a1e-3 refresh component gate cannot establish the1e-12 ideal mass premise, the1e-6 encrypted component gates, or final model accuracy.
- Attention LayerNorm's observed magnitude49.37 needs a candidate public bound64 for direct refresh. That is an observation, not a bound for every input. The pending full-slot proof uses bound16. Bound64 and the new p60/d12/d14 chains remain unvalidated. Base Q/scale changes from33.4 atp56 to about1.58 atp60; bootstrap adjusts scale internally, but the changed conditioning needs its own numerical proof.
- The separate p60 LayerNorm file proof passed all131072 checked slots at1e-6 on N32768/Q26/P3/d13. It does not transfer to N131072. Existing QK, FFN and LayerNorm ring degrees, exact chains, scales, layouts and key identities cannot be relabeled into a common ciphertext context. RowTile F4096 grows from4 or8 rows to16 rows per tile, and must be validated there.
- The native408 MRPC audit found at least one existing full-model valid-token domain violation in **every example**. Even the layer0 MLP variance domain misses one native value. These fixed domains were not changed here. This conditional level plan therefore is not a valid full-MRPC inference certificate; it does not repair domain coverage or measure approximate trajectories.

`report.json` contains all512 K/profile schedules, exact primes, scale examples and resource details. `inputs/` preserves the public certificates and separately scoped native/precision60 reports. `attempt1/` and `attempt2/` preserve earlier public reports before final scope and initial-C1 certificate additions. Run `python verify_archive.py` without importing crypto to check the final archive. To rerun the planner/tests, extract `sources.tar.gz` into a fresh directory and use CPU-only execution there; live sources are not needed.
