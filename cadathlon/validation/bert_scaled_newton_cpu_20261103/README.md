The seven pinned BERT cases passed the unchanged probability, context, block,
sum, and declared-input-domain gates using fixed s=4 normalized Newton updates.
The run completed in 45.10 seconds on CPU 152–155; all seven tracked source hashes
were unchanged. This is a NumPy planning result, not homomorphic execution.

For the five squared-probability normalization rounds, C=K/4 is public and the
candidate stores z≈1/(Cx). It computes x*z before public C and p²*z before final
C, retaining 13 iterations, the original exponential approximation, five
squarings, all domains, and all accuracy gates. The initial mean-exponential
reciprocal remains unchanged. No clipping or value-dependent iteration occurs.

The maximum probability difference from the unchanged circuit was 1.721e-15.
The maximum native context error was 5.260e-10 and native block error 2.496e-10.
All 14,100 intermediate range records passed a declared binary64 diagnostic
allowance of 64 machine epsilons times the interval scale. The 2,316 records with
strict last-bit excursions remain in the raw report; the largest excursion was
4.441e-16. No values were clipped. Observed Newton intermediates ranged from
-2.56 to 4; the declared z ceiling is 8.

The exact recurrence is e_next=e² for e=1-Cxz. After the first update,
0<z≤1/(Cx)≤8, xz≤4/K, and1≤2-Cxz<2. The final probability bound uses the
correlation x=sum(p²): C*p_i²*z≤p_i²/x≤1. Independent loose magnitude bounds do
not establish these relationships. CKKS and refresh errors may break them.

The separate public-domain grid covers K=1…128 and has worst relative residual
1.266e-14. An artificial additive-step sensitivity calculation illustrates
e_next=e²-Cxη; its η is not an encryption or refresh noise model. It provides no
precision guarantee.

The reviewed unfused softmax depth estimate is 232: 227 for the earlier schedule
plus five final public scales. This remains an abstract estimate without a
complete ciphertext schedule. Legacy arithmetic counters in the raw report
list 148 ciphertext products and 87 public affine stages; the separately identified
mean anchor mask would add one affine stage. Full-period 65,536-slot refresh,
unseen-input domains, and the actual encrypted attention/block remain unproved.

`summary.json` contains the command, gates, case results, and provenance.
`report.json.gz` preserves all intermediate observations and original source
hashes. `tracked_sources.tar.gz` contains the exact seven tracked sources plus
the separately hashed public algebra checker. Model weights are not included.
The checker keeps its original paths; replay in a new copy with corresponding
source and pinned-model paths. Earlier factored-map archives remain unchanged.
