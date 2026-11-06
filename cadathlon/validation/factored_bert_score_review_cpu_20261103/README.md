Independent semantic and cost review of the full-period factored BERT score map.

The isolated CPU run passed all 128 forward/inverse maps against independently
indexed oracles with hostile padding, verified exactly-once query/key/head
coverage, and checked explicit rotation-based reduction and broadcast across all
32 independent 2,048-slot chunks. Fixed-softmax comparisons covered sequences
1, 17, and 128 with all-valid, interleaved, and last-key masks. All source hashes
were unchanged during the run. This is NumPy arithmetic, not encrypted execution.

The map uses 30 distinct rotation indices, 2,272 rotation calls and 4,352 public
multiplications per query tile, with nine multiplication levels per direction.
The forward anchor filter could be folded into an adjacent head reduction; that
fusion is not implemented here.

A concrete segmented sum uses seven left rotations, one anchor mask/rescale,
then seven right rotations to broadcast. Seven such reductions add 98 rotations
and seven anchor multiplications per score ciphertext; their indices already
belong to the map union. Inactive denominator groups receive a public one only
after broadcast. Masked keys inside live groups retain the common denominator
and contribute a zero numerator.

The simulator's abstract softmax depth of 226 omits a distinct mean anchor mask
when the input key mask is retained. The illustrated schedule needs 227 levels
before other implementation details, scale alignment, or refresh are accounted
for. Folding public key validity into the score maps could remove that extra
level. Neither figure is a validated homomorphic execution budget.

Uniform probabilities over 128 keys require a reciprocal of 128; the declared
denominator interval permits values approaching 256. Existing bound-16 refresh
evidence does not cover those intermediate values. This layout has period 65,536
and no alias averaging, so period-128 refresh evidence also does not transfer.

Run command:

```
taskset -c 152-155 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/jianming_gatech/miniconda3/envs/jaxite/bin/python /tmp/factored-bert-score-review-20261103/check.py
```

The checker records the original source paths and hashes. Its imported candidate
and coefficient manifest are preserved here as `reviewed_score_layout.py` and
`coefficients_manifest.json`; adjust the two paths in a new copy when replaying.
The original report and exact script remain unchanged in this archive.
