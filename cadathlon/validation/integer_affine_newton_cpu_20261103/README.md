# Integer affine and two Newton updates: isolated CPU feasibility

The final attempt passed the unchanged **1e-6 absolute gate** over all 8192
slots at the actual secure **N16384, Q8/P3, dnum4** profile. Actual log2(QP)
is **318.991937256538**, with integer bit length 319 below the 438-bit
128-bit classical ceiling. There is no refresh, TPU run, full inverse,
normalization chain, softmax or model claim.

The isolated helper uses the existing validated HEAdd/HESub facades to form
an integer addition chain and the existing plaintext-add facade for a bias
encoded at the ciphertext's actual scale. It preserves level, scale and NSD,
and removes `_public_*` semantic contracts from the returned wrapper without
changing its input. Integer multiplication is exact residue arithmetic; bias
encoding retains ordinary CKKS rounding error.

Six variants, `(-32,2)`, `(-1,0)`, `(0,2)`, `(1,0)`, `(7,-.125)` and `(32,2)`,
matched an independent residue oracle bit for bit. Their worst decoded error
was **1.36936e-10**. Signed, zero and small inputs were included, alongside
invalid coefficient/bias/metadata and secret-bearing-context rejection.

For the two-update C32 circuit, a public Decimal.sqrt planner solves the scale
recurrence backward from the requested final scale. A real initial plaintext
affine/rescale uses that planned scale; no ciphertext scale is relabeled.
The subsequent product, integer correction and product consume two more
levels. Exact binary64 metadata agrees at every stage:

| Stage | Level | Maximum absolute error against the float circuit |
|---|---:|---:|
| First Newton update | 2 | 2.32462e-10 |
| Denominator × first update | 1 | 7.40201e-12 |
| `2 - 32*product` | 1 | 2.36864e-10 |
| Second Newton update | 0 | 8.93677e-10 |

The initial output scale is 7.1943214263262456e16; the tracked final scale is
7.199603266083226e16. The 258 public denominator points cover the first-round
interval endpoints, interior and inactive midpoint, and repeat across all
physical slots. The error against the true inverse after just two updates is
**3.78723**: this is an operation/scale feasibility proof, not convergence.
All client decoding occurs after the complete fixed server circuit.

The server holds no secret key; both key-generation module aliases and all
server codec entry points are guarded. Input payloads and metadata are
unchanged. Separate client/server objects share one dedicated process; this
does not provide an OS sandbox or an all-input CKKS error bound.

`run-v2/` completed in **10.58 seconds**, peak host RSS **785652 KiB**, with all
**96 exact source files unchanged**, no cleanup errors and observed/reaped
exit 0 (session19609). This is an uncalibrated CPU observation. The byte audit
in `audit.json` checks both archive inventories against before/after hashes.
The final public planner checkpoint passed **10 tests** with unchanged helper
and test hashes (`public-tests.json`, log and XML).

`run-v1/` is preserved unchanged, including observed exit 1 (session96953),
its report/log and exact source archive. Its six residue checks passed, then
an overly broad test assertion failed before client decoding: the existing
generic multiplication copied injected sentinel contracts from its input to
the product. The corrected assertion checks the new affine outputs and final
result, and explicitly records the inherited product attributes. Arithmetic,
scale checks, numerical gates and input-immutability checks did not change.

The runtime snapshot copies core/common Python bytes from the sealed
`cross-optimized-twelve-20261103` checkout; `base_manifest.json` records their
exact identities. Only the helper, driver and public tests are new isolated
files. The candidate has not been promoted and neither ongoing encrypted run
was modified.

Reproduce using an extracted `run-v2/sources.tar.gz` in a fresh directory:

```sh
taskset -c 152-155 env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/jianming_gatech/miniconda3/envs/jaxite/bin/python \
  check_component.py --output fresh-result
```
