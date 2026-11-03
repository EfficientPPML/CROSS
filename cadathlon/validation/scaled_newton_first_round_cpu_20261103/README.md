# Complete first-round scaled Newton inverse, CPU

The genuine encrypted eight-update first-round inverse **passed** the fixed
absolute error gate of **1e-6**. The maximum error across all eight updates
against the corresponding binary64 circuit was **2.3182593e-8**. The final
result differed from the true inverse `4 / (128 * x)` by **2.3083293e-8**.
The process returned exit **0**; its wrapper and child were subsequently
confirmed absent. Cleanup completed without errors.

This is one complete first-round inverse, with no bootstrap. It does not
validate the remaining four normalization rounds, complete softmax, a BERT
block, all-input CKKS error bounds, or accelerator performance.

The [raw report](run/report.json) contains the actual primes, all inputs,
per-update checks, phase times, environment and source hashes. The
[independent consistency audit](summary.json) verifies these against both
exact source archives and the captured runtime origin.

| Property | Observed value |
| --- | --- |
| Ring | N65536, CD2, depth24, Q50/P7, dnum8 |
| Actual log2(QP) | 1613.9436645924673 |
| Integer QP bits / archived classical 128-bit ceiling | 1614 / 1747 |
| Public interval | [0.00390625, 0.041910546372478145] |
| Inputs | 258 distinct endpoints, adjacent values, linear/geometric grid points and inactive public midpoint |
| Decoded slots checked per update | 32768 |
| Newton parameters | K128, s4, C32, eight updates |
| Output levels | 23, 20, 17, 14, 11, 8, 5, 2 |
| Noise scale degree | 1 throughout |
| Exact binary64 circuit versus true inverse | 8.8817842e-16 |
| Total CPU process time | 2359.2458 seconds |
| Peak host RSS | 11899048 KiB |

The initial estimate is public and derived from the declared interval. The
first update is evaluated as an affine polynomial in encrypted `x`. Each
subsequent update computes `x*z`, then `2 - 32*(x*z)`, then multiplies by `z`,
using real ciphertext operations and rescaling. Actual scale recurrence is
checked against the dropped modulus pairs. The decoded final maximum was
approximately 8.0000000187; this small noise excursion above the ideal bound
8 is retained in the report.

The server context, parameter cache and parameter dictionary contain no
secret key. Both imported key-generation module names are guarded, and server
encryption/decryption methods are prohibited. Client decryption starts only
after all eight updates finish. Client and server are separate objects in
one dedicated process; this is not an OS isolation proof.

The copied runtime includes the combined candidate, HEMul scoped NTT
extension and guarded dispatch policy. It is not the unchanged live runtime.
It predates the optimized adapter wiring, rescale provider and lazy BConv
follow-ups. [Origin metadata and patches](origin/origin_report.json) make
this distinction explicit: 92 runtime files match the source snapshot byte
for byte, and the actual run independently verifies all 95 captured files
before and after execution.

The timing and host RSS describe this CPU development run on cores152–155
with JAX's CPU backend and one OMP/BLAS thread. Other development work was
concurrent; these are not calibrated speed or accelerator-memory claims.
Saved checkpoint ciphertexts permit every update to be checked after the
complete server circuit and contribute to the reported memory use.

`preflight/` preserves the earlier parameter/resource-only result separately
from the actual encrypted result in `run/`. `sources.tar.gz` in each directory
contains exact source bytes captured before that attempt. The initially
failed archive verifier log is retained as
`verification_initial_failure.log`: its AST parser mistakenly called a list
as a function. The corrected verifier passed without rerunning or altering
the encrypted proof.

To audit the existing evidence, run `python verify_result.py`. To reproduce
the arithmetic, extract `run/sources.tar.gz` into a fresh directory, then run
its driver with a fresh output path:

```sh
taskset -c 152-155 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/jianming_gatech/miniconda3/envs/jaxite/bin/python \
  check_first_round.py --output /tmp/first-round-new-attempt
```

Fresh encryption randomness can change numerical errors; the gate remains
1e-6. Neither the archive audit nor this documentation asserts a fresh
arithmetic rerun.
