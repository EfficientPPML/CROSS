# Secure full-slot factored score routing

One real CKKS forward map and inverse passed on CPU for the worst public
tile/offset pair, **tile 7 / offset 15**, over all **65,536 slots**. This is a
routing component proof, not complete attention, bootstrap, a BERT block or
encrypted full-model inference.

| Check | Output level | Maximum absolute error | Fixed gate |
| --- | ---: | ---: | ---: |
| Forward map, all slots | 9 | 1.6297651e-9 | 1e-6 |
| Inverse against source-mask oracle, all slots | 0 | 2.2142514e-9 | 1e-6 |

The input has 192 public source anchors and hostile nonzero inactive padding
of magnitude 4–8. The oracle checks inactive slots as well as anchors. The
actual profile is N131072, CD2, Q38/P10, dnum4, layout256×512, with actual
log2(QP)=1368.7311337680203 and the library's 128-bit classical security gate.
Scale and NSD1 checks pass through levels 18→9→0. The circuit performs
34 plaintext multiplications and actual composite rescales across 18 path
levels, 18 rotations and 16 additions. Independent branch masks precede
rotation; no source-coordinate mask is silently replaced by a destination mask.

The client generated fresh keys; the separate server context had public keys
only. Both `key_gen` import spellings were guarded against key generation,
and server encryption/decryption methods were forbidden. Only the client
decrypted the completed forward and inverse outputs. These are object and
API boundaries within one process, not an OS sandbox or cross-process protocol.

The authoritative tool session 46728 exited **0** and was reaped. Both the
time wrapper (735608) and Python process (735609) were absent at verification.
Cleanup reported no errors and removed the owned public-key directory.
Internal elapsed time was **9981.8024 seconds**; GNU time reported
**2:46:23** wall time, rounded to one second, and **17,788,432 KiB** peak RSS.
The run used CPU cores 144–147 with four OMP/BLAS threads. Other independently
assigned jobs ran on disjoint affinities during parts of the run; this is not
a controlled optimization timing comparison or a TPU performance result.

[report.json](report.json) and [run.log](run.log) retain the numerical gates,
every phase, actual rings, operation counts, resource use and cleanup outcome.
[sources.tar.gz](sources.tar.gz) contains all **97 exact measured source and
patch files**, whose hashes match before/after records and the isolated files
at archive time. [verification.json](verification.json) records the independent
whitelist verification. No key arrays, ciphertext payloads or private inputs
are archived.

The measured checkout is the earlier isolated combined runtime candidate plus
HEMul scoped-NTT and guarded-dispatch patches. It **predates the later 32-file
runtime promotion**; this archive must not be relabeled as a proof against that
later source set. Exact optional flags and patch bytes are retained. The
[preflight](preflight/report.json) is parameter/storage planning only and has a
different driver hash from before the final exception-publication fix; its
original source archive is preserved separately. The
[adapter check](adapter_semantics.json) establishes exact NumPy schedule parity,
not an additional encrypted run.

To reproduce, extract `sources.tar.gz` into a fresh directory. It includes the
driver, layout, isolated checkout and input patches; do not apply the patches
again. With the archived JAX/NumPy versions available, run from that directory:

```sh
/usr/bin/time -v env JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 144-147 \
  /path/to/jaxite/python check_secure_routing.py --output /fresh/output/path
```

The fresh output path must not exist. The driver performs its own disk-space
preflight and owns only its freshly created key directory. No healthy-run
deadline was imposed. Numerical success for this one map does not validate all
128 encrypted maps, score formation, full-period refresh, polynomial softmax,
model accuracy or a practical end-to-end runtime.
