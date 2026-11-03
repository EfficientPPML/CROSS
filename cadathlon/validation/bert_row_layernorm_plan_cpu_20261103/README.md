The public full-width BERT LayerNorm schedule passes eight ideal cases: the
unchanged layer-zero attention and MLP inverse-square-root polynomials, each
with one, two, three or four active rows. All 16,384 physical slots are checked
against independent NumPy means, variances and Chebyshev evaluation. Maximum
arithmetic difference is 7.11e-15 at the fixed 1e-9 ideal gate. No keys or
ciphertexts were created.

The layout keeps 768 logical features in each 4096-slot row. Rolling prefix
sums are retained only at row anchors, then broadcast within each row. This
excludes deliberately hostile feature padding from the moments without a
separate input mask. The variance is broadcast across the entire feature block;
inactive rows receive the fixed domain midpoint before inverse-square-root
evaluation. The final learned affine masks unused features and rows. Intermediate
padding is allowed to be nonzero and is not confused with logical output.

Both fixed polynomials consume six levels. The full component consumes eleven:
input 12 → mean 11 → variance 9 → inverse 3 → product 2 → learned affine 1. It uses
42 rotations and 22 distinct rotation keys. The proposed N32768/CD2/depth12,
dnum 8 ring has Q26/P4, actual log2(QP)=853.2333533, integer bit length 854 below
the library's ceiling 881, and exact auxiliary-modulus coverage. Q26 yields
seven actual key-switch partitions, despite the requested dnum 8. Raw rotation
keys total 1,211,105,280 bytes (about 1.13 GiB), plus one 55,050,240-byte
relinearization key. These exclude NTT/control/compiler buffers and are not
peak-memory measurements.

A separate native-only preflight uses the previously fixed public pair 0 and
row positions 0/1 from the pinned full checkpoint. Attention-input variances
range 0.48868–0.97495; MLP-input variances range 3.75288–6.47133. Both fit their
unchanged domains. Independent exact LayerNorm agrees with native PyTorch
within 7.11e-15. Fixed-polynomial output differences from native are
2.65040e-6 / 3.83900e-6, reported as approximation diagnostics rather than a
full-model quality gate. No P2 logit or MRPC gate was evaluated.

[report.json](report.json), [plan.py](plan.py) and [sources.tar.gz](sources.tar.gz)
preserve all 75 exact public-planning sources. The
[native report](native_preflight.json) and [native source archive](native_sources.tar.gz)
preserve its separate five-source snapshot, asset identities and tokenization
hash. Both commands exited zero and their tool sessions were reaped. Public
arithmetic took 0.52697s after imports; native preprocessing took 2.63736s after
top-level imports. These are uncalibrated CPU observations, excluding process
startup, and do not measure homomorphic execution.

The native fixture computes upstream attention and MLP operations to obtain
LayerNorm inputs. That trusted component preprocessing exceeds the allowed
embedding-only P2 client boundary; it is not an implementation of a full-model
stage. Domains remain sampled public contracts, not universal bounds for hidden
inputs. Real scale/noise behavior, encrypted padding leakage, full-period
refresh and full-model composition remain unproved. This ring differs from
other component rings; their ciphertexts cannot simply be chained across rings.

Reproduce the public check with `JAX_PLATFORMS=cpu` and the archived NumPy/JAX
environment, and the native preflight with the pinned local checkpoint and
transformers 4.51.3 environment. Both scripts currently retain their recorded
absolute source/asset locations. No live runtime or release files were changed.
